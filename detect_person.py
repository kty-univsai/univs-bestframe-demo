import torch
import cv2
import time
import math
import aiohttp
import asyncio
import numpy as np
from ultralytics import YOLO
from onvif import ONVIFCamera
from db_operations import insert_frame
from db_pool import close_connection_pool  # 종료 시 커넥션 풀 닫기

SERVER_URL = "http://localhost:7800"
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJvcmdfaWQiOiIyNSIsIm9yZ19ncm91cF9pZCI6ImRlNTNhNzIyLTkzNDMtNDllMC1hMmVlLTQ0ZWFjNjlhZmU1NiIsIm5hbWUiOiJ1bml2cyIsImVtYWlsIjoia3R5QHVuaXZzLmFpIiwiaWF0IjoxNzM2Mzk1NDc5LCJleHAiOjM0NzI3OTA5NTh9.XzxfCy3V0wc8MpYO6m6LvT98UESKOrMXayITTJdncpA"

# 비동기 HTTP 요청 함수
async def send_frame_async(image_data):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(SERVER_URL, data={'image': image_data}) as response:
            result = await response.text()
            return f"📡 서버 응답: {response.status}, {result}"

# 비동기 HTTP 요청 함수
async def send_human_async(image_data, rect):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(SERVER_URL+ "/event/generate", data={'image': image_data}) as response:
            result = await response.text()
            return f"📡 서버 응답: {response.status}, {result}"
        
# 비동기 HTTP 요청 함수
async def send_vehicle_async(image_data, rect):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(SERVER_URL + "/bestframe/vehicle", data={'image': image_data}) as response:
            result = await response.text()
            return f"📡 서버 응답: {response.status}, {result}"        


frame_skip = 30

ip = '192.168.0.100'  # 카메라 IP
port = 80             # ONVIF 서비스 포트 (보통 80, 8080 등)
user = 'admin'        # ONVIF 계정
passwd = 'admin'      # ONVIF 계정 비밀번호



def is_overlapping_with_center_offset(rect1, rect2):
    # rect1의 중심좌표 계산
    x1_c = (rect1[0] + rect1[2]) / 2
    y1_c = (rect1[1] + rect1[3]) / 2

    # rect2의 중심좌표 계산
    x2_c = (rect2[0] + rect2[2]) / 2
    y2_c = (rect2[1] + rect2[3]) / 2

    vehicle_w = rect2[2] - rect2[0] / 2

    # 두 사각형 중심 간 거리 계산 (유클리드 거리)
    distance = math.sqrt((x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2)
    return distance < vehicle_w 

async def main():
    model = YOLO('yolo11n.pt', verbose=False)  # COCO 사전 학습
    model.overrides['conf'] = 0.25  # confidence threshold 설정

    if torch.cuda.is_available():
        model.to('cuda')
        print("GPU를 사용합니다.")
    else:
        print("GPU가 없어서 CPU로 실행됩니다.")
    
    # 3. 웹캠 열기 (0번 장치)
    cap = cv2.VideoCapture("sample.webm")
    if not cap.isOpened():
        print(".")
        return

    raw_frame = 0
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(".")
            break
        
        if raw_frame % frame_skip == 0:
            results = model.predict(frame)
            clone_frame = frame.copy()
            
            tasks = []            
            vehicles = []
            humans = []
            for r in results:
                object_idx = 1
                            
            
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]  # COCO 클래스명
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)  # 바운딩 박스 좌표
                 
                    # "person", "car"만 필터링
                    if label in ["person", "car"]:
                        x1, y1, x2, y2 = xyxy

                        # if label == "person":
                        #     persons.append({
                        #         "object_type": label,
                        #         "rect": (x1, y1, x2, y2)
                        #     })

                        # if label == "car":
                        #     cars.append({
                        #         "object_type": label,
                        #         "rect": (x1, y1, x2, y2)
                        #     })


                        cropped_frame = clone_frame[y1:y2, x1:x2]
                        _, img_encoded = cv2.imencode('.jpg', cropped_frame)
                        
                        if label == "car":
                            tasks.append(send_vehicle_async(img_encoded.tobytes(), (x1, y1, x2, y2)))
                        if label == "person":
                            tasks.append(send_human_async(img_encoded.tobytes(), (x1, y1, x2, y2)))


                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                        cv2.putText(frame, f"{label}", 
                                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, (0,255,0), 2)                    

                    object_idx += 1   
                
                # for person in persons:
                #     for car in cars:
                #         print("Person overlap + " + str(is_overlapping_with_center_offset(person['rect'], car['rect'])))
            results = await asyncio.gather(*tasks)

            # 결과 출력
            for res in results:
                print(res)
        


        if raw_frame % frame_skip == 0:
            # filename = "./frames/frame_" + str(current_frame) + ".jpg"
            _, img_encoded = cv2.imencode('.jpg', frame)
            # 비동기 HTTP 요청 실행 (서버 응답을 기다리지 않음)
            send_frame_async(img_encoded.tobytes())

            current_frame += 1

        # cv2.imshow("YOLOv11n - Person & Car", frame)        
        raw_frame += 1
       
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
            break

    cap.release()
    cv2.destroyAllWindows()

asyncio.run(main())