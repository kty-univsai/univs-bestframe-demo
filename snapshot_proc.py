import torch
import cv2
import time
import math
import requests
import asyncio
import json
import numpy as np
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from onvif import ONVIFCamera
from db_operations import insert_frame
from db_pool import close_connection_pool  # 종료 시 커넥션 풀 닫기

SERVER_URL = "http://localhost:7800"
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJvcmdfaWQiOiIyNSIsIm9yZ19ncm91cF9pZCI6ImRlNTNhNzIyLTkzNDMtNDllMC1hMmVlLTQ0ZWFjNjlhZmU1NiIsIm5hbWUiOiJ1bml2cyIsImVtYWlsIjoia3R5QHVuaXZzLmFpIiwiaWF0IjoxNzM2Mzk1NDc5LCJleHAiOjM0NzI3OTA5NTh9.XzxfCy3V0wc8MpYO6m6LvT98UESKOrMXayITTJdncpA"

def convert_to_native_types(data):
    if isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, np.int64):
        return int(data)  # Convert numpy.int64 to Python int
    else:
        return data


async def send_frame_async(image_data, metadata):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    metadata_native = convert_to_native_types(metadata)
    json_string = json.dumps(metadata_native)
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(SERVER_URL + "/bestframe/frame", data={'image': image_data, 'metadata': json_string}) as response:
            if response.status == 200:
                json_data = await response.json()
                return {
                    "json": json_data,
                    "type": "frame"
                }                        
            else:
                return None

async def send_human_async(image_data, rect):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(SERVER_URL+ "/event/generate", data={'image': image_data}) as response:
            if response.status == 200:
                json_data = await response.json()
                if json_data.get("code") == "success":
                    return {
                        "json": json_data,
                        "type": "human",
                        "rect": rect
                    }
                else:
                    return None    
            else:
                return None
        
async def send_car_async(image_data, rect):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(SERVER_URL + "/bestframe/car", data={'image': image_data}) as response:
            if response.status == 200:
                json_data = await response.json()
                if json_data.get("code") == "success":
                    return {
                        "json": json_data,
                        "type": "car",
                        "rect": rect
                    }
                else:
                    return None        
            else:
                return None

image_url = "http://admin:dbslqjtm!@192.168.0.109/stw-cgi/video.cgi?msubmenu=snapshot&action=view"

def is_overlapping_with_center_offset(rect1, rect2):
    # rect1의 중심좌표 계산
    x1_c = (rect1[0] + rect1[2]) / 2
    y1_c = (rect1[1] + rect1[3]) / 2

    # rect2의 중심좌표 계산
    x2_c = (rect2[0] + rect2[2]) / 2
    y2_c = (rect2[1] + rect2[3]) / 2

    car_w = rect2[2] - rect2[0] / 2

    # 두 사각형 중심 간 거리 계산 (유클리드 거리)
    distance = math.sqrt((x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2)

    return distance < car_w 

# 웹에서 이미지 불러오는 함수
def load_image_from_url(url):
    response = requests.get(url)
    image_array = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# 이미지 전처리
def preprocess_image(image, input_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = cv2.resize(image, input_size) 
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
    image = transform(image)
    image = image.unsqueeze(0)  
    return image

async def main():
    model = YOLO('yolo11n.pt', verbose=False)  # COCO 사전 학습
    model.overrides['conf'] = 0.25  # confidence threshold 설정

    if torch.cuda.is_available():
        model.to('cuda')
        print("GPU를 사용합니다.")
    else:
        print("GPU가 없어서 CPU로 실행됩니다.")    

    while True:
 
        frame = load_image_from_url(image_url)

        results = model(frame)
        clone_frame = frame.copy()
        
        tasks = []            
        cars = []
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


                    cropped_frame = frame[y1:y2, x1:x2]
                    _, img_encoded = cv2.imencode('.jpg', cropped_frame)
                    
                    if label == "car":
                        tasks.append(send_car_async(img_encoded.tobytes(), (x1, y1, x2, y2)))
                    if label == "person":
                        tasks.append(send_human_async(img_encoded.tobytes(), (x1, y1, x2, y2)))


                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                    # cv2.putText(frame, f"{label}", 
                    #             (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    #             0.6, (0,255,0), 2)                    

                object_idx += 1   
            
            
        results = await asyncio.gather(*tasks)
        valid_results = list(filter(None, results))

        # 결과 출력
        for res in valid_results:
            if res['type'] == 'car':
                cars.append(res)
            elif res['type'] == 'human':
                humans.append(res) 
        
        overlap_trigger = False
        human_metadata = []
        car_metadata = []

        for human in humans:
            overlap_car = [] 
            
            h1 = {
                "id": human['json'].get("data", {}).get("id", -1)
            }
            if human['json'].get("data", {}).get("faceSamples", {}) != None:
                h1["face_image_path"] = human['json'].get("data", {}).get("faceSamples", {}).get("filePath", "")
            if human['json'].get("data", {}).get("bodySamples", {}) != None:
                h1["body_image_path"] = human['json'].get("data", {}).get("bodySamples", {}).get("filePath", "")
            
            for car in cars:
                if is_overlapping_with_center_offset(human['rect'], car['rect']):
                    overlap_trigger = True
                    overlap_car.append(car['json'].get("data", {}).get("id")) 
            h1["overlap"] = overlap_car
            h1["rect"] = [human['rect'][0],human['rect'][1], human['rect'][2], human['rect'][3]]
            human_metadata.append(h1)
        
        for car in cars:
            v1 = {
                "id": car['json'].get("data", {}).get("id", -1),
                "image_path": car['json'].get("data", {}).get("samples", {}).get("filePath", ""),
                "rect": [car['rect'][0],car['rect'][1], car['rect'][2], car['rect'][3]]
            }
            car_metadata.append(v1)
        
        metadata= {
            "human": human_metadata,
            "car": car_metadata,
            "events": {
                "vehicle_overlap": overlap_trigger
            },
            "width": frame.shape[1],
            "height": frame.shape[0]
        }

        _, img_encoded = cv2.imencode('.jpg', frame)
        # 비동기 HTTP 요청 실행 (서버 응답을 기다리지 않음)
        await send_frame_async(img_encoded.tobytes(), metadata)

        if cv2.waitKey(1000) & 0xFF == 27:  # ESC 종료
            break

    cv2.destroyAllWindows()

asyncio.run(main())