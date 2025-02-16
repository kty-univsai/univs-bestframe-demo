import torch
import cv2
import math
import aiohttp
import asyncio
import json
import numpy as np
import os
from dotenv import load_dotenv
from ultralytics import YOLO
from onvif_snapshot import get_onvif_snapshot

# .env 파일 로드
load_dotenv()

SERVER_URL = os.getenv("API_SERVER_URL")
BEARER_TOKEN = os.getenv("API_TOKEN")

ip = os.getenv("CCTV_IP")          # 카메라 IP
port = 80                          # ONVIF 서비스 포트 (기본값 80)
user = os.getenv("CCTV_ID")        # ONVIF 사용자
password = os.getenv("CCTV_PW")    # ONVIF 비밀번호


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
        

def is_overlap(boxA, boxB):
    Ax1, Ay1, Ax2, Ay2 = boxA
    Bx1, By1, Bx2, By2 = boxB

    inter_x1 = max(Ax1, Bx1)
    inter_y1 = max(Ay1, By1)
    inter_x2 = min(Ax2, Bx2)
    inter_y2 = min(Ay2, By2)

    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
        return True
    return False


async def main():

    model = YOLO('yolo11l.pt', verbose=False)  # COCO 사전 학습
    model.overrides['conf'] = 0.25  # confidence threshold 설정
    model.overrides['imgsz']=1024

    if torch.cuda.is_available():
        model.to('cuda')
        print("GPU를 사용합니다.")
    else:
        print("GPU가 없어서 CPU로 실행됩니다.")    

    while True:
        frame = get_onvif_snapshot(ip, port, user, password)
        results = model(frame)
    
        tasks = []            
        humans = []
        for r in results:
            object_idx = 1
                        
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]  # COCO 클래스명
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  # 바운딩 박스 좌표                


                # "person", "car"만 필터링
                if label in ["person"]:
                    x1, y1, x2, y2 = xyxy

                    cropped_frame = frame[y1:y2, x1:x2]
                    _, img_encoded = cv2.imencode('.jpg', cropped_frame)

                    if label == "person":
                        tasks.append(send_human_async(img_encoded.tobytes(), (x1, y1, x2, y2)))              

                object_idx += 1   
            
            
        results = await asyncio.gather(*tasks)
        valid_results = list(filter(None, results))

        # 결과 출력
        for res in valid_results:
            if res['type'] == 'human':
                humans.append(res) 
        
        overlap_trigger = False
        human_metadata = []

        for human in humans:                        
            h1 = {
                "id": human['json'].get("data", {}).get("id", -1)
            }
            if human['json'].get("data", {}).get("faceSamples", {}) != None:
                h1["face_image_path"] = human['json'].get("data", {}).get("faceSamples", {}).get("filePath", "")
            if human['json'].get("data", {}).get("bodySamples", {}) != None:
                h1["body_image_path"] = human['json'].get("data", {}).get("bodySamples", {}).get("filePath", "")
            
            h1["rect"] = [human['rect'][0],human['rect'][1], human['rect'][2], human['rect'][3]]
            human_metadata.append(h1)
                
        metadata= {
            "human": human_metadata,            
            "events": {
                "vehicle_overlap": overlap_trigger
            },
            "width": frame.shape[1],
            "height": frame.shape[0]
        }

        _, img_encoded = cv2.imencode('.jpg', frame)
        # 비동기 HTTP 요청 실행 (서버 응답을 기다리지 않음)
        await send_frame_async(img_encoded.tobytes(), metadata)

        if cv2.waitKey() & 0xFF == 27:  # ESC 종료
            break

    cv2.destroyAllWindows()


asyncio.run(main())