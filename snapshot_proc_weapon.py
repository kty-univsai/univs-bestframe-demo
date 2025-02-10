import torch
import cv2
import math
import aiohttp
import asyncio
import json
import numpy as np

from ultralytics import YOLO
from onvif_snapshot import get_onvif_snapshot


SERVER_URL = "http://localhost:7800"
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJvcmdfaWQiOiIyNSIsIm9yZ19ncm91cF9pZCI6ImRlNTNhNzIyLTkzNDMtNDllMC1hMmVlLTQ0ZWFjNjlhZmU1NiIsIm5hbWUiOiJ1bml2cyIsImVtYWlsIjoia3R5QHVuaXZzLmFpIiwiaWF0IjoxNzM2Mzk1NDc5LCJleHAiOjM0NzI3OTA5NTh9.XzxfCy3V0wc8MpYO6m6LvT98UESKOrMXayITTJdncpA"

ip = "192.168.0.96"         # 카메라 IP
port = 80                   # ONVIF 서비스 포트 (기본값 80)
user = "admin"              # ONVIF 사용자
password = "dbslqjtm1"      # ONVIF 비밀번호


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
        
async def send_weapon_async(image_data, rect):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
    async with aiohttp.ClientSession(headers=headers) as session:

        async with session.post(SERVER_URL + "/bestframe/weapon", data={'image': image_data}) as response:
            if response.status == 200:
                json_data = await response.json()
                if json_data.get("code") == "success":
                    return {
                        "json": json_data,
                        "type": "weapon",
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

    aa = torch.load("best.pt", weights_only=False)    
    model = YOLO(aa)  # COCO 사전 학습
    print("sex11")
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
        weapons = []
        humans = []
        for r in results:                                    
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]  # COCO 클래스명
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  # 바운딩 박스 좌표                
                
                if label in ["person"]:
                    x1, y1, x2, y2 = xyxy

                    cropped_frame = frame[y1:y2, x1:x2]
                    _, img_encoded = cv2.imencode('.jpg', cropped_frame)

                    if label == "person":
                        tasks.append(send_human_async(img_encoded.tobytes(), (x1, y1, x2, y2)))                              
            
        # for wr in weapon_results:            
        #     for box in r.boxes:
        #         cls_id = int(box.cls[0])
        #         conf = float(box.conf[0])
        #         label = model.names[cls_id] 
        #         xyxy = box.xyxy[0].cpu().numpy().astype(int)  # 바운딩 박스 좌표                


        #         # "person", "car"만 필터링
        #         if label in ["person", "car"]:
        #             x1, y1, x2, y2 = xyxy

        #             cropped_frame = frame[y1:y2, x1:x2]
        #             _, img_encoded = cv2.imencode('.jpg', cropped_frame)
                                        
        #             tasks.append(send_weapon_async(img_encoded.tobytes(), (x1, y1, x2, y2)))

            
        results = await asyncio.gather(*tasks)
        valid_results = list(filter(None, results))

        # 결과 출력
        for res in valid_results:
            if res['type'] == 'weapon':
                weapons.append(res)
            elif res['type'] == 'human':
                humans.append(res) 
        
        overlap_trigger = False
        human_metadata = []
        weapon_metadata = []

        for human in humans:
            overlap_weapon = [] 
            
            h1 = {
                "id": human['json'].get("data", {}).get("id", -1)
            }
            if human['json'].get("data", {}).get("faceSamples", {}) != None:
                h1["face_image_path"] = human['json'].get("data", {}).get("faceSamples", {}).get("filePath", "")
            if human['json'].get("data", {}).get("bodySamples", {}) != None:
                h1["body_image_path"] = human['json'].get("data", {}).get("bodySamples", {}).get("filePath", "")
            
            for weapon in weapons:
                if is_overlap(human['rect'], weapon['rect']):
                    overlap_trigger = True
                    overlap_weapon.append(weapon['json'].weapon("data", {}).get("id")) 
            h1["overlap"] = overlap_weapon
            h1["rect"] = [human['rect'][0],human['rect'][1], human['rect'][2], human['rect'][3]]
            human_metadata.append(h1)
        
        for weapon in weapons:
            v1 = {
                "id": weapon['json'].get("data", {}).get("id", -1),
                "image_path": weapon['json'].get("data", {}).get("samples", {}).get("filePath", ""),
                "rect": [weapon['rect'][0],weapon['rect'][1], weapon['rect'][2], weapon['rect'][3]]
            }
            weapon_metadata.append(v1)
        
        metadata= {
            "human": human_metadata,
            "weapon": weapon_metadata,
            "events": {
                "vehicle_overlap": overlap_trigger
            },
            "width": frame.shape[1],
            "height": frame.shape[0]
        }

        _, img_encoded = cv2.imencode('.jpg', frame)
        # 비동기 HTTP 요청 실행 (서버 응답을 기다리지 않음)
        await send_frame_async(img_encoded.tobytes(), metadata)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
            break

    cv2.destroyAllWindows()


asyncio.run(main())