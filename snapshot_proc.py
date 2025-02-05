import torch
import cv2
import math
import aiohttp
import asyncio
import json
import numpy as np
import ultimateAlprSdk

from ultralytics import YOLO
from onvif_snapshot import get_onvif_snapshot


SERVER_URL = "http://localhost:7800"
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJvcmdfaWQiOiIyNSIsIm9yZ19ncm91cF9pZCI6ImRlNTNhNzIyLTkzNDMtNDllMC1hMmVlLTQ0ZWFjNjlhZmU1NiIsIm5hbWUiOiJ1bml2cyIsImVtYWlsIjoia3R5QHVuaXZzLmFpIiwiaWF0IjoxNzM2Mzk1NDc5LCJleHAiOjM0NzI3OTA5NTh9.XzxfCy3V0wc8MpYO6m6LvT98UESKOrMXayITTJdncpA"

ip = "192.168.0.232"     # 카메라 IP
port = 80               # ONVIF 서비스 포트 (기본값 80)
user = "admin"          # ONVIF 사용자
password = "dbslqjtm!2"      # ONVIF 비밀번호


JSON_CONFIG = {
    "debug_level": "info",
    "debug_write_input_image_enabled": False,
    "debug_internal_data_path": ".",
    
    "num_threads": -1,
    "gpgpu_enabled": True,
    "max_latency": -1,

    "klass_vcr_gamma": 1.5,
    
    "detect_roi": [0, 0, 0, 0],
    "detect_minscore": 0.1,

    "car_noplate_detect_min_score": 0.8,
    
    "pyramidal_search_enabled": True,
    "pyramidal_search_sensitivity": 0.28,
    "pyramidal_search_minscore": 0.3,
    "pyramidal_search_min_image_size_inpixels": 800,
    
    "recogn_rectify_enabled": True,
    "recogn_minscore": 0.3,
    "recogn_score_type": "min",

    "assets_folder": "/home/bearkim/samples/ultimateALPR-SDK/assets",
    "charset": "korean", 
    "openvino_enabled": False
}


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
        
async def send_car_async(image_data, image_byte, rect):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        
        result = ultimateAlprSdk.UltAlprSdkEngine_process(
                    0,
                    image_byte, # type(x) == bytes
                    width,
                    height,
                    0, # stride
                    1 # exifOrientation (already rotated in load_image -> use default value: 1)
        )        


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
    # LPR init
    ultimateAlprSdk.UltAlprSdkEngine_init(json.dumps(JSON_CONFIG))

    model = YOLO('yolo11m.pt', verbose=False)  # COCO 사전 학습
    model.overrides['conf'] = 0.25  # confidence threshold 설정

    if torch.cuda.is_available():
        model.to('cuda')
        print("GPU를 사용합니다.")
    else:
        print("GPU가 없어서 CPU로 실행됩니다.")    

    while True:
        frame = get_onvif_snapshot(ip, port, user, password)
        results = model(frame)
    
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
                        tasks.append(send_car_async(img_encoded.tobytes(), cropped_frame.tobytes(), (x1, y1, x2, y2)))
                    if label == "person":
                        tasks.append(send_human_async(img_encoded.tobytes(), (x1, y1, x2, y2)))              

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
                if is_overlap(human['rect'], car['rect']):
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

        if cv2.waitKey() & 0xFF == 27:  # ESC 종료
            break

    cv2.destroyAllWindows()
    ultimateAlprSdk.UltAlprSdkEngine_deInit()

asyncio.run(main())