import torch
import cv2
import time
import math
from ultralytics import YOLO
from onvif import ONVIFCamera
from db_operations import insert_frame
from db_pool import close_connection_pool  # 종료 시 커넥션 풀 닫기



frame_skip = 30

ip = '192.168.0.100'  # 카메라 IP
port = 80             # ONVIF 서비스 포트 (보통 80, 8080 등)
user = 'admin'        # ONVIF 계정
passwd = 'admin'      # ONVIF 계정 비밀번호



def is_overlapping_with_center_offset(rect1, rect2, offset):
    # rect1의 중심좌표 계산
    x1_c = (rect1[0] + rect1[2]) / 2
    y1_c = (rect1[1] + rect1[3]) / 2

    # rect2의 중심좌표 계산
    x2_c = (rect2[0] + rect2[2]) / 2
    y2_c = (rect2[1] + rect2[3]) / 2

    # 두 사각형 중심 간 거리 계산 (유클리드 거리)
    distance = math.sqrt((x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2)
    return distance 

def main():
    model = YOLO('yolo11n.pt')  # COCO 사전 학습
    model.overrides['conf'] = 0.25  # confidence threshold 설정

    if torch.cuda.is_available():
        model.to('cuda')
        print("GPU를 사용합니다.")
    else:
        print("GPU가 없어서 CPU로 실행됩니다.")
    
    # 3. 웹캠 열기 (0번 장치)
    cap = cv2.VideoCapture("sample.mp4")
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
            
            for r in results:
                object_idx = 1
                
                persons = []
                cars = []
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]  # COCO 클래스명
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)  # 바운딩 박스 좌표
                 
                    # "person", "car"만 필터링
                    if label in ["person", "car"]:
                        x1, y1, x2, y2 = xyxy

                        if label == "person":
                            persons.append({
                                "object_type": label,
                                "rect": (x1, y1, x2, y2)
                            })

                        if label == "car":
                            cars.append({
                                "object_type": label,
                                "rect": (x1, y1, x2, y2)
                            })

                        cropped_frame = clone_frame[y1:y2, x1:x2]
                        frameshot_filename = "./frameshots/frame" + str(current_frame) + "_" + label + "_" + str(object_idx)  + ".jpg"
                        cv2.imwrite(frameshot_filename, cropped_frame)            

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                        cv2.putText(frame, f"{label}", 
                                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, (0,255,0), 2)                    

                    object_idx += 1   
                
                for person in persons:
                    for car in cars:
                        print("Person overlap + " + str(is_overlapping_with_center_offset(person['rect'], car['rect'], 100)))



        if raw_frame % frame_skip == 0:
            filename = "./frames/frame_" + str(current_frame) + ".jpg"
            cv2.imwrite(filename, frame)
            current_frame += 1

        cv2.imshow("YOLOv11n - Person & Car", frame)        
        raw_frame += 1
       
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
