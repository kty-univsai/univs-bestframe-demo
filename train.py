from ultralytics import YOLO

# 1) 이미 학습된 모델 또는 사전학습 모델 로드
model = YOLO("./yolo11l_weapon.pt")  # 또는 "yolov8s.pt"

# 2) 새 데이터셋으로 추가 학습
model.train(
    data="./dataset.yaml",  # 새로 만든 data.yaml 파일 경로
    epochs=30,
    imgsz=640,
    batch=8,
    device='1',  # 특정 GPU 사용 시 (예: '0' or 'cpu')
    # 기타 하이퍼파라미터 설정 가능
)