# Yolo11 기반에 Bestframe 데모

## 개발환경
1. python 3.9
2. anaconda 환경 권장

## 환경설정
윈도우는 직접 환경 설정을 참고하기 바람

#### 리눅스 (우분투 권장)
```
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
conda remove --name myenv --all -y
conda create --name myenv python=3.9 -y
conda activate myenv
conda install -c conda-forge tensorflow -y
```

## Python  

```sh
pip install torch torchvision torchaudio opencv-python ultralytics onvif-zeep psycopg2 asyncio aiohttp
```

기초 예제 코드
```sh
python3 video_analysis.py 
```
