# 1) Miniconda 베이스 이미지 사용
FROM continuumio/miniconda3:latest

USER root
RUN apt-get update && \
    apt-get install -y \ 
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpq-dev gcc

# 2) 작업 디렉터리 설정
WORKDIR /app

# 3) environment.yml 복사
COPY environment.yml .

# 4) conda 환경 생성
#    --yes: 설치시 사용자 입력 없이 바로 진행
#    --name 옵션을 써도 되지만, environment.yml에 name이 있을 경우 자동으로 해당 이름으로 설치됨
RUN conda env create -f environment.yml --yes
ENV PATH /opt/conda/envs/myenv/bin:$PATH

RUN pip install python-dotenv

# 5) SHELL을 변경해서 docker 컨테이너 안에서 conda activate를 사용 가능하게 하거나
#    혹은 아래처럼 'conda run'을 통해서 환경을 실행하도록 할 수 있음.
# SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# 6) 프로젝트 코드 복사
COPY . .

# 7) 컨테이너 실행 시 명령
#    conda run --no-capture-output -n myenv python your_script.py
CMD ["python", "snapshot_proc_person.py"]
