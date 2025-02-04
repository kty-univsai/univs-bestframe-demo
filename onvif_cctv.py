import requests
import cv2
import numpy as np
from onvif import ONVIFCamera
from requests.auth import HTTPDigestAuth

def get_onvif_snapshot(ip, port, user, password, profile_index=0):
    """
    ONVIF 카메라에 접속하여 특정 프로필의 스냅샷 이미지를 가져옵니다.
    - ip, port: 카메라 IP와 ONVIF 포트 (기본 80 혹은 8899 등)
    - user, password: ONVIF 사용자 계정
    - profile_index: 사용할 프로필 인덱스 (기본 0번)
    """
    # 1) ONVIFCamera 객체 생성
    camera = ONVIFCamera(ip, port, user, password)
    
    # 2) Media 서비스 생성 → 프로필 목록 가져오기
    media_service = camera.create_media_service()
    profiles = media_service.GetProfiles()

    if not profiles:
        raise ValueError("프로필이 없습니다. 카메라 설정을 확인하세요.")

    # 사용할 프로필 선택 (기본 0번)
    profile_token = profiles[profile_index].token

    # 3) 스냅샷 URL 얻기
    snapshot_uri = media_service.GetSnapshotUri({'ProfileToken': profile_token})
    snapshot_url = snapshot_uri.Uri  # 실제 스냅샷 URL

    # 4) HTTP 요청을 통해 스냅샷 이미지 가져오기
    # 일부 카메라는 BasicAuth, 일부는 DigestAuth 필요
    # 여기서는 DigestAuth 예시
    response = requests.get(snapshot_url, auth=HTTPDigestAuth(user, password), stream=True, verify=False)
    # --> 만약 BasicAuth 라면: auth=(user, password)

    if response.status_code != 200:
        raise ConnectionError(f"스냅샷 요청 실패, 상태 코드: {response.status_code}")

    # 5) 이미지를 메모리에서 OpenCV 포맷으로 디코딩
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("스냅샷 이미지를 디코딩할 수 없습니다.")

    return image

if __name__ == "__main__":
    # 카메라 정보 입력
    ip = "192.168.0.109"     # 카메라 IP
    port = 80               # ONVIF 서비스 포트 (기본값 80)
    user = "admin"          # ONVIF 사용자
    password = "Dbslqjtm!"      # ONVIF 비밀번호

    try:
        # 스냅샷 가져오기
        snapshot_img = get_onvif_snapshot(ip, port, user, password)

        # 이미지 파일로 저장
        cv2.imwrite("snapshot.jpg", snapshot_img)
        print("✅ 스냅샷을 'snapshot.jpg'로 저장했습니다.")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
