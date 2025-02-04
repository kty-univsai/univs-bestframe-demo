import re
from onvif import ONVIFCamera, exceptions as onvif_exceptions
from wsdiscovery.discovery import ThreadedWSDiscovery
from wsdiscovery import QName

def discover_onvif_devices():
    """
    WS-Discovery를 이용해 네트워크 상의 ONVIF 장치를 검색하고,
    각 장치가 제공하는 XAddr 목록과 EPR(고유 식별자) 정보를 반환합니다.
    """
    # WS-Discovery 스레드 시작
    wsd = ThreadedWSDiscovery()
    wsd.start()

    # ONVIF NetworkVideoTransmitter 타입(카메라/비디오 장치) 검색
    scope = QName("tdn:NetworkVideoTransmitter")
    services = wsd.searchServices(types=scope)

    discovered = []
    for service in services:
        xaddrs = service.getXAddrs()  # 예: ['http://192.168.0.10:80/onvif/device_service', ...]
        epr = service.getEPR()        # 장치의 고유 식별자
        discovered.append((xaddrs, epr))

    wsd.stop()
    return discovered

def parse_xaddr_to_ip_port(xaddr: str):
    """
    XAddr 예: "http://192.168.0.100:80/onvif/device_service"
    정규식을 이용해 IP와 포트를 추출합니다.
    """
    pattern = r"http://([\d\.]+):(\d+)/"
    match = re.search(pattern, xaddr)
    if match:
        ip = match.group(1)
        port = int(match.group(2))
        return ip, port
    return None, None

def get_device_info(ip, port, user, password):
    """
    onvif-zeep 의 ONVIFCamera를 사용해 장치 기본 정보를 가져옵니다.
    """
    try:
        camera = ONVIFCamera(ip, port, user, password)  # 예: 192.168.0.100, 80
        devicemgmt_service = camera.create_devicemgmt_service()
        info = devicemgmt_service.GetDeviceInformation()
        return {
            "Manufacturer": info.Manufacturer,
            "Model": info.Model,
            "FirmwareVersion": info.FirmwareVersion,
            "SerialNumber": info.SerialNumber,
            "HardwareId": info.HardwareId
        }
    except onvif_exceptions.ONVIFError as e:
        return {"error": f"장치 연결 오류: {e}"}
    except Exception as e:
        return {"error": f"알 수 없는 오류: {e}"}

if __name__ == "__main__":
    # 1) WS-Discovery로 장치 검색
    devices = discover_onvif_devices()

    # 2) 각 장치에 대해 정보 조회
    default_user = "admin"       # 카메라 별 ONVIF 계정
    default_password = "Dbslqjtm!"

    for idx, (xaddrs, epr) in enumerate(devices, start=1):
        print(f"\n--- [{idx}]번 장치 ---")
        print("EPR :", epr)
        if not xaddrs:
            print("XAddr가 없습니다.")
            continue

        # 여러 XAddr가 있을 수 있으나, 여기서는 첫 번째만 사용
        xaddr = xaddrs[0]
        ip, port = parse_xaddr_to_ip_port(xaddr)

        if ip is None or port is None:
            print(f"IP/Port 추출 실패: {xaddr}")
            continue

        print(f"연결 시도 -> IP: {ip}, Port: {port}")
        info = get_device_info(ip, port, default_user, default_password)
        if "error" in info:
            print(info["error"])
        else:
            print("Manufacturer  :", info["Manufacturer"])
            print("Model         :", info["Model"])
            print("Firmware      :", info["FirmwareVersion"])
            print("SerialNumber  :", info["SerialNumber"])
            print("HardwareId    :", info["HardwareId"])
