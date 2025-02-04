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

if __name__ == "__main__":
    devices = discover_onvif_devices()
    if not devices:
        print("❌ ONVIF 카메라를 찾지 못했습니다.")
    else:
        print(f"✅ 검색된 ONVIF 장치 수: {len(devices)}")
        for idx, (xaddrs, epr) in enumerate(devices, start=1):
            print(f"[{idx}] EPR: {epr}")
            print(f"    XAddr: {xaddrs}")
