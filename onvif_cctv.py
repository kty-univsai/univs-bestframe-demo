from onvif.discovery import discover_devices

def discover_onvif_cameras():
    """
    네트워크에서 ONVIF 프로토콜을 사용하는 장치를 검색하여 반환합니다.
    """
    print("🔍 네트워크에서 ONVIF 카메라를 검색 중입니다...")
    
    # 'NetworkVideoTransmitter': 영상 전송 장치(카메라, NVR 등)만 검색
    devices = discover_devices(device_type='NetworkVideoTransmitter')  
    
    if not devices:
        print("❌ ONVIF 카메라를 찾지 못했습니다.")
        return []
    
    print(f"✅ 검색된 ONVIF 장치 수: {len(devices)}")
    for idx, device in enumerate(devices, start=1):
        print(f"[{idx}] XAddrs: {device.XAddrs}, EPR: {device.EPR}")
        
    return devices

if __name__ == "__main__":
    # 네트워크 상의 ONVIF 장치 검색
    discovered_devices = discover_onvif_cameras()