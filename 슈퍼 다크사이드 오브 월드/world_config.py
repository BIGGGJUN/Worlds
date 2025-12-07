# world_config.py
# 월드 생성에 필요한 상수 및 설정값을 관리합니다.

class Config:
    # 1. 맵 설정
    WIDTH = 512       # 데모용 해상도 (실제 사용시 2048 이상 권장)
    HEIGHT = 256
    SEED = 42         # 재현성을 위한 시드값
    
    # 2. 판 구조 설정
    NUM_PLATES = 20         # 생성할 지각 판의 개수
    OCEAN_PERCENTAGE = 0.65 # 전체 맵 중 해양의 비율 목표치
    
    # 3. 노이즈 및 고도 설정
    NOISE_SCALE = 50.0      # 노이즈 주파수 스케일
    NOISE_OCTAVES = 6       # 노이즈 디테일 레벨
    MOUNTAIN_HEIGHT = 0.8   # 산맥으로 취급할 최소 높이 (0.0 ~ 1.0)
    
    # 4. 침식 설정
    EROSION_ITERATIONS = 1  # 단순화된 모델에서는 1회로 충분
    MIN_RIVER_FLOW = 100    # 강으로 표시하기 위한 최소 유량 누적치
    
    # 5. 기후 설정
    EQUATOR_TEMP = 30.0     # 적도 평균 기온 (섭씨)
    POLE_TEMP = -20.0       # 극지방 평균 기온 (섭씨)
    LAPSE_RATE = 15.0       # 고도 1.0(최대) 상승 시 기온 하강 폭