import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.ndimage import gaussian_filter, zoom
import random
from world_config import Config

class WorldGrid:
    """
    전체 월드 데이터를 관리하는 컨테이너 클래스입니다.
    여러 레이어(고도, 기온, 습도 등)를 Numpy 배열로 보유합니다.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # 2D 데이터 그리드 초기화
        self.elevation = np.zeros((height, width))    # 고도 (-1.0 ~ 1.0, 0은 해수면)
        self.rainfall = np.zeros((height, width))     # 강수량
        self.temperature = np.zeros((height, width))  # 기온
        self.flux = np.zeros((height, width))         # 유량 (강 생성용)
        self.biome_id = np.zeros((height, width), dtype=int) # 생태계 ID
        
        # 시각화용 마스크
        self.is_land = np.zeros((height, width), dtype=bool)

    def normalize_elevation(self):
        """고도를 0.0 ~ 1.0 사이로 정규화 (선택적)"""
        min_val = np.min(self.elevation)
        max_val = np.max(self.elevation)
        if max_val - min_val > 0:
            self.elevation = (self.elevation - min_val) / (max_val - min_val)

class PlateGenerator:
    """
    판 구조론에 기반한 대륙의 기초 형태를 생성합니다.
    Voronoi 다이어그램을 사용하여 판을 나누고, 대륙/해양 판을 구분합니다.
    """
    def __init__(self, grid: WorldGrid, config: Config):
        self.grid = grid
        self.config = config

    def generate_plates(self):
        print("1. 판 구조 생성 중... (Voronoi Tectonics)")
        
        # 1. 무작위 판 중심점 생성
        points = []
        for _ in range(self.config.NUM_PLATES):
            px = random.randint(0, self.grid.width - 1)
            py = random.randint(0, self.grid.height - 1)
            points.append([py, px]) # y, x 순서 주의
        
        points = np.array(points)
        
        # 2. 각 픽셀이 어느 판에 속하는지 계산 (KDTree 사용으로 가속)
        # 맵의 모든 좌표 생성
        y_indices, x_indices = np.indices((self.grid.height, self.grid.width))
        coords = np.stack((y_indices.ravel(), x_indices.ravel()), axis=-1)
        
        tree = cKDTree(points)
        # k=1: 가장 가까운 점 1개 찾기
        dists, plate_indices = tree.query(coords, k=1)
        
        plate_map = plate_indices.reshape((self.grid.height, self.grid.width))
        
        # 3. 각 판을 대륙(Land) 또는 해양(Ocean)으로 무작위 할당
        plate_types = []
        for _ in range(self.config.NUM_PLATES):
            # 설정된 해양 비율에 따라 판의 타입 결정 (True=Ocean, False=Land)
            is_ocean = random.random() < self.config.OCEAN_PERCENTAGE
            plate_types.append(-1.0 if is_ocean else 1.0) # 해양은 낮게, 대륙은 높게
            
        # 4. 기초 고도 할당
        base_elevation = np.zeros_like(self.grid.elevation)
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                pid = plate_map[y, x]
                # 판의 중심에서 멀어질수록 고도가 약간 낮아지거나 높아지는 변형 추가 가능
                base_elevation[y, x] = plate_types[pid]
        
        # 5. 경계 부드럽게 처리 (가우시안 블러)
        self.grid.elevation = gaussian_filter(base_elevation, sigma=self.config.WIDTH / 100)

class ElevationGenerator:
    """
    Fractal Noise를 사용하여 지형의 세부 디테일을 추가합니다.
    """
    def __init__(self, grid: WorldGrid, config: Config):
        self.grid = grid
        self.config = config

    def _generate_noise(self, scale):
        """간단한 노이즈 생성 (저해상도 노이즈 -> 업스케일링 방식)"""
        # Scipy의 zoom을 이용한 빠른 프랙탈 노이즈 근사
        small_h = int(self.grid.height / scale)
        small_w = int(self.grid.width / scale)
        if small_h < 1: small_h = 1
        if small_w < 1: small_w = 1
        
        noise = np.random.rand(small_h, small_w)
        # 맵 크기로 확대 (Bicubic interpolation 효과)
        zoomed = zoom(noise, (self.grid.height / small_h, self.grid.width / small_w))
        # 크기 맞춤 (zoom 오차 보정)
        return zoomed[:self.grid.height, :self.grid.width]

    def apply_detail(self):
        print("2. 지형 디테일 추가 중... (Fractal Noise)")
        
        detail_noise = np.zeros_like(self.grid.elevation)
        amplitude = 0.5
        frequency = 10.0
        
        for _ in range(self.config.NOISE_OCTAVES):
            layer = self._generate_noise(frequency)
            detail_noise += layer * amplitude
            amplitude *= 0.5
            frequency *= 0.6 # 주파수 변화
            
        # 기존 판 구조 고도에 노이즈 합성 (가중치 조절)
        # 판 구조 60% + 노이즈 40%
        self.grid.elevation = self.grid.elevation * 0.6 + detail_noise * 0.4
        
        # 해수면(0) 기준으로 대륙/해양 마스크 업데이트
        # 약간의 오프셋을 주어 해수면 조정
        self.grid.elevation -= 0.2 
        self.grid.is_land = self.grid.elevation > 0

class ErosionSimulator:
    """
    하천 생성 및 수력 침식 시뮬레이션
    """
    def __init__(self, grid: WorldGrid, config: Config):
        self.grid = grid
        self.config = config

    def simulate(self):
        print("3. 침식 및 하천 생성 중... (Hydraulic Erosion)")
        # 1. 간단한 유량 계산 (Flow Accumulation)
        # 높은 곳에서 낮은 곳으로 흐름을 누적
        
        # 경사도 계산 (Gradient)
        dy, dx = np.gradient(self.grid.elevation)
        
        # 유량 초기화 (강우량에 비례한다고 가정, 여기선 균일)
        self.grid.flux = np.ones_like(self.grid.elevation)
        
        # 단순화된 내리막 추적 (반복적 이동)
        # 실제로는 재귀적 탐색이 필요하나, 반복문으로 근사
        num_steps = int(max(self.grid.width, self.grid.height) / 2)
        
        # 벡터화된 방식 대신 이해하기 쉬운 단순 이동 시뮬레이션 (느리지만 확실함)
        # 성능을 위해 매우 간소화된 버전: 
        # "자신보다 낮은 이웃 중 가장 낮은 곳으로 내 물을 보낸다"
        
        # 여기서는 빠른 데모를 위해 고도가 높은 순서대로 정렬하여 흐름을 계산
        flat_indices = np.argsort(self.grid.elevation.ravel())[::-1] # 내림차순 정렬
        
        h, w = self.grid.height, self.grid.width
        elev = self.grid.elevation
        flux = self.grid.flux
        
        # 이웃 오프셋 (8방향)
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Python Loop은 느리므로, Numpy 연산을 위해 생략하거나
        # 매우 축소된 알고리즘을 사용해야 함.
        # 여기서는 'Ridge Noise'를 역으로 이용하여 강 위치를 추정하는 트릭 사용
        # (실제 Path finding은 Python에서 수초~수분 소요됨)
        
        # 대안: 골짜기(Valley) 탐지
        # 2차 미분(Laplacian)이 높은 곳이 골짜기일 확률이 높음
        from scipy.ndimage import laplace
        lap = laplace(elev)
        # 골짜기이면서 해수면보다 높은 곳
        rivers = (lap > 0.05) & (elev > 0)
        self.grid.flux[rivers] = 100 # 시각화를 위해 강으로 표시
        
        # 강 주변 침식 (강이 있는 곳의 고도를 깎음)
        self.grid.elevation[rivers] -= 0.05

class ClimateSimulator:
    """
    기온과 습도를 계산하여 기후대를 형성합니다.
    """
    def __init__(self, grid: WorldGrid, config: Config):
        self.grid = grid
        self.config = config

    def calculate_climate(self):
        print("4. 기후 시뮬레이션 중...")
        h, w = self.grid.height, self.grid.width
        
        # 1. 위도별 기본 기온 계산
        # 0 ~ 1.0 (0=북극, 0.5=적도, 1.0=남극)
        y_indices = np.linspace(0, 1, h)[:, np.newaxis]
        # 적도(0.5)에서 가장 뜨겁고 극지방에서 차가움
        lat_heat = 1.0 - np.abs(y_indices - 0.5) * 2 
        
        base_temp = self.config.POLE_TEMP + (self.config.EQUATOR_TEMP - self.config.POLE_TEMP) * lat_heat
        base_temp = np.tile(base_temp, (1, w)) # 가로로 복사
        
        # 2. 고도에 따른 기온 하강 (Lapse Rate)
        # 고도 0.0 ~ 1.0 가정. 해수면(0) 이상인 곳만 적용
        altitude_factor = np.maximum(0, self.grid.elevation)
        temp_map = base_temp - (altitude_factor * self.config.LAPSE_RATE)
        
        self.grid.temperature = temp_map
        
        # 3. 습도 계산 (바다에서의 거리 + 바람)
        # 간단하게: 해수면 이하는 습도 100%, 내륙으로 갈수록 감소
        is_water = ~self.grid.is_land
        # 거리 변환 (Distance Transform) - 바다로부터의 거리
        from scipy.ndimage import distance_transform_edt
        dist_from_water = distance_transform_edt(self.grid.is_land)
        
        # 거리가 멀수록 습도 낮음 (지수 감소)
        max_dist = np.max(dist_from_water) if np.max(dist_from_water) > 0 else 1
        moisture = 1.0 - (dist_from_water / max_dist)
        
        # 노이즈를 섞어서 불규칙성 추가
        noise = np.random.rand(h, w) * 0.2
        self.grid.rainfall = moisture + noise

class BiomeClassifier:
    """
    기온과 강수량을 기반으로 휘태커(Whittaker) 다이어그램 스타일의 바이옴을 결정합니다.
    """
    def __init__(self, grid: WorldGrid):
        self.grid = grid
        
        # 바이옴 색상 매핑 (R, G, B)
        self.biome_colors = {
            0: (0.1, 0.1, 0.4), # Deep Ocean
            1: (0.2, 0.4, 0.8), # Ocean
            2: (0.9, 0.9, 0.6), # Beach/Sand
            3: (0.8, 0.1, 0.1), # Scorched
            4: (0.9, 0.6, 0.2), # Desert
            5: (0.5, 0.7, 0.2), # Grassland
            6: (0.1, 0.5, 0.1), # Forest
            7: (0.05, 0.3, 0.05), # Jungle
            8: (0.7, 0.7, 0.8), # Tundra
            9: (0.95, 0.95, 1.0), # Snow
        }

    def classify(self):
        print("5. 생태계(Biome) 분류 중...")
        h, w = self.grid.height, self.grid.width
        
        for y in range(h):
            for x in range(w):
                elev = self.grid.elevation[y, x]
                temp = self.grid.temperature[y, x]
                rain = self.grid.rainfall[y, x]
                
                # 1. 해양 처리
                if elev <= 0:
                    if elev < -0.2:
                        self.grid.biome_id[y, x] = 0 # Deep Ocean
                    else:
                        self.grid.biome_id[y, x] = 1 # Ocean
                    continue
                
                # 2. 해변
                if elev < 0.05:
                    self.grid.biome_id[y, x] = 2 # Beach
                    continue
                
                # 3. 육지 바이옴 분류 (간소화된 로직)
                if temp < -5:
                    self.grid.biome_id[y, x] = 9 # Snow
                elif temp < 5:
                    self.grid.biome_id[y, x] = 8 # Tundra
                elif temp > 25:
                    if rain < 0.3: self.grid.biome_id[y, x] = 3 # Scorched
                    elif rain < 0.5: self.grid.biome_id[y, x] = 4 # Desert
                    elif rain < 0.8: self.grid.biome_id[y, x] = 6 # Forest
                    else: self.grid.biome_id[y, x] = 7 # Jungle
                else: # 온대 (5 ~ 25도)
                    if rain < 0.4: self.grid.biome_id[y, x] = 5 # Grassland
                    elif rain < 0.7: self.grid.biome_id[y, x] = 6 # Forest
                    else: self.grid.biome_id[y, x] = 7 # Jungle (Rainforest)