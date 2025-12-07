import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.ndimage import gaussian_filter, zoom, map_coordinates, distance_transform_edt
from PIL import Image, ImageTk
import random
import time

class WorldGenerator:
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
        self.seed = 0
        
        # 지도 데이터 배열
        self.elevation = None
        self.moisture = None
        self.temperature = None
        self.rivers = None
        self.lakes = None
        self.color_map = None

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_fractal_noise(self, shape, res, octaves=6, persistence=0.5, lacunarity=2.0):
        """
        여러 주파수의 노이즈를 겹쳐서 자연스러운 지형 텍스처를 생성합니다.
        """
        noise = np.zeros(shape)
        frequency = 1
        amplitude = 1
        max_value = 0
        
        for _ in range(octaves):
            grid_h = max(1, int(res[0] * frequency))
            grid_w = max(1, int(res[1] * frequency))
            
            base = np.random.rand(grid_h, grid_w)
            
            # 큐빅 보간법으로 확대
            zoom_factor = (shape[0]/base.shape[0], shape[1]/base.shape[1])
            layer = zoom(base, zoom_factor, order=3)
            
            layer = layer[:shape[0], :shape[1]]
            
            noise += layer * amplitude
            max_value += amplitude
            
            amplitude *= persistence
            frequency *= lacunarity
        
        return noise / max_value

    def generate_tectonics(self, num_plates=20):
        """
        판 구조론 시뮬레이션 개선:
        1. 보로노이 셀로 판 생성
        2. 판의 경계(Collision Zones)를 찾아 산맥 후보지로 설정
        """
        # 1. 판 중심점 생성
        points = np.column_stack((np.random.randint(0, self.height, num_plates),
                                  np.random.randint(0, self.width, num_plates)))
        
        # 2. 저해상도에서 보로노이 계산 (속도 최적화)
        scale = 4
        h_small, w_small = self.height // scale, self.width // scale
        y, x = np.indices((h_small, w_small))
        y = y * scale
        x = x * scale
        
        # 각 픽셀에서 가장 가까운 포인트 2개 찾기 (경계 계산을 위해)
        # KDTree를 사용하여 가장 가까운 2개의 점까지의 거리를 구함
        tree = KDTree(points)
        coords = np.stack((y.ravel(), x.ravel()), axis=-1)
        dist, idx = tree.query(coords, k=2)
        
        dist = dist.reshape(h_small, w_small, 2)
        idx = idx.reshape(h_small, w_small, 2)
        
        # 판 인덱스 맵
        plate_map = idx[:, :, 0]
        
        # 경계 강도 계산: (두 번째 가까운 거리 - 첫 번째 가까운 거리)가 작을수록 경계에 가깝다
        # 0에 가까울수록 경계, 클수록 판의 중심
        border_dist = dist[:, :, 1] - dist[:, :, 0]
        # 정규화 및 반전 (1.0 = 경계선, 0.0 = 판 중심)
        border_val = 1.0 / (1.0 + border_dist * 0.05) 
        
        # 원래 크기로 확대
        plate_map = zoom(plate_map, scale, order=0)[:self.height, :self.width]
        border_val = zoom(border_val, scale, order=1)[:self.height, :self.width]
        
        return plate_map, border_val

    def apply_domain_warping(self, input_array, intensity=50.0, scale=4):
        """
        좌표 자체를 노이즈로 비틀어 지형을 자연스럽게 만듭니다.
        """
        h, w = input_array.shape
        warp_x = self.generate_fractal_noise((h, w), (scale, scale), octaves=2)
        warp_y = self.generate_fractal_noise((h, w), (scale, scale), octaves=2)
        
        y, x = np.indices((h, w))
        map_y = y + (warp_y - 0.5) * intensity
        map_x = x + (warp_x - 0.5) * intensity
        
        return map_coordinates(input_array, [map_y, map_x], order=1, mode='nearest')

    def generate_world(self, sea_level=0.4, precip_mod=0.0, temp_mod=0.0, num_plates=15):
        """
        개선된 월드 생성 파이프라인
        """
        print(f"Generating world with Seed: {self.seed}")
        
        # 1. 판 구조 및 경계 산맥 생성 (Tectonic Borders)
        plate_map, border_val = self.generate_tectonics(num_plates)
        
        # 판별 기본 높이 (대륙 vs 해양)
        plate_base_height = np.random.rand(num_plates)
        # 대륙/해양 이분화 (0.2 or 0.7 근처로 몰리게)
        plate_base_height = np.where(plate_base_height < 0.6, 
                                   plate_base_height * 0.3,       # 해양
                                   0.4 + plate_base_height * 0.4) # 대륙
        
        base_elevation = plate_base_height[plate_map]
        
        # 경계선에 산맥 솟아오르게 하기 (충돌 존)
        # 대륙판인 경우에만 경계를 융기시킴
        is_continent = base_elevation > 0.3
        mountain_ranges = border_val * is_continent * 0.6 
        
        # 베이스 지형 합성
        self.elevation = base_elevation + mountain_ranges
        
        # 도메인 워핑으로 찌그러트리기 (직선 경계 제거)
        self.elevation = self.apply_domain_warping(self.elevation, intensity=self.width * 0.1, scale=4)
        self.elevation = gaussian_filter(self.elevation, sigma=2)

        # 2. 프랙탈 노이즈로 디테일 추가 (Ridged Multifractal)
        # 산맥을 더 뾰족하게 만드는 노이즈
        noise_large = self.generate_fractal_noise((self.height, self.width), (8, 8), octaves=8)
        ridged_noise = np.abs(noise_large - 0.5) * 2
        ridged_noise = np.power(ridged_noise, 2) # 계곡을 더 넓게
        
        noise_small = self.generate_fractal_noise((self.height, self.width), (20, 20), octaves=8)
        
        # 최종 합성: 베이스(판구조) + 릿지 노이즈(산맥 디테일) + 작은 노이즈(질감)
        self.elevation += ridged_noise * 0.3 + noise_small * 0.1
        
        # 해수면 근처 평탄화 (해안선 부드럽게)
        # 3. 침식 시뮬레이션 (Erosion) - 강이 땅을 깎음
        self.rivers = np.zeros((self.height, self.width))
        self.lakes = np.zeros((self.height, self.width))
        
        # 해상도에 비례하여 강 갯수 조절
        num_droplets = int(np.sqrt(self.width * self.height) * 30)
        self.simulate_erosion(sea_level, num_droplets=num_droplets)

        # 높이 재정규화 (침식 후)
        self.elevation = (self.elevation - np.min(self.elevation)) / (np.max(self.elevation) - np.min(self.elevation))

        # 4. 기후 (기온 & 습도)
        self.generate_climate(sea_level, temp_mod, precip_mod)

        # 5. 렌더링
        self.render_map(sea_level)

    def simulate_erosion(self, sea_level, num_droplets):
        """
        강 생성 및 침식 (Carving)
        강이 흐르는 경로의 지형 높이를 실제로 깎아내려 V자 계곡을 만듦.
        """
        # 경사도 계산을 위한 준비
        h, w = self.height, self.width
        
        # 무작위 시작점들 (육지 높은 곳 위주로 선택하면 좋으나 랜덤도 무방)
        # 완전히 랜덤한 위치에서 시작
        starts_y = np.random.randint(0, h, num_droplets)
        starts_x = np.random.randint(0, w, num_droplets)
        
        # 속도를 위해 단순 반복문 사용 (복잡한 물리 시뮬레이션 대신 근사치)
        erosion_rate = 0.005 # 깎이는 정도
        
        for i in range(num_droplets):
            cy, cx = starts_y[i], starts_x[i]
            
            # 바다에서 시작하면 스킵
            if self.elevation[cy, cx] < sea_level:
                continue
                
            path_len = 0
            while path_len < 300:
                # 현재 위치 침식 (계곡 형성)
                self.elevation[cy, cx] -= erosion_rate
                self.rivers[cy, cx] += 1
                
                # 주변 8방향 중 가장 낮은 곳 찾기
                min_h = self.elevation[cy, cx]
                nx, ny = cx, cy
                found_lower = False
                
                # 3x3 탐색
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0: continue
                        py, px = cy + dy, cx + dx
                        if 0 <= py < h and 0 <= px < w:
                            if self.elevation[py, px] < min_h:
                                min_h = self.elevation[py, px]
                                ny, nx = py, px
                                found_lower = True
                
                # 더 낮은 곳이 없으면(웅덩이/국소 최저점) 호수 형성 후 종료
                if not found_lower:
                    self.lakes[cy, cx] += 1
                    # 웅덩이를 메워줌 (Depression filling - 너무 깊어지지 않게)
                    self.elevation[cy, cx] += erosion_rate * 5 
                    break
                
                # 바다를 만나면 종료
                if min_h < sea_level:
                    break
                    
                cy, cx = ny, nx
                path_len += 1

    def generate_climate(self, sea_level, temp_mod, precip_mod):
        # 위도별 기온
        y_grid = np.linspace(-1, 1, self.height).reshape(-1, 1)
        latitude_temp = 1.0 - np.abs(y_grid)
        latitude_temp = np.repeat(latitude_temp, self.width, axis=1)
        
        # 고도에 따른 기온 감소 (Lapse rate)
        self.temperature = latitude_temp - (self.elevation * 0.7) + temp_mod
        self.temperature = np.clip(self.temperature, 0, 1)

        # 습도 계산 (수증기 이동 시뮬레이션은 복잡하므로 거리 기반 근사)
        # 1. 노이즈 베이스
        moisture_noise = self.generate_fractal_noise((self.height, self.width), (4, 4), octaves=4)
        
        # 2. 수원(바다, 강, 호수)으로부터의 거리
        water_source = (self.elevation < sea_level) | (self.rivers > 10) | (self.lakes > 0)
        # 거리가 멀수록 습도 급격히 감소
        dist_moisture = distance_transform_edt(~water_source)
        dist_moisture = np.exp(-dist_moisture * 0.05) # 지수 함수적 감소 (해안가는 습함, 내륙은 급격히 건조)
        
        self.moisture = (moisture_noise * 0.3) + (dist_moisture * 0.7) + (precip_mod * 0.2)
        
        # 적도(중앙) 수렴대 비 보정
        equator_rain = np.repeat(1.0 - np.abs(y_grid), self.width, axis=1)
        self.moisture += equator_rain * 0.2
        
        self.moisture = np.clip(self.moisture, 0, 1)

    def render_map(self, sea_level):
        """
        자연스러운 색상 매핑
        """
        r_layer = np.zeros((self.height, self.width), dtype=np.uint8)
        g_layer = np.zeros((self.height, self.width), dtype=np.uint8)
        b_layer = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # 1. 물 (바다 & 호수)
        # 호수 및 강
        water_body = (self.lakes > 0) | (self.rivers > 20)
        
        water_mask = (self.elevation < sea_level)
        deep_water = self.elevation < (sea_level * 0.5)
        shallow_water = (self.elevation >= (sea_level * 0.5)) & water_mask
        
        # 바다 색상
        r_layer[deep_water] = 20; g_layer[deep_water] = 40; b_layer[deep_water] = 100
        r_layer[shallow_water] = 40; g_layer[shallow_water] = 110; b_layer[shallow_water] = 190
        
        # 2. 육지
        land_mask = ~water_mask
        beach_mask = land_mask & (self.elevation < sea_level + 0.02)
        
        t = self.temperature
        m = self.moisture
        e = self.elevation
        
        # 기본 육지 색
        r_layer[land_mask] = 120; g_layer[land_mask] = 120; b_layer[land_mask] = 100

        # 생태계 분류 (Whittaker 변형)
        # 극지방/한대
        snow = land_mask & (t < 0.2)
        r_layer[snow] = 240; g_layer[snow] = 245; b_layer[snow] = 255 # 눈
        
        tundra = land_mask & (t >= 0.2) & (t < 0.35)
        r_layer[tundra] = 180; g_layer[tundra] = 190; b_layer[tundra] = 170

        # 온대/열대
        # 사막
        desert = land_mask & (t >= 0.35) & (m < 0.25)
        r_layer[desert] = 230; g_layer[desert] = 210; b_layer[desert] = 160
        
        # 초원
        grass = land_mask & (t >= 0.35) & (m >= 0.25) & (m < 0.5)
        r_layer[grass] = 130; g_layer[grass] = 180; b_layer[grass] = 100
        
        # 숲
        forest = land_mask & (t >= 0.35) & (m >= 0.5) & (m < 0.75)
        r_layer[forest] = 34; g_layer[forest] = 139; b_layer[forest] = 34
        
        # 우림
        rainforest = land_mask & (t >= 0.35) & (m >= 0.75)
        r_layer[rainforest] = 10; g_layer[rainforest] = 90; b_layer[rainforest] = 20
        
        # 3. 고산 지대 (식생 덮어쓰기)
        high_rock = land_mask & (e > 0.8)
        r_layer[high_rock] = 90; g_layer[high_rock] = 85; b_layer[high_rock] = 80
        
        high_snow = land_mask & (e > 0.9)
        r_layer[high_snow] = 250; g_layer[high_snow] = 250; b_layer[high_snow] = 250
        
        # 해변
        r_layer[beach_mask] = 210; g_layer[beach_mask] = 200; b_layer[beach_mask] = 160

        # 강과 호수 그리기 (마지막에)
        r_layer[water_body] = 60; g_layer[water_body] = 120; b_layer[water_body] = 220
        
        self.color_map = np.dstack((r_layer, g_layer, b_layer))

    def get_pil_image(self):
        return Image.fromarray(self.color_map)


class MapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procedural Earth-Like World Generator (Realistic Erosion)")
        self.root.geometry("1200x900")
        
        # 설정 변수
        self.seed_var = tk.StringVar(value=str(random.randint(1, 9999)))
        self.sea_level_var = tk.DoubleVar(value=0.45)
        self.precip_var = tk.DoubleVar(value=0.0)
        self.temp_var = tk.DoubleVar(value=0.0)
        self.resolution_var = tk.StringVar(value="800")
        
        self.generator = WorldGenerator(width=800, height=800)
        
        self._setup_ui()
        self.generate_map()

    def _setup_ui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.image_frame = ttk.Frame(self.root, padding="10")
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # --- 컨트롤 패널 ---
        ttk.Label(control_frame, text="월드 생성 옵션", font=("Arial", 14, "bold")).pack(pady=10)
        
        ttk.Label(control_frame, text="Seed (시드)").pack(anchor="w")
        seed_entry = ttk.Entry(control_frame, textvariable=self.seed_var)
        seed_entry.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="🎲 랜덤 시드", command=self.randomize_seed).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # 해상도 조절
        ttk.Label(control_frame, text="해상도 (Map Size)").pack(anchor="w")
        res_combo = ttk.Combobox(control_frame, textvariable=self.resolution_var, values=["500", "800", "1000", "1500"], state="readonly")
        res_combo.pack(fill=tk.X, pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Label(control_frame, text="해수면 높이").pack(anchor="w")
        sl_slider = ttk.Scale(control_frame, from_=0.1, to=0.9, variable=self.sea_level_var, orient=tk.HORIZONTAL)
        sl_slider.pack(fill=tk.X)
        
        ttk.Label(control_frame, text="강수량 (건조 <-> 습함)").pack(anchor="w", pady=(10, 0))
        precip_slider = ttk.Scale(control_frame, from_=-0.5, to=0.5, variable=self.precip_var, orient=tk.HORIZONTAL)
        precip_slider.pack(fill=tk.X)
        
        ttk.Label(control_frame, text="지구 평균 기온 (한랭 <-> 온난)").pack(anchor="w", pady=(10, 0))
        temp_slider = ttk.Scale(control_frame, from_=-0.5, to=0.5, variable=self.temp_var, orient=tk.HORIZONTAL)
        temp_slider.pack(fill=tk.X)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=20)

        gen_btn = ttk.Button(control_frame, text="🌍 자연스러운 월드 생성", command=self.generate_map)
        gen_btn.pack(fill=tk.X, pady=10)
        
        self.info_label = ttk.Label(control_frame, text="Ready", wraplength=150)
        self.info_label.pack(pady=10)

        self.canvas = tk.Canvas(self.image_frame, bg="#202020", width=800, height=800)
        self.canvas.pack(anchor="center", expand=True)

    def randomize_seed(self):
        self.seed_var.set(str(random.randint(1, 100000)))

    def generate_map(self):
        try:
            seed = int(self.seed_var.get())
            sea_level = self.sea_level_var.get()
            precip = self.precip_var.get()
            temp = self.temp_var.get()
            resolution = int(self.resolution_var.get())
            
            self.generator.width = resolution
            self.generator.height = resolution
            
            self.info_label.config(text=f"생성 중... ({resolution}x{resolution})\n지질 시뮬레이션 중...")
            self.root.update()
            
            start_time = time.time()
            
            self.generator.set_seed(seed)
            self.generator.generate_world(
                sea_level=sea_level,
                precip_mod=precip,
                temp_mod=temp,
                num_plates=25
            )
            
            pil_img = self.generator.get_pil_image()
            
            preview_size = 800
            display_img = pil_img.resize((preview_size, preview_size), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(display_img)
            
            self.canvas.config(width=preview_size, height=preview_size)
            self.canvas.create_image(preview_size//2, preview_size//2, image=self.tk_img)
            
            elapsed = time.time() - start_time
            self.info_label.config(text=f"완료!\n소요 시간: {elapsed:.2f}초\nSeed: {seed}\n크기: {resolution}x{resolution}")
            
        except ValueError:
            self.info_label.config(text="오류: 값을 확인해주세요.")

if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = MapApp(root)
    root.mainloop()