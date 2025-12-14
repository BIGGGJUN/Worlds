import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Slider, Button
import random
import math

class PolygonalMap:
    def __init__(self, num_points=2000, map_size=10.0, seed=None, island_factor=1.0):
        """
        Polygonal Map Generator
        참고: Polygonal Map Generation for Games (Amit Patel)
        """
        self.num_points = num_points
        self.map_size = map_size
        self.seed = seed
        self.island_factor = island_factor # 섬 크기 조절 인자 (기본값 높임)
        
        self.reset_data()

    def reset_data(self):
        """데이터 초기화"""
        self.points = None
        self.vertices = None
        self.regions = []
        self.is_land = {}
        self.elevation = {}
        self.moisture = {}
        self.biomes = {}
        self.rivers = []
        self.neighbors = {}
        self.vor = None

    def generate(self, seed=None):
        """전체 맵 생성 파이프라인 실행"""
        # 시드 설정 (UI에서 리롤 시 새로운 시드 적용)
        current_seed = seed if seed is not None else self.seed
        if current_seed is not None:
            random.seed(current_seed)
            np.random.seed(current_seed)
        
        self.reset_data()
        
        # 1. 메쉬 생성
        self._generate_mesh()
        
        # 2. 섬 형태 정의
        self._assign_island_shape()
        
        # 3. 고도 계산
        self._assign_elevation()
        
        # 4. 수분 및 강 계산
        self._assign_moisture_and_rivers()
        
        # 5. 바이옴 결정
        self._assign_biomes()

    def _generate_mesh(self):
        # 1. 초기 랜덤 포인트
        points = np.random.rand(self.num_points, 2) * self.map_size
        
        # 2. Lloyd Relaxation (셀 균일화)
        for _ in range(2):
            vor = Voronoi(points)
            new_points = []
            for i, region_index in enumerate(vor.point_region):
                region = vor.regions[region_index]
                if -1 in region or len(region) == 0:
                    new_points.append(points[i])
                    continue
                poly_verts = vor.vertices[region]
                centroid = np.mean(poly_verts, axis=0)
                new_points.append(centroid)
            points = np.array(new_points)
            points = np.clip(points, 0, self.map_size)

        # 3. 최종 Voronoi
        self.vor = Voronoi(points)
        self.points = self.vor.points
        self.vertices = self.vor.vertices
        
        self.regions = [
            self.vor.regions[self.vor.point_region[i]]
            for i in range(len(self.points))
        ]
        
        # 이웃 정보 구축
        for (p1, p2) in self.vor.ridge_points:
            if p1 not in self.neighbors: self.neighbors[p1] = []
            if p2 not in self.neighbors: self.neighbors[p2] = []
            self.neighbors[p1].append(p2)
            self.neighbors[p2].append(p1)

    def _assign_island_shape(self):
        """
        중심 거리 기반 섬 생성. 
        이전 코드의 문제점: threshold가 너무 낮아 육지가 거의 생성되지 않음.
        수정: island_factor를 도입하여 육지 비율을 조정 가능하게 함.
        """
        center = np.array([0.5, 0.5]) * self.map_size
        
        for i, point in enumerate(self.points):
            # 중심에서의 거리 (0~1 로 정규화, 대각선 끝이 1.0이 되도록 조정)
            # map_size / 2 가 반지름.
            radius = self.map_size * 0.5
            d = np.linalg.norm(point - center) / radius
            
            # 노이즈 생성 (좌표 정규화하여 사용)
            nx = point[0] / self.map_size
            ny = point[1] / self.map_size
            
            # 여러 주파수의 노이즈를 섞어 더 자연스럽게 만듦
            noise = (math.sin(nx * 6) + math.cos(ny * 6)) * 0.1
            noise += (math.sin(nx * 12 + 2) + math.cos(ny * 12 + 2)) * 0.05
            
            # 육지 판별 기준
            # distance가 threshold보다 작으면 육지
            # island_factor가 1.0이면 기본 반지름(0.5)보다 조금 더 넓게 설정
            # factor가 클수록 육지가 넓어짐
            base_threshold = 0.5 * self.island_factor 
            threshold = base_threshold + noise
            
            self.is_land[i] = d < threshold

            # 맵 가장자리는 강제로 바다
            margin = 0.02 * self.map_size
            if (point[0] < margin or point[0] > self.map_size - margin or 
                point[1] < margin or point[1] > self.map_size - margin):
                self.is_land[i] = False

    def _assign_elevation(self):
        # 초기화
        for i in range(len(self.points)):
            self.elevation[i] = 0.0
            
        queue = []
        # 해안선 찾기
        for i in range(len(self.points)):
            if not self.is_land[i]:
                self.elevation[i] = 0.0
            else:
                is_coast = False
                if i in self.neighbors:
                    for n in self.neighbors[i]:
                        if not self.is_land.get(n, False):
                            is_coast = True
                            break
                if is_coast:
                    self.elevation[i] = 0.1
                    queue.append(i)
                else:
                    self.elevation[i] = float('inf')

        # 고도 전파
        while queue:
            current = queue.pop(0)
            if current not in self.neighbors: continue
                
            for n in self.neighbors[current]:
                if self.is_land[n] and self.elevation[n] == float('inf'):
                    # 약간의 랜덤성을 더해 불규칙한 지형 생성
                    self.elevation[n] = self.elevation[current] + 0.05 + random.random() * 0.05
                    queue.append(n)
        
        # 정규화 및 shaping
        land_elevs = [self.elevation[i] for i in self.elevation if self.is_land[i]]
        if land_elevs:
            max_elev = max(land_elevs)
            for i in self.elevation:
                if self.is_land[i]:
                    self.elevation[i] /= max_elev
                    self.elevation[i] = self.elevation[i] ** 1.3 # 뾰족하게

    def _assign_moisture_and_rivers(self):
        # 강 생성
        potential_sources = [
            i for i in range(len(self.points)) 
            if self.is_land[i] and self.elevation[i] > 0.8
        ]
        
        if not potential_sources:
            potential_sources = [i for i in range(len(self.points)) if self.is_land[i]]
        
        # 수원지 개수 조절
        num_springs = max(1, int(len(potential_sources) * 0.2))
        springs = random.sample(potential_sources, min(len(potential_sources), num_springs))
        
        for start_node in springs:
            current = start_node
            while True:
                if current not in self.neighbors: break
                
                neighbors = self.neighbors[current]
                lowest = current
                min_e = self.elevation[current]
                
                for n in neighbors:
                    if self.elevation[n] < min_e:
                        min_e = self.elevation[n]
                        lowest = n
                
                if lowest == current: break
                if not self.is_land[lowest]: # 바다 도착
                    break
                    
                self.rivers.append((current, lowest))
                current = lowest

        # 수분 계산
        queue = []
        for i in range(len(self.points)):
            self.moisture[i] = 0.0
            is_river = False
            for (p1, p2) in self.rivers:
                if i == p1 or i == p2:
                    is_river = True
                    break
            
            if not self.is_land[i]:
                self.moisture[i] = 1.0
            elif is_river:
                self.moisture[i] = 0.9
                queue.append(i)
        
        visited = set(queue)
        while queue:
            curr = queue.pop(0)
            if curr not in self.neighbors: continue
            
            curr_moist = self.moisture[curr]
            if curr_moist <= 0.1: continue
            
            for n in self.neighbors[curr]:
                if self.is_land[n] and n not in visited:
                    self.moisture[n] = curr_moist * 0.9
                    visited.add(n)
                    queue.append(n)
                    
        for i in range(len(self.points)):
            if self.is_land[i]:
                # 고도에 따른 강우량 보정
                self.moisture[i] = min(1.0, self.moisture[i] + self.elevation[i] * 0.1)

    def _assign_biomes(self):
        for i in range(len(self.points)):
            if not self.is_land[i]:
                self.biomes[i] = 'OCEAN'
                continue

            e = self.elevation[i]
            m = self.moisture[i]

            if e > 0.8:
                if m > 0.5: self.biomes[i] = 'SNOW'
                elif m > 0.2: self.biomes[i] = 'TUNDRA'
                else: self.biomes[i] = 'SCORCHED'
            elif e > 0.5:
                if m > 0.66: self.biomes[i] = 'TAIGA'
                elif m > 0.33: self.biomes[i] = 'SHRUBLAND'
                else: self.biomes[i] = 'TEMPERATE_DESERT'
            elif e > 0.25:
                if m > 0.83: self.biomes[i] = 'TEMPERATE_RAIN_FOREST'
                elif m > 0.50: self.biomes[i] = 'TEMPERATE_DECIDUOUS_FOREST'
                elif m > 0.16: self.biomes[i] = 'GRASSLAND'
                else: self.biomes[i] = 'TEMPERATE_DESERT'
            else:
                if m > 0.66: self.biomes[i] = 'TROPICAL_RAIN_FOREST'
                elif m > 0.33: self.biomes[i] = 'TROPICAL_SEASONAL_FOREST'
                elif m > 0.16: self.biomes[i] = 'GRASSLAND'
                else: self.biomes[i] = 'SUBTROPICAL_DESERT'

# ---------------------------------------------------------
# UI 및 렌더링 로직
# ---------------------------------------------------------

def run_gui():
    # 기본 설정
    init_points = 1000
    init_size = 10.0
    init_island = 1.3 # 기본 섬 크기를 키움
    
    gen = PolygonalMap(num_points=init_points, map_size=init_size, island_factor=init_island)
    gen.generate(seed=42)

    # UI 레이아웃 설정
    fig = plt.figure(figsize=(12, 9))
    ax_map = plt.axes([0.05, 0.25, 0.9, 0.7]) # 맵 영역 (좌, 하, 폭, 높이)
    
    # 위젯 영역
    ax_points = plt.axes([0.25, 0.15, 0.5, 0.03])
    ax_island = plt.axes([0.25, 0.10, 0.5, 0.03])
    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])

    # 슬라이더 생성
    s_points = Slider(ax_points, 'Points', 500, 3000, valinit=init_points, valstep=100)
    s_island = Slider(ax_island, 'Island Size', 0.5, 2.5, valinit=init_island)
    
    # 버튼 생성
    btn_reroll = Button(ax_button, 'Re-roll Map')

    biome_colors = {
        'OCEAN': '#44447a', 'SNOW': '#ffffff', 'TUNDRA': '#ddddbb',
        'SCORCHED': '#999999', 'TAIGA': '#ccd4bb', 'SHRUBLAND': '#c4ccbb',
        'TEMPERATE_DESERT': '#e4e8ca', 'TEMPERATE_RAIN_FOREST': '#a4c4a8',
        'TEMPERATE_DECIDUOUS_FOREST': '#b4c9a9', 'GRASSLAND': '#c4d4aa',
        'TROPICAL_RAIN_FOREST': '#9cbba9', 'TROPICAL_SEASONAL_FOREST': '#a9cca4',
        'SUBTROPICAL_DESERT': '#e9ddc7'
    }

    def update_plot():
        ax_map.clear()
        ax_map.set_xlim(0, gen.map_size)
        ax_map.set_ylim(0, gen.map_size)
        ax_map.set_aspect('equal')
        ax_map.axis('off')
        
        # 다각형 그리기
        polygons = []
        colors = []
        for i, region_idx in enumerate(gen.vor.point_region):
            region = gen.vor.regions[region_idx]
            if -1 in region or len(region) == 0: continue
            
            poly = gen.vor.vertices[region]
            polygons.append(poly)
            
            b = gen.biomes.get(i, 'OCEAN')
            colors.append(biome_colors.get(b, '#000000'))
            
        coll = PolyCollection(polygons, facecolors=colors, edgecolors='none')
        ax_map.add_collection(coll)
        
        # 강 그리기
        river_segs = []
        for p1_idx, p2_idx in gen.rivers:
            river_segs.append([gen.points[p1_idx], gen.points[p2_idx]])
            
        if river_segs:
            from matplotlib.collections import LineCollection
            lc = LineCollection(river_segs, colors='#225588', linewidths=1.5, alpha=0.7)
            ax_map.add_collection(lc)
            
        ax_map.set_title(f"Polygonal Map (Points: {gen.num_points}, Island Factor: {gen.island_factor:.2f})")
        fig.canvas.draw_idle()

    def on_change(val):
        # 슬라이더 값 변경 시 호출되지 않고 버튼 클릭 시에만 반영하도록 할 수도 있지만
        # 여기서는 버튼을 눌렀을 때만 전체 재생성하도록 함 (성능 때문)
        pass

    def on_click(event):
        # 파라미터 업데이트
        gen.num_points = int(s_points.val)
        gen.island_factor = s_island.val
        
        # 재생성 (새로운 랜덤 시드)
        new_seed = random.randint(0, 999999)
        print(f"Regenerating... Points: {gen.num_points}, Island: {gen.island_factor}, Seed: {new_seed}")
        gen.generate(seed=new_seed)
        update_plot()

    # 이벤트 연결
    s_points.on_changed(on_change)
    s_island.on_changed(on_change)
    btn_reroll.on_clicked(on_click)

    # 초기 그리기
    update_plot()
    plt.show()

if __name__ == "__main__":
    run_gui()