import numpy as np
import matplotlib.pyplot as plt
from world_config import Config
from procgen_engine import (
    WorldGrid, 
    PlateGenerator, 
    ElevationGenerator, 
    ErosionSimulator, 
    ClimateSimulator, 
    BiomeClassifier
)

def main():
    # 1. 설정 및 그리드 초기화
    config = Config()
    np.random.seed(config.SEED)
    
    print(f"--- 월드 생성 시작: {config.WIDTH}x{config.HEIGHT} ---")
    
    world = WorldGrid(config.WIDTH, config.HEIGHT)
    
    # 2. 파이프라인 실행
    # (1) 판 구조 형성
    plate_gen = PlateGenerator(world, config)
    plate_gen.generate_plates()
    
    # (2) 고도 디테일
    elev_gen = ElevationGenerator(world, config)
    elev_gen.apply_detail()
    
    # (3) 침식 및 강
    erosion_gen = ErosionSimulator(world, config)
    erosion_gen.simulate()
    
    # (4) 기후
    climate_gen = ClimateSimulator(world, config)
    climate_gen.calculate_climate()
    
    # (5) 바이옴
    biome_gen = BiomeClassifier(world)
    biome_gen.classify()
    
    print("--- 생성 완료. 시각화 중... ---")
    
    # 3. 시각화
    visualize(world, biome_gen.biome_colors)

def visualize(world, colors):
    """Matplotlib를 사용하여 결과 렌더링"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # (1) 고도 맵
    ax1 = axes[0, 0]
    im1 = ax1.imshow(world.elevation, cmap='terrain', vmin=-0.5, vmax=1.0)
    ax1.set_title("Elevation (Terrain)")
    plt.colorbar(im1, ax=ax1)
    
    # (2) 기온 맵
    ax2 = axes[0, 1]
    im2 = ax2.imshow(world.temperature, cmap='coolwarm')
    ax2.set_title("Temperature (C)")
    plt.colorbar(im2, ax=ax2)
    
    # (3) 습도/강수량 맵
    ax3 = axes[1, 0]
    im3 = ax3.imshow(world.rainfall, cmap='Blues')
    ax3.set_title("Rainfall / Moisture")
    plt.colorbar(im3, ax=ax3)
    
    # (4) 최종 바이옴 맵
    ax4 = axes[1, 1]
    # RGB 이미지를 생성
    rgb_map = np.zeros((world.height, world.width, 3))
    for y in range(world.height):
        for x in range(world.width):
            bid = world.biome_id[y, x]
            rgb_map[y, x] = colors.get(bid, (0,0,0))
            
    # 강 표시 (하얀색 덮어쓰기)
    rivers = world.flux > 50
    rgb_map[rivers] = (0.0, 0.5, 1.0) # 강 색상
            
    ax4.imshow(rgb_map)
    ax4.set_title("Biome & Rivers")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()