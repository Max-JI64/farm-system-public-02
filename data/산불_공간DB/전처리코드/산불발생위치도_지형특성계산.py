import pandas as pd
import numpy as np
import rasterio
from pyproj import Transformer
from tqdm import tqdm
import os
import math

def calculate_topography(window, dx, dy):
    """
    3x3 window를 받아서 고도, 경사도, 사면방향, TPI를 계산합니다.
    """
    # -9999 등 노데이터(결측치)가 포함된 경우 계산하지 않음
    if np.any(window < -1000) or window.shape != (3, 3):
        return np.nan, np.nan, np.nan, np.nan
        
    z = window[1, 1] # 중심 픽셀 고도
    
    # TPI (Topographic Position Index): 중심 고도 - 3x3 전체 평균 고도
    tpi = z - np.mean(window)
    
    # 기울기 계산 (가장 일반적인 중심 차분법)
    # dx, dy는 픽셀의 미터(m) 단위 크기 (예: 90m)
    dz_dx = (window[1, 2] - window[1, 0]) / (2 * dx)
    dz_dy = (window[2, 1] - window[0, 1]) / (2 * dy)
    
    # 경사도 (Slope) - Degree 단위
    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope = math.degrees(slope_rad)
    
    # 사면방향 (Aspect) - Degree 단위 (0: 북, 90: 동, 180: 남, 270: 서)
    # 남쪽이 Y축 아래쪽이므로 방향 고려
    aspect_rad = math.atan2(dz_dx, -dz_dy)
    aspect = (math.degrees(aspect_rad) + 360) % 360
    
    return z, slope, aspect, tpi

def main():
    base_dir = r'd:\farm-system-public-02\jsw\data\산불발생위치도_지형특성계산'
    csv_path = os.path.join(base_dir, '산불_공간DB_위경도.csv')
    dem_path = os.path.join(base_dir, '한반도90m_GRS80.img')
    twi_dir = os.path.join(base_dir, 'TWI')
    
    print("1. 화재 데이터 로딩 중...")
    df = pd.read_csv(csv_path, encoding='utf-8')
    # 위도 경도가 있는 데이터만 필터링
    df = df.dropna(subset=['위도', '경도']).copy()
    print(f"-> 총 {len(df)}건의 좌표 데이터를 분석합니다.")
    
    # 좌표 변환기 (EPSG:4326 WGS84 -> DEM / TWI 좌표계)
    trans_dem = Transformer.from_crs("epsg:4326", "epsg:5186", always_xy=True)
    trans_twi = Transformer.from_crs("epsg:4326", "epsg:5179", always_xy=True)
    
    # 결과 담을 리스트
    elevations, slopes, aspects, tpis, twis = [], [], [], [], []
    
    # 2. DEM 데이터 메모리 로딩
    print("2. 전국 DEM 래스터 데이터 로딩 중 (시간이 조금 걸립니다)...")
    with rasterio.open(dem_path) as src_dem:
        dem_transform = src_dem.transform
        dem_array = src_dem.read(1)
        # 픽셀 크기 (m 단위)
        dx = abs(dem_transform.a)
        dy = abs(dem_transform.e)
        dem_rows, dem_cols = dem_array.shape

    # 3. TWI 파일 객체 목록 준비
    print("3. TWI 파일 목록 구성 중...")
    twi_files = [f for f in os.listdir(twi_dir) if f.lower().endswith('.tif')]
    twi_sources = []
    for f in twi_files:
        src = rasterio.open(os.path.join(twi_dir, f))
        twi_sources.append(src)
        
    print(f"-> 총 {len(twi_sources)}개의 TWI 타일 파일이 준비되었습니다.")

    # 4. 각 좌표별 지표 추출
    print("4. 위경도별 지형 특성 변수 추출 시작...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        lon, lat = row['경도'], row['위도']
        
        # 4-1. DEM 기반 추출 (고도, 경사도, 사면방향, TPI)
        x_dem, y_dem = trans_dem.transform(lon, lat)
        
        try:
            r, c = rasterio.transform.rowcol(dem_transform, x_dem, y_dem)
            # 가장자리가 아닌 경우 3x3 윈도우 추출 가능
            if 1 <= r < dem_rows-1 and 1 <= c < dem_cols-1:
                window = dem_array[r-1:r+2, c-1:c+2]
                z, slope, aspect, tpi = calculate_topography(window, dx, dy)
                
                elevations.append(z)
                slopes.append(slope)
                aspects.append(aspect)
                tpis.append(tpi)
            else:
                elevations.append(np.nan)
                slopes.append(np.nan)
                aspects.append(np.nan)
                tpis.append(np.nan)
        except Exception:
            elevations.append(np.nan)
            slopes.append(np.nan)
            aspects.append(np.nan)
            tpis.append(np.nan)
            
        # 4-2. TWI 기반 추출
        x_twi, y_twi = trans_twi.transform(lon, lat)
        twi_val = np.nan
        
        # 여러 개의 TWI 타일 중 좌표가 속하는 타일 찾기
        for src in twi_sources:
            bounds = src.bounds
            if bounds.left <= x_twi <= bounds.right and bounds.bottom <= y_twi <= bounds.top:
                try:
                    # 해당 좌표의 픽셀값 하나만 샘플링
                    val = next(src.sample([(x_twi, y_twi)]))[0]
                    if val > -1000: # 유효한 값인 경우만
                        twi_val = val
                    break # 찾았으면 다른 타일은 볼 필요 없음
                except:
                    pass
        twis.append(twi_val)

    # 5. 데이터프레임에 병합 후 저장
    df['고도(m)'] = elevations
    df['경사도(도)'] = slopes
    df['사면방향_sin'] = np.sin(np.radians(aspects)) # 동/서 축
    df['사면방향_cos'] = np.cos(np.radians(aspects)) # 남/북 축
    df['TPI(지형위치지수)'] = tpis
    df['TWI(지형다습지수)'] = twis
    
    output_path = os.path.join(base_dir, '산불발생위치도_지형특성계산.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n[성공] 추출 완료! 결과가 다음 경로에 저장되었습니다:")
    print(f"-> {output_path}")
    
    # 파일 닫기
    for src in twi_sources:
        src.close()

if __name__ == '__main__':
    main()
