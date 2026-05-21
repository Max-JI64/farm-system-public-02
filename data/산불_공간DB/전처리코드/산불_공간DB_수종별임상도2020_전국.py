import pandas as pd
import geopandas as gpd
import pyogrio
import glob
import time
import os
import warnings

warnings.filterwarnings('ignore')

# 지역코드 (시도) 매핑표
SIDO_DICT = {
    "11": "서울", "26": "부산", "27": "대구", "28": "인천", "29": "광주",
    "30": "대전", "31": "울산", "36": "세종", "41": "경기", "42": "강원",
    "43": "충북", "44": "충남", "45": "전북", "46": "전남", "47": "경북",
    "48": "경남", "50": "제주"
}

# 기존 매핑 테이블 재사용
MAPPING_DICT = {
    "AGCLS_CD": {"1": "1영급", "2": "2영급", "3": "3영급", "4": "4영급", "5": "5영급", "6": "6영급", "7": "7영급", "8": "8영급", "9": "9영급"},
    "DMCLS_CD": {"0": "치수", "1": "소경목", "2": "중경목", "3": "대경목"},
    "DNST_CD": {"A": "소", "B": "중", "C": "밀"},
    "FRTP_CD": {"0": "무립목지/비산림", "1": "침엽수림", "2": "활엽수림", "3": "혼효림", "4": "죽림"},
    "KOFTR_GROU": {
        "10": "기타침엽수", "11": "소나무", "12": "잣나무", "13": "낙엽송", "14": "리기다소나무", 
        "15": "곰솔", "16": "전나무", "17": "편백나무", "18": "삼나무", "19": "가문비나무", 
        "20": "비자나무", "21": "은행나무", "30": "기타활엽수", "31": "상수리나무", "32": "신갈나무", 
        "33": "굴참나무", "34": "기타 참나무류", "35": "오리나무", "36": "고로쇠나무", "37": "자작나무", 
        "38": "박달나무", "39": "밤나무", "40": "물푸레나무", "41": "서어나무", "42": "때죽나무", 
        "43": "호두나무", "44": "백합나무", "45": "포플러", "46": "벚나무", "47": "느티나무", 
        "48": "층층나무", "49": "아까시나무", "60": "기타상록활엽수", "61": "가시나무", "62": "구실잣밤나무", 
        "63": "녹나무", "64": "굴거리나무", "65": "황칠나무", "66": "사스레피나무", "67": "후박나무", 
        "68": "새덕이", "77": "침활혼효림", "78": "죽림", "81": "미립목지", "82": "제지", 
        "91": "주거지", "92": "초지", "93": "경작지", "94": "수체", "95": "과수원", "99": "기타"
    }
}

def decode_filename(filename):
    """
    파일명(예: FRT001102_420111.zip)의 6자리 숫자를 해독합니다.
    42(강원도) + 01(침엽수) + 11(소나무)
    """
    code = filename.split('_')[1].split('.')[0]
    sido_cd = code[:2]
    frtp_cd = code[2:4]  # 01, 02, 03, 99
    koftr_cd = code[4:]
    
    sido_nm = SIDO_DICT.get(sido_cd, "알수없음")
    
    # 중분류 (파일 분류 기준)
    frtp_dict = {"01": "침엽수림", "02": "활엽수림", "03": "혼효림", "99": "기타"}
    frtp_nm = frtp_dict.get(frtp_cd, "알수없음")
    
    koftr_nm = MAPPING_DICT["KOFTR_GROU"].get(koftr_cd, "기타수종")
    
    return f"[{sido_nm}] {frtp_nm} - {koftr_nm}"

def main():
    print("🌲 산불 발생 좌표 기준 임상도(시도별 ZIP) 추출 스크립트 🌲")
    
    base_dir = r'd:\farm-system-public-02\jsw\data\임상도\수종별임상도(나무종류지도)_시도'
    fire_csv_path = r'd:\farm-system-public-02\jsw\data\산불발생위치도_전국\산불_공간DB_위경도.csv'
    output_csv = os.path.join(base_dir, '산불_공간DB_수종별임상도2020_전국.csv')

    zip_files = sorted(glob.glob(os.path.join(base_dir, '*.zip')))
    print(f"-> 총 {len(zip_files)}개의 ZIP 파일을 발견했습니다.")

    print(f"\n1. 산불 공간DB 로딩 중...")
    fire_df = pd.read_csv(fire_csv_path, encoding='utf-8-sig') 
    fire_gdf = gpd.GeoDataFrame(
        fire_df, geometry=gpd.points_from_xy(fire_df['경도'], fire_df['위도']), crs="EPSG:4326"
    ).to_crs("EPSG:5179")

    columns_to_read = ['FRTP_CD', 'KOFTR_GROU', 'DMCLS_CD', 'AGCLS_CD', 'DNST_CD']
    matched_results = []
    start_time = time.time()
    
    print("\n2. ZIP 파일 압축해제 없이 다이렉트 매핑 시작...")
    for i, zip_path in enumerate(zip_files, 1):
        filename = os.path.basename(zip_path)
        desc = decode_filename(filename)
        
        # 파일이 작으면 진행상황이 너무 많이 출력되므로, 매칭된 건수만 출력하도록 조정
        # print(f"   [{i}/{len(zip_files)}] {desc} 탐색 중...", end='\r')
        
        try:
            # zip:// 프로토콜을 사용하면 압축을 풀지 않고도 메모리에서 바로 Shapefile을 읽을 수 있습니다.
            chunk_gdf = gpd.read_file(f"zip://{zip_path}", engine='pyogrio', columns=columns_to_read)
            joined = gpd.sjoin(fire_gdf, chunk_gdf, how='inner', predicate='within')
            
            if not joined.empty:
                matched_results.append(joined)
                print(f"   [{i}/{len(zip_files)}] 🎯 매칭 성공! {desc} (매칭: {len(joined)}건)")
        except Exception as e:
            print(f"\n   [{i}/{len(zip_files)}] ⚠️ {filename} 읽기 실패: {e}")

    print(f"\n3. 전체 스캔 완료! (총 소요 시간: {time.time() - start_time:.1f}초)")
    
    if matched_results:
        all_matched = pd.concat(matched_results, ignore_index=True)
        all_matched = all_matched.drop_duplicates(subset=['fire_id'])
        final_df = pd.merge(fire_df, all_matched[columns_to_read + ['fire_id']], on='fire_id', how='left')
    else:
        print("   -> ⚠️ 매칭된 산불 좌표가 없습니다.")
        final_df = fire_df

    print("\n4. 한글명 번역 및 컬럼 정리 중...")
    for col in columns_to_read:
        if col in MAPPING_DICT:
            nm_col = col.replace('_CD', '_NM').replace('_GROU', '_NM')
            final_df[nm_col] = final_df[col].astype(str).str.strip().map(MAPPING_DICT[col])
                
    rename_dict = {
        'FRTP_CD': '임상구분코드', 'FRTP_NM': '임상구분',
        'KOFTR_GROU': '수종코드', 'KOFTR_NM': '수종',
        'DMCLS_CD': '경급코드', 'DMCLS_NM': '경급',
        'AGCLS_CD': '영급코드', 'AGCLS_NM': '영급',
        'DNST_CD': '소밀도코드', 'DNST_NM': '소밀도'
    }
    final_df = final_df.rename(columns=rename_dict)
                
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n🎉 최종 저장 완료: {output_csv}")

if __name__ == '__main__':
    main()
