from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "data" / "학습데이터").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_project_root()
DATA_DIR = ROOT / "data" / "학습데이터"
ANALYSIS_DIR = ROOT / "jsw" / "Analysis" / "new_machine_learning"

BASE_PATH = DATA_DIR / "학습데이터_최종_캐나다지수.csv"
D2D3_PATH = DATA_DIR / "학습데이터_로지스틱_D2D3.csv"
FFMC_DAILY_PATH = DATA_DIR / "캐나다_FFMC_선행연구_일단위.csv"
FWI_DAILY_PATH = DATA_DIR / "캐나다_FWI_일단위.csv"

OUTPUT_PATH = DATA_DIR / "최종_머신러닝_학습데이터.csv"
DICTIONARY_PATH = ANALYSIS_DIR / "최종_머신러닝_변수사전.csv"
README_PATH = ANALYSIS_DIR / "최종_머신러닝_학습데이터_설명.md"

FACTOR_COLUMNS = ["F1_score", "F2_score", "F3_score", "F4_score", "F5_score"]
CANADIAN_INDEX_COLUMNS = [
    "FFMC",
    "FFMC_10일평균",
    "Indexed_FFMC",
    "FFMC_논문식_발생확률",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI",
]

SPLIT_METADATA_COLUMNS = [
    "fire_id",
    "원본_fire_id",
    "후보점ID",
    "source_fire_id",
    "모델링_그룹ID",
]

# 기존 최종 데이터에 이미 존재하는 도로·임도·산림 log1p 변수와 의미가
# 중복되지 않는 D1 파생변수만 추가한다.
ADDITIONAL_ENGINEERED_COLUMNS = [
    "log1p_시가화거리_m",
    "log1p_농업거리_m",
    "log1p_등산로거리_m",
    "월_sin",
    "월_cos",
    "시간_sin",
    "시간_cos",
]

LANDCOVER_COLUMNS = [
    "토지피복_L1_CODE",
    "토지피복_L1_NAME",
    "토지피복_L2_CODE",
    "토지피복_L2_NAME",
    "토지피복_매칭방식",
    "토지피복_산림지역",
    "토지피복_시가화건조지역",
    "토지피복_농업지역",
    "토지피복_초지",
    "토지피복_나지",
    "토지피복_도로",
    "토지피복_활엽수림",
    "토지피복_침엽수림",
    "토지피복_혼효림",
    "토지피복_산림유형",
    "비산림_WUI_접경후보",
]

FOREST_COLUMNS = [
    "임상도_zip",
    "임상도_매칭수",
    "임상도_매칭여부",
    "임상도_출처",
    "임상구분코드",
    "임상구분",
    "수종코드",
    "수종",
    "경급코드",
    "경급",
    "영급코드",
    "영급",
    "소밀도코드",
    "소밀도",
    "임상_영급_숫자",
    "임상_경급_숫자",
    "임상_소밀도_순서",
    "임상_산림여부",
    "임상_침엽수림",
    "임상_활엽수림",
    "임상_혼효림",
    "임상_소나무류",
    "임상_침엽수_수종",
    "임상_수종_대분류",
]

D2D3_ADDITION_COLUMNS = (
    SPLIT_METADATA_COLUMNS
    + ADDITIONAL_ENGINEERED_COLUMNS
    + LANDCOVER_COLUMNS
    + FOREST_COLUMNS
)

LEGACY_TO_D2D3_LOG_COLUMNS = {
    "log1p_도로_최단거리_m": "log1p_도로거리_m",
    "log1p_임도_최단거리_m": "log1p_임도거리_m",
    "log1p_산림지역_최단거리_m": "log1p_산림거리_m",
}


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def validate_source_data(base: pd.DataFrame, d2d3: pd.DataFrame) -> None:
    required_base = {
        "샘플ID",
        "기준시각",
        "기상셀ID",
        "Target",
        *FACTOR_COLUMNS,
        *CANADIAN_INDEX_COLUMNS,
    }
    required_d2d3 = {"샘플ID", *D2D3_ADDITION_COLUMNS}
    missing_base = sorted(required_base - set(base.columns))
    missing_d2d3 = sorted(required_d2d3 - set(d2d3.columns))
    if missing_base:
        raise KeyError(f"기본 데이터 누락 컬럼: {missing_base}")
    if missing_d2d3:
        raise KeyError(f"D2D3 데이터 누락 컬럼: {missing_d2d3}")

    for name, frame in [("base", base), ("d2d3", d2d3)]:
        if frame["샘플ID"].isna().any():
            raise ValueError(f"{name}: 샘플ID 결측이 있습니다.")
        if frame["샘플ID"].duplicated().any():
            raise ValueError(f"{name}: 샘플ID 중복이 있습니다.")

    if set(base["샘플ID"]) != set(d2d3["샘플ID"]):
        raise ValueError("기본 데이터와 D2D3 데이터의 샘플ID 집합이 다릅니다.")
    if len(base) != 17045:
        raise ValueError(f"예상 행 수 17,045와 다릅니다: {len(base):,}")
    if base["Target"].value_counts().sort_index().to_dict() != {0: 15492, 1: 1553}:
        raise ValueError("Target 분포가 기존 확정값과 다릅니다.")

    base_target = base.set_index("샘플ID")["Target"].sort_index()
    d2d3_target = d2d3.set_index("샘플ID")["Target"].sort_index()
    if not base_target.equals(d2d3_target):
        raise ValueError("기본 데이터와 D2D3 데이터의 Target 값이 다릅니다.")

    for legacy, current in LEGACY_TO_D2D3_LOG_COLUMNS.items():
        left = pd.to_numeric(base[legacy], errors="coerce")
        right = pd.to_numeric(
            d2d3.set_index("샘플ID").loc[base["샘플ID"], current].reset_index(drop=True),
            errors="coerce",
        )
        if not np.allclose(left, right, equal_nan=True, rtol=0.0, atol=1e-12):
            raise ValueError(f"중복 log1p 변수 값이 일치하지 않습니다: {legacy} vs {current}")


def load_daily_canadian_indices() -> pd.DataFrame:
    ffmc = read_csv(FFMC_DAILY_PATH)
    fwi = read_csv(FWI_DAILY_PATH)
    for frame in [ffmc, fwi]:
        frame["날짜"] = pd.to_datetime(frame["날짜"], errors="raise").dt.normalize()
        if frame.duplicated(["기상셀ID", "날짜"]).any():
            raise ValueError("일단위 캐나다지수에 기상셀ID+날짜 중복이 있습니다.")

    ffmc_check = ffmc[["기상셀ID", "날짜", "FFMC"]].merge(
        fwi[["기상셀ID", "날짜", "FFMC"]],
        on=["기상셀ID", "날짜"],
        how="inner",
        suffixes=("_FFMC트랙", "_FWI트랙"),
        validate="one_to_one",
    )
    if len(ffmc_check) != len(ffmc) or len(ffmc_check) != len(fwi):
        raise ValueError("FFMC 트랙과 FWI 트랙의 기상셀ID+날짜 범위가 다릅니다.")
    max_diff = (
        ffmc_check["FFMC_FFMC트랙"] - ffmc_check["FFMC_FWI트랙"]
    ).abs().max()
    if float(max_diff) > 1e-9:
        raise ValueError(f"FFMC 트랙 간 값이 다릅니다: max_abs_diff={max_diff}")

    daily = ffmc[
        [
            "기상셀ID",
            "날짜",
            "FFMC",
            "FFMC_10일평균",
            "Indexed_FFMC",
            "FFMC_논문식_발생확률",
        ]
    ].merge(
        fwi[["기상셀ID", "날짜", "DMC", "DC", "ISI", "BUI", "FWI"]],
        on=["기상셀ID", "날짜"],
        how="inner",
        validate="one_to_one",
    )
    if daily[CANADIAN_INDEX_COLUMNS].isna().any().any():
        raise ValueError("일단위 캐나다지수에 결측이 있습니다.")
    return daily


def attach_leak_safe_canadian_indices(
    base: pd.DataFrame, daily: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    data = base.drop(columns=CANADIAN_INDEX_COLUMNS).copy()
    data["기준시각"] = pd.to_datetime(data["기준시각"], errors="raise")

    before_noon = data["기준시각"].dt.hour.lt(12)
    reference_date = data["기준시각"].dt.normalize()
    reference_date = reference_date.where(
        ~before_noon, reference_date - pd.Timedelta(days=1)
    )
    data["캐나다지수_기준날짜"] = reference_date
    data["캐나다지수_정책"] = "latest_available_noon_LST"

    daily_for_merge = daily.rename(columns={"날짜": "캐나다지수_기준날짜"})
    data = data.merge(
        daily_for_merge,
        on=["기상셀ID", "캐나다지수_기준날짜"],
        how="left",
        validate="many_to_one",
    )
    missing = data[CANADIAN_INDEX_COLUMNS].isna().sum()
    if int(missing.sum()) > 0:
        raise ValueError(f"원문 재현형 캐나다지수 병합 결측: {missing.to_dict()}")

    current_date = data["기준시각"].dt.normalize()
    day_gap = (current_date - data["캐나다지수_기준날짜"]).dt.days
    if not day_gap.loc[before_noon].eq(1).all():
        raise ValueError("12시 이전 표본에 D-1 정오 지수가 적용되지 않았습니다.")
    if not day_gap.loc[~before_noon].eq(0).all():
        raise ValueError("12시 이후 표본에 당일 정오 지수가 적용되지 않았습니다.")

    audit = {
        "before_noon_n": int(before_noon.sum()),
        "at_or_after_noon_n": int((~before_noon).sum()),
        "before_noon_positive_n": int((before_noon & data["Target"].eq(1)).sum()),
        "at_or_after_noon_positive_n": int(
            ((~before_noon) & data["Target"].eq(1)).sum()
        ),
    }
    return data, audit


def attach_d2d3_additions(data: pd.DataFrame, d2d3: pd.DataFrame) -> pd.DataFrame:
    additions = d2d3[["샘플ID", *D2D3_ADDITION_COLUMNS]].copy()
    output = data.merge(additions, on="샘플ID", how="left", validate="one_to_one")
    if len(output) != len(data):
        raise ValueError("D2D3 변수 병합 후 행 수가 변경됐습니다.")
    return output


EXACT_DESCRIPTIONS = {
    "샘플ID": "각 학습 표본을 유일하게 식별하는 ID",
    "기준시각": "산불 발생 또는 비발생 대조 표본의 기준 날짜와 시간",
    "위도": "표본 위치의 위도",
    "경도": "표본 위치의 경도",
    "기상셀ID": "표본에 연결된 시간단위 기상 격자 ID",
    "기후지형유형": "영동 해안형·영서 내륙형·고지 산간형 등 기후·지형 구분",
    "월_key": "기준시각에서 추출한 월(1~12)",
    "시간_key": "기준시각에서 추출한 시각(0~23)",
    "샘플유형": "Target_1, Target_0A, Target_0B1, Target_0B2 표본 유형",
    "Target": "산불 발생 여부 목적변수(발생 1, 비발생 0)",
    "실험안": "음성표본 생성 및 실험 설계 구분",
    "샘플가중치": "샘플링 설계를 보정하기 위한 학습 가중치",
    "시간샘플링방식": "비발생 표본의 시간 선택 방식",
    "공간층": "비발생 표본이 추출된 공간 후보층",
    "고도(m)": "표본 위치의 해발고도",
    "경사도(도)": "표본 위치의 지형 경사각",
    "사면방향_sin": "사면방향을 사인으로 순환 변환한 값",
    "사면방향_cos": "사면방향을 코사인으로 순환 변환한 값",
    "TPI(지형위치지수)": "주변 지형 대비 표본 위치의 상대적 고저를 나타내는 TPI",
    "시점_기온_C": "기준시각의 기온",
    "시점_풍속_m_s": "기준시각의 풍속",
    "시점_습도_pct": "기준시각의 상대습도",
    "시점_풍향_deg": "기준시각 풍향의 각도",
    "풍향_sin": "기준시각 풍향의 사인 변환값",
    "풍향_cos": "기준시각 풍향의 코사인 변환값",
    "서풍계열_여부": "기준시각 풍향이 서풍 계열이면 1, 아니면 0",
    "시점_현지기압_hPa": "기준시각의 현지기압",
    "시점_해면기압_hPa": "기준시각의 해면기압",
    "기압변동_3h": "기준시각 현지기압과 3시간 전 현지기압의 차이",
    "월_sin": "월을 12개월 주기로 사인 변환한 값",
    "월_cos": "월을 12개월 주기로 코사인 변환한 값",
    "시간_sin": "시간을 24시간 주기로 사인 변환한 값",
    "시간_cos": "시간을 24시간 주기로 코사인 변환한 값",
    "캐나다지수_기준날짜": "해당 표본에 실제 연결된 정오 캐나다지수 날짜",
    "캐나다지수_정책": "캐나다지수 as-of 병합 정책 코드",
    "FFMC": "정오 기상과 전일 상태로 계산한 미세연료수분지수",
    "FFMC_10일평균": "기상셀별 FFMC의 최근 10일 이동평균",
    "Indexed_FFMC": "FFMC 10일 평균을 선행연구 기준 4단계로 변환한 지수",
    "FFMC_논문식_발생확률": "Indexed_FFMC를 강원도 선행연구 로지스틱 식에 대입한 참고값",
    "DMC": "중간 깊이 유기물층의 누적 건조도를 나타내는 Duff Moisture Code",
    "DC": "깊은 유기물층의 장기 건조도를 나타내는 Drought Code",
    "ISI": "FFMC와 풍속을 결합한 초기확산지수",
    "BUI": "DMC와 DC를 결합한 가용연료 누적건조지수",
    "FWI": "ISI와 BUI를 결합한 최종 화재기상위험지수",
    "fire_id": "Target 1 산불 사건 식별자",
    "원본_fire_id": "대조 표본이 연결된 원본 산불 사건 식별자",
    "후보점ID": "공간 대조 후보점 식별자",
    "source_fire_id": "음성표본 생성 시 참조한 산불 사건 식별자",
    "모델링_그룹ID": "연결된 양성·대조 표본을 같은 fold에 배치하기 위한 그룹 ID",
    "토지피복_L1_CODE": "세분류 토지피복도의 대분류 코드",
    "토지피복_L1_NAME": "세분류 토지피복도의 대분류 명칭",
    "토지피복_L2_CODE": "세분류 토지피복도의 중분류 코드",
    "토지피복_L2_NAME": "세분류 토지피복도의 중분류 명칭",
    "토지피복_매칭방식": "공간점이 토지피복 폴리곤에 연결된 방식",
    "토지피복_산림유형": "활엽수림·침엽수림·혼효림·비산림·미상 구분",
    "비산림_WUI_접경후보": "생활권-WUI 표본 중 토지피복상 비산림인 접경 후보 여부",
    "임상도_zip": "표본에 매칭된 수종별 임상도 원천 zip 파일명",
    "임상도_매칭수": "표본 위치와 공간적으로 매칭된 임상도 객체 수",
    "임상도_매칭여부": "임상도 매칭이 존재하면 1, 아니면 0",
    "임상도_출처": "임상도 매칭 자료의 출처 또는 미매칭 표시",
    "임상구분코드": "임상도의 산림 형태 코드",
    "임상구분": "임상도의 산림 형태 명칭",
    "수종코드": "임상도의 대표 수종 코드",
    "수종": "임상도의 대표 수종 명칭",
    "경급코드": "임목 흉고직경 등급 코드",
    "경급": "임목 흉고직경 등급 명칭",
    "영급코드": "임령 등급 코드",
    "영급": "임령 등급 명칭",
    "소밀도코드": "임목 밀도 등급 코드",
    "소밀도": "임목 밀도 등급 명칭",
    "임상_영급_숫자": "영급 코드를 모델용 숫자로 변환한 값",
    "임상_경급_숫자": "경급 코드를 모델용 숫자로 변환한 값",
    "임상_소밀도_순서": "소밀도를 소·중·밀 순서의 숫자로 변환한 값",
    "임상_산림여부": "임상구분상 산림이면 1, 아니면 0",
    "임상_침엽수림": "임상구분상 침엽수림이면 1, 아니면 0",
    "임상_활엽수림": "임상구분상 활엽수림이면 1, 아니면 0",
    "임상_혼효림": "임상구분상 혼효림이면 1, 아니면 0",
    "임상_소나무류": "대표 수종이 소나무류이면 1, 아니면 0",
    "임상_침엽수_수종": "대표 수종이 침엽수 계열이면 1, 아니면 0",
    "임상_수종_대분류": "수종을 소나무류·기타침엽수·활엽수 등으로 재분류한 값",
}

DISTANCE_LABELS = {
    "도로": "도로",
    "시가화": "시가화 지역",
    "농업": "농업 지역",
    "임도": "임도",
    "등산로": "등산로",
    "산림": "산림 지역",
    "산림지역": "산림 지역",
}


def describe_column(column: str) -> str:
    if column in EXACT_DESCRIPTIONS:
        return EXACT_DESCRIPTIONS[column]

    if column.endswith("_최단거리_m"):
        key = column.removesuffix("_최단거리_m")
        return f"표본 위치에서 가장 가까운 {DISTANCE_LABELS.get(key, key)}까지 거리"

    if column.startswith("log1p_"):
        key = column.removeprefix("log1p_").replace("_최단거리_m", "").replace("_m", "")
        return f"{DISTANCE_LABELS.get(key, key)} 거리의 log(1+x) 변환값"

    rolling = re.fullmatch(r"직전(24|48)h_(평균|최대|최소)?(풍속|기온_C|습도|강수량합)", column)
    if rolling:
        hours, statistic, variable = rolling.groups()
        statistic = statistic or ""
        return f"기준시각 이전 {hours}시간의 {variable} {statistic}값".strip()

    lagged = re.fullmatch(r"D-(1|2|3)_(최소습도_pct|평균습도_pct|강수량합_mm)", column)
    if lagged:
        lag, variable = lagged.groups()
        return f"기준일 {lag}일 전의 일단위 {variable}"

    if column.startswith("토지피복_"):
        label = column.removeprefix("토지피복_")
        return f"토지피복 분류가 {label}에 해당하면 1, 아니면 0"

    raise KeyError(f"변수 설명이 정의되지 않았습니다: {column}")


def column_group(column: str) -> str:
    if column in {"샘플ID", "기준시각", "위도", "경도", "기상셀ID"}:
        return "식별·시공간"
    if column in {"Target", "샘플유형", "실험안", "샘플가중치", "시간샘플링방식", "공간층"}:
        return "목적변수·표본설계"
    if column in SPLIT_METADATA_COLUMNS:
        return "분할 메타데이터"
    if column in {"기후지형유형", "월_key", "시간_key", "월_sin", "월_cos", "시간_sin", "시간_cos"}:
        return "계절·시간"
    if column in CANADIAN_INDEX_COLUMNS or column.startswith("캐나다지수_"):
        return "캐나다 FWI"
    if column in LANDCOVER_COLUMNS:
        return "토지피복"
    if column in FOREST_COLUMNS:
        return "임상도"
    if any(token in column for token in ["거리", "고도", "경사", "사면방향", "TPI"]):
        return "공간·지형"
    return "기상"


def model_usage(column: str) -> str:
    if column == "Target":
        return "목적변수"
    if column == "샘플가중치":
        return "가중치로만 사용"
    if column in {"샘플ID", "기준시각", "기상셀ID", *SPLIT_METADATA_COLUMNS}:
        return "분할·추적용, X 제외"
    if column in {"샘플유형", "실험안", "시간샘플링방식", "공간층"}:
        return "누수 위험, 반드시 X 제외"
    if column in {"캐나다지수_기준날짜", "캐나다지수_정책"}:
        return "검증용, X 제외"
    if column in {"위도", "경도"}:
        return "기본 X 제외"
    if column in {"월_key", "시간_key"}:
        return "주기변환 사용 시 X 제외"
    if column == "시점_풍향_deg":
        return "sin/cos 사용 시 X 제외"
    if column in FOREST_COLUMNS:
        return "매칭률 점검 후 후보"
    if column in {
        "토지피복_L1_CODE",
        "토지피복_L1_NAME",
        "토지피복_L2_CODE",
        "토지피복_L2_NAME",
    }:
        return "코드·명칭 중 표현 택1"
    if column.endswith("_최단거리_m"):
        return "log1p 변수와 비교 후 택1"
    return "모델 후보"


def source_for_column(column: str) -> str:
    if column in CANADIAN_INDEX_COLUMNS:
        return "캐나다_FFMC/FWI_일단위.csv 재병합"
    if column.startswith("캐나다지수_"):
        return "최종 데이터 생성 스크립트"
    if column in D2D3_ADDITION_COLUMNS:
        return "학습데이터_로지스틱_D2D3.csv"
    return "학습데이터_최종_캐나다지수.csv"


def unit_for_column(column: str) -> str:
    if column in {"기준시각"}:
        return "YYYY-MM-DD HH:MM:SS"
    if column == "캐나다지수_기준날짜":
        return "YYYY-MM-DD"
    if column in {"위도", "경도", "시점_풍향_deg", "경사도(도)"}:
        return "degree"
    if column.endswith("_m") or column.endswith("(m)"):
        return "m"
    if "기온" in column:
        return "°C"
    if "풍속" in column and "여부" not in column:
        return "m/s"
    if "습도" in column:
        return "%"
    if "강수량" in column:
        return "mm"
    if "기압" in column:
        return "hPa"
    return "-"


def make_variable_dictionary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for number, column in enumerate(data.columns, start=1):
        rows.append(
            {
                "순번": number,
                "변수명": column,
                "그룹": column_group(column),
                "자료형": str(data[column].dtype),
                "단위_형식": unit_for_column(column),
                "모델사용": model_usage(column),
                "설명": describe_column(column),
                "출처": source_for_column(column),
                "결측수": int(data[column].isna().sum()),
                "고유값수": int(data[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.fillna("").astype(str)
    headers = display.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        escaped = [value.replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def build_readme(
    data: pd.DataFrame,
    dictionary: pd.DataFrame,
    canadian_audit: dict[str, int],
    old_base: pd.DataFrame,
) -> str:
    target_counts = data["Target"].value_counts().sort_index()
    sample_counts = data["샘플유형"].value_counts()
    group_summary = (
        dictionary.groupby("그룹", sort=False)
        .size()
        .rename("변수수")
        .reset_index()
    )
    landcover_counts = {
        str(key): int(value)
        for key, value in data["토지피복_매칭방식"]
        .value_counts(dropna=False)
        .items()
    }
    forest_counts = {
        str(key): int(value)
        for key, value in data["임상도_매칭여부"]
        .value_counts(dropna=False)
        .items()
    }

    comparison = []
    for column in CANADIAN_INDEX_COLUMNS:
        old = pd.to_numeric(old_base[column], errors="coerce")
        new = pd.to_numeric(data[column], errors="coerce")
        changed = ~np.isclose(old, new, equal_nan=True, rtol=0.0, atol=1e-12)
        comparison.append(
            {
                "지수": column,
                "변경행수": int(changed.sum()),
                "변경비율": f"{changed.mean():.4f}",
            }
        )
    comparison_df = pd.DataFrame(comparison)

    validation = pd.DataFrame(
        [
            {"검증항목": "전체 행 수", "결과": f"{len(data):,}", "통과": "예"},
            {"검증항목": "전체 변수 수", "결과": f"{len(data.columns):,}", "통과": "예"},
            {
                "검증항목": "샘플ID 중복",
                "결과": f"{int(data['샘플ID'].duplicated().sum()):,}",
                "통과": "예",
            },
            {
                "검증항목": "캐나다지수 결측",
                "결과": f"{int(data[CANADIAN_INDEX_COLUMNS].isna().sum().sum()):,}",
                "통과": "예",
            },
            {
                "검증항목": "요인점수 잔존",
                "결과": ", ".join(sorted(set(FACTOR_COLUMNS) & set(data.columns))) or "없음",
                "통과": "예",
            },
            {
                "검증항목": "12시 이전 D-1 정오 적용",
                "결과": f"{canadian_audit['before_noon_n']:,}행",
                "통과": "예",
            },
            {
                "검증항목": "12시 이후 당일 정오 적용",
                "결과": f"{canadian_audit['at_or_after_noon_n']:,}행",
                "통과": "예",
            },
        ]
    )

    variable_table = dictionary[
        ["순번", "변수명", "그룹", "자료형", "단위_형식", "모델사용", "설명", "출처", "결측수"]
    ]

    lines = [
        "# 최종 머신러닝 학습데이터 설명",
        "",
        "## 1. 산출물",
        "",
        f"- 최종 CSV: `{OUTPUT_PATH.relative_to(ROOT)}`",
        f"- 생성 코드: `{Path(__file__).resolve().relative_to(ROOT)}`",
        f"- 변수사전 CSV: `{DICTIONARY_PATH.relative_to(ROOT)}`",
        "",
        "## 2. 생성 목적",
        "",
        "기존 `학습데이터_최종_캐나다지수.csv`의 샘플과 기본 변수를 유지하면서 다음 문제를 수정한 새 머신러닝 입력 데이터다.",
        "",
        "- 같은 날짜의 정오 캐나다지수를 모든 시간대에 붙이던 시간 누수를 제거했다.",
        "- 더 이상 사용하지 않는 요인점수 `F1_score`~`F5_score`를 제거했다.",
        "- D2D3 데이터에서 분할 메타데이터, 추가 거리·주기 파생변수, 토지피복, 임상도 변수를 `샘플ID`로 결합했다.",
        "- 모델 입력 변수와 분할·검증·누수 위험 변수를 변수사전에서 명시적으로 구분했다.",
        "",
        "## 3. 캐나다 산불지수 적용 정책",
        "",
        "Van Wagner(1987)의 표준 FWI는 매일 정오 지방표준시(LST)의 기온·상대습도·풍속과 정오까지 직전 24시간 강수량으로 계산하는 일단위 지수다.",
        "이 데이터에서는 각 표본 시점에 실제로 이용 가능한 가장 최근 정오 지수를 연결했다.",
        "",
        "| 기준시각 | 연결 지수 | 이유 |",
        "| --- | --- | --- |",
        "| 00:00~11:59 | 전날 정오 지수 | 당일 정오 관측은 아직 미래 정보 |",
        "| 12:00~23:59 | 당일 정오 지수 | 당일 정오 관측이 이용 가능한 시점 |",
        "",
        f"- 12시 이전: {canadian_audit['before_noon_n']:,}행, Target 1 {canadian_audit['before_noon_positive_n']:,}행",
        f"- 12시 이후: {canadian_audit['at_or_after_noon_n']:,}행, Target 1 {canadian_audit['at_or_after_noon_positive_n']:,}행",
        "- `캐나다지수_기준날짜`로 실제 연결 날짜를 검증할 수 있다.",
        "- `캐나다지수_정책` 값은 `latest_available_noon_LST`로 고정했다.",
        "",
        "### 기존 값 대비 변경",
        "",
        markdown_table(comparison_df),
        "",
        "12시 이전 표본은 전날 정오 값으로 교체되므로 값이 달라진다. 12시 이후 표본은 기존과 동일한 당일 정오 값을 사용한다.",
        "",
        "## 4. 데이터 규모",
        "",
        f"- 행 수: {len(data):,}",
        f"- 변수 수: {len(data.columns):,}",
        f"- Target 0: {int(target_counts.get(0, 0)):,}",
        f"- Target 1: {int(target_counts.get(1, 0)):,}",
        f"- Target 1 비율: {float(data['Target'].mean()):.4%}",
        f"- 기간: {data['기준시각'].min()} ~ {data['기준시각'].max()}",
        "",
        "### 표본 유형",
        "",
        markdown_table(
            sample_counts.rename_axis("샘플유형").rename("행수").reset_index()
        ),
        "",
        "### 변수 그룹",
        "",
        markdown_table(group_summary),
        "",
        "## 5. D2D3에서 추가한 변수",
        "",
        f"- 분할·추적 메타데이터: {len(SPLIT_METADATA_COLUMNS)}개",
        f"- 추가 거리·주기 파생변수: {len(ADDITIONAL_ENGINEERED_COLUMNS)}개",
        f"- 토지피복 변수: {len(LANDCOVER_COLUMNS)}개",
        f"- 임상도 변수: {len(FOREST_COLUMNS)}개",
        f"- 총 추가 변수: {len(D2D3_ADDITION_COLUMNS)}개",
        "",
        "기존 데이터에 이미 존재하는 아래 log1p 변수는 중복 추가하지 않았다.",
        "",
        "- `log1p_도로_최단거리_m` ↔ `log1p_도로거리_m`",
        "- `log1p_임도_최단거리_m` ↔ `log1p_임도거리_m`",
        "- `log1p_산림지역_최단거리_m` ↔ `log1p_산림거리_m`",
        "",
        "### 공간자료 매칭 상태",
        "",
        f"- 토지피복 매칭: {landcover_counts}",
        f"- 임상도 매칭 여부: {forest_counts}",
        "- 임상도는 기존 생성 당시 28개 표본만 매칭되어 정보가 매우 희소하다. 데이터에는 보존하지만 기본 모델 입력으로 바로 사용하지 말고 결측·희소성 검토 후 결정한다.",
        "",
        "## 6. 모델링 시 반드시 제외하거나 분리할 변수",
        "",
        "- 목적변수: `Target`",
        "- 누수 위험: `샘플유형`, `실험안`, `시간샘플링방식`, `공간층`",
        "- 추적·분할 전용: `샘플ID`, `기준시각`, `기상셀ID`, `fire_id`, `원본_fire_id`, `후보점ID`, `source_fire_id`, `모델링_그룹ID`",
        "- 캐나다지수 검증 전용: `캐나다지수_기준날짜`, `캐나다지수_정책`",
        "- 샘플 가중치: `샘플가중치`는 X가 아니라 학습 가중치 인자로만 사용",
        "- 위도·경도: 공간 암기를 막기 위해 기본적으로 X에서 제외하고 별도 민감도 분석에서만 검토",
        "",
        "월·시간은 `월_key`, `시간_key`보다 `월_sin/cos`, `시간_sin/cos` 사용을 우선한다. 풍향도 각도 원값보다 `풍향_sin/cos` 사용을 우선한다.",
        "",
        "## 7. 검증 결과",
        "",
        markdown_table(validation),
        "",
        "## 8. 전체 변수표",
        "",
        markdown_table(variable_table),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    base = read_csv(BASE_PATH)
    d2d3 = read_csv(D2D3_PATH)
    validate_source_data(base, d2d3)

    original_base = base.copy()
    base = base.drop(columns=FACTOR_COLUMNS)
    daily = load_daily_canadian_indices()
    data, canadian_audit = attach_leak_safe_canadian_indices(base, daily)
    data = attach_d2d3_additions(data, d2d3)

    if any(column in data.columns for column in FACTOR_COLUMNS):
        raise ValueError("요인점수 컬럼이 제거되지 않았습니다.")
    if data["샘플ID"].duplicated().any() or data["샘플ID"].isna().any():
        raise ValueError("최종 데이터의 샘플ID 무결성 검증에 실패했습니다.")
    if data["Target"].value_counts().sort_index().to_dict() != {0: 15492, 1: 1553}:
        raise ValueError("최종 데이터 Target 분포가 변경됐습니다.")

    dictionary = make_variable_dictionary(data)
    if len(dictionary) != len(data.columns):
        raise ValueError("변수사전 행 수와 최종 데이터 변수 수가 다릅니다.")

    data_to_write = data.copy()
    data_to_write["캐나다지수_기준날짜"] = data_to_write[
        "캐나다지수_기준날짜"
    ].dt.strftime("%Y-%m-%d")
    data_to_write.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    dictionary.to_csv(DICTIONARY_PATH, index=False, encoding="utf-8-sig")

    readme = build_readme(data, dictionary, canadian_audit, original_base)
    README_PATH.write_text(readme, encoding="utf-8")

    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"행/열: {data.shape[0]:,} × {data.shape[1]:,}")
    print(f"저장 완료: {DICTIONARY_PATH}")
    print(f"저장 완료: {README_PATH}")


if __name__ == "__main__":
    main()
