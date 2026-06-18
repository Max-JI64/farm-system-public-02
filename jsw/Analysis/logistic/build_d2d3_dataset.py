from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import Point


warnings.filterwarnings("ignore")


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "학습데이터" / "학습데이터_로지스틱_D1.csv").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_project_root()
DATA_DIR = ROOT / "data" / "학습데이터"
RAW_DIR = ROOT / "원천데이터"
LOGISTIC_DIR = ROOT / "jsw" / "Analysis" / "logistic"
OUTPUT_DIR = LOGISTIC_DIR / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
METRIC_DIR = OUTPUT_DIR / "metrics"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

D1_PATH = DATA_DIR / "학습데이터_로지스틱_D1.csv"
OUT_PATH = DATA_DIR / "학습데이터_로지스틱_D2D3.csv"
LANDCOVER_PATH = (
    RAW_DIR
    / "강원도_토지피복도_세분류"
    / "강원도_토지피복도_세분류_병합_1m.gpkg"
)
FOREST_ZIP_DIR = RAW_DIR / "임상도" / "수종별임상도(나무종류지도)_시도"


FOREST_MAPPING = {
    "AGCLS_CD": {
        "1": "1영급",
        "2": "2영급",
        "3": "3영급",
        "4": "4영급",
        "5": "5영급",
        "6": "6영급",
        "7": "7영급",
        "8": "8영급",
        "9": "9영급",
    },
    "DMCLS_CD": {"0": "치수", "1": "소경목", "2": "중경목", "3": "대경목"},
    "DNST_CD": {"A": "소", "B": "중", "C": "밀"},
    "FRTP_CD": {
        "0": "무립목지/비산림",
        "1": "침엽수림",
        "2": "활엽수림",
        "3": "혼효림",
        "4": "죽림",
    },
    "KOFTR_GROU": {
        "10": "기타침엽수",
        "11": "소나무",
        "12": "잣나무",
        "13": "낙엽송",
        "14": "리기다소나무",
        "15": "곰솔",
        "16": "전나무",
        "17": "편백나무",
        "18": "삼나무",
        "19": "가문비나무",
        "20": "비자나무",
        "21": "은행나무",
        "30": "기타활엽수",
        "31": "상수리나무",
        "32": "신갈나무",
        "33": "굴참나무",
        "34": "기타 참나무류",
        "35": "오리나무",
        "36": "고로쇠나무",
        "37": "자작나무",
        "38": "박달나무",
        "39": "밤나무",
        "40": "물푸레나무",
        "41": "서어나무",
        "42": "때죽나무",
        "43": "호두나무",
        "44": "백합나무",
        "45": "포플러",
        "46": "벚나무",
        "47": "느티나무",
        "48": "층층나무",
        "49": "아까시나무",
        "60": "기타상록활엽수",
        "61": "가시나무",
        "62": "구실잣밤나무",
        "63": "녹나무",
        "64": "굴거리나무",
        "65": "황칠나무",
        "66": "사스레피나무",
        "67": "후박나무",
        "68": "새덕이",
        "77": "침활혼효림",
        "78": "죽림",
        "81": "미립목지",
        "82": "제지",
        "91": "주거지",
        "92": "초지",
        "93": "경작지",
        "94": "수체",
        "95": "과수원",
        "99": "기타",
    },
}


def normalize_code(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def make_points(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df[["샘플ID", "위도", "경도"]].copy(),
        geometry=gpd.points_from_xy(df["경도"], df["위도"]),
        crs="EPSG:4326",
    )


def pick_one_match(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame(columns=["샘플ID"])
    out = joined.sort_values(["샘플ID"]).drop_duplicates("샘플ID", keep="first")
    return pd.DataFrame(out.drop(columns=["geometry"], errors="ignore"))


def attach_landcover(df: pd.DataFrame) -> pd.DataFrame:
    print("D2: 토지피복 원천 GPKG 공간조인")
    points = make_points(df)
    info = pyogrio.read_info(LANDCOVER_PATH)
    layer = info["layer_name"]
    landcover = gpd.read_file(
        LANDCOVER_PATH,
        layer=layer,
        columns=["L1_CODE", "L1_NAME", "L2_CODE", "L2_NAME"],
        engine="pyogrio",
    )
    points_proj = points.to_crs(landcover.crs)
    joined = gpd.sjoin(
        points_proj,
        landcover,
        how="left",
        predicate="within",
    )
    matched = pick_one_match(joined)
    matched["토지피복_매칭방식"] = np.where(
        matched["L1_NAME"].notna(), "within", "unmatched"
    )

    missing_ids = set(df["샘플ID"]) - set(
        matched.loc[matched["L1_NAME"].notna(), "샘플ID"]
    )
    if missing_ids:
        print(f"  within 미매칭 {len(missing_ids):,}건: 30m nearest 보완")
        missing_points = points_proj.loc[points_proj["샘플ID"].isin(missing_ids)]
        nearest = gpd.sjoin_nearest(
            missing_points,
            landcover,
            how="left",
            max_distance=30,
            distance_col="토지피복_nearest_dist_m",
        )
        nearest = pick_one_match(nearest)
        nearest["토지피복_매칭방식"] = np.where(
            nearest["L1_NAME"].notna(), "nearest_30m", "unmatched"
        )
        matched = pd.concat(
            [matched.loc[matched["L1_NAME"].notna()], nearest],
            ignore_index=True,
        )

    keep = [
        "샘플ID",
        "L1_CODE",
        "L1_NAME",
        "L2_CODE",
        "L2_NAME",
        "토지피복_매칭방식",
    ]
    matched = matched[[c for c in keep if c in matched.columns]].drop_duplicates(
        "샘플ID", keep="first"
    )
    matched = matched.rename(
        columns={
            "L1_CODE": "토지피복_L1_CODE",
            "L1_NAME": "토지피복_L1_NAME",
            "L2_CODE": "토지피복_L2_CODE",
            "L2_NAME": "토지피복_L2_NAME",
        }
    )
    out = df.merge(matched, on="샘플ID", how="left", validate="one_to_one")
    for col in ["토지피복_L1_CODE", "토지피복_L1_NAME", "토지피복_L2_CODE", "토지피복_L2_NAME"]:
        out[col] = out[col].fillna("미상")
    out["토지피복_매칭방식"] = out["토지피복_매칭방식"].fillna("unmatched")

    out["토지피복_산림지역"] = out["토지피복_L1_NAME"].eq("산림지역").astype(np.int8)
    out["토지피복_시가화건조지역"] = out["토지피복_L1_NAME"].eq("시가화건조지역").astype(np.int8)
    out["토지피복_농업지역"] = out["토지피복_L1_NAME"].eq("농업지역").astype(np.int8)
    out["토지피복_초지"] = out["토지피복_L1_NAME"].eq("초지").astype(np.int8)
    out["토지피복_나지"] = out["토지피복_L1_NAME"].eq("나지").astype(np.int8)
    out["토지피복_도로"] = out["토지피복_L2_NAME"].eq("도로").astype(np.int8)
    out["토지피복_활엽수림"] = out["토지피복_L2_NAME"].eq("활엽수림").astype(np.int8)
    out["토지피복_침엽수림"] = out["토지피복_L2_NAME"].eq("침엽수림").astype(np.int8)
    out["토지피복_혼효림"] = out["토지피복_L2_NAME"].eq("혼효림").astype(np.int8)
    out["토지피복_산림유형"] = np.select(
        [
            out["토지피복_L2_NAME"].eq("활엽수림"),
            out["토지피복_L2_NAME"].eq("침엽수림"),
            out["토지피복_L2_NAME"].eq("혼효림"),
            out["토지피복_L1_NAME"].eq("미상"),
        ],
        ["활엽수림", "침엽수림", "혼효림", "미상"],
        default="비산림",
    )
    out["비산림_WUI_접경후보"] = (
        out["공간층"].eq("생활권-WUI") & out["토지피복_산림지역"].eq(0)
    ).astype(np.int8)
    return out


def forest_zip_inventory(points_5179: gpd.GeoDataFrame) -> pd.DataFrame:
    inventory_path = FEATURE_DIR / "d2d3_forest_zip_inventory.csv"
    if inventory_path.exists():
        inventory = pd.read_csv(inventory_path, encoding="utf-8-sig")
    else:
        rows = []
        for path in sorted(FOREST_ZIP_DIR.glob("FRT001102_*.zip")):
            try:
                info = pyogrio.read_info(f"zip://{path.resolve()}")
                minx, miny, maxx, maxy = info["total_bounds"]
                rows.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "minx": minx,
                        "miny": miny,
                        "maxx": maxx,
                        "maxy": maxy,
                        "features": info["features"],
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "error": repr(exc),
                    }
                )
        inventory = pd.DataFrame(rows)
        inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")

    minx, miny, maxx, maxy = points_5179.total_bounds
    pad = 1000.0
    if "error" in inventory.columns:
        usable = inventory.loc[inventory["error"].isna()].copy()
    else:
        usable = inventory.copy()
    for col in ["minx", "miny", "maxx", "maxy", "features"]:
        usable[col] = pd.to_numeric(usable[col], errors="coerce")
    usable = usable.dropna(subset=["minx", "miny", "maxx", "maxy"])
    intersects = (
        usable["maxx"].ge(minx - pad)
        & usable["minx"].le(maxx + pad)
        & usable["maxy"].ge(miny - pad)
        & usable["miny"].le(maxy + pad)
    )
    candidates = usable.loc[intersects].copy()
    candidates.to_csv(
        FEATURE_DIR / "d2d3_forest_zip_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return candidates


def attach_forest(df: pd.DataFrame) -> pd.DataFrame:
    print("D3: 수종별 임상도 zip 공간조인")
    points = make_points(df).to_crs("EPSG:5179")
    candidates = forest_zip_inventory(points)
    print(f"  bbox 후보 zip: {len(candidates):,}개")

    attr_cols = ["FRTP_CD", "KOFTR_GROU", "DMCLS_CD", "AGCLS_CD", "DNST_CD"]
    matched_parts = []
    for i, row in enumerate(candidates.itertuples(index=False), start=1):
        path = Path(row.path)
        try:
            forest = gpd.read_file(
                f"zip://{path.resolve()}",
                engine="pyogrio",
                columns=attr_cols,
            )
            if forest.empty:
                continue
            if forest.crs is None:
                forest = forest.set_crs("EPSG:5179")
            elif str(forest.crs).upper() != "EPSG:5179":
                forest = forest.to_crs("EPSG:5179")
            joined = gpd.sjoin(
                points,
                forest,
                how="inner",
                predicate="within",
            )
            if not joined.empty:
                part = joined[["샘플ID", *attr_cols]].copy()
                part["임상도_zip"] = path.name
                matched_parts.append(part)
                print(f"  [{i}/{len(candidates)}] {path.name}: {len(part):,}건")
        except Exception as exc:
            print(f"  [{i}/{len(candidates)}] {path.name}: 실패 {exc}")

    if matched_parts:
        matched = pd.concat(matched_parts, ignore_index=True)
        matched["임상도_매칭수"] = matched.groupby("샘플ID")["샘플ID"].transform("size")
        matched = matched.drop_duplicates("샘플ID", keep="first")
    else:
        matched = pd.DataFrame(columns=["샘플ID", *attr_cols, "임상도_zip", "임상도_매칭수"])

    out = df.merge(matched, on="샘플ID", how="left", validate="one_to_one")
    out["임상도_매칭수"] = out["임상도_매칭수"].fillna(0).astype(int)
    out["임상도_매칭여부"] = out["임상도_매칭수"].gt(0).astype(np.int8)
    out["임상도_출처"] = np.where(out["임상도_매칭여부"].eq(1), "2020_수종별zip", "미매칭")

    code_to_new = {
        "FRTP_CD": "임상구분코드",
        "KOFTR_GROU": "수종코드",
        "DMCLS_CD": "경급코드",
        "AGCLS_CD": "영급코드",
        "DNST_CD": "소밀도코드",
    }
    name_cols = {
        "FRTP_CD": "임상구분",
        "KOFTR_GROU": "수종",
        "DMCLS_CD": "경급",
        "AGCLS_CD": "영급",
        "DNST_CD": "소밀도",
    }
    for source, target in code_to_new.items():
        out[target] = out[source].map(normalize_code).fillna("")
        out[name_cols[source]] = out[target].map(FOREST_MAPPING[source]).fillna("미상")

    out["임상_영급_숫자"] = pd.to_numeric(out["영급코드"], errors="coerce").fillna(0).astype(int)
    out["임상_경급_숫자"] = pd.to_numeric(out["경급코드"], errors="coerce").fillna(-1).astype(int)
    out["임상_소밀도_순서"] = out["소밀도코드"].map({"A": 1, "B": 2, "C": 3}).fillna(0).astype(int)
    out["임상_산림여부"] = out["임상구분코드"].isin(["1", "2", "3", "4"]).astype(np.int8)
    out["임상_침엽수림"] = out["임상구분코드"].eq("1").astype(np.int8)
    out["임상_활엽수림"] = out["임상구분코드"].eq("2").astype(np.int8)
    out["임상_혼효림"] = out["임상구분코드"].eq("3").astype(np.int8)
    out["임상_소나무류"] = out["수종코드"].isin(["11", "12", "14", "15"]).astype(np.int8)
    out["임상_침엽수_수종"] = out["수종코드"].isin(
        ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    ).astype(np.int8)
    out["임상_수종_대분류"] = np.select(
        [
            out["수종코드"].isin(["11", "12", "14", "15"]),
            out["임상_침엽수_수종"].eq(1),
            out["수종코드"].isin([str(v) for v in range(30, 69)]),
            out["수종코드"].eq("77"),
            out["수종코드"].isin(["81", "82", "91", "92", "93", "94", "95", "99"]),
            out["수종코드"].eq(""),
        ],
        ["소나무류", "기타침엽수", "활엽수", "혼효림", "비산림", "미상"],
        default="기타",
    )
    out = out.drop(columns=[c for c in attr_cols if c in out.columns])
    return out


def write_audits(df: pd.DataFrame) -> None:
    new_cols = [
        c
        for c in df.columns
        if c.startswith("토지피복_")
        or c.startswith("임상")
        or c in ["산림유형", "비산림_WUI_접경후보", "수종", "영급", "경급", "소밀도", "임상구분"]
    ]
    df[["샘플ID", "Target", "샘플유형", "기후지형유형", *new_cols]].to_csv(
        FEATURE_DIR / "d2d3_feature_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target_counts": df["Target"].value_counts(dropna=False).sort_index().to_dict(),
        "sample_type_counts": df["샘플유형"].value_counts(dropna=False).to_dict(),
        "landcover_match": df["토지피복_매칭방식"].value_counts(dropna=False).to_dict(),
        "landcover_l1": df["토지피복_L1_NAME"].value_counts(dropna=False).to_dict(),
        "landcover_l2": df["토지피복_L2_NAME"].value_counts(dropna=False).head(30).to_dict(),
        "forest_match": df["임상도_출처"].value_counts(dropna=False).to_dict(),
        "forest_type": df["임상구분"].value_counts(dropna=False).to_dict(),
        "tree_species_top": df["수종"].value_counts(dropna=False).head(30).to_dict(),
    }
    (FEATURE_DIR / "d2d3_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    by_target = []
    for col in [
        "토지피복_L1_NAME",
        "토지피복_L2_NAME",
        "토지피복_산림유형",
        "임상구분",
        "수종",
        "임상_수종_대분류",
    ]:
        table = (
            df.groupby(["Target", col], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        table["variable"] = col
        by_target.append(table.rename(columns={col: "category"}))
    pd.concat(by_target, ignore_index=True).to_csv(
        METRIC_DIR / "d2d3_category_distribution_by_target.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    print("D2D3 데이터 생성 시작")
    df = pd.read_csv(D1_PATH, encoding="utf-8-sig", low_memory=False)
    original_shape = df.shape
    if not df["샘플ID"].is_unique:
        raise ValueError("D1 샘플ID가 유일하지 않습니다.")

    df = attach_landcover(df)
    df = attach_forest(df)

    if df.shape[0] != original_shape[0]:
        raise ValueError(f"행 수 변경 발생: {original_shape[0]} -> {df.shape[0]}")
    if not df["샘플ID"].is_unique:
        raise ValueError("D2D3 샘플ID가 유일하지 않습니다.")
    if df["Target"].isna().any():
        raise ValueError("Target 결측 발생")

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    write_audits(df)
    print(f"저장 완료: {OUT_PATH}")
    print(f"행/열: {df.shape[0]:,} × {df.shape[1]:,}")
    print("D2D3 데이터 생성 완료")


if __name__ == "__main__":
    main()
