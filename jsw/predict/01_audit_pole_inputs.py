from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predict_common import OUTPUT_DIR, POLE_CSV, POLE_SHP, ensure_output_dir, validation_frame, write_csv, write_json


def load_csv_points(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    rename_map = {}
    if "lon" not in data.columns and "경도" in data.columns:
        rename_map["경도"] = "lon"
    if "lat" not in data.columns and "위도" in data.columns:
        rename_map["위도"] = "lat"
    data = data.rename(columns=rename_map)
    if "pole_id" not in data.columns:
        data.insert(0, "pole_id", np.arange(len(data), dtype=np.int64))
    required = {"pole_id", "lon", "lat"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Pole CSV 필수 컬럼 누락: {missing}")
    return data


def load_shp_points(path: Path) -> pd.DataFrame:
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("Pole shapefile CRS가 없습니다. .prj 파일을 확인하세요.")
    gdf = gdf.to_crs("EPSG:4326")
    if "pole_id" not in gdf.columns:
        gdf.insert(0, "pole_id", np.arange(len(gdf), dtype=np.int64))
    out = pd.DataFrame(gdf.drop(columns="geometry"))
    out["lon"] = gdf.geometry.x
    out["lat"] = gdf.geometry.y
    return out


def compare_csv_shp(csv_points: pd.DataFrame, shp_points: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {
        "csv_rows": int(len(csv_points)),
        "shp_rows": int(len(shp_points)),
        "row_count_equal": bool(len(csv_points) == len(shp_points)),
    }
    if "pole_id" in csv_points.columns and "pole_id" in shp_points.columns:
        csv_ids = set(csv_points["pole_id"].astype(str))
        shp_ids = set(shp_points["pole_id"].astype(str))
        result["csv_only_pole_ids"] = int(len(csv_ids - shp_ids))
        result["shp_only_pole_ids"] = int(len(shp_ids - csv_ids))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Pole 입력 CSV/SHP 감사 및 표준 좌표 테이블 생성")
    parser.add_argument("--csv", type=Path, default=POLE_CSV)
    parser.add_argument("--shp", type=Path, default=POLE_SHP)
    parser.add_argument("--prefer", choices=["csv", "shp"], default="shp")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)

    csv_points = load_csv_points(args.csv) if args.csv.exists() else None
    shp_points = load_shp_points(args.shp) if args.shp.exists() else None
    if csv_points is None and shp_points is None:
        raise FileNotFoundError("Pole CSV와 SHP를 모두 찾지 못했습니다.")

    selected = shp_points if args.prefer == "shp" and shp_points is not None else csv_points
    if selected is None:
        selected = shp_points if shp_points is not None else csv_points
    assert selected is not None

    selected = selected.copy()
    selected["pole_id"] = selected["pole_id"].astype(str)
    selected = selected.drop_duplicates("pole_id", keep="first").reset_index(drop=True)

    audit_items = [
        ("selected_source", args.prefer if (args.prefer == "csv" and csv_points is not None) or (args.prefer == "shp" and shp_points is not None) else "fallback", True),
        ("selected_rows", f"{len(selected):,}", len(selected) > 0),
        ("pole_id_duplicates_after_dedup", "0", True),
        ("lon_missing", int(selected["lon"].isna().sum()), int(selected["lon"].isna().sum()) == 0),
        ("lat_missing", int(selected["lat"].isna().sum()), int(selected["lat"].isna().sum()) == 0),
        ("lon_range", f"{selected['lon'].min():.6f} ~ {selected['lon'].max():.6f}", selected["lon"].between(124, 132).all()),
        ("lat_range", f"{selected['lat'].min():.6f} ~ {selected['lat'].max():.6f}", selected["lat"].between(36, 39.5).all()),
    ]

    if csv_points is not None:
        audit_items.append(("csv_rows", f"{len(csv_points):,}", len(csv_points) > 0))
        audit_items.append(("csv_pole_id_duplicates", int(csv_points["pole_id"].astype(str).duplicated().sum()), True))
    if shp_points is not None:
        audit_items.append(("shp_rows", f"{len(shp_points):,}", len(shp_points) > 0))
        audit_items.append(("shp_pole_id_duplicates", int(shp_points["pole_id"].astype(str).duplicated().sum()), True))

    if csv_points is not None and shp_points is not None:
        comparison = compare_csv_shp(csv_points, shp_points)
        for key, value in comparison.items():
            audit_items.append((key, value, True))

    write_csv(validation_frame(audit_items), output_dir / "input_audit__gangwon_poles.csv")
    write_csv(selected[["pole_id", "lon", "lat"] + [c for c in selected.columns if c not in {"pole_id", "lon", "lat"}]], output_dir / "pole_input_points.csv")
    write_json(
        {
            "script": "01_audit_pole_inputs.py",
            "csv": str(args.csv),
            "shp": str(args.shp),
            "prefer": args.prefer,
            "output_dir": str(output_dir),
            "selected_rows": int(len(selected)),
        },
        output_dir / "run_manifest__01_audit_pole_inputs.json",
    )
    print(f"완료: {output_dir / 'pole_input_points.csv'}")


if __name__ == "__main__":
    main()
