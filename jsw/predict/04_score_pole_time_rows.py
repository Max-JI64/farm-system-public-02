from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predict_common import (
    FINAL_LGBM_FEATURES,
    OUTPUT_DIR,
    STATIC_REQUIRED_COLUMNS,
    TRAIN_DATA_DIR,
    WEATHER_REQUIRED_COLUMNS,
    CANADIAN_INDEX_COLUMNS,
    ensure_output_dir,
    read_csv,
    validation_frame,
    write_csv,
    write_json,
)


ROOT = OUTPUT_DIR.parents[2]
ML_OUTPUT_DIR = ROOT / "jsw" / "Analysis" / "new_machine_learning" / "outputs"
STEP2_DIR = ML_OUTPUT_DIR / "step2_tuned_single_models"
STEP3_DIR = ML_OUTPUT_DIR / "step3_f2_threshold"
TRAIN_DATA_PATH = TRAIN_DATA_DIR / "최종_머신러닝_학습데이터.csv"
SELECTED_PARAMS_PATH = STEP2_DIR / "selected_params__TUNE_LGBM_ALL_ALL_LC_NONE.json"
FINAL_THRESHOLDS_PATH = STEP3_DIR / "final_thresholds.json"

STATIC_META_COLUMNS = [
    "pole_id",
    "lon",
    "lat",
    "기상셀ID",
    "기후지형유형",
    "토지피복_L1_NAME",
    "토지피복_L2_NAME",
    "토지피복_매칭방식",
    "DEM_보완방식",
    "DEM_보완거리_m",
    "static_feature_complete",
]

WEATHER_META_COLUMNS = [
    "기상셀ID",
    "기준시각",
    "캐나다지수_기준날짜",
    "캐나다지수_정책",
]

INT_PARAM_KEYS = {"n_estimators", "num_leaves", "max_depth", "min_child_samples"}


def log(message: str) -> None:
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def make_one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_deployment_pipeline(params: dict[str, Any], *, random_state: int, n_jobs: int):
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    categorical_features = [column for column in ["기후지형유형"] if column in FINAL_LGBM_FEATURES]
    numeric_features = [column for column in FINAL_LGBM_FEATURES if column not in categorical_features]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    transformers.append(
        (
            "numeric",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            numeric_features,
        )
    )
    transformers.append(
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                    ("onehot", make_one_hot_encoder()),
                ]
            ),
            categorical_features,
        )
    )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    estimator = LGBMClassifier(
        **params,
        objective="binary",
        class_weight=None,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def load_selected_lgbm_params(path: Path) -> dict[int, dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("candidate_id") != "TUNE_LGBM_ALL_ALL_LC_NONE":
        raise ValueError(f"예상한 LightGBM 후보가 아닙니다: {obj.get('candidate_id')}")
    if obj.get("feature_set") != "WS_ALL__CANADA_ALL__LC_USED":
        raise ValueError(f"예상한 최종 피처셋이 아닙니다: {obj.get('feature_set')}")
    if obj.get("imbalance_option") != "none":
        raise ValueError(f"예상한 imbalance 옵션이 아닙니다: {obj.get('imbalance_option')}")
    return {int(k): v for k, v in obj["selected_params_by_outer_fold"].items()}


def representative_median_params(params_by_fold: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """fold별 선택 params에서 단일 배포용 대표값을 만든다.

    최종 후보가 nested outer fold마다 다른 params를 갖기 때문에, 단일 모델 배포 시
    특정 fold 하나를 임의로 고르지 않고 각 하이퍼파라미터의 중앙값을 사용한다.
    """

    keys = sorted({key for params in params_by_fold.values() for key in params})
    representative: dict[str, Any] = {}
    for key in keys:
        values = [params[key] for _, params in sorted(params_by_fold.items())]
        if all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            median = float(np.median(np.asarray(values, dtype=float)))
            if key in INT_PARAM_KEYS:
                representative[key] = int(round(median))
            else:
                representative[key] = median
        else:
            counts = pd.Series(values).value_counts()
            representative[key] = counts.index[0]
    return representative


def load_f2_threshold(path: Path) -> float:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return float(obj["operating_points"]["best_f2"]["threshold"])


def train_deployment_models(
    *,
    train_data_path: Path,
    params_by_fold: dict[int, dict[str, Any]],
    model_mode: str,
    random_state: int,
    n_jobs: int,
):
    log(f"최종 학습데이터 로드: {train_data_path}")
    data = read_csv(train_data_path, usecols=FINAL_LGBM_FEATURES + ["Target"])
    missing = sorted(set(FINAL_LGBM_FEATURES + ["Target"]) - set(data.columns))
    if missing:
        raise KeyError(f"학습데이터 누락 컬럼: {missing}")

    x = data[FINAL_LGBM_FEATURES].copy()
    y = data["Target"].astype(int)

    models = []
    if model_mode == "single_median_params":
        median_params = representative_median_params(params_by_fold)
        log("배포용 LightGBM 학습 중: model_mode=single_median_params")
        model = build_deployment_pipeline(
            median_params,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        model.fit(x, y)
        models.append(("single_median_params", model))
        return models

    if model_mode != "five_param_ensemble":
        raise ValueError(f"지원하지 않는 model_mode: {model_mode}")

    for fold, params in sorted(params_by_fold.items()):
        log(f"배포용 LightGBM 학습 중: model_mode=five_param_ensemble param_set_outer_fold={fold}")
        model = build_deployment_pipeline(
            params,
            random_state=random_state + fold,
            n_jobs=n_jobs,
        )
        model.fit(x, y)
        models.append((fold, model))
    return models


def load_or_train_models(
    *,
    model_cache: Path,
    reuse_model_cache: bool,
    train_data_path: Path,
    params_by_fold: dict[int, dict[str, Any]],
    model_mode: str,
    random_state: int,
    n_jobs: int,
):
    import joblib

    if reuse_model_cache and model_cache.exists():
        log(f"배포용 모델 캐시 로드: {model_cache}")
        return joblib.load(model_cache)

    models = train_deployment_models(
        train_data_path=train_data_path,
        params_by_fold=params_by_fold,
        model_mode=model_mode,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, model_cache)
    log(f"배포용 모델 캐시 저장: {model_cache}")
    return models


def predict_ensemble(models, x: pd.DataFrame) -> np.ndarray:
    pred = np.zeros(len(x), dtype=np.float64)
    for _, model in models:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            pred += model.predict_proba(x[FINAL_LGBM_FEATURES])[:, 1]
    pred /= len(models)
    return np.clip(pred, 1e-12, 1 - 1e-12)


def available_usecols(path: Path, requested: list[str]) -> list[str]:
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
    return [column for column in requested if column in header]


def load_static_features(path: Path, limit_poles: int | None = None) -> pd.DataFrame:
    usecols = available_usecols(path, list(dict.fromkeys(STATIC_META_COLUMNS + STATIC_REQUIRED_COLUMNS)))
    data = read_csv(path, usecols=usecols)
    if limit_poles is not None:
        data = data.head(limit_poles).copy()
    if "static_feature_complete" in data.columns:
        data = data[data["static_feature_complete"].astype(str).isin(["1", "1.0", "True", "true"])].copy()
    missing = sorted(set(["pole_id", "기상셀ID"] + STATIC_REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise KeyError(f"Pole 정적 피처 누락 컬럼: {missing}")
    return data.reset_index(drop=True)


def load_weather_population(path: Path, max_weather_rows_per_cell: int | None = None) -> pd.DataFrame:
    usecols = available_usecols(
        path,
        list(dict.fromkeys(WEATHER_META_COLUMNS + WEATHER_REQUIRED_COLUMNS + CANADIAN_INDEX_COLUMNS)),
    )
    data = read_csv(path, usecols=usecols)
    missing = sorted(
        set(["기상셀ID", "기준시각"] + WEATHER_REQUIRED_COLUMNS + CANADIAN_INDEX_COLUMNS) - set(data.columns)
    )
    if missing:
        raise KeyError(f"날씨 모집단 누락 컬럼: {missing}")
    data["기준시각"] = pd.to_datetime(data["기준시각"], errors="raise")
    data = data.sort_values(["기상셀ID", "기준시각"]).reset_index(drop=True)
    if max_weather_rows_per_cell is not None:
        data = data.groupby("기상셀ID", group_keys=False).head(max_weather_rows_per_cell).reset_index(drop=True)
    return data


def estimate_scoring_rows(static: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    pole_counts = static.groupby("기상셀ID").size().rename("pole_n")
    weather_counts = weather.groupby("기상셀ID").size().rename("weather_row_n")
    estimate = pd.concat([pole_counts, weather_counts], axis=1).fillna(0).reset_index()
    estimate["pole_n"] = estimate["pole_n"].astype(np.int64)
    estimate["weather_row_n"] = estimate["weather_row_n"].astype(np.int64)
    estimate["score_row_n"] = estimate["pole_n"] * estimate["weather_row_n"]
    return estimate.sort_values("score_row_n", ascending=False).reset_index(drop=True)


def build_feature_block(static_chunk: pd.DataFrame, weather_chunk: pd.DataFrame) -> pd.DataFrame:
    n_weather = len(weather_chunk)
    left = static_chunk.loc[static_chunk.index.repeat(n_weather)].reset_index(drop=True)
    right = pd.concat([weather_chunk] * len(static_chunk), ignore_index=True)

    overlap = [column for column in right.columns if column in left.columns and column != "기상셀ID"]
    if overlap:
        right = right.drop(columns=overlap)
    block = pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1)
    return block[FINAL_LGBM_FEATURES]


def summarize_score_matrix(
    static_chunk: pd.DataFrame,
    score_matrix: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    meta_cols = [
        column
        for column in [
            "pole_id",
            "lon",
            "lat",
            "기상셀ID",
            "기후지형유형",
            "토지피복_L1_NAME",
            "토지피복_L2_NAME",
            "토지피복_매칭방식",
            "DEM_보완방식",
            "DEM_보완거리_m",
        ]
        if column in static_chunk.columns
    ]
    out = static_chunk[meta_cols].copy().reset_index(drop=True)
    out["scored_time_rows"] = score_matrix.shape[1]
    out["mean_score"] = score_matrix.mean(axis=1)
    out["p90_score"] = np.quantile(score_matrix, 0.90, axis=1)
    out["p95_score"] = np.quantile(score_matrix, 0.95, axis=1)
    out["p99_score"] = np.quantile(score_matrix, 0.99, axis=1)
    out["max_score"] = score_matrix.max(axis=1)
    out["f2_threshold_exceed_count"] = (score_matrix >= threshold).sum(axis=1)
    out["f2_threshold_exceed_rate"] = out["f2_threshold_exceed_count"] / score_matrix.shape[1]
    return out


def assign_vulnerability_groups(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out = out.sort_values(["p95_score", "f2_threshold_exceed_rate", "max_score"], ascending=False).reset_index(drop=True)
    n = len(out)
    out["p95_rank"] = np.arange(1, n + 1)
    out["p95_percentile_rank"] = out["p95_rank"] / n
    conditions = [
        out["p95_percentile_rank"].le(0.05),
        out["p95_percentile_rank"].le(0.10),
        out["p95_percentile_rank"].le(0.20),
    ]
    choices = ["매우 높음(top5%)", "높음(top5~10%)", "중간(top10~20%)"]
    out["vulnerability_group"] = np.select(conditions, choices, default="낮음(나머지)")
    return out


def checkpoint_part_path(checkpoint_dir: Path, cell_id: str, pole_start: int, pole_end: int) -> Path:
    safe_cell_id = str(cell_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    return checkpoint_dir / f"part__{safe_cell_id}__pole_{pole_start:08d}_{pole_end:08d}.csv"


def write_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(path)


def read_checkpoint_parts(part_paths: list[Path]) -> pd.DataFrame:
    missing = [path for path in part_paths if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"누락된 checkpoint part가 있습니다: {preview}")
    frames = [pd.read_csv(path, encoding="utf-8-sig", low_memory=False) for path in part_paths]
    if not frames:
        raise RuntimeError("checkpoint part가 없습니다.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="최종 LightGBM 5-parameter-set ensemble로 Pole×기상시각 score를 chunk 계산하고 Pole별 취약성 요약 생성"
    )
    parser.add_argument("--static", type=Path, default=OUTPUT_DIR / "pole_static_features_model_ready.csv")
    parser.add_argument("--weather", type=Path, default=OUTPUT_DIR / "weather_population_10to5_09to16.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--selected-params", type=Path, default=SELECTED_PARAMS_PATH)
    parser.add_argument("--thresholds", type=Path, default=FINAL_THRESHOLDS_PATH)
    parser.add_argument("--model-cache", type=Path, default=None)
    parser.add_argument(
        "--model-mode",
        choices=["single_median_params", "five_param_ensemble"],
        default="single_median_params",
        help="single_median_params는 단일 LightGBM, five_param_ensemble은 fold별 params 5개 LightGBM 평균",
    )
    parser.add_argument("--reuse-model-cache", action="store_true")
    parser.add_argument("--random-state", type=int, default=20260622)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--pole-chunk-size", type=int, default=1000)
    parser.add_argument("--weather-chunk-size", type=int, default=64)
    parser.add_argument("--limit-poles", type=int, default=None, help="검증용 Pole 수 제한")
    parser.add_argument("--max-weather-rows-per-cell", type=int, default=None, help="검증용 셀별 날씨 row 수 제한")
    parser.add_argument("--cell-ids", nargs="*", default=None, help="특정 기상셀ID만 실행")
    parser.add_argument("--confirm-full-run", action="store_true", help="제한 없는 전체 scoring 실행 확인")
    parser.add_argument("--estimate-only", action="store_true", help="score row 수 추정만 수행")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Pole 요약 part checkpoint 저장 폴더. 기본값은 output-dir/pole_vulnerability_summary_parts",
    )
    parser.add_argument(
        "--no-resume-checkpoints",
        action="store_true",
        help="이미 존재하는 checkpoint part를 재사용하지 않고 다시 계산",
    )
    parser.add_argument(
        "--progress-every-weather-chunks",
        type=int,
        default=20,
        help="긴 pole chunk 내부에서 weather chunk 몇 개마다 진행 로그를 출력할지 설정. 0이면 내부 진행 로그 생략",
    )
    args = parser.parse_args()

    started = time.time()
    output_dir = ensure_output_dir(args.output_dir)
    model_cache = args.model_cache or (output_dir / f"deployment_lgbm__{args.model_mode}.joblib")

    log(f"Pole 정적 피처 로드: {args.static}")
    static = load_static_features(args.static, args.limit_poles)
    log(f"날씨 모집단 로드: {args.weather}")
    weather = load_weather_population(args.weather, args.max_weather_rows_per_cell)

    if args.cell_ids:
        cell_ids = set(args.cell_ids)
        static = static[static["기상셀ID"].isin(cell_ids)].reset_index(drop=True)
        weather = weather[weather["기상셀ID"].isin(cell_ids)].reset_index(drop=True)

    estimate = estimate_scoring_rows(static, weather)
    total_score_rows = int(estimate["score_row_n"].sum())
    write_csv(estimate, output_dir / "pole_time_score_row_estimate.csv")
    log(f"예상 score row 수: {total_score_rows:,}")
    log(f"Pole 수: {len(static):,} | 날씨 row 수: {len(weather):,} | 기상셀 수: {estimate['기상셀ID'].nunique():,}")
    log(f"모델 모드: {args.model_mode}")

    full_run_requested = (
        args.limit_poles is None
        and args.max_weather_rows_per_cell is None
        and not args.cell_ids
    )
    if args.estimate_only or (full_run_requested and not args.confirm_full_run):
        audit = validation_frame(
            [
                ("estimate_only", True, True),
                ("full_run_requires_confirm_full_run", full_run_requested and not args.confirm_full_run, True),
                ("static_rows", f"{len(static):,}", len(static) > 0),
                ("weather_rows", f"{len(weather):,}", len(weather) > 0),
                ("estimated_score_rows", f"{total_score_rows:,}", total_score_rows > 0),
            ]
        )
        write_csv(audit, output_dir / "pole_scoring_audit.csv")
        write_json(
            {
                "script": "04_score_pole_time_rows.py",
                "mode": "estimate_only",
                "estimated_score_rows": total_score_rows,
                "note": "제한 없는 전체 실행은 --confirm-full-run을 명시해야 수행됩니다.",
            },
            output_dir / "run_manifest__04_score_pole_time_rows.json",
        )
        log("전체 실행 보호로 scoring은 수행하지 않았습니다. 전체 실행은 --confirm-full-run을 붙여 실행하세요.")
        return

    params_by_fold = load_selected_lgbm_params(args.selected_params)
    threshold = load_f2_threshold(args.thresholds)
    models = load_or_train_models(
        model_cache=model_cache,
        reuse_model_cache=args.reuse_model_cache,
        train_data_path=args.train_data,
        params_by_fold=params_by_fold,
        model_mode=args.model_mode,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    checkpoint_dir = args.checkpoint_dir or (output_dir / f"pole_vulnerability_summary_parts__{args.model_mode}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoints = not args.no_resume_checkpoints

    expected_part_paths: list[Path] = []
    completed_score_rows = 0
    computed_score_rows_this_run = 0
    reused_checkpoint_score_rows = 0
    checkpoint_chunks_expected = 0
    checkpoint_chunks_completed = 0
    checkpoint_chunks_reused = 0
    processed_cells = 0
    skipped_cells = 0

    weather_by_cell = {cell: frame.reset_index(drop=True) for cell, frame in weather.groupby("기상셀ID", sort=True)}
    static_groups = [(cell_id, frame.reset_index(drop=True)) for cell_id, frame in static.groupby("기상셀ID", sort=True)]
    expected_checkpoint_chunks = sum(
        int(np.ceil(len(frame) / args.pole_chunk_size))
        for cell_id, frame in static_groups
        if cell_id in weather_by_cell and not weather_by_cell[cell_id].empty
    )
    log(
        "진행 계획: "
        f"cells={len(static_groups):,}, checkpoint_chunks={expected_checkpoint_chunks:,}, "
        f"checkpoint_dir={checkpoint_dir}"
    )

    global_chunk_index = 0
    for cell_index, (cell_id, cell_static) in enumerate(static_groups, start=1):
        cell_weather = weather_by_cell.get(cell_id)
        if cell_weather is None or cell_weather.empty:
            skipped_cells += 1
            continue

        processed_cells += 1
        log(
            f"[cell {cell_index:,}/{len(static_groups):,}] {cell_id} | "
            f"pole={len(cell_static):,} | weather={len(cell_weather):,} | "
            f"rows={len(cell_static) * len(cell_weather):,} | elapsed={format_duration(time.time() - started)}"
        )

        for pole_start in range(0, len(cell_static), args.pole_chunk_size):
            global_chunk_index += 1
            pole_end = min(pole_start + args.pole_chunk_size, len(cell_static))
            static_chunk = cell_static.iloc[pole_start:pole_end].reset_index(drop=True)
            part_path = checkpoint_part_path(checkpoint_dir, str(cell_id), pole_start, pole_end)
            expected_part_paths.append(part_path)
            checkpoint_chunks_expected += 1
            chunk_score_rows = len(static_chunk) * len(cell_weather)

            if resume_checkpoints and part_path.exists():
                completed_score_rows += chunk_score_rows
                reused_checkpoint_score_rows += chunk_score_rows
                checkpoint_chunks_completed += 1
                checkpoint_chunks_reused += 1
                progress_pct = completed_score_rows / total_score_rows * 100 if total_score_rows else 0.0
                log(
                    f"  [chunk {global_chunk_index:,}/{expected_checkpoint_chunks:,}] checkpoint 재사용 | "
                    f"pole={pole_start:,}:{pole_end:,} | completed_rows={completed_score_rows:,}/{total_score_rows:,} "
                    f"({progress_pct:.2f}%) | elapsed={format_duration(time.time() - started)}"
                )
                continue

            log(
                f"  [chunk {global_chunk_index:,}/{expected_checkpoint_chunks:,}] 계산 시작 | "
                f"pole={pole_start:,}:{pole_end:,} | score_rows={chunk_score_rows:,}"
            )
            score_matrix = np.empty((len(static_chunk), len(cell_weather)), dtype=np.float32)

            weather_chunk_count = int(np.ceil(len(cell_weather) / args.weather_chunk_size))
            for weather_start in range(0, len(cell_weather), args.weather_chunk_size):
                weather_chunk = cell_weather.iloc[weather_start : weather_start + args.weather_chunk_size].reset_index(drop=True)
                x_block = build_feature_block(static_chunk, weather_chunk)
                pred = predict_ensemble(models, x_block).astype(np.float32)
                score_matrix[:, weather_start : weather_start + len(weather_chunk)] = pred.reshape(
                    len(static_chunk),
                    len(weather_chunk),
                )
                computed_score_rows_this_run += len(pred)
                if args.progress_every_weather_chunks > 0:
                    weather_chunk_index = weather_start // args.weather_chunk_size + 1
                    is_last_weather_chunk = weather_start + args.weather_chunk_size >= len(cell_weather)
                    if (
                        weather_chunk_index % args.progress_every_weather_chunks == 0
                        or is_last_weather_chunk
                    ):
                        chunk_done_rows = min(weather_start + len(weather_chunk), len(cell_weather)) * len(static_chunk)
                        total_done_if_committed = completed_score_rows + chunk_done_rows
                        progress_pct = total_done_if_committed / total_score_rows * 100 if total_score_rows else 0.0
                        log(
                            f"    weather {weather_chunk_index:,}/{weather_chunk_count:,} | "
                            f"chunk_rows={chunk_done_rows:,}/{chunk_score_rows:,} | "
                            f"global_progress~{progress_pct:.2f}% | elapsed={format_duration(time.time() - started)}"
                        )

            part_summary = summarize_score_matrix(static_chunk, score_matrix, threshold=threshold)
            write_checkpoint(part_summary, part_path)
            completed_score_rows += chunk_score_rows
            checkpoint_chunks_completed += 1
            progress_pct = completed_score_rows / total_score_rows * 100 if total_score_rows else 0.0
            log(
                f"  [chunk {global_chunk_index:,}/{expected_checkpoint_chunks:,}] 저장 완료 | "
                f"{part_path.name} | completed_rows={completed_score_rows:,}/{total_score_rows:,} "
                f"({progress_pct:.2f}%) | elapsed={format_duration(time.time() - started)}"
            )

    if not expected_part_paths:
        raise RuntimeError("score를 계산할 checkpoint part가 없습니다.")

    log(f"checkpoint part 병합 중: {len(expected_part_paths):,}개")
    summary = read_checkpoint_parts(expected_part_paths)
    summary = assign_vulnerability_groups(summary)
    groups = (
        summary.groupby("vulnerability_group", as_index=False)
        .agg(
            pole_n=("pole_id", "count"),
            p95_min=("p95_score", "min"),
            p95_max=("p95_score", "max"),
            exceed_rate_mean=("f2_threshold_exceed_rate", "mean"),
        )
        .sort_values("p95_max", ascending=False)
    )

    summary_path = output_dir / "pole_vulnerability_summary.csv"
    groups_path = output_dir / "pole_vulnerability_groups.csv"
    write_csv(summary, summary_path)
    write_csv(groups, groups_path)

    audit = validation_frame(
        [
            ("static_rows", f"{len(static):,}", len(static) > 0),
            ("weather_rows", f"{len(weather):,}", len(weather) > 0),
            ("estimated_score_rows", f"{total_score_rows:,}", total_score_rows > 0),
            ("completed_score_rows", f"{completed_score_rows:,}", completed_score_rows == total_score_rows),
            ("computed_score_rows_this_run", f"{computed_score_rows_this_run:,}", True),
            ("reused_checkpoint_score_rows", f"{reused_checkpoint_score_rows:,}", True),
            ("checkpoint_chunks_expected", checkpoint_chunks_expected, checkpoint_chunks_expected > 0),
            ("checkpoint_chunks_completed", checkpoint_chunks_completed, checkpoint_chunks_completed == checkpoint_chunks_expected),
            ("checkpoint_chunks_reused", checkpoint_chunks_reused, True),
            ("processed_cells", processed_cells, processed_cells > 0),
            ("skipped_cells_without_weather", skipped_cells, skipped_cells == 0),
            ("summary_rows", f"{len(summary):,}", len(summary) == len(static[static["기상셀ID"].isin(weather_by_cell)])),
            ("f2_threshold", threshold, threshold > 0),
            ("score_nan_cells", int(summary[["mean_score", "p90_score", "p95_score", "max_score"]].isna().sum().sum()), True),
        ]
    )
    write_csv(audit, output_dir / "pole_scoring_audit.csv")
    write_json(
        {
            "script": "04_score_pole_time_rows.py",
            "mode": "scoring",
            "static_file": str(args.static),
            "weather_file": str(args.weather),
            "train_data": str(args.train_data),
            "selected_params": str(args.selected_params),
            "thresholds": str(args.thresholds),
            "model_cache": str(model_cache),
            "model_mode": args.model_mode,
            "model_strategy": (
                "single LightGBM trained on full development data using median representative params from five outer-fold selected parameter sets"
                if args.model_mode == "single_median_params"
                else "five LightGBM pipelines trained on full development data using the five outer-fold selected parameter sets; score is arithmetic mean"
            ),
            "f2_threshold": threshold,
            "static_rows": int(len(static)),
            "weather_rows": int(len(weather)),
            "estimated_score_rows": total_score_rows,
            "completed_score_rows": int(completed_score_rows),
            "computed_score_rows_this_run": int(computed_score_rows_this_run),
            "reused_checkpoint_score_rows": int(reused_checkpoint_score_rows),
            "checkpoint_dir": str(checkpoint_dir),
            "resume_checkpoints": resume_checkpoints,
            "checkpoint_chunks_expected": checkpoint_chunks_expected,
            "checkpoint_chunks_completed": checkpoint_chunks_completed,
            "checkpoint_chunks_reused": checkpoint_chunks_reused,
            "pole_chunk_size": args.pole_chunk_size,
            "weather_chunk_size": args.weather_chunk_size,
            "elapsed_seconds": time.time() - started,
        },
        output_dir / "run_manifest__04_score_pole_time_rows.json",
    )
    log(f"완료: {summary_path}")
    log(f"등급 요약: {groups_path}")


if __name__ == "__main__":
    main()
