from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = REPO_ROOT / "jsw/강원_재_EDA"
OUT_DIR = MODULE_DIR / "outputs/Step4"
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

font_path = "C:/Windows/Fonts/malgun.ttf"
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()

sns.set_theme(
    style="whitegrid",
    font=font_name,
    rc={
        "font.family": font_name,
        "font.sans-serif": [font_name],
        "axes.unicode_minus": False,
    },
)
plt.rcParams["font.family"] = font_name
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def stat_row(stats: pd.DataFrame, layer: str, variable: str) -> pd.Series:
    row = stats[stats["spatial_layer"].eq(layer) & stats["variable"].eq(variable)]
    if row.empty:
        raise KeyError(f"Missing S4-05 stat row: {layer} / {variable}")
    return row.iloc[0]


def pct_value(lc: pd.DataFrame, layer: str, frame: str, name: str, column: str) -> float:
    row = lc[
        lc["spatial_layer"].eq(layer)
        & lc["comparison_frame"].eq(frame)
        & lc["L1_NAME"].eq(name)
    ]
    if row.empty:
        return 0.0
    return float(row[column].iloc[0])


def format_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 1e-6:
        return "<1e-6"
    return f"{value:.3g}"


def build_handoff(stats: pd.DataFrame) -> pd.DataFrame:
    wui = "생활권-WUI"
    access = "산림 접근권"
    interior = "산림 내부"

    road = stat_row(stats, wui, "도로_최단거리_m")
    urban = stat_row(stats, wui, "시가화_최단거리_m")
    forest = stat_row(stats, wui, "산림_최단거리_m")
    slope = stat_row(stats, wui, "경사도(도)")
    elevation = stat_row(stats, wui, "고도(m)")
    agriculture = stat_row(stats, wui, "농업_최단거리_m")
    tpi = stat_row(stats, wui, "TPI(지형위치지수)")
    forest_road = stat_row(stats, wui, "임도_최단거리_m")
    trail = stat_row(stats, wui, "등산로_최단거리_m")
    trail_access = stat_row(stats, access, "등산로_최단거리_m")
    tpi_interior = stat_row(stats, interior, "TPI(지형위치지수)")

    rows = [
        {
            "변수명": "도로_최단거리_m",
            "등급": "EDA 핵심",
            "Cliff's delta (WUI)": road["cliffs_delta"],
            "권장 피처 처리 방식": "도로 10m 이내 여부 더미 변수(is_road_ultra_close) 및 선형 거리 변수 병용",
            "선정/제외 사유": (
                f"WUI 내부에서 가장 큰 접근성 효과(delta={road['cliffs_delta']:.3f}, "
                f"q={format_p(road['welch_q_value_fdr'])})"
            ),
        },
        {
            "변수명": "시가화_최단거리_m",
            "등급": "후속 모델링 후보",
            "Cliff's delta (WUI)": urban["cliffs_delta"],
            "권장 피처 처리 방식": "시가화 10m 이내 여부(is_urban_edge) 및 선형 거리",
            "선정/제외 사유": (
                f"순위 기반 근접 편향은 크지만 평균은 롱테일 영향을 받음(delta={urban['cliffs_delta']:.3f})"
            ),
        },
        {
            "변수명": "산림_최단거리_m",
            "등급": "후속 모델링 후보",
            "Cliff's delta (WUI)": forest["cliffs_delta"],
            "권장 피처 처리 방식": "산림 내부(0m)와 산림 외부 경계 거리 분리",
            "선정/제외 사유": (
                f"WUI 안에서 발생지가 산림 피복 내부보다 경계 바깥 쪽으로 치우침(delta={forest['cliffs_delta']:.3f})"
            ),
        },
        {
            "변수명": "경사도(도)",
            "등급": "후속 모델링 후보",
            "Cliff's delta (WUI)": slope["cliffs_delta"],
            "권장 피처 처리 방식": "선형 경사도 변수와 공간층 상호작용 후보",
            "선정/제외 사유": f"WUI 발생지가 완만한 사면 쪽으로 이동(delta={slope['cliffs_delta']:.3f})",
        },
        {
            "변수명": "고도(m)",
            "등급": "후속 모델링 후보",
            "Cliff's delta (WUI)": elevation["cliffs_delta"],
            "권장 피처 처리 방식": "기후지형유형 권역 층화 결합 피처",
            "선정/제외 사유": f"발생지가 상대적으로 저지대에 분포(delta={elevation['cliffs_delta']:.3f})",
        },
        {
            "변수명": "농업_최단거리_m",
            "등급": "보고서 보조",
            "Cliff's delta (WUI)": agriculture["cliffs_delta"],
            "권장 피처 처리 방식": "선형 거리 변수 또는 WUI 보조 설명 변수",
            "선정/제외 사유": f"WUI 내부 효과가 도로·시가화보다 작음(delta={agriculture['cliffs_delta']:.3f})",
        },
        {
            "변수명": "TPI(지형위치지수)",
            "등급": "후속 모델링 후보",
            "Cliff's delta (WUI)": tpi["cliffs_delta"],
            "권장 피처 처리 방식": "공간층별 상호작용 후보로만 사용",
            "선정/제외 사유": (
                f"WUI에서는 음수 방향(delta={tpi['cliffs_delta']:.3f}); 산림 내부 양수 방향은 "
                f"q={format_p(tpi_interior['welch_q_value_fdr'])}로 경계적 탐색 신호"
            ),
        },
        {
            "변수명": "임도_최단거리_m",
            "등급": "보고서 보조",
            "Cliff's delta (WUI)": forest_road["cliffs_delta"],
            "권장 피처 처리 방식": "선형 거리 변수, 산림 접근권 보조 변수",
            "선정/제외 사유": f"WUI 내부 효과는 약하고 방향도 도로와 다름(delta={forest_road['cliffs_delta']:.3f})",
        },
        {
            "변수명": "등산로_최단거리_m",
            "등급": "보고서 보조",
            "Cliff's delta (WUI)": trail["cliffs_delta"],
            "권장 피처 처리 방식": "산림 접근권 층에서만 조건부 탐색 변수",
            "선정/제외 사유": (
                f"산림 접근권 방향 후보(delta={trail_access['cliffs_delta']:.3f})이나 "
                f"FDR q={format_p(trail_access['welch_q_value_fdr'])}로 확정 근거는 부족"
            ),
        },
        {
            "변수명": "TWI(지형다습지수)",
            "등급": "제외/보류",
            "Cliff's delta (WUI)": np.nan,
            "권장 피처 처리 방식": "N/A",
            "선정/제외 사유": "수문학적 흐름 누적 연산 한계로 대조군에서 계산하지 않아 주 비교 변수에서 보류",
        },
        {
            "변수명": "addr_type (임야번지)",
            "등급": "제외/보류",
            "Cliff's delta (WUI)": np.nan,
            "권장 피처 처리 방식": "N/A",
            "선정/제외 사유": "실제 토지피복 산림지역과 괴리가 커 취약성 프록시로 사용하지 않음",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    print("--- S4-06: 최종 EDA 공간 요약 시작 ---")

    stats = pd.read_csv(TABLE_DIR / "S4-05_effect_size_by_variable.csv", encoding="utf-8-sig")
    lc = pd.read_csv(TABLE_DIR / "S4-05_landcover_crosstab.csv", encoding="utf-8-sig")
    fire = pd.read_csv(TABLE_DIR / "S4-03_spatial_layer_assignment.csv", encoding="utf-8-sig")
    controls = pd.read_csv(TABLE_DIR / "S4-04_spatial_control_pool.csv", encoding="utf-8-sig")

    matched_fire_ids = set(controls["matched_fire_id"].dropna())
    fire_primary = fire[fire["fire_id"].isin(matched_fire_ids)].copy()
    fire_wui = fire_primary[fire_primary["spatial_layer_500"].eq("생활권-WUI")].copy()
    control_wui = controls[controls["spatial_layer_500"].eq("생활권-WUI")].copy()

    road = stat_row(stats, "생활권-WUI", "도로_최단거리_m")
    urban = stat_row(stats, "생활권-WUI", "시가화_최단거리_m")
    forest = stat_row(stats, "생활권-WUI", "산림_최단거리_m")
    slope = stat_row(stats, "생활권-WUI", "경사도(도)")
    tpi = stat_row(stats, "생활권-WUI", "TPI(지형위치지수)")
    tpi_interior = stat_row(stats, "산림 내부", "TPI(지형위치지수)")
    trail_access = stat_row(stats, "산림 접근권", "등산로_최단거리_m")

    road_10_fire_pct = float((fire_wui["도로_최단거리_m"] <= 10).mean() * 100)
    road_10_control_pct = float((control_wui["도로_최단거리_m"] <= 10).mean() * 100)

    frame = "eligible_comparable" if "comparison_frame" in lc.columns else "primary_raw"
    urban_fire_pct = pct_value(lc, "생활권-WUI", frame, "시가화건조지역", "fire_pct")
    grass_fire_pct = pct_value(lc, "생활권-WUI", frame, "초지", "fire_pct")
    forest_fire_pct = pct_value(lc, "생활권-WUI", frame, "산림지역", "fire_pct")
    forest_control_pct = pct_value(lc, "생활권-WUI", frame, "산림지역", "control_pct")

    summary_data = [
        {
            "요약 메시지": (
                f"도로 초근접성 편향: WUI 내 발생지의 {road_10_fire_pct:.1f}%가 도로 10m 이내에 있으며 "
                f"대조군은 {road_10_control_pct:.1f}%에 그침"
            ),
            "근거 표": (
                f"S4-05_effect_size_by_variable.csv "
                f"(WUI 도로 Cliff's delta {road['cliffs_delta']:.3f}, q={format_p(road['welch_q_value_fdr'])})"
            ),
            "근거 플롯": "S4-05_key_variable_ecdf.png",
            "해석 가능 범위": "동일 기상셀·동일 공간층 조건에서도 발생지가 도로 초근접 구간에 강하게 몰림",
            "해석 금지 문장": "도로변 담뱃재 투기나 차량 화재가 직접 주원인이라고 인과 단정하는 것",
        },
        {
            "요약 메시지": (
                f"비교가능 토지피복 프레임에서 WUI 발생지는 시가화 {urban_fire_pct:.1f}%, "
                f"초지 {grass_fire_pct:.1f}%, 산림지역 {forest_fire_pct:.1f}%로 구성됨"
            ),
            "근거 표": (
                f"S4-05_landcover_crosstab.csv ({frame}; 대조군 산림지역 {forest_control_pct:.1f}%)"
            ),
            "근거 플롯": "N/A",
            "해석 가능 범위": "대조군과 동일한 제외 규칙을 맞춘 뒤에도 발생지는 비산림 WUI 접경 피복 비중이 큼",
            "해석 금지 문장": "건물 자체가 산불을 유도하거나 특정 토지피복이 원인이라고 해석하는 것",
        },
        {
            "요약 메시지": (
                f"WUI에서는 저고도·완만한 경사·음수 TPI 방향이 관찰되지만, 산림 내부 TPI 양수 방향은 "
                f"q={format_p(tpi_interior['welch_q_value_fdr'])}의 경계적 탐색 신호임"
            ),
            "근거 표": (
                f"S4-05_effect_size_by_variable.csv "
                f"(WUI 경사 delta {slope['cliffs_delta']:.3f}, WUI TPI {tpi['cliffs_delta']:.3f}, "
                f"산림 내부 TPI {tpi_interior['cliffs_delta']:.3f})"
            ),
            "근거 플롯": "S4-05_effect_size_forest.png",
            "해석 가능 범위": "공간층에 따라 지형 지표의 방향이 달라질 수 있다는 후보를 제시함",
            "해석 금지 문장": "산림 내부 능선부 취약성이 통계적으로 확정되었다고 단정하는 것",
        },
        {
            "요약 메시지": (
                f"산림 접근권 등산로 최단거리는 delta {trail_access['cliffs_delta']:.3f}의 방향성은 있으나 "
                f"FDR q={format_p(trail_access['welch_q_value_fdr'])}로 유의한 결과가 아님"
            ),
            "근거 표": "S4-05_effect_size_by_variable.csv",
            "근거 플롯": "S4-05_effect_size_forest.png",
            "해석 가능 범위": "소표본 보조 신호로만 기록하고 후속 자료에서 재검증함",
            "해석 금지 문장": "등산로 인접성이 산림 접근권에서 확정적 취약 인자라고 표현하는 것",
        },
    ]
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(TABLE_DIR / "S4-06_final_spatial_eda_summary.csv", index=False, encoding="utf-8-sig")
    print("S4-06_final_spatial_eda_summary.csv 생성 완료")

    df_handoff = build_handoff(stats)
    df_handoff.to_csv(TABLE_DIR / "S4-06_spatial_variable_handoff.csv", index=False, encoding="utf-8-sig")
    print("S4-06_spatial_variable_handoff.csv 생성 완료")

    limits_data = [
        {
            "구분": "도로/인프라 인접성",
            "해석 가능 범위": "동일 기상셀·동일 공간층 하에서 발생지가 대조군보다 도로 및 시가화 구역에 더 가까운 분포를 보임",
            "해석 금지/주의 사항": "도로변 담뱃재 투기나 차량 화재가 직접 원인이라고 인과 단정하는 것",
            "대안 및 보완책": "물리적 노출 및 발원 취약 지대로만 기술하고, 발화 원인은 소방 원인자료와 별도 교차 분석할 것",
        },
        {
            "구분": "토지피복과 WUI 경계",
            "해석 가능 범위": "대조군과 같은 제외 프레임에서도 발생지가 비산림 WUI 접경 피복에 더 많이 놓임",
            "해석 금지/주의 사항": "토지피복 유형 자체가 산불 원인이라고 해석하거나, 비대칭 제외 규칙의 원시 비율만으로 결론 내리는 것",
            "대안 및 보완책": "`eligible_comparable` 프레임의 민감도 비율과 원시 비율을 함께 보고할 것",
        },
        {
            "구분": "지형 지표",
            "해석 가능 범위": "WUI에서는 저고도·완만한 경사·음수 TPI 방향이 관찰되고, 산림 내부 TPI는 양수 방향의 탐색 신호를 보임",
            "해석 금지/주의 사항": "산림 내부 능선부 취약성을 확정하거나 특정 고도·TPI가 산불을 유발한다고 해석하는 것",
            "대안 및 보완책": "공간층별 상호작용 후보로 모델링에서 검증하고, 소표본 층은 효과크기와 q값을 함께 제시할 것",
        },
        {
            "구분": "대조군 부족 71건",
            "해석 가능 범위": "71건의 대조군 부족은 표본 프레임 품질 한계이며, 기술통계 민감도에서 큰 차이는 관찰되지 않음",
            "해석 금지/주의 사항": "대조군이 없는 위험지라고 해석하거나, 정식 검정 없이 선택 편향이 없다고 입증했다고 표현하는 것",
            "대안 및 보완책": "주 분석 표본과 전체 발생지의 기술통계 차이를 민감도 표로 제시할 것",
        },
        {
            "구분": "등산로/산림 접근권",
            "해석 가능 범위": "소표본에서 방향성 후보가 보였으나 FDR 기준 유의 결과는 아님",
            "해석 금지/주의 사항": "등산로 인접성을 확정적 취약 인자 또는 등산객 원인으로 쓰는 것",
            "대안 및 보완책": "후속 모델링에서는 조건부 후보로만 포함하고 별도 검증 기준을 둘 것",
        },
    ]
    df_limits = pd.DataFrame(limits_data)
    df_limits.to_csv(TABLE_DIR / "S4-06_interpretation_limits.csv", index=False, encoding="utf-8-sig")
    print("S4-06_interpretation_limits.csv 생성 완료")

    print("S4-06 최종 요약 플롯 생성 중...")
    df_plot = df_handoff.dropna(subset=["Cliff's delta (WUI)"]).copy()
    df_plot["abs_delta"] = df_plot["Cliff's delta (WUI)"].abs()
    df_plot = df_plot.sort_values(by="abs_delta", ascending=True)

    colors = []
    for grade in df_plot["등급"]:
        if grade == "EDA 핵심":
            colors.append("#d73027")
        elif grade == "후속 모델링 후보":
            colors.append("#fc8d59")
        elif grade == "보고서 보조":
            colors.append("#fee090")
        else:
            colors.append("#e0f3f8")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        df_plot["변수명"],
        df_plot["Cliff's delta (WUI)"],
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    for bar, grade in zip(bars, df_plot["등급"]):
        width = bar.get_width()
        x_pos = width + 0.02 if width >= 0 else width - 0.12
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            grade,
            va="center",
            ha="left" if width >= 0 else "right",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("S4-06 후속 모델링 이관 변수 등급 및 WUI 효과크기(Cliff's delta) 요약", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("WUI 내 Cliff's delta 효과크기", fontsize=11)
    ax.set_ylabel("변수명", fontsize=11)
    ax.set_xlim(-0.95, 0.65)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "S4-06_final_spatial_summary_plot.png", dpi=200)
    plt.close()
    print("S4-06_final_spatial_summary_plot.png 생성 완료")
    print("--- S4-06: 최종 EDA 공간 요약 완료 ---")


if __name__ == "__main__":
    main()
