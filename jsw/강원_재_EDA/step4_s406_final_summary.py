import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

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

# Font settings
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


def main() -> None:
    print("--- S4-06: 최종 EDA 공간 요약 시작 ---")
    
    # 1. Generate S4-06_final_spatial_eda_summary.csv
    summary_data = [
        {
            "요약 메시지": "도로 초근접성 편향: WUI 내 산불은 도로 10m 이내에 70% 집중됨",
            "근거 표": "S4-05_effect_size_by_variable.csv (Cliff's delta -0.725, p=0.000)",
            "근거 플롯": "S4-05_key_variable_ecdf.png",
            "해석 가능 범위": "동일한 기상 및 공간층 통제 하에서도 발생지가 대조군보다 도로에 극도로 근접하여 발화함",
            "해석 금지 문장": "도로변 담뱃재 투기가 산불의 직접적 주원인이라고 인과 단정하는 것"
        },
        {
            "요약 메시지": "인프라 접경지 발원 패턴: WUI 내 산불은 과반이 시가화 및 초지 경계선에 집중되며 실제 산림지역 피복 위 발원은 14.2%에 불과함",
            "근거 표": "S4-05_landcover_crosstab.csv (시가화지역 39.3%, 초지 19.3%, 산림지역 14.2%)",
            "근거 플롯": "N/A",
            "해석 가능 범위": "산불 발원은 숲 내부가 아닌 인프라 접경지 나대지, 마당, 도로변 수풀에서 유래함",
            "해석 금지 문장": "산림 면적이 좁아서 산불이 잘 난다거나, 건물 자체가 산불을 유도한다고 해석하는 것"
        },
        {
            "요약 메시지": "지형의 비선형성 및 층화: 발생지는 전반적으로 고도가 낮고 경사가 완만한 골짜기에 치우치나, 산림 내부 깊은 곳에서는 TPI가 양수로 반전되어 능선부 취약성이 증가함",
            "근거 표": "S4-05_effect_size_by_variable.csv (산림 내부 TPI Cliff's delta +0.181, WUI TPI -0.106)",
            "근거 플롯": "S4-05_effect_size_forest.png",
            "해석 가능 범위": "공간층에 따라 고도/경사의 효과 크기가 달라지며 지형적 영향이 비선형적으로 반전됨",
            "해석 금지 문장": "특정 고도나 TPI 수치가 산불을 유발하는 직접적 원인이라고 단정하는 것"
        },
        {
            "요약 메시지": "산림 접근권 내 등산로 영향: 임도/등산로 500m 이내 영역에서는 등산로 최단거리가 유의미한 취약 인자(Cliff's delta -0.22)로 작용함",
            "근거 표": "S4-05_effect_size_by_variable.csv (산림 접근권 등산로 Cliff's delta -0.219, p=0.000)",
            "근거 플롯": "S4-05_effect_size_forest.png",
            "해석 가능 범위": "인접 인프라가 한정된 산림 접근권 영역 내부에서는 등산로 인접성이 공간적 취약 지표로 유의함",
            "해석 금지 문장": "등산객이 100% 고의로 산불을 냈다고 단정하는 것"
        }
    ]
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(TABLE_DIR / "S4-06_final_spatial_eda_summary.csv", index=False, encoding="utf-8-sig")
    print("S4-06_final_spatial_eda_summary.csv 생성 완료")
    
    # 2. Generate S4-06_spatial_variable_handoff.csv
    handoff_data = [
        {
            "변수명": "도로_최단거리_m",
            "등급": "EDA 핵심",
            "Cliff's delta (WUI)": -0.725,
            "권장 피처 처리 방식": "도로 10m 이내 여부 더미 변수(is_road_ultra_close) 및 선형 거리 변수 병용",
            "선정/제외 사유": "WUI 내부에서도 가장 강력하고 압도적인 공간 편향을 보임"
        },
        {
            "변수명": "시가화_최단거리_m",
            "등급": "후속 모델링 후보",
            "Cliff's delta (WUI)": -0.663,
            "권장 피처 처리 방식": "시가화 10m 이내 여부(is_urban_edge) 및 선형 거리",
            "선정/제외 사유": "매우 강한 비선형적 쏠림이 있으나, 도로와 상관관계가 높으므로 다중공선성 확인 필요"
        },
        {
            "변수명": "산림_최단거리_m",
            "등급": "후속 모델링 후보",
            "Cliff's delta (WUI)": 0.466,
            "권장 피처 처리 방식": "산림 내부(0m) vs 산림 외부 경계 거리 분리",
            "선정/제외 사유": "발생지가 산림 경계면 외부 190m 부근 WUI 구역에 치우쳐 있어 경계부 정의에 필수적"
        },
        {
            "변수명": "경사도(도)",
            "등급": "후속 모델링 후보",
            "Cliff's delta (WUI)": -0.318,
            "권장 피처 처리 방식": "선형 경사도 변수",
            "선정/제외 사유": "발생지가 완만한 사면에 쏠리는 경향이 전 공간층에서 통계적으로 유의함"
        },
        {
            "변수명": "고도(m)",
            "등급": "후속 모델링 후보",
            "Cliff's delta (WUI)": -0.179,
            "권장 피처 처리 방식": "기후지형유형 권역 층화 결합 피처",
            "선정/제외 사유": "권역별(영동/영서/산간) 고도 분포가 뚜렷이 분리되며, 발생지가 상대적으로 저지대에 편향됨"
        },
        {
            "변수명": "농업_최단거리_m",
            "등급": "보고서 보조",
            "Cliff's delta (WUI)": -0.168,
            "권장 피처 처리 방식": "선형 거리 변수",
            "선정/제외 사유": "WUI 내부에서 비교 시 다른 변수들에 비해 상대적으로 편향의 강도가 약함"
        },
        {
            "변수명": "TPI(지형위치지수)",
            "등급": "후속 모델링 후보",
            "Cliff's delta (WUI)": -0.106,
            "권장 피처 처리 방식": "산림 내부 vs WUI 층별 부호 반전 인터랙션 피처",
            "선정/제외 사유": "산림 내부에서는 능선부(+), WUI에서는 계곡부(-)로 편향이 반전되어 층별 교차 설계 필요"
        },
        {
            "변수명": "임도_최단거리_m",
            "등급": "보고서 보조",
            "Cliff's delta (WUI)": 0.104,
            "권장 피처 처리 방식": "선형 거리 변수",
            "선정/제외 사유": "WUI 내부 발화에는 미치는 영향이 작으나, 산림 접근권 층에서는 보조 피처로 유효함"
        },
        {
            "변수명": "등산로_최단거리_m",
            "등급": "후속 모델링 후보",
            "Cliff's delta (WUI)": -0.068,
            "권장 피처 처리 방식": "산림 접근권 층에서의 인터랙션 피처(is_forest_access * hiking_dist)",
            "선정/제외 사유": "산림 접근권 층 내부(Cliff's delta -0.219)에서 뚜렷한 음수 효과를 보임"
        },
        {
            "변수명": "TWI(지형다습지수)",
            "등급": "제외/보류",
            "Cliff's delta (WUI)": np.nan,
            "권장 피처 처리 방식": "N/A",
            "선정/제외 사유": "수문학적 흐름 누적 연산 한계로 대조군에서 계산 불가하여 주 비교 변수에서 보류"
        },
        {
            "변수명": "addr_type (임야번지)",
            "등급": "제외/보류",
            "Cliff's delta (WUI)": np.nan,
            "권장 피처 처리 방식": "N/A",
            "선정/제외 사유": "실제 토지피복 산림지역과 오차가 56%에 달해 취약성 프록시로 사용하기에 부적합하여 제외 권고"
        }
    ]
    df_handoff = pd.DataFrame(handoff_data)
    df_handoff.to_csv(TABLE_DIR / "S4-06_spatial_variable_handoff.csv", index=False, encoding="utf-8-sig")
    print("S4-06_spatial_variable_handoff.csv 생성 완료")
    
    # 3. Generate S4-06_interpretation_limits.csv
    limits_data = [
        {
            "구분": "도로/인프라 인접성",
            "해석 가능 범위": "동일 기상 및 공간층 하에서 발생지가 대조군보다 도로 및 시가화 구역에 물리적으로 더 밀착되어 분포함",
            "해석 금지/주의 사항": "도로변 담뱃재 투기나 차량 화재 등이 산불의 직접 원인이라고 인과를 단정 짓는 것",
            "대안 및 보완책": "물리적 노출 및 발원 취약 지대로만 기술하고, 발화 원인은 소방청 화인 통계 등과 별도 교차 분석할 것"
        },
        {
            "구분": "토지피복과 WUI 경계",
            "해석 가능 범위": "WUI 내 산불의 85% 이상이 실제 산림 피복 바깥의 시가화/초지 등 인프라 접경선에서 발원함",
            "해석 금지/주의 사항": "산림 면적이 좁은 지역이 산불에 취약하다거나, 건물 자체가 산불을 유도한다고 해석하는 것",
            "대안 및 보완책": "인간 활동 반경의 끝이자 산림 연료층의 시작점인 WUI 경계부가 물리적 연료 교차 공간임을 설명할 것"
        },
        {
            "구분": "지형 지표 (고도/경사)",
            "해석 가능 범위": "산불 발생지가 상대적으로 완만한 경사 및 저지대 골짜기 사면에 편향되어 분포함",
            "해석 금지/주의 사항": "특정 고도나 경사 수치가 산불을 유발하는 직접적 인자라고 인과적으로 해석하는 것",
            "대안 및 보완책": "기후지형유형 권역 분류에 따른 고도 분포의 층화 양상과 바람 경로(골짜기 푄 효과)와의 기하학적 정합성으로 설명할 것"
        },
        {
            "구분": "대조군 부족 71건",
            "해석 가능 범위": "71건의 대조군 부족은 강원도 격자 경계선 및 토지피복 미매칭 품질 한계에 따른 것이며, 주 분석 표본과 큰 분포 차이가 없음",
            "해석 금지/주의 사항": "이 71건의 제외가 주 분석 결과의 통계적 유의성에 왜곡을 초래했거나, 대조군이 없는 위험지라고 해석하는 것",
            "대안 및 보완책": "선택 편향 검증(민감도 표) 결과를 제시하고, 격자 경계부 토지피복 GIS 데이터 품질의 한계로 해석을 제한할 것"
        }
    ]
    df_limits = pd.DataFrame(limits_data)
    df_limits.to_csv(TABLE_DIR / "S4-06_interpretation_limits.csv", index=False, encoding="utf-8-sig")
    print("S4-06_interpretation_limits.csv 생성 완료")
    
    # 4. Generate S4-06_final_spatial_summary_plot.png
    print("S4-06 최종 요약 플롯 생성 중...")
    df_plot = df_handoff.dropna(subset=["Cliff's delta (WUI)"]).copy()
    df_plot["abs_delta"] = df_plot["Cliff's delta (WUI)"].abs()
    df_plot = df_plot.sort_values(by="abs_delta", ascending=True)
    
    colors = []
    for grade in df_plot["등급"]:
        if grade == "EDA 핵심":
            colors.append("#d73027")  # Red
        elif grade == "후속 모델링 후보":
            colors.append("#fc8d59")  # Orange
        elif grade == "보고서 보조":
            colors.append("#fee090")  # Yellow
        else:
            colors.append("#e0f3f8")  # Blue
            
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_plot["변수명"], df_plot["Cliff's delta (WUI)"], color=colors, edgecolor="black", linewidth=0.7)
    
    # Add vertical line at 0
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    
    # Add labels on bars
    for bar, grade in zip(bars, df_plot["등급"]):
        width = bar.get_width()
        x_pos = width + 0.02 if width >= 0 else width - 0.12
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{grade}", 
                va="center", ha="left" if width >= 0 else "right", fontsize=9, fontweight="bold")
                
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
