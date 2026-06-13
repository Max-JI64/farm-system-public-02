from __future__ import annotations

import json
import uuid
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent
NOTEBOOK_PATH = HERE / "Step1_강원도_기후지형및캐나다지수_기본분석.ipynb"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def reset_cell(cell: dict) -> dict:
    cell = dict(cell)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def reviewed_cells() -> list[dict]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = [reset_cell(cell) for cell in notebook["cells"][:41]]

    cells[22] = code_cell(
        """
        grid_dem = weather_grid.to_crs(dem_crs)
        elevation_min = cell_points_dem["DEM_고도_m"].min()
        elevation_max = cell_points_dem["DEM_고도_m"].max()
        elevation_norm = Normalize(elevation_min, elevation_max)
        elevation_cmap = plt.get_cmap("terrain")

        fig, ax = plt.subplots(figsize=(9, 9))
        grid_dem.plot(ax=ax, facecolor="#f7f7f7", edgecolor="#666666", linewidth=0.55)
        ax.scatter(
            cell_points_dem.geometry.x,
            cell_points_dem.geometry.y,
            c=cell_points_dem["DEM_고도_m"],
            cmap=elevation_cmap,
            norm=elevation_norm,
            s=38,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        ax.set_title("S1-01 기상셀 경계와 중심점 DEM 표본")
        ax.set_axis_off()
        fig.colorbar(
            plt.cm.ScalarMappable(norm=elevation_norm, cmap=elevation_cmap),
            ax=ax,
            shrink=0.75,
            label="중심점 DEM 고도 (m)",
        )
        cell_map_path = PLOT_DIR / "S1-01_cell_center_sampling_map.png"
        fig.savefig(cell_map_path, dpi=180, bbox_inches="tight")
        plt.show()

        with rasterio.open(DATA_PATHS["dem"]) as dem:
            dem_image = dem.read(1, masked=True)
            dem_extent = [dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top]

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(dem_image, cmap="terrain", extent=dem_extent, origin="upper")
        grid_dem.boundary.plot(ax=ax, color="white", linewidth=0.35, alpha=0.65)
        ax.scatter(
            cell_points_dem.geometry.x,
            cell_points_dem.geometry.y,
            c=cell_points_dem["DEM_고도_m"],
            cmap=elevation_cmap,
            norm=elevation_norm,
            s=34,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        ax.set_title("S1-01 DEM 배경과 92개 중심점")
        ax.set_axis_off()
        fig.colorbar(
            plt.cm.ScalarMappable(norm=elevation_norm, cmap=elevation_cmap),
            ax=ax,
            shrink=0.75,
            label="중심점 DEM 고도 (m)",
        )
        dem_map_path = PLOT_DIR / "S1-01_dem_sampling_map.png"
        fig.savefig(dem_map_path, dpi=180, bbox_inches="tight")
        plt.show()
        """
    )
    cells[24] = code_cell(
        """
        hourly_month = hourly_weather["일시"].dt.to_period("M")
        hourly_monthly = (
            hourly_weather.assign(연월=hourly_month)
            .groupby([CELL_KEY, "연월"], as_index=False)
            .size()
        )
        hourly_monthly["예상수"] = hourly_monthly["연월"].dt.days_in_month * 24
        hourly_monthly["완전성_pct"] = 100 * hourly_monthly["size"] / hourly_monthly["예상수"]

        canadian_month = canadian_indices["날짜"].dt.to_period("M")
        canadian_monthly = (
            canadian_indices.assign(연월=canadian_month)
            .groupby([CELL_KEY, "연월"], as_index=False)
            .size()
        )
        canadian_monthly["예상수"] = canadian_monthly["연월"].dt.days_in_month
        canadian_monthly["완전성_pct"] = (
            100 * canadian_monthly["size"] / canadian_monthly["예상수"]
        )

        hourly_matrix = hourly_monthly.pivot(
            index=CELL_KEY, columns="연월", values="완전성_pct"
        )
        canadian_matrix = canadian_monthly.pivot(
            index=CELL_KEY, columns="연월", values="완전성_pct"
        )
        hourly_matrix.columns = hourly_matrix.columns.astype(str)
        canadian_matrix.columns = canadian_matrix.columns.astype(str)

        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(
            hourly_matrix,
            ax=ax,
            cmap="Blues",
            vmin=99,
            vmax=100,
            yticklabels=8,
            cbar_kws={"label": "완전성 (%)"},
        )
        ax.set_title("S1-01 시간기상 셀별 월간 완전성")
        ax.set_xlabel("연월")
        ax.tick_params(axis="x", labelrotation=90)
        hourly_completeness_path = PLOT_DIR / "S1-01_hourly_completeness_heatmap.png"
        fig.savefig(hourly_completeness_path, dpi=180, bbox_inches="tight")
        plt.show()

        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(
            canadian_matrix,
            ax=ax,
            cmap="Greens",
            vmin=99,
            vmax=100,
            yticklabels=8,
            cbar_kws={"label": "완전성 (%)"},
        )
        ax.set_title("S1-01 캐나다 지수 셀별 월간 완전성")
        ax.set_xlabel("연월")
        ax.tick_params(axis="x", labelrotation=90)
        canadian_completeness_path = (
            PLOT_DIR / "S1-01_canadian_completeness_heatmap.png"
        )
        fig.savefig(canadian_completeness_path, dpi=180, bbox_inches="tight")
        plt.show()
        """
    )
    cells[26] = code_cell(
        """
        summary_metrics = pd.DataFrame({
            "지표": [
                "기상셀 수",
                "시간기상 행수",
                "셀당 예상 시간수",
                "시간기상 중복 키 수",
                "시간기상 누락 시간 수",
                "시간기상 변수 결측 수",
                "캐나다 지수 행수",
                "셀당 예상 날짜수",
                "캐나다 지수 중복 키 수",
                "캐나다 지수 누락 날짜 수",
                "캐나다 지수 변수 결측 수",
                "중심점 셀 외부 수",
                "DEM 샘플 실패 수",
            ],
            "값": [
                len(meta_cell_set),
                len(hourly_weather),
                expected_hours_per_cell,
                int(hourly_weather[[CELL_KEY, "일시"]].duplicated().sum()),
                int(hourly_cell_coverage["누락시간수"].sum()),
                int(hourly_weather.isna().sum().sum()),
                len(canadian_indices),
                expected_days_per_cell,
                int(canadian_indices[[CELL_KEY, "날짜"]].duplicated().sum()),
                int(canadian_cell_coverage["누락날짜수"].sum()),
                int(canadian_indices.isna().sum().sum()),
                int((~cell_points_wgs84["중심점_셀내부"]).sum()),
                int((~cell_points_dem["DEM_샘플성공"]).sum()),
            ],
        })
        summary_metrics.to_csv(
            TABLE_DIR / "S1-01_summary_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        assert source_audit["존재"].all()
        assert dataset_audit["중복키수"].sum() == 0
        assert dataset_audit["결측키행수"].sum() == 0
        assert dataset_audit["메타에는있고자료에는없는셀수"].sum() == 0
        assert dataset_audit["자료에만있는셀수"].sum() == 0
        assert hourly_cell_coverage["누락시간수"].sum() == 0
        assert canadian_cell_coverage["누락날짜수"].sum() == 0
        assert hourly_weather.isna().sum().sum() == 0
        assert canadian_indices.isna().sum().sum() == 0
        assert cell_points_wgs84["중심점_셀내부"].all()
        assert cell_points_dem["DEM_샘플성공"].all()
        for path in [
            cell_map_path,
            dem_map_path,
            hourly_completeness_path,
            canadian_completeness_path,
        ]:
            assert path.exists() and path.stat().st_size > 0

        display(summary_metrics)
        print("S1-01 validation passed")
        """
    )
    cells[32] = code_cell(
        """
        grid_region = weather_grid.merge(
            cell_climate[[CELL_KEY, "DEM_고도_m"]],
            on=CELL_KEY,
            how="left",
            validate="one_to_one",
        ).to_crs(dem_crs)
        points_region = gpd.GeoDataFrame(
            cell_climate.copy(),
            geometry=gpd.points_from_xy(
                cell_climate["중심경도_wgs84"],
                cell_climate["중심위도_wgs84"],
            ),
            crs="EPSG:4326",
        ).to_crs(dem_crs)

        fig, ax = plt.subplots(figsize=(9, 9))
        for region in REGION_ORDER:
            subset = grid_region.loc[grid_region["기후권역"] == region]
            subset.plot(
                ax=ax,
                facecolor=REGION_PALETTE[region],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.85,
            )
        region_handles = [
            Patch(
                facecolor=REGION_PALETTE[region],
                edgecolor="white",
                label=f"{region} ({int((cell_climate['기후권역'] == region).sum())}셀)",
            )
            for region in REGION_ORDER
        ]
        ax.legend(handles=region_handles, title="기후권역", loc="lower left")
        ax.set_title("S1-02 영동·영서 기상셀 공간 분포")
        ax.set_axis_off()
        s102_region_map_path = PLOT_DIR / "S1-02_region_map.png"
        fig.savefig(s102_region_map_path, dpi=180, bbox_inches="tight")
        plt.show()

        elevation_norm_s102 = Normalize(
            cell_climate["DEM_고도_m"].min(),
            cell_climate["DEM_고도_m"].max(),
        )
        elevation_cmap_s102 = plt.get_cmap("terrain")
        fig, ax = plt.subplots(figsize=(9, 9))
        grid_region.plot(ax=ax, facecolor="#f7f7f7", edgecolor="#808080", linewidth=0.45)
        for region, marker in zip(REGION_ORDER, ["o", "s"]):
            subset = points_region.loc[points_region["기후권역"] == region]
            ax.scatter(
                subset.geometry.x,
                subset.geometry.y,
                c=subset["DEM_고도_m"],
                cmap=elevation_cmap_s102,
                norm=elevation_norm_s102,
                marker=marker,
                s=48,
                edgecolor="black",
                linewidth=0.5,
                label=region,
                zorder=3,
            )
        ax.legend(title="기후권역", loc="lower left")
        ax.set_title("S1-02 권역별 중심점 DEM 고도")
        ax.set_axis_off()
        fig.colorbar(
            plt.cm.ScalarMappable(
                norm=elevation_norm_s102,
                cmap=elevation_cmap_s102,
            ),
            ax=ax,
            shrink=0.78,
            label="중심점 DEM 고도 (m)",
        )
        s102_dem_map_path = PLOT_DIR / "S1-02_region_dem_points.png"
        fig.savefig(s102_dem_map_path, dpi=180, bbox_inches="tight")
        plt.show()
        """
    )
    cells[34] = code_cell(
        """
        np.random.seed(42)
        s102_box_paths = []
        for column, label, unit in VARIABLE_SPECS:
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            sns.boxplot(
                data=cell_climate,
                x="기후권역",
                y=column,
                order=REGION_ORDER,
                hue="기후권역",
                palette=REGION_PALETTE,
                legend=False,
                width=0.55,
                showfliers=False,
                ax=ax,
            )
            sns.stripplot(
                data=cell_climate,
                x="기후권역",
                y=column,
                order=REGION_ORDER,
                hue="기후권역",
                palette=REGION_PALETTE,
                legend=False,
                jitter=0.16,
                size=4,
                alpha=0.72,
                edgecolor="black",
                linewidth=0.3,
                ax=ax,
            )
            ax.set_title(f"S1-02 영동·영서 {label}")
            ax.set_xlabel("")
            ax.set_ylabel(unit)
            path = PLOT_DIR / f"S1-02_boxplot_{column}.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.show()
            s102_box_paths.append(path)
        """
    )
    cells[36] = code_cell(
        """
        s102_ecdf_paths = []
        for column, label, unit in VARIABLE_SPECS:
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            sns.ecdfplot(
                data=cell_climate,
                x=column,
                hue="기후권역",
                hue_order=REGION_ORDER,
                palette=REGION_PALETTE,
                linewidth=2.1,
                ax=ax,
            )
            ax.set_title(f"S1-02 영동·영서 {label} ECDF")
            ax.set_xlabel(unit)
            ax.set_ylabel("누적비율")
            path = PLOT_DIR / f"S1-02_ecdf_{column}.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.show()
            s102_ecdf_paths.append(path)
        """
    )
    cells[38] = code_cell(
        """
        effect_plot = region_tests.sort_values("Cohen_d_영동_minus_영서").copy()
        y_positions = np.arange(len(effect_plot))

        fig, ax = plt.subplots(figsize=(8, 6))
        cohen_values = effect_plot["Cohen_d_영동_minus_영서"]
        ax.barh(
            y_positions,
            cohen_values,
            color=np.where(
                cohen_values >= 0,
                REGION_PALETTE["영동"],
                REGION_PALETTE["영서"],
            ),
            alpha=0.85,
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(-0.8, color="#777777", linestyle="--", linewidth=0.8)
        ax.axvline(0.8, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_yticks(y_positions, effect_plot["표시명"])
        ax.set_xlabel("Cohen's d (영동 - 영서)")
        ax.set_title("S1-02 영동·영서 평균 차이 효과크기")
        s102_cohen_path = PLOT_DIR / "S1-02_effect_cohen_d.png"
        fig.savefig(s102_cohen_path, dpi=180, bbox_inches="tight")
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        cliff_values = effect_plot["Cliff_delta_영동_minus_영서"]
        ax.scatter(
            cliff_values,
            y_positions,
            c=np.where(
                cliff_values >= 0,
                REGION_PALETTE["영동"],
                REGION_PALETTE["영서"],
            ),
            s=75,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(-0.474, color="#777777", linestyle="--", linewidth=0.8)
        ax.axvline(0.474, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_yticks(y_positions, effect_plot["표시명"])
        ax.set_xlabel("Cliff's delta (영동 - 영서)")
        ax.set_title("S1-02 영동·영서 순위 기반 효과크기")
        s102_cliff_path = PLOT_DIR / "S1-02_effect_cliff_delta.png"
        fig.savefig(s102_cliff_path, dpi=180, bbox_inches="tight")
        plt.show()
        """
    )
    cells[40] = code_cell(
        """
        s102_plot_paths = [
            s102_region_map_path,
            s102_dem_map_path,
            *s102_box_paths,
            *s102_ecdf_paths,
            s102_cohen_path,
            s102_cliff_path,
        ]
        s102_summary = pd.DataFrame({
            "지표": [
                "영동 셀 수",
                "영서 셀 수",
                "셀별 관측일수 최솟값",
                "셀별 관측일수 최댓값",
                "분석 변수 수",
                "Welch FDR 유의 변수 수",
                "Mann-Whitney FDR 유의 변수 수",
                "S1-02 결과표 수",
                "S1-02 개별 플롯 수",
            ],
            "값": [
                int((cell_climate["기후권역"] == "영동").sum()),
                int((cell_climate["기후권역"] == "영서").sum()),
                int(cell_climate["관측일수"].min()),
                int(cell_climate["관측일수"].max()),
                len(ANALYSIS_COLUMNS),
                int(region_tests["Welch_FDR_0.05"].sum()),
                int(region_tests["MannWhitney_FDR_0.05"].sum()),
                4,
                len(s102_plot_paths),
            ],
        })
        s102_summary.to_csv(
            TABLE_DIR / "S1-02_summary_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        assert len(cell_climate) == 92
        assert int((cell_climate["기후권역"] == "영동").sum()) == 21
        assert int((cell_climate["기후권역"] == "영서").sum()) == 71
        assert cell_climate["관측일수"].min() == 731
        assert cell_climate["관측일수"].max() == 731
        assert cell_climate[ANALYSIS_COLUMNS].isna().sum().sum() == 0
        assert len(region_tests) == len(ANALYSIS_COLUMNS)
        assert np.isfinite(
            region_tests[
                [
                    "Welch_p",
                    "MannWhitney_p",
                    "Cohen_d_영동_minus_영서",
                    "Cliff_delta_영동_minus_영서",
                ]
            ].to_numpy()
        ).all()
        for path in s102_plot_paths:
            assert path.exists() and path.stat().st_size > 0

        display(s102_summary)
        print("S1-02 validation passed")
        """
    )

    cells.extend(
        [
            markdown_cell(
                """
                ## 20. 검토 반영 분석 모듈

                S1-03 이후 분석은 반복관측 독립성, 이상 셀 민감도, 지수의 구성적 상관,
                선행연구 단계의 제한적 의미를 반영한 검토 버전으로 실행한다.
                결과 해석은 대응 진행예정로그에만 기록한다.
                """
            ),
            code_cell(
                """
                from step1_reviewed_analysis import (
                    run_s103,
                    run_s104,
                    run_s105,
                    run_s106,
                    run_s107,
                    run_s108,
                    run_s109,
                    run_s110,
                    run_s1a01,
                )
                """
            ),
            markdown_cell("## 21. S1-03 상호배타적 계절 비교"),
            code_cell(
                """
                s103 = run_s103(
                    hourly_weather=hourly_weather,
                    climate_type=climate_type,
                    table_dir=TABLE_DIR,
                    plot_dir=PLOT_DIR,
                )
                display(s103["paired_tests"])
                display(s103["date_tests"])
                """
            ),
            markdown_cell("## 22. S1-04 군집 이상치·민감도·안정성 감사"),
            code_cell(
                """
                s104 = run_s104(
                    cell_climate=cell_climate,
                    climate_type=climate_type,
                    weather_grid=weather_grid,
                    table_dir=TABLE_DIR,
                    plot_dir=PLOT_DIR,
                )
                display(s104["quality_audit"].query("품질검토대상"))
                display(s104["scores"])
                display(s104["crosstab_k3"])
                display(s104["stability"].describe())
                """
            ),
            markdown_cell("## 23. S1-05 3개 기후지형유형 기술적 분리 재검증"),
            code_cell(
                """
                s105 = run_s105(
                    full_data=s104["data"],
                    clean_data=s104["clean"],
                    table_dir=TABLE_DIR,
                    plot_dir=PLOT_DIR,
                )
                display(s105["global_tests"])
                display(s105["pairwise"])
                """
            ),
            markdown_cell("## 24. S1-06 월별 기후와 풍향 분포"),
            code_cell(
                """
                s106 = run_s106(
                    hourly_weather=hourly_weather,
                    climate_type=climate_type,
                    table_dir=TABLE_DIR,
                    plot_dir=PLOT_DIR,
                )
                display(s106["monthly_summary"].head(18))
                display(s106["monthly_tests"].head())
                """
            ),
            markdown_cell("## 25. S1-07 고도 연관성·층화·비선형 민감도"),
            code_cell(
                """
                s107 = run_s107(
                    full_data=s104["data"],
                    clean_data=s104["clean"],
                    table_dir=TABLE_DIR,
                    plot_dir=PLOT_DIR,
                )
                display(s107["correlations"])
                display(s107["nonlinear"])
                """
            ),
            markdown_cell("## 26. S1-08 셀 단위 캐나다 지수 배경 분포"),
            code_cell(
                """
                s108 = run_s108(
                    canadian_indices=canadian_indices,
                    climate_type=climate_type,
                    table_dir=TABLE_DIR,
                    plot_dir=PLOT_DIR,
                )
                display(s108["regional_tests"])
                display(s108["season_summary"].head(24))
                display(s108["season_tests"])
                """
            ),
            markdown_cell("## 27. S1-09 FFMC 10일평균과 선행연구 단계"),
            code_cell(
                """
                s109 = run_s109(
                    canadian_data=s108["data"],
                    weather_grid=weather_grid,
                    table_dir=TABLE_DIR,
                    plot_dir=PLOT_DIR,
                )
                display(s109["ffmc10_summary"])
                display(s109["crosstab"].round(1))
                """
            ),
            markdown_cell("## 28. S1-10 지수 구성 정합성과 층화 상관"),
            code_cell(
                """
                s110 = run_s110(
                    canadian_data=s108["data"],
                    table_dir=TABLE_DIR,
                    plot_dir=PLOT_DIR,
                )
                display(s110["construction"])
                display(s110["cross_corr"].round(2))
                display(
                    s110["stratified"].query(
                        "기상변수 == '기온_C' and FWI지수 == 'FWI'"
                    )
                )
                """
            ),
            markdown_cell("## 29. S1-A01 품질검토대상 셀 원천 요약"),
            code_cell(
                """
                s1a01 = run_s1a01(
                    hourly_weather=hourly_weather,
                    full_data=s104["data"],
                    quality_flag_ids=s104["quality_flag_ids"],
                    table_dir=TABLE_DIR,
                )
                display(s1a01)
                """
            ),
            markdown_cell("## 30. 수정 분석 완료 검증"),
            code_cell(
                """
                required_tables = [
                    "S1-03_fire_vs_nonfire_paired_tests.csv",
                    "S1-04_cluster_sensitivity_scores.csv",
                    "S1-04_k3_seed_stability.csv",
                    "S1-05_kruskal_sensitivity.csv",
                    "S1-06_monthly_type_tests.csv",
                    "S1-07_altitude_spearman_sensitivity.csv",
                    "S1-08_fwi_cell_level_kruskal.csv",
                    "S1-09_ffmc10_season_cell_summary.csv",
                    "S1-10_weather_fwi_stratified_spearman.csv",
                    "S1-A01_flagged_cell_source_audit.csv",
                ]
                missing_tables = [
                    name for name in required_tables if not (TABLE_DIR / name).exists()
                ]
                assert not missing_tables, missing_tables
                assert len(s103["paired_tests"]) == 5
                assert len(s104["clean"]) < len(s104["data"])
                assert s108["regional_tests"]["N_셀"].eq(92).all()
                assert set(s109["crosstab"].columns) == {
                    "1단계", "2단계", "3단계", "4단계"
                }
                print("Reviewed S1-03 through S1-10 and S1-A01 validation passed")
                """
            ),
        ]
    )
    return cells


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    notebook["cells"] = reviewed_cells()
    notebook["nbformat_minor"] = 5
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"written: {NOTEBOOK_PATH.name} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
