from pathlib import Path
import csv


BASE_DIR = Path(__file__).resolve().parent

CSV_2020_PATH = BASE_DIR / "수종별임상도(나무종류지도)_시도" / "산불_공간DB_수종별임상도2020_전국.csv"
CSV_2023_PATH = BASE_DIR / "임상도(15000).gdb" / "산불_공간DB_임상도2023_전국.csv"
OUTPUT_CSV_PATH = BASE_DIR / "산불_공간DB_임상도_전국통합.csv"

BASE_COLUMNS = ["fire_id", "위도", "경도"]
FOREST_COLUMNS = [
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
]
CODE_COLUMNS = ["임상구분코드", "수종코드", "경급코드", "영급코드", "소밀도코드"]
OUTPUT_COLUMNS = BASE_COLUMNS + FOREST_COLUMNS + ["임상도_출처"]


def clean_value(value: str | None) -> str:
    if value is None:
        return ""

    value = value.strip()
    if value.lower() in {"nan", "none", "<na>"}:
        return ""

    return value


def read_csv_by_fire_id(csv_path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = {}
        order = []

        for row in reader:
            cleaned_row = {column: clean_value(value) for column, value in row.items()}
            fire_id = cleaned_row["fire_id"]
            rows[fire_id] = cleaned_row
            order.append(fire_id)

    return order, rows


def has_forest_value(row: dict[str, str] | None) -> bool:
    if row is None:
        return False

    return any(clean_value(row.get(column)) for column in CODE_COLUMNS)


def build_empty_forest_values() -> dict[str, str]:
    return {column: "" for column in FOREST_COLUMNS}


def pick_forest_values(
    row_2020: dict[str, str] | None,
    row_2023: dict[str, str] | None,
) -> tuple[dict[str, str], str]:
    if has_forest_value(row_2023):
        return {column: clean_value(row_2023.get(column)) for column in FOREST_COLUMNS}, "2023"

    if has_forest_value(row_2020):
        return {column: clean_value(row_2020.get(column)) for column in FOREST_COLUMNS}, "2020_보완"

    return build_empty_forest_values(), "결측"


def pick_base_values(
    fire_id: str,
    row_2020: dict[str, str] | None,
    row_2023: dict[str, str] | None,
) -> dict[str, str]:
    base_row = row_2023 or row_2020 or {}
    return {
        "fire_id": fire_id,
        "위도": clean_value(base_row.get("위도")),
        "경도": clean_value(base_row.get("경도")),
    }


def count_source(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {}
    for row in rows:
        source = row["임상도_출처"]
        counts[source] = counts.get(source, 0) + 1
    return counts


def main() -> None:
    print("임상도 통합을 시작합니다.")
    print("통합 규칙: 2023 우선 -> 2020 보완 -> 결측")

    order_2020, rows_2020 = read_csv_by_fire_id(CSV_2020_PATH)
    order_2023, rows_2023 = read_csv_by_fire_id(CSV_2023_PATH)

    fire_id_order = []
    seen_fire_ids = set()
    for fire_id in order_2023 + order_2020:
        if fire_id not in seen_fire_ids:
            fire_id_order.append(fire_id)
            seen_fire_ids.add(fire_id)

    integrated_rows = []
    for fire_id in fire_id_order:
        row_2020 = rows_2020.get(fire_id)
        row_2023 = rows_2023.get(fire_id)

        forest_values, source = pick_forest_values(row_2020, row_2023)
        output_row = pick_base_values(fire_id, row_2020, row_2023)
        output_row.update(forest_values)
        output_row["임상도_출처"] = source
        integrated_rows.append(output_row)

    with OUTPUT_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(integrated_rows)

    print(f"저장 완료: {OUTPUT_CSV_PATH}")
    print(f"전체 행 수: {len(integrated_rows):,}")
    for source, count in sorted(count_source(integrated_rows).items()):
        print(f"{source}: {count:,}")


if __name__ == "__main__":
    main()
