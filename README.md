# farm-system-public-02

산불 및 기상 관련 데이터 분석 프로젝트 작업공간입니다.

> 이 구조도는 `D:\farm-system-public-02\data\날씨데이터` 경로를 제외하고 작성했습니다.

## 작업공간 구조

```text
farm-system-public-02/
├─ .gitignore
├─ README.md
├─ data/
│  ├─ README.md
│  ├─ 산불_공간DB/
│  │  ├─ README.md
│  │  ├─ 산불발생위치도_지형특성계산.csv
│  │  ├─ 전국_등산로.csv
│  │  ├─ 전국_산불소화시설.csv
│  │  ├─ 전국_소방서_좌표.csv
│  │  ├─ 전국_소방용수시설.csv
│  │  ├─ 전국_임도망도.csv
│  │  ├─ 산불발생_토지피복도/
│  │  │  ├─ landcover_300m_clipped_강원특별자치도.gpkg
│  │  │  ├─ landcover_300m_clipped_경기도.gpkg
│  │  │  ├─ landcover_300m_clipped_경상남도.gpkg
│  │  │  ├─ landcover_300m_clipped_경상북도.gpkg
│  │  │  ├─ landcover_300m_clipped_광주광역시.gpkg
│  │  │  ├─ landcover_300m_clipped_대구광역시.gpkg
│  │  │  ├─ landcover_300m_clipped_대전광역시.gpkg
│  │  │  ├─ landcover_300m_clipped_부산광역시.gpkg
│  │  │  ├─ landcover_300m_clipped_서울특별시.gpkg
│  │  │  ├─ landcover_300m_clipped_세종특별자치시.gpkg
│  │  │  ├─ landcover_300m_clipped_울산광역시.gpkg
│  │  │  ├─ landcover_300m_clipped_인천광역시.gpkg
│  │  │  ├─ landcover_300m_clipped_전라남도.gpkg
│  │  │  ├─ landcover_300m_clipped_전북특별자치도.gpkg
│  │  │  ├─ landcover_300m_clipped_제주특별자치도.gpkg
│  │  │  ├─ landcover_300m_clipped_충청남도.gpkg
│  │  │  └─ landcover_300m_clipped_충청북도.gpkg
│  │  └─ 전처리코드/
│  │     ├─ 산불발생위치도_지형특성계산.py
│  │     ├─ 전국_등산로.py
│  │     ├─ 전국_산불소화시설.ipynb
│  │     ├─ 전국_소방서_좌표.ipynb
│  │     ├─ 전국_소방용수시설.ipynb
│  │     └─ 전국_임도망도.py
│  ├─ 산불_전국/
│  │  ├─ README.md
│  │  ├─ 산불_공간DB_위경도.csv
│  │  ├─ 산불발생위치도_전국.csv
│  │  └─ 전처리코드/
│  │     └─ data_preprocessing.ipynb
│  └─ 예측데이터/
│     ├─ README.md
│     └─ gangwon_poles_4326.csv
├─ jgy/
│  └─ README.md
├─ jsw/
│  ├─ data/
│  ├─ 주제에 대한 생각.md
│  └─ 과거 수상작/
│     ├─ 심층분석_보고서.md
│     ├─ 기상현상과 화재발생에 대한 상관분석/
│     │  ├─ 과제3 우수상.md
│     │  ├─ 과제3 장려상.md
│     │  ├─ 과제3 최우수상.md
│     │  └─ 과제3 특별상.md
│     └─ 소방데이터와 날씨 빅데이터를 융합한 119신고 건수 예측/
│        ├─ 우수상.md
│        ├─ 장려상.md
│        ├─ 최우수상.md
│        └─ 특별상.md
└─ kgr/
   └─ README.md
```

## 주요 폴더 설명

| 경로 | 내용 |
|---|---|
| `data/` | 프로젝트 공용 데이터 저장 영역입니다. |
| `data/산불_공간DB/` | 산불 발생 위치와 공간 데이터베이스 구축에 필요한 CSV, GPKG, 전처리 코드가 있습니다. |
| `data/산불_공간DB/산불발생_토지피복도/` | 시도별 토지피복도 GPKG 파일이 있습니다. |
| `data/산불_공간DB/전처리코드/` | 산불 공간 DB 생성을 위한 Python 및 Jupyter Notebook 전처리 코드가 있습니다. |
| `data/산불_전국/` | 전국 산불 발생 위치 및 위경도 기반 산불 데이터가 있습니다. |
| `data/산불_전국/전처리코드/` | 전국 산불 데이터 전처리 노트북이 있습니다. |
| `data/예측데이터/` | 예측 모델 입력 또는 공간 예측 격자 생성에 활용되는 좌표 기반 데이터가 있습니다. |
| `jgy/` | 개인 작업 폴더입니다. 현재 `README.md`만 있습니다. |
| `jsw/` | 주제 정리, 과거 수상작 분석 자료, 개인 작업 데이터 폴더가 있습니다. |
| `kgr/` | 개인 작업 폴더입니다. 현재 `README.md`만 있습니다. |

## 데이터 파일 요약

| 파일 | 형식 | 용도 |
|---|---|---|
| `data/산불_공간DB/산불발생위치도_지형특성계산.csv` | CSV | 산불 발생 위치에 지형 특성을 계산해 결합한 데이터입니다. |
| `data/산불_공간DB/전국_등산로.csv` | CSV | 전국 등산로 공간 정보 데이터입니다. |
| `data/산불_공간DB/전국_산불소화시설.csv` | CSV | 전국 산불소화시설 좌표 및 속성 데이터입니다. |
| `data/산불_공간DB/전국_소방서_좌표.csv` | CSV | 전국 소방서 위치 좌표 데이터입니다. |
| `data/산불_공간DB/전국_소방용수시설.csv` | CSV | 전국 소방용수시설 공간 정보 데이터입니다. |
| `data/산불_공간DB/전국_임도망도.csv` | CSV | 전국 임도망도 공간 정보 데이터입니다. |
| `data/산불_공간DB/산불발생_토지피복도/*.gpkg` | GPKG | 시도별 산불 발생 주변 토지피복도 공간 데이터입니다. |
| `data/산불_전국/산불발생위치도_전국.csv` | CSV | 전국 산불 발생 위치 원천 또는 정리 데이터입니다. |
| `data/산불_전국/산불_공간DB_위경도.csv` | CSV | 공간 DB 연계를 위한 산불 발생 위경도 데이터입니다. |
| `data/예측데이터/gangwon_poles_4326.csv` | CSV | WGS84 좌표계의 강원 지역 전신주 위치 데이터입니다. |

## 제외된 경로

다음 경로는 요청에 따라 이 README의 구조도와 설명 대상에서 제외했습니다.

```text
data/날씨데이터/
```

