# 공모전 대주제 : 기상 데이티 및 공간정보 기반 전력설비 인근 화재 위험도 분석
- 공간데이터 융합 분석을 바탕으로 전력설비 인근의 화재(산불) 발생 위험도를 모델링
- 기상변수를 포함, 각종 데이터로부터 추출한 화재발생 인자를 토대로 설비별 위험수준을 정량적으로 제시

## 진행 원칙
- 결측치가 있다는 이유만으로 변수 전체를 제외하지 않음
- 결측치 보간/대체는 하지 않음
- 변수별 EDA에서는 해당 변수의 결측 행만 제외하고, 강원특별자치도와 타 지역의 유효 표본 수를 함께 확인
- 분석 및 시각화 코드를 작성하기 전에, 기본적으로 데이터 구조(컬럼명, unique 값 등을 통한 지역명 파악, 데이터 형태 등)를 pandas로 먼저 확인한 후 분석에 적합한 코드를 생성함
- **노트북(.ipynb) 수정 및 검증 방식**: 파일 깨짐 방지를 위해 직접 편집하지 않고, 1) 임시 py 스크립트로 개별 로직 사전 테스트, 2) 전체 셀 코드를 모은 통합 스크립트(`run_nb_cells.py`) 실행을 통한 인코딩 및 런타임 무결성 검증, 3) 파이썬 `json` 모듈을 이용한 셀 노드 안전 주입 방식을 고수한다. 작업 시 생성된 백업 파일(`*backup*.ipynb` 등)은 검증 직후 즉시 삭제한다.

### ipynb 확인 원칙
`.ipynb` 파일은 출력 이미지, HTML, 위젯 상태, base64 데이터가 많이 포함될 수 있으므로 전체 JSON을 그대로 읽지 않는다.
Codex가 노트북 내용을 확인할 때는 셀 번호, 셀 타입, 입력 코드/마크다운, 필요한 짧은 `text/plain` 출력만 확인하고, 플롯 이미지와 대용량 출력은 제외한다.

### Matplotlib 한글 폰트 설정
`koreanize_matplotlib`만으로 한글이 깨질 경우, Windows 기본 맑은 고딕 폰트를 직접 등록해서 사용한다.
폰트 설정 이후에 `sns.set_theme(style="whitegrid")`를 다시 호출하면 `plt.rcParams["font.family"]`가 `['sans-serif']`로 덮여 한글이 다시 깨진다.
따라서 seaborn 스타일 설정과 폰트 설정은 아래 셀 하나로 처리하고, 이후에는 `sns.set_theme()`를 단독으로 다시 실행하지 않는다.

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

font_path = "C:/Windows/Fonts/malgun.ttf"
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()

sns.set_theme(
    style="whitegrid",
    font=font_name,
    rc={
        "font.family": font_name,
        "font.sans-serif": [font_name],
        "axes.unicode_minus": False
    }
)

plt.rcParams["font.family"] = font_name
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False

pd.set_option("display.max_columns", None)
print("현재 matplotlib font.family:", plt.rcParams["font.family"])
print("현재 matplotlib font.sans-serif:", plt.rcParams["font.sans-serif"][:3])
```

정상 설정이면 출력이 아래처럼 나와야 한다.

```text
현재 matplotlib font.family: ['Malgun Gothic']
현재 matplotlib font.sans-serif: ['Malgun Gothic']
```

## 분석에 대한 내용
### [강풍 및 건조 기후로 인한 산불 위험 배경]
- 시기: 초겨울에서 늦봄 (11월 ~ 익년 5월)
- 기상: 이동성 고기압에 따른 급격한 건조 및 강풍 발생
    - ※ 특히 강원 지역은 태백산맥 인근 **'양간지풍'**의 영향으로 위험도 높음
- 관련근거: 산림보호법에 따른 산불조심기간 운영 (매년 2월 ~ 5월, 산림청)

### [한국전력공사의 대응]
- 현장관리: 전력설비 화재 예방을 위해 현장 순시 및 대비활동 강화
- 효율화: 설비 산불발생 위험성 기반의 '우선순위 선정 모델' 개발 및 활용
    - -> 고위험 지역을 대상으로 집중 점검 및 설비 보강공사 추진 등

## 프로젝트 주제
아직 미정