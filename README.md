
# MOF 흡착량 예측 프로그램 개발 지침

## 1. 담당 업무

| 담당 | 파일 | 구현 기능 |
|---|---|---|
| 1 | `util/cif_input.py` | CIF 로딩 및 구조 검사 |
| 2 | `util/cif_vis.py` | 결정구조 시각화 |
| 3 | `util/model.py` | CGCNN 체크포인트 로딩, 그래프 생성, CO₂/N₂ 추론 |
| 4 | `util/plotting.py` | 예측 결과 및 DB 그래프 |
| 5 | `util/db_search.py` | DB 검색·정렬·흡착량 비율 계산 |
| 6 | `util/report.py` | 결과 PDF 생성 |
| 7 | `main.py`, `app/`, `requirements.txt` | 6개 모듈 연결, 웹 UI 및 통합 테스트 |

## 2. 폴더 구조

```text
MOF-project/
├── README.md
├── requirements.txt
├── main.py
├── app/
├── util/
│   ├── __init__.py
│   ├── cif_input.py
│   ├── cif_vis.py
│   ├── model.py
│   ├── plotting.py
│   ├── db_search.py
│   └── report.py
├── models/
│   ├── cgcnn_scratch_CO2_N2.pt
│   └── cgcnn_scratch_model.py
├── data/
│   ├── mof_cif.parquet
│   └── cif/
└── outputs/
```

## 3. 함수 인터페이스

각 담당자는 하나의 `.py` 파일에 외부 호출용 `def` 함수 하나를 구현한다. 모든 함수는 `main.py`에서 `import`하여 호출할 수 있어야 한다.

| 모듈 | 함수 | 입력 | 반환 |
|---|---|---|---|
| `cif_input.py` | `load_cif(cif_path)` | CIF 파일 경로 | `pymatgen.Structure` |
| `cif_vis.py` | `visualize_structure(structure, output_path)` | `Structure`, 이미지 저장 경로 | 이미지 경로 `str` |
| `model.py` | `predict_adsorption(structure, pressure, gas)` | `Structure`, 압력 `float`, `"CO2"`/`"N2"` | 역정규화된 흡착량 `float` |
| `plotting.py` | `plot_adsorption(results, output_path)` | 결과 `DataFrame`, 이미지 저장 경로 | 이미지 경로 `str` |
| `db_search.py` | `search_database(db_path, filters, sort_by=None, ascending=True, calculate_selectivity=False)` | DB 경로, 검색·정렬 조건, 선택성 계산 여부 | 결과 `DataFrame` |
| `report.py` | `create_report(results, images, output_path)` | 결과 `DataFrame`, 이미지 경로 목록, PDF 저장 경로 | PDF 경로 `str` |

## 4. 데이터 전처리

**입력 압력 정규화**

$$
p_{\text{scaled}} = \frac{\log(1+\max(p,0))-\mu_p}{\sigma_p}
$$

**출력 흡착량 역정규화**

$$
\hat{y} = \hat{y}_{\text{scaled}}\cdot\sigma_y+\mu_y
$$

- $\mu_p,\sigma_p$: 체크포인트의 `pressure_mean`, `pressure_scale`
- $\mu_y,\sigma_y$: 체크포인트의 `preprocessor_state["y_stats"][gas]`
- 정규화 및 역정규화는 `model.py` 내부에서 처리하며, `predict_adsorption()`은 역정규화된 흡착량을 반환한다.

## 5. 협업 규칙

1. 1~6번 담당자는 자신의 `util/*.py` 파일을 개발하며, `main.py`, `app/`, `requirements.txt`는 7번 담당자가 관리한다.
2. 공통 함수명, 입력 인자, 반환 타입을 유지한다. 내부 보조 함수와 클래스는 자유롭게 구현한다.
3. 각 모듈은 다른 파일에서 `import`하여 사용할 수 있어야 하며, 웹 UI나 `import` 시 자동 실행되는 코드를 포함하지 않는다.
4. 예측 결과 `DataFrame`은 `mof_id`, `gas`, `pressure`, `adsorption` 컬럼을 사용한다. 입력/출력 데이터의 정규화/역정규화는 위의 데이터 전처리 설정을 따른다.
5. 개인 PC의 절대 경로를 사용하지 않고, 필요한 경로는 함수 인자로 전달받는다.
6. 각 담당자는 `.py` 파일, 필요 패키지 목록을 GitHub을 통해 제출한다.
