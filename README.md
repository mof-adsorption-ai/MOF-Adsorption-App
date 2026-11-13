# MOF 흡착량 예측 프로그램 개발 지침 (26.11.6 ver.)

## 1. 담당 업무

| 담당 | 파일 | 구현 기능 | 담당자 |
|---|---|---|---|
| 1 | `main.py`, `app/`, `requirements.txt` | 프로그램 총괄 및 기능 통합 | 문충만 |
| 2 | `utils/cif_vis.py` | MOF 결정구조 시각화 | 김지현 |
| 3 | `utils/preprocessing.py` | MOF 결정구조의 그래프 변환 | 정명수 |
| 4 | `utils/model.py` | CGCNN 모델을 이용한 흡착량 예측 | 이승호 |
| 5 | `utils/plotting.py` | 흡착성능 및 MOF DB 시각화 | 최종민 |
| 6 | `utils/db_search.py` | MOF DB 검색 및 정렬 | 이용호 |
| 7 | `utils/report.py` | 계산 결과 저장 및 보고서 생성 | 임동민 |

### 세부 구현 내용

| 담당 | 세부 구현 내용 |
|---|---|
| **1. 프로그램 총괄** | 웹 UI 개발, 사용자 입력 처리, 모듈 연결 및 통합 테스트 |
| **2. 결정구조 시각화** | MOF 결정구조 3D 시각화 및 이미지 저장 |
| **3. 그래프 변환** | MOF 결정구조를 CGCNN 입력 그래프로 변환 |
| **4. 흡착량 예측** | 모델 로딩, 압력 정규화, CO₂/N₂ 흡착량 예측 및 역정규화 |
| **5. 흡착성능 시각화** | 압력별 흡착량 및 MOF DB 비교 그래프 생성 |
| **6. MOF DB 검색** | 조건별 검색·정렬 및 CO₂/N₂ 흡착량 비율 계산 |
| **7. 결과 저장** | 예측 결과 CSV 저장 및 PDF 보고서 생성 |

## 2. 폴더 구조

```text
MOF-project/
├── README.md
├── requirements.txt
├── main.py
├── app/
│   └── __init__.py
├── utils/
│   ├── __init__.py
│   ├── cif_input.py
│   ├── cif_vis.py
│   ├── preprocessing.py
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

각 담당자는 자신의 `.py` 파일에 외부 호출용 대표 `def` 함수 하나를 구현한다. 내부 보조 함수와 클래스는 자유롭게 구현할 수 있으며, 모든 대표 함수는 `main.py`에서 `import`하여 호출할 수 있어야 한다.

| 모듈 | 함수 | 입력 | 반환 |
|---|---|---|---|
| `main.py` | `main()` | 웹 UI 사용자 입력 | `None` |
| `cif_input.py` | `load_cif(cif_path)` | CIF 파일 경로 | `Structure` |
| `cif_vis.py` | `visualize_structure(structure, output_path)` | `Structure`, 저장 경로 | 이미지 경로 `str` |
| `preprocessing.py` | `prepare_model_input(structure, checkpoint_path)` | `Structure`, 체크포인트 경로 | CGCNN 입력 데이터 `dict` |
| `model.py` | `predict_adsorption(graph_data, mof_id, pressures, gases, checkpoint_path)` | 그래프 데이터, MOF ID, 압력·기체 목록, 체크포인트 경로 | 예측 결과 `DataFrame` |
| `plotting.py` | `plot_adsorption(results, output_path, db_results=None)` | 예측 결과, 저장 경로, DB 결과 | 이미지 경로 `str` |
| `db_search.py` | `search_database(db_path, filters, sort_by=None, ascending=True, calculate_selectivity=False)` | DB 경로, 검색·정렬 조건 | 검색 결과 `DataFrame` |
| `report.py` | `create_report(results, images, graph_info, output_path, db_results=None)` | 예측 결과, 이미지 목록, 구조 정보, 저장 경로, DB 결과 | PDF·CSV 경로 `dict` |

### 모듈별 입력 및 출력

| 담당 | 입력 데이터 | 출력 데이터 |
|---|---|---|
| **1** | 사용자 입력 및 각 모듈의 실행 결과 | 웹 UI 표시 |
| **2** | MOF 결정구조 | 구조 이미지 경로 |
| **3** | MOF 결정구조, 체크포인트 | CGCNN 입력 그래프 |
| **4** | CGCNN 입력 그래프, 압력, 기체 | CO₂/N₂ 흡착량 예측 결과 |
| **5** | 예측 결과 및 MOF DB 데이터 | 흡착량 그래프 이미지 경로 |
| **6** | MOF DB 및 검색 조건 | 검색·정렬 결과 `DataFrame` |
| **7** | 예측 결과, 구조 정보, 이미지 및 DB 결과 | PDF·CSV 파일 경로 |

### 모듈 연결 순서

```text
                   1. 프로그램 총괄 및 기능 통합
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
       2. 결정구조 시각화    3. 그래프 변환    6. MOF DB 검색
              │               │               │
              │               ▼               │
              │          4. 흡착량 예측          │
              │               │               │
              │               ▼               │
              │         5. 흡착성능 시각화        │
              │               │               │
              └───────────────┼───────────────┘
                              │
                              ▼
                    7. 결과 저장 및 보고서 생성
```

## 4. 데이터 전처리

**결정구조 그래프 변환 (3번)**

| 항목 | 설정 |
|---|---|
| 원자 특성 | 체크포인트의 `atom_init` 사용 |
| 이웃 탐색 | 주기 경계 조건(PBC) 적용, 반경 8.0 Å 이내에서 가까운 순서로 최대 12개 |
| 거리 특성 | 0~8.0 Å의 Gaussian 중심, 간격·폭 0.2 Å (41차원) |
| 이웃 부족 시 | 이웃 인덱스 `0`, 거리 `9.0 Å`로 패딩한 뒤 Gaussian 변환 |

$$
G_k(d)=\exp\left[-\left(\frac{d-0.2k}{0.2}\right)^2\right],\quad k=0,1,\ldots,40
$$

**입력 압력 정규화 (4번)**

$$
p_{\text{scaled}} = \frac{\log(1+\max(p,0))-\mu_p}{\sigma_p}
$$

**출력 흡착량 역정규화 (4번)**

$$
\hat{y} = \hat{y}_{\text{scaled}}\cdot\sigma_y+\mu_y
$$

- $\mu_p,\sigma_p$: 체크포인트의 `pressure_mean`, `pressure_scale`
- $\mu_y,\sigma_y$: 체크포인트의 `preprocessor_state["y_stats"][gas]`

## 5. 협업 규칙

1. 1번 담당자는 `main.py`, `app/`, `requirements.txt`를 관리하며, 2~7번 담당자는 자신의 `utils/*.py` 파일을 개발한다. 
2. 공통 함수명, 입력 인자, 반환 타입을 유지한다. 내부 보조 함수와 클래스는 자유롭게 구현한다.
3. 각 모듈은 다른 파일에서 `import`하여 사용할 수 있어야 하며, 웹 UI나 `import` 시 자동 실행되는 코드를 포함하지 않는다.
4. 입력/출력 데이터의 정규화 및 역정규화는 위의 데이터 전처리 설정을 따른다.
5. 개인 PC의 절대 경로를 사용하지 않고, 필요한 경로는 함수 인자로 전달받는다.
6. 각 담당자는 `.py` 파일과 필요 패키지 목록을 GitHub을 통해 제출한다.

## 로컬 실행 환경 (Windows)

이 작업 폴더는 Python 3.13과 프로젝트 전용 `.venv`를 사용합니다.

```powershell
# 처음 설치할 때
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 앱 실행 (가상환경 활성화 불필요)
.\run.cmd
```

브라우저에서 http://localhost:8501 을 엽니다. 종료는 터미널에서 Ctrl+C를 누릅니다.
VS Code에서는 Python 인터프리터를 `.venv\Scripts\python.exe`로 선택하고,
실행 및 디버그에서 `MOF App (Streamlit)`을 선택해 F5로 실행할 수 있습니다.

샘플 CIF는 `data/cif/`에 있습니다. `data/mof_cif.parquet`가 작은 Git LFS
포인터 파일이면 DB 검색을 위해 Git LFS의 실제 데이터를 받아야 합니다.
예측 후 `흡착량 그래프 생성`을 누르면 CO₂·N₂ 그래프를 확인하고 PNG로
다운로드할 수 있습니다. DB 검색 결과가 있으면 비교 곡선도 표시됩니다.
DB 비교에는 예측과 동일한 압력·흡착량 단위를 사용하고, MOF·기체별로
하나의 온도 조건만 포함해야 합니다.

그래프 모듈 테스트는 다음 명령으로 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
