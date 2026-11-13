"""Streamlit entry point. Run with: streamlit run main.py."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "cgcnn_scratch_model.pt"
DATABASE = ROOT / "data" / "mof_cif.parquet"


def show_error(error: Exception) -> None:
    """Keep unfinished team modules from interrupting the page."""
    if isinstance(error, NotImplementedError):
        st.info("이 기능은 담당 모듈 구현 후 연결됩니다. 현재는 페이지 시안을 확인할 수 있습니다.")
    elif isinstance(error, ImportError):
        st.error("기능 실행에 필요한 패키지를 불러올 수 없습니다. 설치 환경을 확인해 주세요.")
        st.caption(str(error))
    else:
        st.error(f"처리 중 오류가 발생했습니다: {error}")


def render_structure() -> None:
    structure = st.session_state.get("structure")
    if structure is None:
        st.info("왼쪽에서 CIF 파일을 업로드하면 결정구조 정보를 확인할 수 있습니다.")
        return

    a, b, c = st.columns(3)
    a.metric("화학식", structure.composition.reduced_formula)
    b.metric("원자 수", len(structure))
    c.metric("단위격자 부피", f"{structure.volume:,.2f} Å³")
    with st.expander("격자 정보 · CIF 원문"):
        st.write(dict(zip(("a (Å)", "b (Å)", "c (Å)"), structure.lattice.abc)))
        st.write(dict(zip(("α (°)", "β (°)", "γ (°)"), structure.lattice.angles)))
        st.code(st.session_state["cif_bytes"].decode("utf-8", errors="replace"), language="text")

    regenerate = st.button("구조 이미지 다시 생성", type="primary", use_container_width=True)
    if regenerate or not st.session_state.get("structure_image_attempted"):
        st.session_state["structure_image_attempted"] = True
        st.session_state.pop("structure_image_error", None)
        try:
            from utils.cif_vis import visualize_structure

            with st.spinner("결정구조 이미지를 생성하고 있습니다…"):
                path = Path(st.session_state["workdir"].name) / "structure.png"
                image_path = visualize_structure(structure, path)
                st.session_state["structure_image"] = Path(image_path).read_bytes()
        except Exception as error:
            st.session_state["structure_image_error"] = error
    if st.session_state.get("structure_image_error") is not None:
        show_error(st.session_state["structure_image_error"])
    if st.session_state.get("structure_image"):
        st.image(st.session_state["structure_image"], caption="MOF 결정구조", use_container_width=True)
        st.caption("단위격자 내 원자 배치와 원소별 색상을 표시한 정적 이미지입니다. 결합선은 표시하지 않습니다.")
        st.download_button(
            "구조 이미지 PNG 다운로드", st.session_state["structure_image"],
            file_name=f"{st.session_state['mof_id']}_structure.png", mime="image/png",
        )


def render_prediction() -> None:
    st.subheader("흡착량 예측")
    st.caption("결정구조와 압력 조건을 입력하여 CO₂·N₂ 흡착량을 계산합니다.")
    with st.form("prediction_form"):
        gases = st.multiselect("대상 기체", ["CO2", "N2"], default=["CO2", "N2"])
        st.caption("압력 단위는 모델 학습 데이터의 기준을 따릅니다. 모듈 통합 시 확정합니다.")
        left, right = st.columns(2)
        start = left.number_input("시작 압력", min_value=0.0, value=0.0, step=0.1)
        end = right.number_input("종료 압력", min_value=0.0, value=1.0, step=0.1)
        count = st.slider("계산 지점 수", 2, 100, 11)
        submitted = st.form_submit_button(
            "흡착량 계산", type="primary", use_container_width=True,
            disabled=st.session_state.get("structure") is None,
        )
    if st.session_state.get("structure") is None:
        st.info("예측을 시작하려면 CIF 파일을 먼저 업로드해 주세요.")
    if submitted:
        if not gases or end <= start:
            st.warning("기체를 하나 이상 선택하고 종료 압력을 시작 압력보다 크게 설정해 주세요.")
        else:
            try:
                from utils.preprocessing import prepare_model_input
                from utils.model import predict_adsorption

                pressures = [start + (end - start) * i / (count - 1) for i in range(count)]
                with st.spinner("구조를 그래프로 변환하고 흡착량을 계산하고 있습니다…"):
                    graph = prepare_model_input(st.session_state["structure"], CHECKPOINT)
                    results = predict_adsorption(
                        graph, st.session_state["mof_id"], pressures, gases, CHECKPOINT
                    )
                st.session_state["results"] = results
                st.session_state.pop("adsorption_image", None)
            except Exception as error:
                show_error(error)

    results = st.session_state.get("results")
    if results is None:
        st.caption("계산 완료 후 예측 결과 표, 흡착 등온선과 CSV 다운로드가 표시됩니다.")
        return
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button("예측 결과 CSV 다운로드", results.to_csv(index=False).encode("utf-8-sig"),
                       file_name="adsorption_results.csv", mime="text/csv")
    if st.button("흡착량 그래프 생성", use_container_width=True):
        try:
            from utils.plotting import plot_adsorption

            with st.spinner("흡착량 그래프를 생성하고 있습니다…"):
                path = Path(st.session_state["workdir"].name) / "adsorption.png"
                st.session_state["adsorption_image"] = plot_adsorption(
                    results, path, db_results=st.session_state.get("db_results")
                )
        except Exception as error:
            show_error(error)
    if st.session_state.get("adsorption_image"):
        st.image(st.session_state["adsorption_image"], use_container_width=True)


def render_database() -> None:
    st.subheader("MOF 데이터베이스 탐색")
    st.caption("검색 결과를 예측 결과와 비교할 수 있습니다. 검색 조건은 DB 모듈의 규격 확정 후 확장합니다.")
    with st.form("database_form"):
        st.write("전체 MOF 조회")
        selectivity = st.checkbox("CO₂ / N₂ 선택도 계산", value=False)
        submitted = st.form_submit_button("데이터베이스 검색", use_container_width=True)
    if submitted:
        try:
            if not DATABASE.is_file():
                raise FileNotFoundError("MOF 데이터베이스 파일을 찾을 수 없습니다.")
            with DATABASE.open("rb") as source:
                is_pointer = source.read(80).startswith(b"version https://git-lfs.github.com/spec/")
            if is_pointer:
                st.warning("현재 DB 파일은 Git LFS 포인터입니다. 실제 Parquet 데이터를 받은 뒤 검색할 수 있습니다.")
            else:
                from utils.db_search import search_database

                with st.spinner("데이터베이스를 검색하고 있습니다…"):
                    st.session_state["db_results"] = search_database(
                        DATABASE, filters={}, calculate_selectivity=selectivity
                    )
                st.session_state.pop("adsorption_image", None)
        except Exception as error:
            show_error(error)
    results = st.session_state.get("db_results")
    if results is not None:
        st.caption(f"검색 결과 {len(results):,}건")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("검색 결과 CSV 다운로드", results.to_csv(index=False).encode("utf-8-sig"),
                           file_name="mof_database_results.csv", mime="text/csv")
    else:
        st.info("검색 결과가 이곳에 표시됩니다.")


def main() -> None:
    st.set_page_config(page_title="MOF Adsorption Lab", page_icon="🧪", layout="wide")
    st.markdown("""
        <style>
        /* Keep the page readable even when Streamlit uses a dark system theme. */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: #f6f7fa; color: #303440; color-scheme: light;
            --text-color: #303440;
            --background-color: #f6f7fa;
            --secondary-background-color: #eef1f9;
            --primary-color: #7c9cf5;
        }
        .block-container { max-width: 1250px; padding-top: 2.5rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] {
            background: #eef1f9; border-right: 1px solid #dfe5f1;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stExpander"] summary {
            color: #303440 !important;
        }
        h1, h2, h3 { color: #202430 !important; letter-spacing: -0.025em; }
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
            color: #5e6575 !important;
        }
        [data-testid="stMetric"], [data-testid="stForm"],
        [data-testid="stExpander"] {
            background: #ffffff; border: 1px solid #dfe5f1;
            border-radius: 6px; padding: 16px;
        }
        [data-baseweb="input"], [data-baseweb="input"] input,
        [data-baseweb="select"] > div, [data-baseweb="textarea"],
        [data-baseweb="textarea"] textarea, [data-baseweb="popover"],
        [data-baseweb="menu"], [role="listbox"], [role="option"] {
            background: #ffffff !important; color: #303440 !important;
            -webkit-text-fill-color: #303440;
        }
        [data-baseweb="input"], [data-baseweb="select"] > div {
            border-color: #b7c5e3 !important;
        }
        input::placeholder, textarea::placeholder {
            color: #697184 !important; -webkit-text-fill-color: #697184;
        }
        [data-baseweb="tag"] { background: #eaf0fd !important; color: #3f5fa8 !important; }
        [data-baseweb="tag"] span { color: #3f5fa8 !important; }
        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff; border: 2px dashed #b7c5e3; border-radius: 12px;
            color: #303440;
        }
        [data-testid="stFileUploaderDropzone"] small { color: #5e6575 !important; }
        button[kind="secondary"], button[kind="secondaryFormSubmit"] {
            background: #ffffff !important; color: #3f5fa8 !important;
            border: 1px solid #b7c5e3 !important; border-radius: 8px;
        }
        button[kind="primary"], button[kind="primaryFormSubmit"] {
            background: #7c9cf5 !important; color: #ffffff !important;
            border: 1px solid #7c9cf5 !important; border-radius: 8px;
        }
        button[kind="primary"] p, button[kind="primaryFormSubmit"] p {
            color: #ffffff !important;
        }
        button[kind="secondary"] p, button[kind="secondaryFormSubmit"] p {
            color: #3f5fa8 !important;
        }
        button:disabled { opacity: 0.55 !important; cursor: not-allowed; }
        button:focus-visible, input:focus-visible, summary:focus-visible {
            outline: 3px solid #7c9cf5 !important; outline-offset: 3px;
        }
        [data-baseweb="tab-list"] { gap: 12px; border-bottom: 1px solid #dfe5f1; }
        [data-baseweb="tab"] {
            background: transparent; color: #5e6575 !important;
            padding: 12px 20px; font-weight: 600;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            background: #eaf0fd; color: #3f5fa8 !important; border-radius: 8px 8px 0 0;
        }
        [data-baseweb="tab-highlight"] { background: #7c9cf5; }
        [data-testid="stAlert"] {
            background: #edf2fc; border: 1px solid #d5dff5; color: #3f5fa8;
        }
        [data-testid="stCode"], [data-testid="stCode"] pre,
        [data-testid="stCode"] code {
            background: #f0f3fa !important; color: #303440 !important;
        }
        hr { border-color: #dfe5f1 !important; }
        /* Dashboard palette: periwinkle header, white panels, pale blue metrics. */
        [data-testid="stMain"] h1 {
            background: #7c9cf5; color: #ffffff !important;
            padding: 20px 24px; border-radius: 6px;
        }
        [data-testid="stSidebar"] h1 { color: #3f5fa8 !important; }
        [data-testid="stMetric"] { background: #eaf0fb; }
        [data-testid="stMetricValue"] { color: #496bc5 !important; }
        button[kind="primary"], button[kind="primaryFormSubmit"] {
            background: #eaf0fd !important; color: #3f5fa8 !important;
            border-color: #b7c5e3 !important;
        }
        button[kind="primary"] p, button[kind="primaryFormSubmit"] p {
            color: #3f5fa8 !important;
        }
        button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
            background: #dce6fc !important; border-color: #7c9cf5 !important;
        }
        [data-baseweb="tab"][aria-selected="true"] p { color: #3f5fa8 !important; }
        [data-testid="stSuccess"] {
            background: #ebf7f1; border-color: #8ed9b8; color: #27654c;
        }
        </style>
        """, unsafe_allow_html=True)
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = TemporaryDirectory(prefix="mof-app-")

    with st.sidebar:
        st.title("🧪 MOF Lab")
        st.caption("결정구조에서 흡착 성능까지")
        st.divider()
        uploaded = st.file_uploader("MOF 결정구조 업로드", type=["cif"],
                                    help="결정구조가 저장된 .cif 파일을 선택하세요.")
        if uploaded is not None:
            contents = uploaded.getvalue()
            token = sha256(contents).hexdigest() + uploaded.name
            if token != st.session_state.get("upload_token"):
                for key in ("structure", "structure_image", "results", "adsorption_image",
                            "cif_bytes", "mof_id", "upload_error",
                            "structure_image_attempted", "structure_image_error"):
                    st.session_state.pop(key, None)
                st.session_state["upload_token"] = token
                try:
                    from utils.cif_input import load_cif

                    path = Path(st.session_state["workdir"].name) / "input.cif"
                    path.write_bytes(contents)
                    st.session_state["structure"] = load_cif(path)
                    st.session_state["cif_bytes"] = contents
                    st.session_state["mof_id"] = Path(uploaded.name).stem
                except Exception as error:
                    st.session_state["upload_error"] = error
            if st.session_state.get("upload_error") is not None:
                show_error(st.session_state["upload_error"])
            elif st.session_state.get("structure") is not None:
                st.success(f"{uploaded.name} 로딩 완료")
        else:
            for key in ("upload_token", "structure", "structure_image", "results",
                        "adsorption_image", "cif_bytes", "mof_id", "upload_error",
                        "structure_image_attempted", "structure_image_error"):
                st.session_state.pop(key, None)
        st.divider()
        st.caption("사용 순서")
        st.markdown("1. CIF 파일 업로드\n2. 결정구조 확인\n3. 조건 설정 및 흡착량 계산\n4. DB 비교 및 결과 다운로드")
        st.caption("페이지 시안 · 계산 및 시각화 기능은 팀 모듈 구현 후 사용 가능합니다.")

    st.title("MOF Adsorption Lab")
    st.write("MOF 결정구조를 살펴보고, 기체 흡착 성능을 예측하고 비교하세요.")
    st.caption("CIF 기반 구조 분석 · CGCNN 기반 CO₂ / N₂ 예측")
    structure_tab, prediction_tab, database_tab = st.tabs(
        ["결정구조", "흡착량 예측", "데이터베이스"]
    )
    with structure_tab:
        st.subheader("결정구조 미리보기")
        render_structure()
    with prediction_tab:
        render_prediction()
    with database_tab:
        render_database()


if __name__ == "__main__":
    main()
