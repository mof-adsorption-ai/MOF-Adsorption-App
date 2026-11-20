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
        return

    a, b, c = st.columns(3)
    a.metric("화학식", structure.composition.reduced_formula)
    b.metric("원자 수", len(structure))
    c.metric("단위격자 부피", f"{structure.volume:,.2f} Å³")
    with st.expander("격자 정보 · CIF 원문"):
        st.write(dict(zip(("a (Å)", "b (Å)", "c (Å)"), structure.lattice.abc)))
        st.write(dict(zip(("α (°)", "β (°)", "γ (°)"), structure.lattice.angles)))
        st.code(st.session_state["cif_bytes"].decode("utf-8", errors="replace"), language="text")

    regenerate = st.button("구조 이미지 다시 생성", use_container_width=True)
    if regenerate or not st.session_state.get("structure_image_attempted"):
        st.session_state.pop("report_files", None)
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
                st.session_state.pop("report_files", None)
                st.session_state["results"] = results
                st.session_state.pop("adsorption_image", None)
            except Exception as error:
                show_error(error)

    results = st.session_state.get("results")
    if results is None:
        st.caption("계산 완료 후 예측 결과 표, 흡착 등온선과 CSV 다운로드가 표시됩니다.")
        return
    render_result_table(results)
    st.download_button("예측 결과 CSV 다운로드", results.to_csv(index=False).encode("utf-8-sig"),
                       file_name="adsorption_results.csv", mime="text/csv")
    if st.button("흡착량 그래프 생성", use_container_width=True):
        try:
            from utils.plotting import plot_adsorption

            with st.spinner("흡착량 그래프를 생성하고 있습니다…"):
                st.session_state.pop("report_files", None)
                path = Path(st.session_state["workdir"].name) / "adsorption.png"
                st.session_state["adsorption_image"] = plot_adsorption(
                    results, path, db_results=st.session_state.get("db_results")
                )
        except Exception as error:
            show_error(error)
    if st.session_state.get("adsorption_image"):
        st.image(st.session_state["adsorption_image"], use_container_width=True)


def render_result_table(results) -> None:
    """Keep long tables compact while preserving all rows in CSV exports."""
    preview = results.head(50).reset_index(drop=True)
    preview.index = preview.index + 1
    st.table(preview)
    if len(results) > len(preview):
        st.caption(f"전체 {len(results):,}건 중 처음 50건을 표시합니다. 전체 데이터는 CSV로 다운로드하세요.")


def render_database() -> None:
    st.header("MOF 데이터베이스 조회")
    st.write("가스와 압력을 지정해 흡착량이 높은 구조 또는 낮은 구조를 찾아보세요.")
    st.caption("CIF 파일 없이 독립적으로 조회할 수 있습니다.")
    conditions, output = st.columns([2, 3], gap="large")
    with conditions:
        st.markdown("**검색 조건**")
        submitted, gas, pressure, extremum, row_count = render_database_form()
    with output:
        st.markdown("**검색 결과**")
        render_database_results(submitted, gas, pressure, extremum, row_count)


def render_database_form() -> tuple[bool, str, float, str, int]:
    with st.form("database_form"):
        left, right = st.columns(2)
        gas = left.selectbox("가스 종류", ["CO2", "N2", "N2 대비 CO2"])
        pressure = right.number_input("조회 압력", min_value=0.0, value=0.1, step=0.01, format="%.4f")
        left, right = st.columns(2)
        extremum = left.selectbox("흡착량 정렬", ["최대", "최소"])
        row_count = right.number_input("결과 행 개수", min_value=1, value=10, step=1)
        st.caption("압력은 DB와 같은 단위로 입력하세요. 저장된 압력만 조회합니다. N2 대비 CO2 = CO2 흡착량 ÷ N2 흡착량.")
        st.caption("기본 DB 저장 압력: CO2 0.01, 0.05, 0.1, 0.5, 2.5 / N2 0.09, 0.9 (298 K). 공통 압력이 없어 기본 DB의 CO2/N2 비율은 직접 조회할 수 없습니다.")
        submitted = st.form_submit_button("MOF 검색", type="primary", use_container_width=True)
    return submitted, gas, pressure, extremum, int(row_count)


def render_database_results(submitted: bool, gas: str, pressure: float, extremum: str, row_count: int) -> None:
    if submitted:
        st.session_state.pop("report_files", None)
        st.session_state.pop("db_results", None)
        st.session_state.pop("db_query", None)
        st.session_state.pop("adsorption_image", None)
        try:
            if not DATABASE.is_file():
                raise FileNotFoundError("MOF 데이터베이스 파일을 찾을 수 없습니다.")
            with DATABASE.open("rb") as source:
                is_pointer = source.read(80).startswith(b"version https://git-lfs.github.com/spec/")
            if is_pointer:
                st.warning("현재 DB 파일은 Git LFS 포인터입니다. 실제 Parquet 데이터를 받은 뒤 검색할 수 있습니다.")
            else:
                from utils.db_search import search_mofs

                with st.spinner("데이터베이스를 검색하고 있습니다…"):
                    st.session_state["db_results"] = search_mofs(
                        gas, pressure, extremum, int(row_count), db_path=DATABASE
                    )
                    st.session_state["db_query"] = f"{gas} · 압력 {pressure:g} · {extremum} 순 · 요청 {int(row_count)}건"
                st.session_state.pop("adsorption_image", None)
        except Exception as error:
            show_error(error)
    results = st.session_state.get("db_results")
    if results is not None:
        st.caption(st.session_state.get("db_query", ""))
        if results.empty:
            st.info("해당 압력에 유효한 흡착량이 없습니다. 다른 압력으로 검색해 주세요.")
            return
        st.caption(f"검색 결과 {len(results):,}건")
        render_result_table(results)
        st.download_button("검색 결과 CSV 다운로드", results.to_csv(index=False).encode("utf-8-sig"),
                           file_name="mof_database_results.csv", mime="text/csv")
    else:
        st.caption("검색 조건을 입력하고 ‘MOF 검색’을 누르면 결과가 여기에 표시됩니다.")


def render_upload() -> None:
    uploaded = st.file_uploader("MOF 결정구조 업로드", type=["cif"],
                                help="결정구조가 저장된 .cif 파일을 선택하세요.")
    if uploaded is not None:
        contents = uploaded.getvalue()
        token = sha256(contents).hexdigest() + uploaded.name
        if token != st.session_state.get("upload_token"):
            for key in ("structure", "structure_image", "results", "adsorption_image",
                        "cif_bytes", "mof_id", "upload_error",
                        "structure_image_attempted", "structure_image_error", "report_files"):
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
                    "structure_image_attempted", "structure_image_error", "report_files"):
            st.session_state.pop(key, None)


def render_report() -> None:
    st.write("구조 정보, 흡착량 예측 결과와 그래프를 PDF 보고서로 정리합니다.")
    results = st.session_state.get("results")
    ready = results is not None and not results.empty
    if not ready:
        st.info("2단계에서 흡착량을 계산하면 보고서를 작성할 수 있습니다.")
    include_database = st.checkbox(
        "데이터베이스 조회 결과 포함", value=False,
        disabled=st.session_state.get("db_results") is None,
    )
    st.caption("PDF 본문은 영문으로 작성됩니다. 예측 데이터는 CSV로도 다운로드할 수 있습니다.")
    if st.button("보고서 작성하기", type="primary", disabled=not ready, use_container_width=True):
        st.session_state.pop("report_files", None)
        try:
            from utils.report import create_report

            directory = Path(st.session_state["workdir"].name)
            images = []
            if st.session_state.get("structure_image"):
                image_path = directory / "structure.png"
                image_path.write_bytes(st.session_state["structure_image"])
                images.append(image_path)
            if st.session_state.get("adsorption_image"):
                images.append(st.session_state["adsorption_image"])
            structure = st.session_state["structure"]
            info = {
                "Formula": structure.composition.reduced_formula,
                "Atoms": len(structure),
                "Cell volume (A^3)": round(structure.volume, 3),
                "Lattice lengths (A)": tuple(round(v, 3) for v in structure.lattice.abc),
            }
            with st.spinner("보고서를 작성하고 있습니다…"):
                files = create_report(
                    results, images, info, directory / "prediction_report.pdf",
                    db_results=st.session_state.get("db_results") if include_database else None,
                )
                st.session_state["report_files"] = {
                    "pdf": Path(files["pdf_path"]).read_bytes(),
                    "csv": Path(files["csv_path"]).read_bytes(),
                    "includes_database": include_database,
                }
        except Exception as error:
            show_error(error)
    files = st.session_state.get("report_files")
    if files and files["includes_database"] == include_database:
        st.success("보고서 작성이 완료되었습니다.")
        left, right = st.columns(2)
        left.download_button("PDF 보고서 다운로드", files["pdf"], file_name="mof_report.pdf", mime="application/pdf", use_container_width=True)
        right.download_button("예측 데이터 CSV 다운로드", files["csv"], file_name="mof_predictions.csv", mime="text/csv", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="MOF Adsorption Lab", page_icon="🧪", layout="wide")
    st.markdown("""
        <style>
        :root {
            --font-title: 28px;
            --font-body: 16px;
            --font-data: 14px;
        }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: #f6f7fb; color: #303440; color-scheme: light;
            --text-color: #303440;
            --background-color: #f6f7fb;
            --secondary-background-color: #edf2ff;
            --primary-color: #7c9cf5;
        }
        .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
        /* Three text sizes across headings, controls and data. */
        [data-testid="stAppViewContainer"] * {
            font-size: var(--font-body) !important;
        }
        [data-testid="stMain"] :is(h1, h2, h3, h4, h5, h6),
        [data-testid="stMain"] :is(h1, h2, h3, h4, h5, h6) * {
            font-size: var(--font-title) !important;
            line-height: 1.35; letter-spacing: -0.025em;
        }
        [data-testid="stMain"] :is(li, small, code, pre, th, td),
        [data-testid="stMain"] :is(li, small, code, pre, th, td) *,
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
        [data-testid="stMetric"], [data-testid="stMetric"] *,
        [data-testid="stDataFrame"], [data-testid="stDataFrame"] *,
        [data-testid="stTable"], [data-testid="stTable"] * {
            font-size: var(--font-data) !important;
        }
        [data-testid="stMain"] h1 {
            background: linear-gradient(110deg, #7698ef, #91adfa);
            color: #172b50; padding: 26px 30px; border-radius: 16px;
            box-shadow: 0 8px 28px #7c9cf526;
        }
        [data-testid="stMain"] h2 { color: #34548f; margin-top: 12px; }
        [data-testid="stCaptionContainer"] { color: #626d80; }
        [data-testid="stVerticalBlockBorderWrapper"] > div,
        [data-testid="stForm"], [data-testid="stExpander"] {
            background: #ffffff; border-color: #e0e7f3;
            border-radius: 14px;
        }
        [data-testid="stForm"] { padding: 22px; }
        [data-testid="stMetric"] {
            background: #edf2ff; border-top: 3px solid #8ed9b8;
            border-radius: 10px; padding: 16px;
        }
        [data-testid="stMetricValue"] { color: #496bc5; }
        /* Explicit colors also cover browsers configured to use a dark theme. */
        [data-testid="stMain"] :is(p, span, label, input, textarea, li, summary),
        [data-testid="stWidgetLabel"] * {
            color: #303440 !important;
        }
        [data-baseweb="input"], [data-baseweb="input"] input,
        [data-baseweb="base-input"], [data-baseweb="base-input"] input,
        [data-baseweb="textarea"],
        [data-baseweb="select"] > div, [data-baseweb="textarea"] textarea,
        [data-baseweb="popover"], [data-baseweb="menu"],
        [role="listbox"], [role="option"],
        [data-testid="stNumberInput"] button {
            background: white !important; color: #303440 !important;
            -webkit-text-fill-color: #303440;
            font-size: var(--font-body) !important;
        }
        [data-baseweb="select"] :is(div, input, span),
        [data-baseweb="popover"] *, [data-baseweb="menu"] *,
        [role="listbox"] *, [role="option"] * {
            color: #303440 !important;
            -webkit-text-fill-color: #303440 !important;
        }
        [data-baseweb="select"] input {
            background: white !important;
        }
        input::placeholder, textarea::placeholder {
            color: #626d80 !important; -webkit-text-fill-color: #626d80 !important;
            opacity: 1;
        }
        [role="option"]:hover, [role="option"][aria-selected="true"] {
            background: #edf2ff !important;
        }
        [role="tooltip"], [role="tooltip"] * {
            background: white !important; color: #303440 !important;
            font-size: var(--font-data) !important;
        }
        [data-baseweb="input"], [data-baseweb="select"] > div { border-color: #b7c5e3; }
        [data-baseweb="tag"] { background: #eaf0fd; color: #3f5fa8; }
        [data-testid="stFileUploaderDropzone"] {
            background: #f5f8ff; border: 2px dashed #b7c5e3; border-radius: 12px;
        }
        button[kind="primary"], button[kind="primaryFormSubmit"] {
            background: #7c9cf5; border: 1px solid #7c9cf5; color: #172b50;
            min-height: 44px; border-radius: 10px;
        }
        button[kind="primary"] p, button[kind="primaryFormSubmit"] p { color: #172b50; }
        button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
            background: #90aaf5; border-color: #5678cd; color: #172b50;
        }
        button[kind="secondary"], button[kind="secondaryFormSubmit"] {
            background: white; border: 1px solid #b7c5e3; color: #3f5fa8;
            min-height: 44px; border-radius: 10px;
        }
        button:disabled { opacity: 0.55; }
        button:focus-visible, input:focus-visible, summary:focus-visible {
            outline: 3px solid #7c9cf5; outline-offset: 3px;
        }
        [data-baseweb="tab-list"] {
            margin: 12px 0 24px; gap: 8px; border: 1px solid #d6e0f0;
            background: #eaf0fc; border-radius: 12px; padding: 6px;
        }
        [data-baseweb="tab"] {
            flex: 1; min-height: 48px; padding: 14px 24px;
            color: #405575; font-weight: 600; border-radius: 8px;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            background: white; color: #29446e; border-radius: 8px;
            box-shadow: inset 0 -3px #8ed9b8, 0 2px 6px #30344010;
        }
        [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { background: transparent; }
        [data-testid="stTable"] { overflow-x: auto; }
        [data-testid="stTable"] th { background: #edf2ff; color: #29446e; }
        [data-testid="stTable"] td { color: #303440; }
        [data-testid="stAlert"] { background: #edf2fc; border: 1px solid #d5dff5; }
        [data-testid="stCode"] pre { background: #f0f3fa; color: #303440; }
        @media (max-width: 700px) {
            .block-container { padding: 1rem; }
            [data-testid="stMain"] h1 { padding: 20px; }
            [data-baseweb="tab"] { padding: 12px; }
        }
        </style>
        """, unsafe_allow_html=True)
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = TemporaryDirectory(prefix="mof-app-")

    st.title("MOF Adsorption Lab")
    st.caption("구조 입력부터 흡착량 예측과 보고서 작성까지 · MOF 데이터베이스 비교")
    prediction_tab, database_tab = st.tabs(["흡착량 예측", "데이터베이스 조회"])
    with prediction_tab:
        with st.container(border=True):
            st.header("1. 구조 입력 (.cif)")
            st.write("MOF 결정구조 파일을 업로드하고 구조 정보를 확인하세요.")
            render_upload()
            render_structure()
        with st.container(border=True):
            st.header("2. 흡착량 계산")
            render_prediction()
        with st.container(border=True):
            st.header("3. 보고서 작성하기")
            render_report()
    with database_tab:
        with st.container(border=True):
            render_database()


if __name__ == "__main__":
    main()
