"""Streamlit UI integrating the team's MOF modules.

Run with: python -m streamlit run main.py
"""

from __future__ import annotations

import json
import logging
import math
from importlib import import_module, util
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def resolve_path(value: str) -> Path:
    path = Path(value.strip()).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_pressures(text: str) -> list[float]:
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError("압력을 하나 이상 입력하세요.")
    values = [float(token) for token in tokens]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("압력은 0 이상의 유한한 숫자로 입력하세요.")
    return sorted(set(values))


def call_module(module: str, function: str, *args, **kwargs):
    return getattr(import_module(f"utils.{module}"), function)(*args, **kwargs)


def build_outputs(result: dict, db_results: pd.DataFrame | None = None) -> dict:
    """Keep output bytes after temporary files have been removed."""
    output = {
        "csv": result["table"].to_csv(index=False).encode("utf-8-sig"),
        "plot": None,
        "pdf": None,
        "warnings": list(result["input_warnings"]),
        "comparison": db_results is not None and not db_results.empty,
    }
    with TemporaryDirectory(prefix="mof-report-") as directory:
        work = Path(directory)
        images = []
        if result["structure_image"] is not None:
            image_path = work / "structure.png"
            image_path.write_bytes(result["structure_image"])
            images.append(str(image_path))
        try:
            plot_path = Path(call_module(
                "plotting", "plot_adsorption", result["table"],
                str(work / "adsorption.png"), db_results=db_results,
            ))
            output["plot"] = plot_path.read_bytes()
            images.append(str(plot_path))
        except Exception as exc:
            LOGGER.exception("Plot generation failed")
            output["warnings"].append(f"그래프 생성 실패 — {exc}")
        try:
            files = call_module(
                "report", "create_report", result["table"], images,
                result["graph_info"], str(work / "report.pdf"), db_results=db_results,
            )
            output["pdf"] = Path(files["pdf_path"]).read_bytes()
            output["csv"] = Path(files["csv_path"]).read_bytes()
        except Exception as exc:
            LOGGER.exception("Report generation failed")
            output["warnings"].append(f"PDF 보고서 생성 실패 — {exc}")
    return output


def run_prediction(cif_bytes: bytes, name: str, pressures: list[float],
                   gases: list[str], checkpoint: Path) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"모델 체크포인트를 찾을 수 없습니다. {checkpoint}")
    mof_id = Path(name.replace("\\", "/")).stem
    with TemporaryDirectory(prefix="mof-predict-") as directory:
        work = Path(directory)
        cif_path = work / "input.cif"
        cif_path.write_bytes(cif_bytes)
        structure = call_module("cif_input", "load_cif", str(cif_path))
        graph = call_module("preprocessing", "prepare_model_input", structure, str(checkpoint))
        table = call_module(
            "model", "predict_adsorption", graph, mof_id, pressures, gases, str(checkpoint)
        )
        required = ["mof_id", "gas", "pressure", "adsorption"]
        if not isinstance(table, pd.DataFrame) or not set(required).issubset(table.columns):
            raise ValueError("모델 결과에 mof_id, gas, pressure, adsorption 컬럼이 필요합니다.")
        table = table[required].copy()
        if len(table) != len(pressures) * len(gases):
            raise ValueError("예측 결과 개수가 요청한 기체·압력 조합과 다릅니다.")
        if not all(math.isfinite(float(value)) for value in table["adsorption"]):
            raise ValueError("예측 결과에 유효하지 않은 흡착량이 있습니다.")
        result = {
            "name": name,
            "mof_id": mof_id,
            "table": table,
            "structure_image": None,
            "structure_data": structure.as_dict(),
            "input_warnings": [],
            "graph_info": {
                "formula": graph["formula"],
                "n_atoms": graph["n_atoms"],
                "lattice_lengths_A": tuple(round(v, 4) for v in structure.lattice.abc),
                "lattice_angles_deg": tuple(round(v, 4) for v in structure.lattice.angles),
                "cell_volume_A3": round(structure.volume, 4),
                "atom_features": str(tuple(graph["atom"].shape)),
                "neighbor_features": str(tuple(graph["nbr"].shape)),
                "checkpoint": checkpoint.name,
            },
        }
        try:
            image_path = call_module(
                "cif_vis", "visualize_structure", structure, str(work / "structure.png")
            )
            result["structure_image"] = Path(image_path).read_bytes()
        except Exception as exc:
            LOGGER.exception("Structure visualization failed")
            result["input_warnings"].append(f"결정구조 이미지 생성 실패 — {exc}")
    result["outputs"] = build_outputs(result)
    return result


def show_prediction(result: dict):
    st.subheader(f"최근 예측 결과 · {result['name']}")
    left, right = st.columns([1, 2])
    with left:
        st.metric("원자 수", result["graph_info"]["n_atoms"])
        st.write(result["graph_info"]["formula"])
        if "structure_data" in result:
            styles = {
                "Ball & Stick": "ball_and_stick",
                "Space-filling Sphere": "space_filling",
                "Atoms only": "atoms_only",
            }
            selected_style = st.selectbox("결정구조 표시 스타일", list(styles))
            try:
                from pymatgen.core import Structure

                structure = Structure.from_dict(result["structure_data"])
                figure = call_module(
                    "cif_vis", "create_interactive_structure",
                    structure, style=styles[selected_style],
                )
                st.plotly_chart(
                    figure, width="stretch",
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "modeBarButtonsToRemove": ["select3d", "lasso3d"],
                    },
                )
                st.caption("드래그로 회전하고 마우스 휠로 확대·축소할 수 있습니다.")
            except Exception as exc:
                LOGGER.exception("Interactive structure rendering failed")
                st.warning(f"결정구조 3D 표시 실패 — {exc}")
        if result["structure_image"] is not None:
            with st.expander("정적 PNG 미리보기"):
                st.image(result["structure_image"], caption="단위격자와 원자 배치")
    with right:
        st.dataframe(result["table"], hide_index=True, width="stretch")
        if result["outputs"]["plot"] is not None:
            st.image(result["outputs"]["plot"], caption="압력별 흡착량")
    with st.expander("구조 및 그래프 정보"):
        st.json(result["graph_info"])

    database = st.session_state.get("db_result")
    compare = st.checkbox(
        "최근 DB 검색 결과를 그래프·보고서에 포함",
        value=False,
        disabled=database is None or database.empty,
        key="compare_database",
    )
    if compare:
        st.caption("DB 검색에서 온도 조건을 하나로 맞추고 예측과 동일한 단위를 사용하세요.")
    if st.button("PDF 보고서 생성", help="현재 예측 결과와 선택한 DB 비교 결과로 그래프와 PDF를 생성합니다."):
        with st.spinner("결과 파일 생성 중"):
            result["outputs"] = build_outputs(result, database if compare else None)
        st.rerun()
    output = result["outputs"]
    if output["comparison"]:
        st.caption("현재 결과 파일에는 생성 당시의 DB 검색 결과가 포함되어 있습니다.")
    for message in output["warnings"]:
        st.warning(message)
    name = result["mof_id"]
    csv_col, pdf_col, plot_col, structure_col = st.columns(4)
    csv_col.download_button(
        "CSV 다운로드", output["csv"], f"{name}_adsorption.csv", "text/csv"
    )
    if output["pdf"] is not None:
        st.success("PDF 보고서가 준비되었습니다. 아래 PDF 다운로드 버튼으로 저장하세요.")
        pdf_col.download_button(
            "PDF 다운로드", output["pdf"], f"{name}_report.pdf", "application/pdf"
        )
    if output["plot"] is not None:
        plot_col.download_button(
            "그래프 PNG", output["plot"], f"{name}_adsorption.png", "image/png"
        )
    if result["structure_image"] is not None:
        structure_col.download_button(
            "결정구조 PNG", result["structure_image"], f"{name}_structure.png", "image/png"
        )


def prediction_page(checkpoint: Path):
    with st.form("prediction_form"):
        upload = st.file_uploader("MOF 결정구조 파일", type=["cif"])
        samples = sorted((ROOT / "data" / "cif").glob("*.cif"))
        sample_name = st.selectbox(
            "또는 샘플 CIF 선택", ["선택 안 함"] + [path.name for path in samples]
        )
        pressure_text = st.text_input(
            "압력 목록", value="0.01, 0.1, 0.5, 1, 5, 10",
            help="쉼표 또는 공백으로 구분하세요. 모델 학습 데이터와 동일한 압력 단위를 사용합니다.",
        )
        gases = st.multiselect("기체", ["CO2", "N2"], default=["CO2", "N2"])
        submitted = st.form_submit_button("흡착량 예측", type="primary")
    st.caption("흡착량은 모델이 역정규화한 값입니다. 압력과 흡착량 단위는 학습 데이터 기준입니다.")

    if submitted:
        st.session_state.pop("prediction_result", None)
        try:
            pressures = parse_pressures(pressure_text)
            if not gases:
                raise ValueError("기체를 하나 이상 선택하세요.")
            if upload is not None:
                content, name = upload.getvalue(), upload.name
            elif sample_name != "선택 안 함":
                content, name = (ROOT / "data" / "cif" / sample_name).read_bytes(), sample_name
            else:
                raise ValueError("CIF 파일을 업로드하거나 샘플을 선택하세요.")
            if not content:
                raise ValueError("빈 CIF 파일입니다.")
            with st.spinner("구조 분석 및 흡착량 예측 중"):
                st.session_state["prediction_result"] = run_prediction(
                    content, name, pressures, gases, checkpoint
                )
        except Exception as exc:
            LOGGER.exception("Prediction failed")
            st.error(f"예측을 완료하지 못했습니다. {exc}")
    result = st.session_state.get("prediction_result")
    if result is not None:
        show_prediction(result)


def database_page(db_path: Path):
    if not db_path.is_file():
        st.info("DB 파일이 없습니다. 사이드바에서 실제 데이터 파일 경로를 지정하세요.")
        return
    with db_path.open("rb") as file:
        is_pointer = file.read(64).startswith(b"version https://git-lfs.github.com/spec")
    if is_pointer:
        st.warning("현재 DB는 Git LFS 포인터입니다. 실제 데이터를 받은 뒤 검색할 수 있습니다.")
        st.code("git lfs install\ngit lfs pull", language="shell")
        return

    with st.form("database_form"):
        filters_text = st.text_area(
            "검색 조건 (JSON)", value="{}",
            help='예시: {"mof_id": "hMOF-0"} 또는 {"pressure": {"gte": 0.1, "lte": 10}}',
        )
        sort_by = st.text_input("정렬 컬럼 (선택)")
        ascending = st.checkbox("오름차순", value=True)
        selectivity = st.checkbox("CO₂/N₂ 흡착량 비율 계산")
        submitted = st.form_submit_button("DB 검색")
    if submitted:
        st.session_state.pop("db_result", None)
        try:
            filters = json.loads(filters_text)
            if not isinstance(filters, dict):
                raise ValueError("검색 조건은 JSON 객체여야 합니다.")
            with st.spinner("DB 검색 중"):
                results = call_module(
                    "db_search", "search_database", str(db_path), filters,
                    sort_by=sort_by.strip() or None, ascending=ascending,
                    calculate_selectivity=selectivity,
                )
            st.session_state["db_result"] = results
            st.session_state["db_result_path"] = str(db_path)
        except Exception as exc:
            LOGGER.exception("Database search failed")
            st.error(f"DB 검색 실패 — {exc}")
    results = st.session_state.get("db_result")
    if results is not None and st.session_state.get("db_result_path") == str(db_path):
        st.write(f"최근 검색 결과 {len(results):,}건")
        st.dataframe(results, hide_index=True, width="stretch")
        st.download_button(
            "검색 결과 CSV", results.to_csv(index=False).encode("utf-8-sig"),
            "database_results.csv", "text/csv",
        )
        st.caption("예측 탭에서 이 검색 결과를 비교 그래프와 보고서에 포함할 수 있습니다.")


def main():
    st.set_page_config(page_title="MOF 흡착량 예측", page_icon="🧪", layout="wide")
    st.title("MOF 흡착량 예측")
    with st.sidebar:
        st.header("데이터 설정")
        checkpoints = sorted((ROOT / "models").glob("*.pt"))
        default_checkpoint = str(checkpoints[0].relative_to(ROOT)) if checkpoints else "models/cgcnn_scratch_model.pt"
        checkpoint_text = st.text_input("모델 체크포인트", value=default_checkpoint)
        db_text = st.text_input("MOF DB 경로", value="data/mof_cif.parquet")
        st.caption("상대 경로는 프로젝트 폴더 기준입니다.")
        missing = [
            name for name in ("torch", "pymatgen", "matplotlib", "pyarrow")
            if util.find_spec(name) is None
        ]
        if missing:
            st.warning("필요 패키지 설치 필요 · " + ", ".join(missing))
            st.code("python -m pip install -r requirements.txt", language="shell")
    if st.session_state.get("active_db_path") != str(resolve_path(db_text)):
        st.session_state.pop("db_result", None)
        st.session_state.pop("db_result_path", None)
        st.session_state["active_db_path"] = str(resolve_path(db_text))
    prediction_tab, database_tab = st.tabs(["흡착량 예측", "MOF DB 검색"])
    with database_tab:
        try:
            database_page(resolve_path(db_text))
        except OSError as exc:
            st.error(f"DB 경로를 확인하세요. {exc}")
    with prediction_tab:
        prediction_page(resolve_path(checkpoint_text))


if __name__ == "__main__":
    main()
