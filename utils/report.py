from __future__ import annotations
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.image import imread

# pyplot 대신 Figure를 직접 만들어 쓰므로, 웹 UI(streamlit) 환경에서도
# 전역 backend 설정에 영향을 주지 않고 PDF를 저장할 수 있다.
# PDF 본문은 matplotlib 기본 폰트가 한글을 지원하지 않아 영문으로 작성한다.

A4 = (8.27, 11.69)
ROWS_PER_PAGE = 35


class ReportFiles(TypedDict):
    pdf_path: str
    csv_path: str


def _resolve_paths(output_path: str | Path, mof_id: str) -> tuple[Path, Path]:
    """output_path가 파일 경로면 그 이름을, 폴더 경로면 '<mof_id>_report'를 기본 이름으로 사용."""
    p = Path(output_path)
    if p.suffix:
        out_dir, stem = p.parent, p.stem
    else:
        out_dir, stem = p, f"{mof_id}_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.pdf", out_dir / f"{stem}.csv"


def _describe_info(graph_info: dict[str, Any]) -> list[tuple[str, str]]:
    """graph_info를 (항목, 값) 목록으로 변환. 텐서/배열은 shape만 표시."""
    rows: list[tuple[str, str]] = []
    for key, value in (graph_info or {}).items():
        if hasattr(value, "shape"):
            rows.append((str(key), f"array, shape={tuple(value.shape)}"))
        else:
            rows.append((str(key), str(value)))
    return rows


def _pivot_results(results: pd.DataFrame) -> pd.DataFrame:
    """pressure x gas 형태의 표. pivot이 불가능하면 원본을 반환."""
    needed = {"gas", "pressure", "adsorption"}
    if results.empty or not needed.issubset(results.columns):
        return results
    try:
        table = results.pivot_table(index="pressure", columns="gas", values="adsorption")
    except Exception:
        return results
    table.columns = [f"{c} adsorption" for c in table.columns]
    return table.reset_index()


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    return out.astype(str)


def _cover_page(pdf: PdfPages, mof_id: str, info_rows: list[tuple[str, str]],
                results: pd.DataFrame) -> None:
    fig = Figure(figsize=A4)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.0, 0.97, "MOF Adsorption Prediction Report", fontsize=20,
            fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.0, 0.93, f"MOF ID: {mof_id}", fontsize=13, va="top", transform=ax.transAxes)
    ax.text(0.0, 0.905, f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            fontsize=10, va="top", color="gray", transform=ax.transAxes)

    summary = list(info_rows)
    if not results.empty:
        if "gas" in results.columns:
            summary.append(("gases", ", ".join(map(str, results["gas"].unique()))))
        if "pressure" in results.columns:
            summary.append(("pressure range",
                            f"{results['pressure'].min():g} - {results['pressure'].max():g}"))
        summary.append(("predictions", str(len(results))))
    if summary:
        height = min(0.04 * (len(summary) + 1), 0.7)
        table = ax.table(cellText=[[k, v] for k, v in summary],
                         colLabels=["Item", "Value"], colWidths=[0.3, 0.7],
                         cellLoc="left", bbox=[0.0, 0.86 - height, 1.0, height])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
    pdf.savefig(fig)


def _table_pages(pdf: PdfPages, title: str, df: pd.DataFrame) -> None:
    if df.empty:
        fig = Figure(figsize=A4)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.0, 0.97, title, fontsize=14, fontweight="bold",
                va="top", transform=ax.transAxes)
        ax.text(0.0, 0.92, "No data.", fontsize=11, va="top", transform=ax.transAxes)
        pdf.savefig(fig)
        return
    text = _fmt(df)
    n_pages = (len(text) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    for page in range(n_pages):
        chunk = text.iloc[page * ROWS_PER_PAGE:(page + 1) * ROWS_PER_PAGE]
        fig = Figure(figsize=A4)
        ax = fig.add_subplot(111)
        ax.axis("off")
        suffix = f" ({page + 1}/{n_pages})" if n_pages > 1 else ""
        ax.text(0.0, 0.97, title + suffix, fontsize=14, fontweight="bold",
                va="top", transform=ax.transAxes)
        height = 0.025 * (len(chunk) + 1)
        table = ax.table(cellText=chunk.values, colLabels=list(chunk.columns),
                         cellLoc="center", bbox=[0.0, 0.94 - height, 1.0, height])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        pdf.savefig(fig)


def _image_page(pdf: PdfPages, image_path: Path) -> None:
    try:
        img = imread(str(image_path))
    except Exception:
        return  # 읽을 수 없는 이미지는 건너뛴다
    fig = Figure(figsize=A4)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.imshow(img)
    ax.set_title(image_path.name, fontsize=10)
    pdf.savefig(fig)


def create_report(
    results: pd.DataFrame,
    images: Sequence[str | Path],
    graph_info: dict[str, Any],
    output_path: str | Path,
    db_results: pd.DataFrame | None = None,
) -> ReportFiles:
    """Returns: {"pdf_path": str, "csv_path": str}."""
    mof_id = "MOF"
    if "mof_id" in results.columns and not results.empty:
        mof_id = str(results["mof_id"].iloc[0])
    pdf_path, csv_path = _resolve_paths(output_path, mof_id)

    # CSV: 예측 결과 원본 (엑셀에서 한글이 깨지지 않도록 utf-8-sig)
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # PDF: 표지 -> 예측 결과 표 -> 이미지들 -> DB 비교 표
    with PdfPages(pdf_path) as pdf:
        _cover_page(pdf, mof_id, _describe_info(graph_info), results)
        _table_pages(pdf, "Predicted Adsorption", _pivot_results(results))
        for image in images:
            image = Path(image)
            if image.is_file():
                _image_page(pdf, image)
        if db_results is not None and not db_results.empty:
            _table_pages(pdf, "MOF Database Comparison", db_results)

    return {"pdf_path": str(pdf_path), "csv_path": str(csv_path)}
