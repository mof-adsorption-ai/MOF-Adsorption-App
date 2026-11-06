from pathlib import Path
from pymatgen.core import Structure


def visualize_structure(structure: Structure, output_path: str | Path) -> str:
    """Returns: 구조 이미지 저장 경로 (str)."""
    raise NotImplementedError
