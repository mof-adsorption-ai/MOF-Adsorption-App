from pathlib import Path
from pymatgen.core import Structure


def load_cif(cif_path: str | Path) -> Structure:
    """CIF 파일을 읽어 pymatgen.Structure를 반환한다."""
    path = Path(cif_path)
    if not path.is_file():
        raise FileNotFoundError(f"CIF 파일을 찾을 수 없습니다: {path}")
    return Structure.from_file(str(path), primitive=False)
