"""CIF 파일 로딩과 MOF 결정구조의 기본 유효성 검사.

``pymatgen.core.Structure`` 형식으로 반환합니다.
외부 사용 예::

    from util.cif_input import load_cif
    structure = load_cif("data/cif/hMOF-0.cif")
"""

from pathlib import Path
import math

from pymatgen.core import Structure

__all__ = ["load_cif"]


def load_cif(cif_path: str | Path) -> Structure:
    """CIF 파일을 읽고 검증한 뒤 pymatgen Structure를 반환합니다.

    Args:
        cif_path: 읽을 .cif 파일의 경로 (str 또는 pathlib.Path).

    Returns:
        원자 종류, 분율좌표, 격자 정보가 포함된 ``pymatgen.Structure``.

    Raises:
        TypeError: 경로 인자가 문자열 또는 Path가 아닌 경우.
        FileNotFoundError: 파일이 존재하지 않는 경우.
        ValueError: CIF 확장자, 결정구조 또는 기본 구조 정보가 유효하지 않은 경우.
        OSError: 파일을 읽을 수 없는 경우.
    """
    if not isinstance(cif_path, (str, Path)):
        raise TypeError("cif_path는 파일 경로 문자열 또는 pathlib.Path여야 합니다.")

    path = Path(cif_path).expanduser()
    if path.suffix.lower() != ".cif":
        raise ValueError(f"CIF 파일(.cif)만 지원합니다: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"CIF 파일을 찾을 수 없습니다: {path}")

    try:
        structure = Structure.from_file(str(path), primitive=False)
    except (OSError, UnicodeError):
        raise
    except Exception as exc:
        raise ValueError(f"CIF 결정구조를 해석하지 못했습니다: {path}") from exc

    if not isinstance(structure, Structure) or len(structure) == 0:
        raise ValueError(f"CIF에 원자 정보가 없습니다: {path}")

    lattice = structure.lattice
    lengths = (lattice.a, lattice.b, lattice.c)
    angles = (lattice.alpha, lattice.beta, lattice.gamma)
    if (not all(math.isfinite(v) and v > 0 for v in lengths)
            or not all(math.isfinite(v) and 0 < v < 180 for v in angles)
            or not math.isfinite(lattice.volume) or lattice.volume <= 0):
        raise ValueError(f"CIF의 격자 길이/각도/부피가 유효하지 않습니다: {path}")

    for index, site in enumerate(structure):
        if not all(math.isfinite(float(v)) for v in site.frac_coords):
            raise ValueError(f"원자 {index}의 분율좌표가 유효하지 않습니다: {path}")
        # 부분 점유/무질서 사이트는 단일 원자번호로 표현할 수 없습니다.
        if not site.is_ordered:
            raise ValueError(
                f"원자 {index}에 부분 점유 또는 복수 원소가 있습니다. "
                f"CGCNN 입력 전에 정리가 필요합니다: {path}"
            )
        try:
            atomic_number = site.specie.number
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"원자 {index}의 원소를 확인할 수 없습니다: {path}") from exc
        if not 1 <= atomic_number <= 118:
            raise ValueError(f"원자 {index}의 원자번호가 유효하지 않습니다: {path}")

    return structure
