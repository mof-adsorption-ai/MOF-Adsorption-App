"""MOF 결정구조를 PNG 이미지로 저장하는 모듈.

팀 공통 인터페이스:
    visualize_structure(structure, output_path) -> str

입력 구조는 1번 담당자의 load_cif()가 반환한 pymatgen Structure 객체입니다.
이 파일은 웹 화면을 만들지 않으며, 다른 모듈에서 불러올 때 자동 실행되지 않습니다.
"""

from pathlib import Path

import matplotlib

# 웹 서버 등 화면이 없는 환경에서도 이미지 파일을 생성하도록 설정합니다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure


# 원소별 표시 색상입니다. 없는 원소는 회색으로 표시합니다.
_ELEMENT_COLORS = {
    "H": "#E9E9E9", "C": "#555B66", "N": "#3F77D5",
    "O": "#E34F54", "F": "#70C77C", "Zn": "#8C79C8",
    "Cu": "#C47A45", "Zr": "#4DB9B5", "Al": "#B4B9C0",
    "S": "#E7C04E", "Cl": "#58B96A",
}

# 원소별 구체 크기입니다. 실제 원자 반지름의 정확한 비율은 아닙니다.
_ELEMENT_SIZES = {
    "H": 24, "C": 65, "N": 75, "O": 80, "F": 75,
    "Zn": 145, "Cu": 135, "Zr": 155, "Al": 115,
}


def visualize_structure(structure, output_path):
    """MOF 결정구조를 3차원 그림으로 그려 PNG 파일로 저장합니다.

    매개변수
    ----------
    structure : pymatgen.core.Structure
        CIF 파일을 읽어 만든 결정구조 데이터입니다.
    output_path : str 또는 pathlib.Path
        생성할 PNG 이미지 경로입니다. 예: 'outputs/mof.png'

    반환값
    ------
    str
        실제로 저장된 이미지 파일 경로입니다.
    """

    # 1단계: 입력 데이터가 올바른지 검사합니다.
    if not isinstance(structure, Structure):
        raise TypeError("structure는 pymatgen.Structure 객체여야 합니다.")
    if len(structure) == 0:
        raise ValueError("표시할 원자가 없는 결정구조입니다.")

    # 2단계: 저장 위치를 준비합니다.
    # 지정한 폴더가 없으면 자동으로 만듭니다.
    image_path = Path(output_path)
    if image_path.suffix.lower() != ".png":
        raise ValueError("output_path는 .png 파일 경로여야 합니다.")
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # 3단계: CIF에 저장된 분율좌표를 하나의 Unit Cell 범위로 옮깁니다.
    # 예: -0.2 -> 0.8, 1.3 -> 0.3
    # 이는 주기적 결정의 동등한 위치로 옮기는 작업입니다.
    # 주의: 경계에 걸친 분자의 결합선은 이 그림에서 표현하지 않습니다.
    fractional_positions = np.mod(np.asarray(structure.frac_coords), 1.0)

    # 4단계: 분율좌표를 실제 3차원 좌표(단위: 옹스트롬, Å)로 바꿉니다.
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    atom_positions = fractional_positions @ lattice

    # 5단계: Unit Cell의 꼭짓점 8개를 계산합니다.
    # (0,0,0)부터 (1,1,1)까지 분율좌표의 조합입니다.
    corner_fractions = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=float)
    corners = corner_fractions @ lattice

    # 6단계: 3차원 그림을 생성합니다.
    fig = plt.figure(figsize=(8, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    try:
        # 7단계: 원소별로 색상과 크기를 다르게 하여 원자를 표시합니다.
        # 혼합 점유 원자 사이트는 대표 원소로 표시합니다.
        elements = [site.specie.symbol if site.is_ordered else
                    max(site.species.items(), key=lambda item: item[1])[0].symbol
                    for site in structure]
        for element in sorted(set(elements)):
            indices = [i for i, symbol in enumerate(elements) if symbol == element]
            points = atom_positions[indices]
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                s=_ELEMENT_SIZES.get(element, 85),
                c=_ELEMENT_COLORS.get(element, "#9B9B9B"),
                edgecolors="#333333", linewidths=0.35,
                depthshade=True, label=element,
            )

        # 8단계: Unit Cell을 이루는 12개의 모서리를 선으로 그립니다.
        # 두 꼭짓점의 분율좌표가 한 축에서만 다르면 서로 연결합니다.
        for i in range(8):
            for j in range(i + 1, 8):
                if np.count_nonzero(corner_fractions[i] != corner_fractions[j]) == 1:
                    p, q = corners[i], corners[j]
                    ax.plot(
                        [p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                        color="#679AB7", linewidth=1.2, alpha=0.85,
                    )

        # 9단계: X, Y, Z 축의 비율을 동일하게 맞춰 구조 왜곡을 줄입니다.
        all_positions = np.vstack((atom_positions, corners))
        minimum = all_positions.min(axis=0)
        maximum = all_positions.max(axis=0)
        center = (minimum + maximum) / 2
        radius = max(float(np.max(maximum - minimum)) / 2, 0.5) * 1.12
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))

        # 10단계: 제목, 축 이름, 범례와 기본 시점을 설정합니다.
        ax.set_title(f"MOF Crystal Structure  |  {len(structure)} atoms", pad=18)
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        ax.view_init(elev=22, azim=38)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),
                  fontsize=9, frameon=False, title="Elements")

        # 11단계: 그림을 PNG 파일로 저장합니다.
        fig.savefig(image_path, dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        # 오류가 발생하더라도 그림 객체를 닫아 메모리 누적을 방지합니다.
        plt.close(fig)

    # 12단계: 팀의 main.py가 사용할 수 있도록 이미지 경로를 반환합니다.
    return str(image_path)
