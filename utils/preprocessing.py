from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import torch
from pymatgen.core import Structure


class GraphData(TypedDict):
    atom: torch.Tensor
    nbr: torch.Tensor
    nbr_idx: torch.Tensor
    n_atoms: int
    formula: str


def prepare_model_input(structure: Structure, checkpoint_path: str | Path) -> GraphData:
    """결정구조를 README의 설정에 따라 단일 CGCNN 그래프로 변환한다.

    반환 텐서는 CPU에 생성된다. ``atom``은 (N, atom_dim) float32,
    ``nbr``은 (N, 12, 41) float32, ``nbr_idx``는 (N, 12) int64이다.
    주기적 이미지의 이웃도 포함하며, 부족한 이웃은 인덱스 0과 거리
    9.0 Å로 채운 뒤 Gaussian 거리 특성으로 변환한다.

    Raises:
        TypeError: structure가 pymatgen Structure가 아닌 경우.
        ValueError: 구조가 비어 있거나 무질서 구조이거나 원자 특성이
            누락되었거나 잘못된 경우.
    """
    if not isinstance(structure, Structure):
        raise TypeError("structure must be a pymatgen Structure.")
    if len(structure) == 0:
        raise ValueError("structure must contain at least one atom.")
    if not structure.is_ordered:
        raise ValueError("Disordered structures are not supported; resolve site occupancies first.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("The checkpoint must be a mapping containing atom_init.")
    atom_init = checkpoint.get("atom_init")
    if not isinstance(atom_init, Mapping) or not atom_init:
        raise ValueError("The checkpoint must contain a non-empty atom_init mapping.")

    atom_features = []
    atom_dim = None
    for site in structure:
        atomic_number = site.specie.Z
        # JSON에서 유래한 체크포인트의 문자열 원자번호 키도 허용한다.
        feature = atom_init.get(atomic_number)
        if feature is None:
            feature = atom_init.get(str(atomic_number))
        if feature is None:
            raise ValueError(
                f"atom_init has no features for {site.specie.symbol} (Z={atomic_number})."
            )
        try:
            feature = torch.as_tensor(feature, dtype=torch.float32, device="cpu")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid atom_init features for Z={atomic_number}.") from exc
        if feature.ndim != 1 or feature.numel() == 0 or not torch.isfinite(feature).all():
            raise ValueError(f"atom_init features for Z={atomic_number} must be a finite vector.")
        if atom_dim is None:
            atom_dim = feature.numel()
        elif feature.numel() != atom_dim:
            raise ValueError("All atom_init feature vectors must have the same dimension.")
        atom_features.append(feature)

    n_atoms = len(structure)
    nbr_idx = torch.zeros((n_atoms, 12), dtype=torch.long)
    distances = torch.full((n_atoms, 12), 9.0, dtype=torch.float32)
    # pymatgen은 격자의 주기적 이미지까지 탐색하고 자기 자신은 제외한다.
    all_neighbors = structure.get_all_neighbors(8.0, include_index=True)
    for atom_index, neighbors in enumerate(all_neighbors):
        nearest = sorted(neighbors, key=lambda neighbor: neighbor.nn_distance)[:12]
        if nearest:
            count = len(nearest)
            nbr_idx[atom_index, :count] = torch.tensor(
                [neighbor.index for neighbor in nearest], dtype=torch.long
            )
            distances[atom_index, :count] = torch.tensor(
                [neighbor.nn_distance for neighbor in nearest], dtype=torch.float32
            )

    centers = torch.arange(41, dtype=torch.float32) * 0.2
    nbr = torch.exp(-((distances.unsqueeze(-1) - centers) / 0.2).square())
    return {
        "atom": torch.stack(atom_features),
        "nbr": nbr,
        "nbr_idx": nbr_idx,
        "n_atoms": n_atoms,
        "formula": structure.composition.reduced_formula,
    }
