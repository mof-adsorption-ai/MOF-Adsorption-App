from __future__ import annotations
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
    """Returns: atom (N, atom_dim), nbr (N, 12, 41), nbr_idx (N, 12), n_atoms, formula."""
    raise NotImplementedError
