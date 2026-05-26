"""Shared physical and dataset constants for the two-mass-spring-damper project.

All numerical values are taken from ``project2_concept.txt`` Sections 2 and 3.
The data generator, the three model files, and the training/eval scripts
import from this module so there is exactly one source of truth.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (fixed; project2_concept.txt Section 2).
# ---------------------------------------------------------------------------
M_R: float = 10.0                                           # robot mass [kg]
M_B: float = 70.0                                           # base mass [kg]
K_B: float = 4000.0                                         # base stiffness [N/m]
ZETA_B: float = 0.25                                        # base damping ratio [-]
D_B: float = 2.0 * math.sqrt(K_B * M_B) * ZETA_B            # base damping [Ns/m]

# ---------------------------------------------------------------------------
# Simulation grid.
# ---------------------------------------------------------------------------
T_HORIZON: float = 10.0                                     # horizon [s]
FS: float = 100.0                                           # sampling frequency [Hz]
N_TIME: int = int(T_HORIZON * FS)                           # samples per trajectory

# ---------------------------------------------------------------------------
# Input / output dimensions.
# ---------------------------------------------------------------------------
N_PARAMS: int = 6                                           # [A, f0, f1, offset, k_r, zeta_r]
N_OUT_CHANNELS: int = 2                                     # (x_r, x_b)

# ---------------------------------------------------------------------------
# Sampled parameter ranges (uniform; project2_concept.txt Section 2).
# ---------------------------------------------------------------------------
PARAM_NAMES: Tuple[str, ...] = ("A", "f0", "f1", "offset", "k_r", "zeta_r")
PARAM_LOW: Tuple[float, ...] = (0.01, 0.10, 1.00, -0.05, 10.0, 0.00)
PARAM_HIGH: Tuple[float, ...] = (0.10, 1.00, 5.00, 0.05, 5000.0, 1.10)
PARAM_BOUNDS: np.ndarray = np.array(
    list(zip(PARAM_LOW, PARAM_HIGH)), dtype=np.float64
)

# ---------------------------------------------------------------------------
# Training subset sizes (nested prefixes; project2_concept.txt Section 3).
# ---------------------------------------------------------------------------
N_VALUES: Tuple[int, ...] = (50, 100, 200, 500, 1000, 2000, 5000)

# ---------------------------------------------------------------------------
# Strata on the amplification ratio r = max_t(x_r + x_b) / max_t|x_c|.
# Tuple format: (name, lower_inclusive, upper_exclusive).
# ---------------------------------------------------------------------------
STRATA: Tuple[Tuple[str, float, float], ...] = (
    ("r_0.0_1.0", 0.0, 1.0),
    ("r_1.0_1.2", 1.0, 1.2),
    ("r_1.2_1.5", 1.2, 1.5),
    ("r_gt_1.5",  1.5, float("inf")),
)
