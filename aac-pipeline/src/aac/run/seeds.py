"""seed_everything(seed): random, numpy, torch (+cuda) with deterministic algorithms.

torch is imported inside the function body on purpose: aac.run must stay free of
module-level torch imports (import-time cost and CPU-only pack/report code)."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np

    np.random.seed(seed % (2**32))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch  # noqa: PLC0415 (deliberately local)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
