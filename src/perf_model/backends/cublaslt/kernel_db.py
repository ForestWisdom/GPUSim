"""Known cuBLASLt algorithm metadata."""

from __future__ import annotations

CUBLASLT_ALGOS = {
    "default_tensorop": {
        "threadblock_shape": (128, 128, 32),
        "warp_shape": (64, 64, 32),
        "instruction_shape": (16, 8, 16),
        "swizzle": "Identity",
    }
}
