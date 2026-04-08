"""Minimal CUTLASS-like swizzle helpers for GEMM CTA mapping."""
from __future__ import annotations

import re
from typing import Tuple


def get_tiled_shape(
    M: int,
    N: int,
    K: int,
    tb_m: int,
    tb_n: int,
    tb_k: int,
    split_k_slices: int,
) -> tuple[int, int, int]:
    tiled_m = (M + tb_m - 1) // tb_m
    tiled_n = (N + tb_n - 1) // tb_n
    tiled_k = max(split_k_slices, 1)
    return tiled_m, tiled_n, tiled_k


def _parse_swizzle(swizzle: str) -> tuple[str, int]:
    if swizzle in ("Horizontal", "SplitKHorizontal"):
        return "horizontal", 1

    match = re.fullmatch(r"(SplitK)?Identity(\d+)?", swizzle)
    if match:
        width = int(match.group(2) or "1")
        if width not in (1, 2, 4, 8):
            raise NotImplementedError(f"Unsupported identity swizzle width: {width}")
        return "identity", width

    raise NotImplementedError(f"Unsupported swizzle: {swizzle}")


def _get_log_tile(swizzle: str, tiled_shape: tuple[int, int, int]) -> int:
    kind, width = _parse_swizzle(swizzle)
    if kind == "horizontal":
        return 0

    _, tiled_n, _ = tiled_shape
    if width >= 8 and tiled_n >= 6:
        return 3
    if width >= 4 and tiled_n >= 3:
        return 2
    if width >= 2 and tiled_n >= 2:
        return 1
    return 0


def get_grid_shape(
    swizzle: str,
    tiled_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    tiled_m, tiled_n, tiled_k = tiled_shape
    kind, _ = _parse_swizzle(swizzle)
    if kind == "identity":
        tile = 1 << _get_log_tile(swizzle, tiled_shape)
        return tiled_m * tile, (tiled_n + tile - 1) // tile, tiled_k
    if kind == "horizontal":
        return tiled_n, tiled_m, tiled_k
    raise AssertionError("unreachable")


def get_tile_offset(
    swizzle: str,
    block_x: int,
    block_y: int,
    block_z: int,
    tiled_shape: tuple[int, int, int],
) -> Tuple[int, int, int]:
    kind, _ = _parse_swizzle(swizzle)

    if kind == "identity":
        log_tile = _get_log_tile(swizzle, tiled_shape)
        tile_m = block_x >> log_tile
        tile_n = (block_y << log_tile) + (block_x & ((1 << log_tile) - 1))
        tile_k = block_z
    elif kind == "horizontal":
        tile_m, tile_n, tile_k = block_y, block_x, block_z
    else:
        raise AssertionError("unreachable")

    return tile_m, tile_n, tile_k
