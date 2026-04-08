"""CUTLASS swizzle placeholders."""

from __future__ import annotations


def decode_swizzle(name: str) -> str:
    supported = {"Identity", "Horizontal", "SplitKIdentity"}
    if name not in supported:
        return "Identity"
    return name
