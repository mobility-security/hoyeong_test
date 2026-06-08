"""Backward-compatible imports for the canonical split utilities."""

from src.utils.split import (
    assert_no_group_leak,
    leak_safe_split,
    leak_safe_trainval_split,
    make_contiguous_groups,
    normal_only_indices,
)

__all__ = [
    'assert_no_group_leak',
    'leak_safe_split',
    'leak_safe_trainval_split',
    'make_contiguous_groups',
    'normal_only_indices',
]
