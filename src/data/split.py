"""Backward-compatible imports for the canonical split utilities."""

from src.utils.split import (
    assert_no_group_leak,
    class_temporal_trainval_split,
    leak_safe_split,
    leak_safe_trainval_split,
    make_contiguous_groups,
    normal_only_indices,
    temporal_trainval_split,
)

__all__ = [
    'assert_no_group_leak',
    'class_temporal_trainval_split',
    'leak_safe_split',
    'leak_safe_trainval_split',
    'make_contiguous_groups',
    'normal_only_indices',
    'temporal_trainval_split',
]
