# Copyright 2026 OpenBMB and the HuggingFace Inc. team.
# Adapted from Transformers v5.18.0 under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0

import itertools

import mlx.core as mx
import numpy as np

from .config import validate_downsample_mode


def _visual_groups(ids, modalities, config):
    markers = (
        config.slice_start_id,
        config.slice_end_id,
        config.image_start_id,
        config.image_end_id,
        config.newline_id,
    )
    groups = []
    slices = None
    previous_end = 0
    for modality, run in itertools.groupby(enumerate(modalities), lambda x: x[1]):
        if modality == 0:
            continue
        if modality not in (1, 2):
            raise ValueError(
                "mm_token_type_ids must contain only text/image/video labels"
            )
        run = list(run)
        start, end = run[0][0], run[-1][0] + 1
        if start == 0:
            raise ValueError("Visual crops require an opening marker")
        token = config.image_token_id if modality == 1 else config.video_token_id
        if any(i != token for i in ids[start:end]):
            raise ValueError("Modality labels do not match visual placeholder tokens")
        if slices is not None and ids[start - 1] == config.slice_start_id:
            if groups[-1][1][-1][0] != modality:
                raise ValueError("A detail slice must have its thumbnail's modality")
            if end >= len(ids) or ids[end] != config.slice_end_id:
                raise ValueError("Visual slice requires a closing marker")
            slices.append((start, end))
        else:
            if ids[start - 1] != config.image_start_id:
                raise ValueError("Visual thumbnail requires an image opening marker")
            if end >= len(ids) or ids[end] != config.image_end_id:
                raise ValueError("Visual thumbnail requires an image closing marker")
            slices = []
            frame = (modality, start, end, slices)
            gap = ids[previous_end : start - 1]
            if groups and all(i in markers for i in gap):
                groups[-1][1].append(frame)
            else:
                groups.append((start - 1, [frame]))
        previous_end = end
    result = []
    for group_start, frames in groups:
        complete = []
        for modality, start, end, slices in frames:
            frame_end = slices[-1][1] if slices else end
            while frame_end < len(ids) and ids[frame_end] in markers:
                frame_end += 1
            complete.append((modality, start, end, frame_end, slices))
        result.append((group_start, complete[-1][3], complete))
    return result


def _grid_positions(origin, grid, merge, canvas=(0, 0), offset=(0, 0)):
    height, width = grid[0] // merge, grid[1] // merge
    canvas_h, canvas_w = canvas[0] or height, canvas[1] or width
    h = np.rint(np.linspace(0, canvas_h - 1, height)).astype(np.int32) + offset[0]
    w = np.rint(np.linspace(0, canvas_w - 1, width)).astype(np.int32) + offset[1]
    h, w = np.meshgrid(h, w, indexing="ij")
    return np.stack((np.zeros_like(h), h, w)).reshape(3, -1) + origin


def canvas_positions(
    input_ids,
    config,
    *,
    mm_token_type_ids=None,
    target_sizes=None,
    target_sizes_videos=None,
    attention_mask=None,
    downsample_mode=None,
):
    ids = np.asarray(input_ids)
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError("Canvas input_ids must be a nonempty [batch, sequence] array")
    valid = (
        np.ones_like(ids, dtype=bool)
        if attention_mask is None
        else np.asarray(attention_mask, dtype=bool)
    )
    if valid.shape != ids.shape or not np.all(valid.any(axis=1)):
        raise ValueError("Canvas attention_mask must preserve tokens in every row")
    has_visuals = target_sizes is not None or target_sizes_videos is not None
    if mm_token_type_ids is None:
        if has_visuals or np.any(
            (ids == config.image_token_id) | (ids == config.video_token_id)
        ):
            raise ValueError(
                "Multimodal canvas requires mm_token_type_ids and crop grids"
            )
        modalities = np.zeros_like(ids)
    else:
        modalities = np.asarray(mm_token_type_ids)
        if modalities.shape != ids.shape:
            raise ValueError("mm_token_type_ids must match input_ids")
    mode = validate_downsample_mode(downsample_mode or config.downsample_mode)
    merge = 2 if mode == "4x" else 4
    grids = {
        1: [] if target_sizes is None else np.asarray(target_sizes).tolist(),
        2: (
            []
            if target_sizes_videos is None
            else np.asarray(target_sizes_videos).tolist()
        ),
    }
    grid_indices = {1: 0, 2: 0}

    def next_grid(modality, span):
        idx = grid_indices[modality]
        if idx >= len(grids[modality]):
            raise ValueError("Visual placeholders require one target grid per crop")
        grid = grids[modality][idx]
        grid_indices[modality] += 1
        if len(grid) != 2 or any(
            type(i) is not int or i <= 0 or i % merge for i in grid
        ):
            raise ValueError(
                "Crop grids must be positive and divisible by visual merge factor"
            )
        if grid[0] * grid[1] // (merge * merge) != span[1] - span[0]:
            raise ValueError("Crop grid does not match its visual placeholder count")
        return grid

    positions = np.zeros((3, *ids.shape), dtype=np.int32)
    for row in range(ids.shape[0]):
        row_ids = ids[row, valid[row]].tolist()
        row_mm = modalities[row, valid[row]].tolist()
        current = np.tile(np.arange(len(row_ids), dtype=np.int32), (3, 1))
        cursor = pos = 0
        for group_start, group_end, frames in _visual_groups(row_ids, row_mm, config):
            if group_start > cursor:
                count = group_start - cursor
                current[:, cursor:group_start] = np.arange(count) + pos
                pos += count
            frame_cursor = group_start
            for modality, thumb_start, thumb_end, frame_end, slices in frames:
                thumb_grid = next_grid(modality, (thumb_start, thumb_end))
                frame_start = thumb_start - 1
                if frame_start > frame_cursor:
                    count = frame_start - frame_cursor
                    current[:, frame_cursor:frame_start] = np.arange(count) + pos
                    pos += count + 1
                frame_end = min(frame_end, group_end)
                origin = pos
                canvas_h, canvas_w = thumb_grid[0] // merge, thumb_grid[1] // merge
                cols = len(slices)
                first_grid = None
                if slices:
                    for k in range(len(slices) - 1):
                        if slices[k + 1][0] - slices[k][1] > 2:
                            cols = k + 1
                            break
                    rows = len(slices) // cols
                    if rows * cols != len(slices):
                        raise ValueError("Visual slices must form complete rows")
                    first_grid = next_grid(modality, slices[0])
                    slice_h, slice_w = first_grid[0] // merge, first_grid[1] // merge
                    canvas_h, canvas_w = rows * slice_h, cols * slice_w
                current[:, frame_start:frame_end] = origin
                current[1:, frame_start] = max(origin - 1, 0)
                if thumb_end < frame_end:
                    current[1, thumb_end] = origin + canvas_h
                    current[2, thumb_end] = origin + canvas_w
                current[:, thumb_start:thumb_end] = _grid_positions(
                    origin, thumb_grid, merge, (canvas_h, canvas_w)
                )
                for k, (start, end) in enumerate(slices):
                    grid = first_grid if k == 0 else next_grid(modality, (start, end))
                    if grid != first_grid:
                        raise ValueError(
                            "Detail crops in a canvas must share grid dimensions"
                        )
                    h_off, w_off = (k // cols) * slice_h, (k % cols) * slice_w
                    current[1, start - 1] = origin + h_off
                    current[2, start - 1] = origin + w_off
                    if end < frame_end:
                        current[1, end] = origin + h_off + slice_h - 1
                        current[2, end] = origin + w_off + slice_w - 1
                    current[:, start:end] = _grid_positions(
                        origin, grid, merge, offset=(h_off, w_off)
                    )
                for k in range(len(slices) - 1):
                    gap_start, gap_end = slices[k][1], slices[k + 1][0]
                    if gap_end - gap_start <= 2:
                        continue
                    for newline in range(gap_start + 1, gap_end - 1):
                        if row_ids[newline] != config.newline_id:
                            raise ValueError(
                                "Slice rows must be separated by newline markers"
                            )
                        current[1, newline] = origin + (k // cols + 1) * slice_h - 1
                        current[2, newline] = origin + cols * slice_w
                pos = origin + max(canvas_h, canvas_w) + 1
                frame_cursor = frame_end
            if frame_cursor < group_end:
                count = group_end - frame_cursor
                current[:, frame_cursor:group_end] = np.arange(count) + pos
                pos += count
            cursor = group_end
        if cursor < len(row_ids):
            current[:, cursor:] = np.arange(len(row_ids) - cursor) + pos
        positions[:, row, valid[row]] = current
    if any(grid_indices[m] != len(grids[m]) for m in grids):
        raise ValueError("Received visual crop grids without matching placeholders")
    deltas = positions.max(axis=(0, 2))[:, None] + 1 - valid.sum(axis=1)[:, None]
    return mx.array(positions), mx.array(deltas, dtype=mx.int32)
