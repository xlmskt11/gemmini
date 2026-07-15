from dataclasses import dataclass
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import numpy as np
except ImportError:
    np = None


TYPE_LABELS = {
    0: "ld",
    1: "ex",
    2: "st",
}

TYPE_Y_RANGES = {
    0: (20, 8),
    1: (10, 8),
    2: (0, 8),
}

TYPE_COLORS = {
    0: "#4e79a7",
    1: "#f28e2b",
    2: "#59a14f",
}

LOOP_EDGE_COLORS = {
    0: "#244f77",
    1: "#9a5200",
}

LOOP_FILL_COLORS = {
    0: "#d7e6f5",
    1: "#fde3bd",
}


@dataclass
class ProfileData:
    intervals: dict
    start: int
    end: int
    max_gemmini: int
    raw_count: int
    plotted_count: int
    merge_gap: int
    loop_annotations: tuple = ()


@dataclass
class LoopAnnotation:
    gemmini: int
    op_type: int
    loop_index: int
    start: int
    end: int
    event_count: int


def offset_profile_data(profile_data, offset):
    if offset == 0:
        return profile_data

    shifted_intervals = {
        key: [(start + offset, duration) for start, duration in ranges]
        for key, ranges in profile_data.intervals.items()
    }
    shifted_annotations = tuple(
        LoopAnnotation(
            annotation.gemmini,
            annotation.op_type,
            annotation.loop_index,
            annotation.start + offset,
            annotation.end + offset,
            annotation.event_count,
        )
        for annotation in profile_data.loop_annotations
    )

    return ProfileData(
        intervals=shifted_intervals,
        start=profile_data.start + offset,
        end=profile_data.end + offset,
        max_gemmini=profile_data.max_gemmini,
        raw_count=profile_data.raw_count,
        plotted_count=profile_data.plotted_count,
        merge_gap=profile_data.merge_gap,
        loop_annotations=shifted_annotations,
    )


def _parse_event(line):
    parts = line.split(",")
    if len(parts) != 4:
        return None

    try:
        gemmini, op_type, start, end = (int(part) for part in parts)
    except ValueError:
        return None

    return gemmini, op_type, start, end


def _first_event(src):
    with open(src, "r") as f:
        for line in f:
            event = _parse_event(line.rstrip())
            if event is not None:
                return event

    return None


def _last_event(src):
    block_size = 8192

    with open(src, "rb") as f:
        f.seek(0, 2)
        position = f.tell()
        buffer = b""

        while position > 0:
            read_size = min(block_size, position)
            position -= read_size
            f.seek(position)
            buffer = f.read(read_size) + buffer

            for raw_line in reversed(buffer.splitlines()):
                try:
                    line = raw_line.decode()
                except UnicodeDecodeError:
                    continue

                event = _parse_event(line.strip())
                if event is not None:
                    return event

    return None


def _auto_merge_gap(src, target_columns):
    if target_columns <= 0:
        return 0

    first = _first_event(src)
    last = _last_event(src)

    if first is None or last is None:
        return 0

    duration = max(1, max(first[3], last[3]) - min(first[2], last[2]))
    return max(0, ceil(duration / target_columns))


def _flush_interval(intervals, key, current):
    start, end = current
    intervals.setdefault(key, []).append((start, end - start))


def read_log(src):
    if np is not None:
        try:
            data = np.loadtxt(src, delimiter=",", dtype=np.int64, ndmin=2)
            if data.size == 0:
                return []
            if data.shape[1] == 4:
                return data.tolist()
        except ValueError:
            pass

    result = []

    with open(src, "r") as f:
        for line in f:
            event = _parse_event(line.rstrip())
            if event is not None:
                result.append(list(event))

    return result


def _coalesce_ranges_numpy(ranges, merge_gap):
    starts = ranges[:, 0]
    ends = ranges[:, 1]

    if len(starts) == 1:
        return [(int(starts[0]), int(ends[0] - starts[0]))]

    previous_max_ends = np.maximum.accumulate(ends)[:-1]
    breaks = np.flatnonzero(starts[1:] > previous_max_ends + merge_gap) + 1
    segment_starts = np.concatenate(([0], breaks))
    segment_start_values = np.minimum.reduceat(starts, segment_starts)
    segment_end_values = np.maximum.reduceat(ends, segment_starts)
    durations = segment_end_values - segment_start_values

    return list(zip(segment_start_values.tolist(), durations.tolist()))


def _normalize_loop_groups(loop_groups):
    if not loop_groups:
        return {}

    return {
        int(op_type): int(group_size)
        for op_type, group_size in loop_groups.items()
        if int(group_size) > 0
    }


def _loop_annotations_numpy(data, loop_groups, include_partial=True):
    loop_groups = _normalize_loop_groups(loop_groups)

    if not loop_groups:
        return ()

    annotations = []
    gemmini_values = np.unique(data[:, 0])

    for gemmini in gemmini_values:
        gemmini_mask = data[:, 0] == gemmini

        for op_type, group_size in loop_groups.items():
            ranges = data[gemmini_mask & (data[:, 1] == op_type), 2:4]

            if len(ranges) == 0:
                continue

            loop_index = 0

            for group_start in range(0, len(ranges), group_size):
                group = ranges[group_start : group_start + group_size]

                if len(group) < group_size and not include_partial:
                    continue

                annotations.append(
                    LoopAnnotation(
                        int(gemmini),
                        int(op_type),
                        loop_index,
                        int(group[:, 0].min()),
                        int(group[:, 1].max()),
                        int(len(group)),
                    )
                )
                loop_index += 1

    return tuple(annotations)


def _flush_loop_annotation(loop_annotations, key, state):
    gemmini, op_type = key
    loop_index, event_count, start, end = state
    loop_annotations.append(
        LoopAnnotation(gemmini, op_type, loop_index, start, end, event_count)
    )


def _read_log_for_plot_numpy(
    src,
    merge_gap=None,
    target_columns=5000,
    loop_groups=None,
    include_partial_loops=True,
):
    if np is None:
        raise RuntimeError("numpy is not available")

    data = np.loadtxt(src, delimiter=",", dtype=np.int64, ndmin=2)

    if data.size == 0:
        raise ValueError(f"No profile events found in {src}")
    if data.shape[1] != 4:
        raise ValueError(f"Expected four columns in {src}")

    start = int(data[:, 2].min())
    end = int(data[:, 3].max())
    max_gemmini = int(data[:, 0].max())
    raw_count = int(data.shape[0])

    if merge_gap is None:
        merge_gap = 0 if target_columns <= 0 else max(0, ceil(max(1, end - start) / target_columns))
    else:
        merge_gap = max(0, merge_gap)

    intervals = {}
    gemmini_values = np.unique(data[:, 0])
    type_values = np.unique(data[:, 1])
    loop_annotations = _loop_annotations_numpy(data, loop_groups, include_partial_loops)

    for gemmini in gemmini_values:
        gemmini_mask = data[:, 0] == gemmini

        for op_type in type_values:
            ranges = data[gemmini_mask & (data[:, 1] == op_type), 2:4]
            if len(ranges) == 0:
                continue

            intervals[(int(gemmini), int(op_type))] = _coalesce_ranges_numpy(ranges, merge_gap)

    plotted_count = sum(len(ranges) for ranges in intervals.values())

    return ProfileData(
        intervals=intervals,
        start=start,
        end=end,
        max_gemmini=max_gemmini,
        raw_count=raw_count,
        plotted_count=plotted_count,
        merge_gap=merge_gap,
        loop_annotations=loop_annotations,
    )


def _read_log_for_plot_python(
    src,
    merge_gap=None,
    target_columns=5000,
    loop_groups=None,
    include_partial_loops=True,
):
    merge_gap = _auto_merge_gap(src, target_columns) if merge_gap is None else max(0, merge_gap)
    loop_groups = _normalize_loop_groups(loop_groups)
    intervals = {}
    current = {}
    loop_states = {}
    loop_annotations = []
    start = None
    end = None
    max_gemmini = -1
    raw_count = 0

    with open(src, "r") as f:
        for line in f:
            event = _parse_event(line.rstrip())
            if event is None:
                continue

            gemmini, op_type, event_start, event_end = event
            key = (gemmini, op_type)
            start = event_start if start is None else min(start, event_start)
            end = event_end if end is None else max(end, event_end)
            max_gemmini = max(max_gemmini, gemmini)
            raw_count += 1

            if key not in current:
                current[key] = [event_start, event_end]
            elif event_start <= current[key][1] + merge_gap:
                current[key][0] = min(current[key][0], event_start)
                current[key][1] = max(current[key][1], event_end)
            else:
                _flush_interval(intervals, key, current[key])
                current[key] = [event_start, event_end]

            group_size = loop_groups.get(op_type)

            if group_size is None:
                continue

            if key not in loop_states:
                loop_states[key] = [0, 1, event_start, event_end]
            else:
                loop_state = loop_states[key]
                if loop_state[1] == 0:
                    loop_state[1] = 1
                    loop_state[2] = event_start
                    loop_state[3] = event_end
                else:
                    loop_state[1] += 1
                    loop_state[2] = min(loop_state[2], event_start)
                    loop_state[3] = max(loop_state[3], event_end)

            if loop_states[key][1] == group_size:
                _flush_loop_annotation(loop_annotations, key, loop_states[key])
                loop_states[key] = [loop_states[key][0] + 1, 0, None, None]

        if include_partial_loops:
            for key, loop_state in loop_states.items():
                if loop_state[1] > 0:
                    _flush_loop_annotation(loop_annotations, key, loop_state)

    if raw_count == 0:
        raise ValueError(f"No profile events found in {src}")

    for key, active_interval in current.items():
        _flush_interval(intervals, key, active_interval)

    plotted_count = sum(len(ranges) for ranges in intervals.values())

    return ProfileData(
        intervals=intervals,
        start=start,
        end=end,
        max_gemmini=max_gemmini,
        raw_count=raw_count,
        plotted_count=plotted_count,
        merge_gap=merge_gap,
        loop_annotations=tuple(loop_annotations),
    )


def read_log_for_plot(
    src,
    merge_gap=None,
    target_columns=5000,
    parser="auto",
    loop_groups=None,
    include_partial_loops=True,
):
    if parser not in ("auto", "numpy", "python"):
        raise ValueError(f"Unknown parser: {parser}")

    if parser in ("auto", "numpy"):
        if np is None:
            if parser == "numpy":
                raise RuntimeError("numpy is not available")
        else:
            try:
                return _read_log_for_plot_numpy(
                    src,
                    merge_gap,
                    target_columns,
                    loop_groups,
                    include_partial_loops,
                )
            except ValueError:
                if parser == "numpy":
                    raise

    return _read_log_for_plot_python(
        src,
        merge_gap,
        target_columns,
        loop_groups,
        include_partial_loops,
    )


def _profile_data_from_rows(rows):
    intervals = {}
    start = None
    end = None
    max_gemmini = -1

    for gemmini, op_type, event_start, event_end in rows:
        intervals.setdefault((gemmini, op_type), []).append((event_start, event_end - event_start))
        start = event_start if start is None else min(start, event_start)
        end = event_end if end is None else max(end, event_end)
        max_gemmini = max(max_gemmini, gemmini)

    if start is None:
        raise ValueError("No profile events to visualize")

    raw_count = len(rows)

    return ProfileData(
        intervals=intervals,
        start=start,
        end=end,
        max_gemmini=max_gemmini,
        raw_count=raw_count,
        plotted_count=raw_count,
        merge_gap=0,
    )


def _draw_loop_boxes(ax, annotations, op_type):
    y_base, height = TYPE_Y_RANGES[op_type]
    box_y = y_base - 0.6
    box_height = height + 1.2
    edge_color = LOOP_EDGE_COLORS.get(op_type, "#333333")
    fill_color = LOOP_FILL_COLORS.get(op_type, "#eeeeee")

    for annotation in annotations:
        width = max(1, annotation.end - annotation.start)
        fill_alpha = 0.22 if annotation.loop_index % 2 == 0 else 0.12

        ax.add_patch(
            Rectangle(
                (annotation.start, box_y),
                width,
                box_height,
                facecolor=fill_color,
                edgecolor="none",
                alpha=fill_alpha,
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (annotation.start, box_y),
                width,
                box_height,
                facecolor="none",
                edgecolor=edge_color,
                linewidth=0.75,
                alpha=0.85,
                zorder=4,
            )
        )


def _draw_loop_labels(ax, annotations, op_type):
    y_base, height = TYPE_Y_RANGES[op_type]
    edge_color = LOOP_EDGE_COLORS.get(op_type, "#333333")

    for annotation in annotations:
        row = annotation.loop_index % 3

        if op_type == 0:
            y = y_base + height + 0.7 + row * 1.25
            va = "bottom"
        elif op_type == 1:
            y = y_base + height - 1.3 - row * 1.45
            va = "center"
        else:
            y = y_base + height + 0.7
            va = "bottom"

        x = annotation.start + (annotation.end - annotation.start) / 2
        ax.text(
            x,
            y,
            f"loop{annotation.loop_index}",
            ha="center",
            va=va,
            fontsize=6.5,
            color="#111111",
            clip_on=True,
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": edge_color,
                "linewidth": 0.6,
                "alpha": 0.88,
            },
        )


def visualize(profile_data, title=None, dpi=120, x_bounds=None):
    if not isinstance(profile_data, ProfileData):
        profile_data = _profile_data_from_rows(profile_data)

    gemmini_count = profile_data.max_gemmini + 1
    fig_height = max(3.0, 2.4 * gemmini_count)
    fig, axes = plt.subplots(
        gemmini_count,
        1,
        figsize=(12, fig_height),
        sharex=True,
        dpi=dpi,
    )

    if gemmini_count == 1:
        axes = [axes]

    x_start = profile_data.start if x_bounds is None else x_bounds[0]
    x_end = profile_data.end if x_bounds is None else x_bounds[1]
    duration = max(1, x_end - x_start)
    padding = max(1, duration * 0.05)

    for gemmini in range(gemmini_count):
        ax = axes[gemmini]
        ax.axvspan(profile_data.start, profile_data.end, facecolor="lightgrey", alpha=0.25)

        for op_type, label in TYPE_LABELS.items():
            ranges = profile_data.intervals.get((gemmini, op_type), [])
            if not ranges:
                continue

            annotations = [
                annotation
                for annotation in profile_data.loop_annotations
                if annotation.gemmini == gemmini and annotation.op_type == op_type
            ]

            if annotations:
                _draw_loop_boxes(ax, annotations, op_type)

            ax.broken_barh(
                ranges,
                TYPE_Y_RANGES[op_type],
                facecolors=TYPE_COLORS[op_type],
                edgecolors="none",
                label=label,
                zorder=3,
            )

            if annotations:
                _draw_loop_labels(ax, annotations, op_type)

        ax.set_xlim(x_start - padding, x_end + padding)
        ax.set_ylim(-2, 34)
        ax.set_xlabel("Time")
        ax.set_yticks([24, 14, 4])
        ax.set_yticklabels(["ld", "ex", "st"])
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_title(f"Gemmini_{gemmini}")

    if title is not None:
        fig.suptitle(title)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(h_pad=3.0)

    print("Overall Start Time:", profile_data.start)
    print("Overall End Time:", profile_data.end)
    print("duration:", profile_data.end - profile_data.start)
    if x_bounds is not None:
        print("X Axis Start Time:", x_start)
        print("X Axis End Time:", x_end)
    print("Raw Events:", profile_data.raw_count)
    print("Plotted Ranges:", profile_data.plotted_count)
    print("Merge Gap:", profile_data.merge_gap)
    if profile_data.loop_annotations:
        print("Loop Labels:", len(profile_data.loop_annotations))

    return fig


def link_x_axes(figures):
    axes = [ax for fig in figures for ax in fig.axes]

    if len(axes) < 2:
        return

    syncing = False

    def on_xlim_changed(changed_ax):
        nonlocal syncing

        if syncing:
            return

        syncing = True
        try:
            xlim = changed_ax.get_xlim()

            for ax in axes:
                if ax is changed_ax:
                    continue

                ax.set_xlim(xlim)
                ax.figure.canvas.draw_idle()
        finally:
            syncing = False

    callback_ids = [
        (ax, ax.callbacks.connect("xlim_changed", on_xlim_changed))
        for ax in axes
    ]

    for fig in figures:
        fig._linked_x_axes = callback_ids
