import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from profile_graph import link_x_axes, offset_profile_data, read_log_for_plot, visualize


def _expand_inputs(paths):
    log_paths = []

    for path in paths:
        candidate = Path(path)

        if candidate.is_dir():
            log_paths.extend(sorted(candidate.glob("*.log")))
        else:
            log_paths.append(candidate)

    return log_paths


def _output_path(output, log_path, multiple_logs):
    if output is None:
        return None

    output = Path(output)

    if multiple_logs or output.suffix == "":
        output.mkdir(parents=True, exist_ok=True)
        return output / f"{log_path.stem}.png"

    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="Visualize Gemmini profiler logs.")
    parser.add_argument("paths", nargs="+", help="Log file(s), or directories containing *.log files")
    parser.add_argument(
        "-o",
        "--output",
        help="Output PNG path. With multiple logs, this is treated as an output directory.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window")
    parser.add_argument(
        "--merge-gap",
        type=int,
        default=None,
        help="Merge same-lane intervals separated by this many cycles. Default: auto based on plot width.",
    )
    parser.add_argument(
        "--target-columns",
        type=int,
        default=5000,
        help="Approximate horizontal detail used for automatic merging. Higher is more detailed but slower.",
    )
    parser.add_argument(
        "--parser",
        choices=("auto", "numpy", "python"),
        default="auto",
        help="Log parser backend. auto uses numpy when available and falls back to pure Python.",
    )
    parser.add_argument(
        "--show-loops",
        action="store_true",
        help="Label loop groups per Gemmini: type 0/load by 16 events and type 1/execute by 256 events.",
    )
    parser.add_argument(
        "--loop-load-size",
        type=int,
        default=16,
        help="Number of load/type-0 events per loop label.",
    )
    parser.add_argument(
        "--loop-execute-size",
        type=int,
        default=256,
        help="Number of execute/type-1 events per loop label.",
    )
    parser.add_argument(
        "--hide-partial-loops",
        action="store_true",
        help="Do not label the final loop group if it has fewer events than its configured group size.",
    )
    time_axis_group = parser.add_mutually_exclusive_group()
    time_axis_group.add_argument(
        "--relative-time",
        action="store_true",
        help="Shift each log so its first event starts at cycle 0.",
    )
    time_axis_group.add_argument(
        "--absolute-time",
        action="store_true",
        help="Keep original cycle values. Multiple logs still share one global x-axis range.",
    )
    parser.add_argument("--dpi", type=int, default=120, help="Figure DPI")
    args = parser.parse_args()

    log_paths = _expand_inputs(args.paths)

    if not log_paths:
        parser.error("No log files found")

    if args.show_loops and (args.loop_load_size <= 0 or args.loop_execute_size <= 0):
        parser.error("--loop-load-size and --loop-execute-size must be positive")

    multiple_logs = len(log_paths) > 1
    loop_groups = None

    if args.show_loops:
        loop_groups = {
            0: args.loop_load_size,
            1: args.loop_execute_size,
        }

    profiles = []

    for log_path in log_paths:
        if not log_path.exists():
            raise FileNotFoundError(log_path)

        profile_data = read_log_for_plot(
            log_path,
            merge_gap=args.merge_gap,
            target_columns=args.target_columns,
            parser=args.parser,
            loop_groups=loop_groups,
            include_partial_loops=not args.hide_partial_loops,
        )
        profiles.append((log_path, profile_data))

    use_relative_time = args.relative_time or (multiple_logs and not args.absolute_time)

    if use_relative_time:
        profiles = [
            (log_path, offset_profile_data(profile_data, -profile_data.start))
            for log_path, profile_data in profiles
        ]

    if multiple_logs or use_relative_time:
        x_bounds = (
            min(profile_data.start for _, profile_data in profiles),
            max(profile_data.end for _, profile_data in profiles),
        )
    else:
        x_bounds = None

    figures = []

    for log_path, profile_data in profiles:
        fig = visualize(profile_data, title=str(log_path), dpi=args.dpi, x_bounds=x_bounds)
        figures.append(fig)

        destination = _output_path(args.output, log_path, multiple_logs)
        if destination is not None:
            fig.savefig(destination, bbox_inches="tight")
            print("Saved:", destination)

        if args.no_show:
            plt.close(fig)

    if not args.no_show:
        link_x_axes(figures)
        plt.show()


if __name__ == "__main__":
    main()
