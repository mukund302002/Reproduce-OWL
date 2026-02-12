# import argparse
# import json
# from pathlib import Path

# try:
#     import matplotlib.pyplot as plt
# except ImportError:
#     plt = None


# def load_results(results_path: Path):
#     with results_path.open("r", encoding="utf-8") as f:
#         data = json.load(f)
#     if isinstance(data, dict):
#         return [data]
#     if not isinstance(data, list):
#         raise ValueError(f"Unsupported JSON format in {results_path}")
#     return data


# def run_label(run, idx):
#     timestamp = run.get("timestamp_utc", "no-time")
#     model = run.get("model", "unknown-model").split("/")[-1]
#     method = run.get("prune_method", "unknown-method")
#     return f"#{idx} | {model} | {method} | {timestamp}"


# def layer_points(run):
#     layerwise = run.get("layerwise_sparsity", [])
#     xs = [int(item["layer_index"]) for item in layerwise]
#     ys = [float(item["sparsity_ratio"]) for item in layerwise]
#     return xs, ys


# def filter_runs(runs, model_filter=None, method_filter=None):
#     out = []
#     for idx, run in enumerate(runs):
#         if "layerwise_sparsity" not in run:
#             continue
#         if model_filter and model_filter not in str(run.get("model", "")):
#             continue
#         if method_filter and method_filter not in str(run.get("prune_method", "")):
#             continue
#         out.append((idx, run))
#     return out


# def plot_single_run(run, idx, output_path: Path):
#     xs, ys = layer_points(run)
#     if not xs:
#         raise ValueError(f"Run #{idx} has empty layerwise_sparsity")

#     if plt is None:
#         plot_single_run_svg(run, idx, output_path.with_suffix(".svg"))
#         return

#     plt.figure(figsize=(12, 5))
#     plt.plot(xs, ys, marker="o", linewidth=2, markersize=3, label="Layer sparsity")

#     target = run.get("target_sparsity_ratio")
#     actual = run.get("actual_sparsity_ratio")
#     if target is not None:
#         plt.axhline(float(target), linestyle="--", linewidth=1.5, label=f"Target: {float(target):.4f}")
#     if actual is not None:
#         plt.axhline(float(actual), linestyle="-.", linewidth=1.5, label=f"Actual: {float(actual):.4f}")

#     plt.title(f"Layerwise Sparsity\n{run_label(run, idx)}")
#     plt.xlabel("Layer Index")
#     plt.ylabel("Sparsity Ratio")
#     plt.ylim(0.0, 1.0)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=180)
#     plt.close()


# def plot_overlay(runs_with_idx, output_path: Path):
#     if not runs_with_idx:
#         raise ValueError("No runs with layerwise_sparsity to plot")

#     if plt is None:
#         plot_overlay_svg(runs_with_idx, output_path.with_suffix(".svg"))
#         return

#     plt.figure(figsize=(13, 6))
#     for idx, run in runs_with_idx:
#         xs, ys = layer_points(run)
#         if not xs:
#             continue
#         short_label = f"#{idx} {run.get('model', '').split('/')[-1]} {run.get('prune_method', '')}"
#         plt.plot(xs, ys, linewidth=1.6, marker="o", markersize=2.5, alpha=0.85, label=short_label)

#     plt.title("Layerwise Sparsity Comparison Across Runs")
#     plt.xlabel("Layer Index")
#     plt.ylabel("Sparsity Ratio")
#     plt.ylim(0.0, 1.0)
#     plt.grid(True, alpha=0.25)
#     plt.legend(fontsize=8, ncol=2)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=180)
#     plt.close()


# def _svg_xy(layer_idx, sparsity, min_x, max_x, width, height, pad):
#     plot_w = width - 2 * pad
#     plot_h = height - 2 * pad
#     x = pad + ((layer_idx - min_x) / (max_x - min_x)) * plot_w if max_x > min_x else width / 2
#     y = pad + (1.0 - sparsity) * plot_h
#     return x, y


# def _write_svg(path: Path, width: int, height: int, elements):
#     header = (
#         f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
#         f"viewBox='0 0 {width} {height}'>"
#     )
#     content = "\n".join(elements)
#     path.write_text(f"{header}\n{content}\n</svg>\n", encoding="utf-8")


# def plot_single_run_svg(run, idx, output_path: Path):
#     xs, ys = layer_points(run)
#     width, height, pad = 1200, 520, 60
#     min_x, max_x = min(xs), max(xs)

#     points = []
#     for x_val, y_val in zip(xs, ys):
#         x, y = _svg_xy(x_val, y_val, min_x, max_x, width, height, pad)
#         points.append(f"{x:.2f},{y:.2f}")

#     elements = [
#         "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
#         f"<text x='{pad}' y='28' font-size='20' font-family='monospace'>Layerwise Sparsity: {run_label(run, idx)}</text>",
#         f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='black'/>",
#         f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='black'/>",
#         f"<polyline fill='none' stroke='#1f77b4' stroke-width='2' points='{' '.join(points)}'/>",
#     ]

#     for x_val, y_val in zip(xs, ys):
#         x, y = _svg_xy(x_val, y_val, min_x, max_x, width, height, pad)
#         elements.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='2.5' fill='#1f77b4'/>")

#     target = run.get("target_sparsity_ratio")
#     if target is not None:
#         _, y = _svg_xy(min_x, float(target), min_x, max_x, width, height, pad)
#         elements.append(f"<line x1='{pad}' y1='{y:.2f}' x2='{width-pad}' y2='{y:.2f}' stroke='#d62728' stroke-dasharray='6,4'/>")
#         elements.append(f"<text x='{width-pad-220}' y='{max(20, y-6):.2f}' font-size='14' font-family='monospace' fill='#d62728'>target={float(target):.4f}</text>")

#     actual = run.get("actual_sparsity_ratio")
#     if actual is not None:
#         _, y = _svg_xy(min_x, float(actual), min_x, max_x, width, height, pad)
#         elements.append(f"<line x1='{pad}' y1='{y:.2f}' x2='{width-pad}' y2='{y:.2f}' stroke='#2ca02c' stroke-dasharray='3,3'/>")
#         elements.append(f"<text x='{width-pad-220}' y='{min(height-10, y+16):.2f}' font-size='14' font-family='monospace' fill='#2ca02c'>actual={float(actual):.4f}</text>")

#     elements.append(f"<text x='{width/2-40:.0f}' y='{height-14}' font-size='14' font-family='monospace'>Layer Index</text>")
#     elements.append(f"<text x='8' y='{height/2:.0f}' font-size='14' font-family='monospace' transform='rotate(-90, 8, {height/2:.0f})'>Sparsity Ratio</text>")
#     _write_svg(output_path, width, height, elements)


# def plot_overlay_svg(runs_with_idx, output_path: Path):
#     width, height, pad = 1280, 620, 70
#     all_x = []
#     for _, run in runs_with_idx:
#         xs, _ = layer_points(run)
#         all_x.extend(xs)
#     min_x, max_x = min(all_x), max(all_x)

#     palette = [
#         "#1f77b4",
#         "#ff7f0e",
#         "#2ca02c",
#         "#d62728",
#         "#9467bd",
#         "#8c564b",
#         "#e377c2",
#         "#7f7f7f",
#         "#bcbd22",
#         "#17becf",
#     ]

#     elements = [
#         "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
#         f"<text x='{pad}' y='30' font-size='22' font-family='monospace'>Layerwise Sparsity Comparison Across Runs</text>",
#         f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='black'/>",
#         f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='black'/>",
#     ]

#     legend_x = width - pad - 340
#     legend_y = 54
#     for j, (idx, run) in enumerate(runs_with_idx):
#         color = palette[j % len(palette)]
#         xs, ys = layer_points(run)
#         points = []
#         for x_val, y_val in zip(xs, ys):
#             x, y = _svg_xy(x_val, y_val, min_x, max_x, width, height, pad)
#             points.append(f"{x:.2f},{y:.2f}")
#         elements.append(f"<polyline fill='none' stroke='{color}' stroke-width='1.7' points='{' '.join(points)}'/>")
#         label = f"#{idx} {run.get('model', '').split('/')[-1]} {run.get('prune_method', '')}"
#         y_lbl = legend_y + j * 18
#         elements.append(f"<line x1='{legend_x}' y1='{y_lbl}' x2='{legend_x+14}' y2='{y_lbl}' stroke='{color}' stroke-width='2'/>")
#         elements.append(f"<text x='{legend_x+20}' y='{y_lbl+4}' font-size='12' font-family='monospace'>{label}</text>")

#     elements.append(f"<text x='{width/2-40:.0f}' y='{height-14}' font-size='14' font-family='monospace'>Layer Index</text>")
#     elements.append(f"<text x='10' y='{height/2:.0f}' font-size='14' font-family='monospace' transform='rotate(-90, 10, {height/2:.0f})'>Sparsity Ratio</text>")
#     _write_svg(output_path, width, height, elements)


# def main():
#     parser = argparse.ArgumentParser(description="Plot layerwise sparsity from results.json")
#     parser.add_argument("--results", type=str, default="OWL/save_test/results.json", help="Path to results.json")
#     parser.add_argument("--output_dir", type=str, default="OWL/save_test/plots", help="Directory for output PNGs")
#     parser.add_argument(
#         "--run_index",
#         type=int,
#         default=-1,
#         help="Run index to plot as single-run figure. Default -1 means latest run with layerwise_sparsity.",
#     )
#     parser.add_argument("--model_filter", type=str, default=None, help="Only include runs where model contains this text")
#     parser.add_argument(
#         "--method_filter",
#         type=str,
#         default=None,
#         help="Only include runs where prune_method contains this text",
#     )
#     args = parser.parse_args()

#     results_path = Path(args.results)
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     runs = load_results(results_path)
#     runs_with_idx = filter_runs(runs, args.model_filter, args.method_filter)
#     if not runs_with_idx:
#         raise ValueError("No runs found with layerwise_sparsity after applying filters")

#     if args.run_index == -1:
#         selected_idx, selected_run = runs_with_idx[-1]
#     else:
#         run_dict = {idx: run for idx, run in runs_with_idx}
#         if args.run_index not in run_dict:
#             raise ValueError(f"run_index {args.run_index} not found among runs with layerwise_sparsity")
#         selected_idx, selected_run = args.run_index, run_dict[args.run_index]

#     ext = ".png" if plt is not None else ".svg"
#     single_out = output_dir / f"layerwise_run_{selected_idx}{ext}"
#     overlay_out = output_dir / f"layerwise_overlay{ext}"

#     plot_single_run(selected_run, selected_idx, single_out)
#     plot_overlay(runs_with_idx, overlay_out)

#     print(f"Saved single-run plot: {single_out}")
#     print(f"Saved overlay plot:    {overlay_out}")
#     print(f"Single-run label:      {run_label(selected_run, selected_idx)}")


# if __name__ == "__main__":
#     main()



import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt


def load_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON format in {path}")


def layer_points(run):
    layerwise = run.get("layerwise_sparsity", [])
    xs = [int(item["layer_index"]) for item in layerwise]
    ys = [float(item["sparsity_ratio"]) for item in layerwise]
    return xs, ys


def run_title(run):
    model = str(run.get("model", "unknown-model")).split("/")[-1]
    method = run.get("prune_method", "unknown-method")
    ts = run.get("timestamp_utc", "no-time")
    return f"{model} | {method} | {ts}"


def _slug(value):
    text = str(value).strip().replace("/", "-")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text)
    text = text.strip("-")
    return text or "na"


def run_filename(run, idx):
    model = _slug(str(run.get("model", "unknown-model")).split("/")[-1])
    method = _slug(run.get("prune_method", "unknown-method"))
    sparsity_type = _slug(run.get("sparsity_type", "unknown-type"))
    target = run.get("target_sparsity_ratio", "na")
    target_txt = _slug(f"{float(target):.4f}" if target != "na" else target)
    return f"layerwise_run_{idx:03d}_{model}_{method}_{sparsity_type}_target-{target_txt}.png"


def build_metadata_text(run):
    # Only include keys that exist, keep it compact but informative
    lines = []

    def add(label, key, fmt=None):
        if key in run and run[key] is not None:
            val = run[key]
            if fmt is not None:
                try:
                    val = fmt(val)
                except Exception:
                    pass
            lines.append(f"{label}: {val}")

    add("Model", "model", lambda v: str(v))
    add("Method", "prune_method", lambda v: str(v))
    add("Sparsity type", "sparsity_type", lambda v: str(v))
    add("Target sparsity", "target_sparsity_ratio", lambda v: f"{float(v):.4f}")
    add("Actual sparsity", "actual_sparsity_ratio", lambda v: f"{float(v):.4f}")
    add("nsamples", "nsamples", lambda v: int(v))
    add("seed", "seed", lambda v: int(v))
    add("ppl_wikitext", "ppl_wikitext", lambda v: f"{float(v):.3f}")
    add("Lamda", "Lamda", lambda v: f"{float(v):.4f}")
    add("Hyper_m", "Hyper_m", lambda v: f"{float(v):.4f}")
    add("outlier_by_activation", "outlier_by_activation", lambda v: bool(v))
    add("outlier_by_wmetric", "outlier_by_wmetric", lambda v: bool(v))

    # Timestamp, if present
    if "timestamp_utc" in run:
        lines.append(f"timestamp_utc: {run['timestamp_utc']}")

    return "\n".join(lines)


def plot_single_run(run, output_path: Path):
    xs, ys = layer_points(run)
    if not xs:
        raise ValueError("Selected run has empty or missing layerwise_sparsity")

    # Sort by layer index to be safe
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    xs, ys = zip(*pairs)

    fig = plt.figure(figsize=(13, 6))
    ax = plt.gca()

    ax.plot(xs, ys, marker="o", linewidth=2, markersize=4)
    ax.set_title("Layerwise Sparsity Ratio (Single Run)\n" + run_title(run), fontsize=12)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Sparsity Ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    # Reference lines
    target = run.get("target_sparsity_ratio", None)
    actual = run.get("actual_sparsity_ratio", None)
    if target is not None:
        ax.axhline(float(target), linestyle="--", linewidth=1.5, label=f"Target: {float(target):.4f}")
    if actual is not None:
        ax.axhline(float(actual), linestyle="-.", linewidth=1.5, label=f"Actual: {float(actual):.4f}")

    ax.legend(loc="lower right")

    # Metadata text box
    meta = build_metadata_text(run)
    ax.text(
        1.02, 0.5, meta,
        transform=ax.transAxes,
        va="center", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.12),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(runs, output_path: Path):
    # Filter to runs that actually have layerwise_sparsity
    filtered = []
    for i, r in enumerate(runs):
        if r.get("layerwise_sparsity"):
            filtered.append((i, r))
    if not filtered:
        raise ValueError("No runs found with layerwise_sparsity")

    fig = plt.figure(figsize=(13, 6))
    ax = plt.gca()

    for idx, run in filtered:
        xs, ys = layer_points(run)
        if not xs:
            continue
        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        xs, ys = zip(*pairs)

        model = str(run.get("model", "")).split("/")[-1]
        method = run.get("prune_method", "")
        label = f"#{idx} {model} {method}"
        ax.plot(xs, ys, linewidth=1.7, marker="o", markersize=3, alpha=0.85, label=label)

    ax.set_title("Layerwise Sparsity Ratio (Overlay of Runs)", fontsize=12)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Sparsity Ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot layerwise sparsity into separate PNG files for each model/setting run"
    )
    parser.add_argument("--results", type=str, default="OWL/save_test/results.json", help="Path to results.json")
    parser.add_argument("--output_dir", type=str, default="OWL/save_test/plots", help="Directory to save PNG outputs")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_results(results_path)

    valid_runs = [(i, r) for i, r in enumerate(runs) if r.get("layerwise_sparsity")]
    if not valid_runs:
        raise ValueError("No runs contain layerwise_sparsity")

    for idx, run in valid_runs:
        out_path = out_dir / run_filename(run, idx)
        plot_single_run(run, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
