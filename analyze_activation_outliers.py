import argparse
import csv
import inspect
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.data import get_loaders
from lib.layerwrapper import WrappedGPT


def get_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise AttributeError("Could not locate transformer layers on model.")


def find_linear_layers(module, name=""):
    if isinstance(module, nn.Linear):
        return {name: module}
    layers = {}
    for child_name, child in module.named_children():
        child_key = child_name if not name else f"{name}.{child_name}"
        layers.update(find_linear_layers(child, child_key))
    return layers


def _first_module_device(module):
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return None


def _lookup_hf_device(model, candidates):
    if not hasattr(model, "hf_device_map"):
        return None
    for key in candidates:
        if key in model.hf_device_map:
            return model.hf_device_map[key]
    return None


def _layer_device(model, layer_idx):
    return _lookup_hf_device(
        model,
        [
            f"model.layers.{layer_idx}",
            f"model.decoder.layers.{layer_idx}",
            f"decoder.layers.{layer_idx}",
        ],
    )


def _embed_device(model):
    return _lookup_hf_device(
        model,
        [
            "model.embed_tokens",
            "model.decoder.embed_tokens",
            "decoder.embed_tokens",
        ],
    )


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_transformer_layers(model)
    embed_dev = _embed_device(model)
    if embed_dev is not None:
        device = embed_dev

    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size
    inps = torch.zeros((nsamples, model.seqlen, hidden_size), dtype=dtype, device=device)
    outs = torch.zeros_like(inps)
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            idx = cache["i"]
            if idx < nsamples:
                inps[idx] = inp
                cache["attention_mask"] = kwargs.get("attention_mask", None)
                cache["position_ids"] = kwargs.get("position_ids", None)
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        if cache["i"] >= nsamples:
            break
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    model.config.use_cache = use_cache
    return inps, outs, cache["attention_mask"], cache["position_ids"]


def get_unique_c4_samples(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    rng = random.Random(seed)
    selected = []
    used_indices = set()
    max_tries = nsamples * 300
    tries = 0

    while len(selected) < nsamples and tries < max_tries:
        tries += 1
        idx = rng.randint(0, len(traindata) - 1)
        if idx in used_indices:
            continue

        text = traindata[idx]["text"]
        trainenc = tokenizer(text, return_tensors="pt")
        if trainenc.input_ids.shape[1] <= seqlen:
            continue

        start = rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        end = start + seqlen
        inp = trainenc.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        selected.append((inp, tar))
        used_indices.add(idx)

    if len(selected) < nsamples:
        raise RuntimeError(
            f"Could only collect {len(selected)} unique C4 samples, requested {nsamples}. "
            "Try reducing --nsamples/--max_samples."
        )
    return selected


def compute_threshold(values: torch.Tensor, mode: str, scale: float, percentile: float):
    if mode == "mean":
        return float(values.mean().item() * scale)
    if mode == "max":
        return float(values.max().item() * scale)
    if mode == "percentile":
        return float(np.percentile(values.numpy(), percentile))
    raise ValueError(f"Unsupported threshold mode: {mode}")


def _build_layer_kwargs(layer, inps, attention_mask, position_ids, rotary_module):
    params = inspect.signature(layer.forward).parameters
    kwargs = {}

    if attention_mask is not None and "attention_mask" in params:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None and "position_ids" in params:
        kwargs["position_ids"] = position_ids

    if "cache_position" in params:
        kwargs["cache_position"] = torch.arange(inps.shape[1], device=inps.device)

    if "position_embeddings" in params:
        if rotary_module is None:
            raise RuntimeError("Layer requires position_embeddings but model has no rotary_emb module.")
        rotary_position_ids = position_ids
        if rotary_position_ids is None:
            rotary_position_ids = torch.arange(inps.shape[1], device=inps.device).unsqueeze(0)
        with torch.no_grad():
            try:
                pos_emb = rotary_module(inps[0].unsqueeze(0), position_ids=rotary_position_ids)
            except RuntimeError:
                rot_dev = _first_module_device(rotary_module)
                if rot_dev is None:
                    raise
                hidden = inps[0].unsqueeze(0).to(rot_dev)
                pos_ids = rotary_position_ids.to(rot_dev)
                pos_emb = rotary_module(hidden, position_ids=pos_ids)
                pos_emb = tuple(x.to(inps.device) for x in pos_emb)
        kwargs["position_embeddings"] = pos_emb

    return kwargs


def collect_layerwise_outliers(
    model,
    inps,
    outs,
    attention_mask,
    position_ids,
    nsamples,
    threshold_mode,
    threshold_scale,
    threshold_percentile,
):
    layers = get_transformer_layers(model)
    stats = []
    raw_values = defaultdict(dict)
    raw_thresholds = defaultdict(dict)

    rotary_module = None
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rotary_module = model.model.rotary_emb

    for layer_idx, layer in enumerate(layers):
        subset = find_linear_layers(layer)

        layer_dev = _layer_device(model, layer_idx)
        if layer_dev is not None:
            inps = inps.to(layer_dev)
            outs = outs.to(layer_dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(layer_dev)
            if position_ids is not None:
                position_ids = position_ids.to(layer_dev)

        layer_kwargs = _build_layer_kwargs(layer, inps, attention_mask, position_ids, rotary_module)

        wrapped_layers = {name: WrappedGPT(mod) for name, mod in subset.items()}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]
        for j in range(nsamples):
            with torch.no_grad():
                layer_out = layer(inps[j].unsqueeze(0), **layer_kwargs)
                if isinstance(layer_out, (tuple, list)):
                    layer_out = layer_out[0]
                outs[j] = layer_out
        for h in handles:
            h.remove()

        for name in sorted(subset.keys()):
            values = torch.sqrt(torch.clamp(wrapped_layers[name].scaler_row.float().cpu(), min=0.0))
            threshold = compute_threshold(values, threshold_mode, threshold_scale, threshold_percentile)
            outlier_mask = values > threshold
            outlier_count = int(outlier_mask.sum().item())
            total = int(values.numel())
            ratio = (100.0 * outlier_count / total) if total else 0.0

            stats.append(
                {
                    "layer": int(layer_idx),
                    "module": name,
                    "num_features": total,
                    "num_outliers": outlier_count,
                    "outlier_ratio_percent": ratio,
                    "threshold": threshold,
                    "mean": float(values.mean().item()),
                    "std": float(values.std(unbiased=False).item()),
                    "p95": float(np.percentile(values.numpy(), 95)),
                    "p99": float(np.percentile(values.numpy(), 99)),
                    "max": float(values.max().item()),
                }
            )

            raw_values[layer_idx][name] = values.numpy().astype(np.float32)
            raw_thresholds[layer_idx][name] = threshold

        inps, outs = outs, inps

    return stats, raw_values, raw_thresholds


def save_stats_json(stats, output_path: Path):
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def save_stats_csv(stats, output_path: Path):
    if not stats:
        return
    keys = list(stats[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(stats)


def save_raw_values_json(raw_values, output_path: Path):
    serializable = {
        str(layer): {name: values.tolist() for name, values in module_map.items()}
        for layer, module_map in raw_values.items()
    }
    output_path.write_text(json.dumps(serializable), encoding="utf-8")


def write_hyperparameters(output_dir: Path, args, loaded_samples: int):
    hp = {
        "model": args.model,
        "seed": args.seed,
        "nsamples": args.nsamples,
        "loaded_samples": loaded_samples,
        "seqlen": args.seqlen,
        "dtype": args.dtype,
        "threshold_mode": args.threshold_mode,
        "threshold_scale": args.threshold_scale,
        "threshold_percentile": args.threshold_percentile,
        "bins": args.bins,
        "per_sample": bool(args.per_sample),
        "max_samples": args.max_samples,
        "save_raw_json": bool(args.save_raw_json),
    }
    (output_dir / "hyperparameters.txt").write_text(json.dumps(hp, indent=2), encoding="utf-8")


def plot_layerwise_ratios(stats, output_path: Path):
    modules = sorted({s["module"] for s in stats})
    fig, ax = plt.subplots(figsize=(14, 7))
    for module in modules:
        points = sorted(
            [(s["layer"], s["outlier_ratio_percent"]) for s in stats if s["module"] == module],
            key=lambda x: x[0],
        )
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=3.5, label=module)

    ax.set_title("Layer-wise Outlier Ratio by Linear Submodule")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Outlier Ratio (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_layer_histograms(raw_values, raw_thresholds, output_dir: Path, bins: int):
    for layer_idx in sorted(raw_values.keys()):
        modules = sorted(raw_values[layer_idx].keys())
        n = len(modules)
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5 * rows))
        axes = np.array(axes).reshape(-1)

        for ax_idx, module in enumerate(modules):
            ax = axes[ax_idx]
            vals = raw_values[layer_idx][module]
            thr = raw_thresholds[layer_idx][module]
            ax.hist(vals, bins=bins, color="#4C78A8", alpha=0.8)
            ax.axvline(thr, color="#E45756", linestyle="--", linewidth=1.6)
            ax.set_title(f"{module} | outliers={(vals > thr).mean() * 100:.2f}%")
            ax.set_xlabel("Activation strength (sqrt scaler_row)")
            ax.set_ylabel("Feature count")
            ax.grid(True, alpha=0.2)

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Layer {layer_idx}: Activation Outlier Distribution", fontsize=14)
        fig.tight_layout()
        out_path = output_dir / f"layer_{layer_idx:02d}_outlier_distribution.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def run_and_save_analysis(
    model,
    inps,
    outs,
    attention_mask,
    position_ids,
    threshold_mode,
    threshold_scale,
    threshold_percentile,
    bins,
    output_dir: Path,
    save_raw_json: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = output_dir / "per_layer_histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    stats, raw_values, raw_thresholds = collect_layerwise_outliers(
        model=model,
        inps=inps,
        outs=outs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        nsamples=inps.shape[0],
        threshold_mode=threshold_mode,
        threshold_scale=threshold_scale,
        threshold_percentile=threshold_percentile,
    )

    stats_json_path = output_dir / "layerwise_outlier_stats.json"
    stats_csv_path = output_dir / "layerwise_outlier_stats.csv"
    ratio_plot_path = output_dir / "layerwise_outlier_ratio.png"

    save_stats_json(stats, stats_json_path)
    save_stats_csv(stats, stats_csv_path)
    plot_layerwise_ratios(stats, ratio_plot_path)
    plot_per_layer_histograms(raw_values, raw_thresholds, hist_dir, bins=bins)

    if save_raw_json:
        raw_json_path = output_dir / "raw_activation_strengths.json"
        save_raw_values_json(raw_values, raw_json_path)
        print(f"Saved raw activation strengths to: {raw_json_path}")

    print(f"Saved stats JSON: {stats_json_path}")
    print(f"Saved stats CSV: {stats_csv_path}")
    print(f"Saved ratio plot: {ratio_plot_path}")
    print(f"Saved per-layer histograms in: {hist_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze activation outliers layer-by-layer for LLaMA/OPT on C4 calibration data.")
    parser.add_argument("--model", type=str, default="decapoda-research/llama-7b-hf", help="HF model id/path")
    parser.add_argument("--cache_dir", type=str, default="llm_weights", help="Model cache directory")
    parser.add_argument("--seed", type=int, default=0, help="Seed for C4 sample selection")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of C4 calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length for calibration")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--threshold_mode", type=str, default="mean", choices=["mean", "max", "percentile"])
    parser.add_argument("--threshold_scale", type=float, default=3.0, help="Scale factor for mean/max threshold modes")
    parser.add_argument("--threshold_percentile", type=float, default=99.5, help="Percentile used when threshold_mode=percentile")
    parser.add_argument("--bins", type=int, default=80, help="Histogram bins")
    parser.add_argument("--output_dir", type=str, default="OWL/save_test/outlier_analysis", help="Output directory")
    parser.add_argument("--save_raw_json", action="store_true", help="Save raw per-feature activation strengths to JSON")
    parser.add_argument(
        "--per_sample",
        action="store_true",
        help="Run full outlier analysis independently for each calibration sample and save one folder per sample.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples in --per_sample mode (default: all nsamples).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}")
    dtype = get_dtype(args.dtype)
    model_kwargs = {
        "cache_dir": args.cache_dir,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **model_kwargs)

    model.seqlen = args.seqlen
    model.eval()

    if args.per_sample:
        requested = args.max_samples if args.max_samples is not None else args.nsamples
        requested = min(max(int(requested), 1), int(args.nsamples))
        print(f"Loading {requested} unique C4 calibration samples for per-sample analysis...")
        dataloader = get_unique_c4_samples(
            nsamples=requested,
            seed=args.seed,
            seqlen=args.seqlen,
            tokenizer=tokenizer,
        )
    else:
        print("Loading C4 calibration set...")
        dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)

    write_hyperparameters(output_dir, args, loaded_samples=len(dataloader))

    start_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Preparing layer input activations...")
    inps, outs, attention_mask, position_ids = prepare_calibration_input(
        model=model,
        dataloader=dataloader,
        device=start_device,
        nsamples=len(dataloader),
    )

    if args.per_sample:
        total_samples = int(inps.shape[0])
        num_to_run = total_samples if args.max_samples is None else min(max(args.max_samples, 0), total_samples)
        print(f"Collecting per-layer outlier distributions per sample ({num_to_run}/{total_samples})...")
        for sample_idx in range(num_to_run):
            print(f"Running sample {sample_idx + 1}/{num_to_run}")
            sample_dir = output_dir / f"sample_{sample_idx:03d}"
            sample_inps = inps[sample_idx : sample_idx + 1].clone()
            sample_outs = torch.zeros_like(sample_inps)
            run_and_save_analysis(
                model=model,
                inps=sample_inps,
                outs=sample_outs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                threshold_mode=args.threshold_mode,
                threshold_scale=args.threshold_scale,
                threshold_percentile=args.threshold_percentile,
                bins=args.bins,
                output_dir=sample_dir,
                save_raw_json=args.save_raw_json,
            )
    else:
        print("Collecting per-layer outlier distributions...")
        run_and_save_analysis(
            model=model,
            inps=inps,
            outs=outs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            threshold_mode=args.threshold_mode,
            threshold_scale=args.threshold_scale,
            threshold_percentile=args.threshold_percentile,
            bins=args.bins,
            output_dir=output_dir,
            save_raw_json=args.save_raw_json,
        )


if __name__ == "__main__":
    main()
