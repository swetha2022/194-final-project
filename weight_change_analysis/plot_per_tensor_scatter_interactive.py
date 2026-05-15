#!/usr/bin/env python3
"""
Interactive HTML scatter plot of per-tensor weight-change norms.

Same inputs as ``plot_per_tensor_scatter.py`` (per-tensor CSVs produced by
``run_analysis.py --per-tensor-norms``) but the output is a single
self-contained ``.html`` file using Plotly.js loaded from a CDN. The HTML
exposes a custom side legend that supports:

  * hover over a tensor-type or model entry -> non-matching points dim
  * click an entry         -> dim filter sticks until you click again (toggle)
  * click the eye icon     -> completely hide that tensor type / model from
                              the plot (toggle); the axes auto-rescale
  * axis scale toggle      -> switch each axis between linear and log at
                              runtime (initial state comes from --log-x/--log-y)
  * outlier toggle         -> percentile-based outlier filter computed within
                              each tensor category (defaults to 1st-99th
                              percentile on each axis); flip on/off in the
                              browser, initial state from --hide-outliers
  * point hover            -> tooltip with tensor name, model, and both norms
  * standard Plotly toolbar (zoom, pan, save to PNG)

The artifact is intended for embedding in a static blog (drop the file into
your site, or load it via ``<iframe>``).

Example:

  python3 weight_change_analysis/plot_per_tensor_scatter_interactive.py \\
      --csvs weight_change_analysis_output/per_tensor_norms/*ckpt_driving_{adamw,muon}.csv \\
      --labels 'Adam pre / AdamW LoRA' 'Adam pre / Muon LoRA' \\
               'Muon pre / AdamW LoRA' 'Muon pre / Muon LoRA' \\
      --x-norm l2 --y-norm l_inf --log-x --log-y \\
      --title 'LoRA finetuning: per-tensor weight changes' \\
      -o weight_change_analysis_output/per_tensor_scatter_lora.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from weight_change_analysis.plot_per_tensor_scatter import (  # noqa: E402
    CATEGORY_LABEL,
    CATEGORY_MARKER,
    CATEGORY_ORDER,
    PER_TENSOR_COLUMN,
    canonical_norm,
    categorize_tensor,
)

# matplotlib marker char -> Plotly marker symbol name.
PLOTLY_SYMBOL: dict[str, str] = {
    "*": "star",
    "^": "triangle-up",
    "v": "triangle-down",
    "D": "diamond",
    "o": "circle",
    "P": "cross",
    "s": "square",
    "X": "x",
}

# Plain-Unicode axis labels (no MathJax dependency in the HTML).
AXIS_LABEL: dict[str, str] = {
    "l_inf": "‖ΔW‖<sub>∞</sub>  (per tensor)",
    "l2": "‖ΔW‖<sub>F</sub>  (per tensor)",
    "rms": "‖ΔW‖<sub>RMS→RMS</sub>  (per 2D tensor)",
}

# Model color palette (categorical, color-blind friendly enough for ~8 series).
MODEL_PALETTE: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def load_rows(csv_path: Path, x_col: str, y_col: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    meta: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or x_col not in reader.fieldnames or y_col not in reader.fieldnames:
            raise SystemExit(
                f"{csv_path}: missing required column(s). "
                f"Need {x_col!r} and {y_col!r}; found {reader.fieldnames!r}."
            )
        for row in reader:
            if not meta:
                meta = {
                    "run_folder": row.get("run_folder", "") or csv_path.stem,
                    "pretrain_optimizer": row.get("pretrain_optimizer", ""),
                    "finetune_type": row.get("finetune_type", ""),
                    "finetune_optimizer": row.get("finetune_optimizer", ""),
                    "name_suffix": row.get("name_suffix", ""),
                }
            try:
                xv = float(row[x_col])
                yv = float(row[y_col])
            except ValueError:
                continue
            if math.isnan(xv) or math.isnan(yv):
                continue
            rows.append(
                {
                    "x": xv,
                    "y": yv,
                    "name": row.get("tensor_name", ""),
                    "shape": row.get("tensor_shape", ""),
                    "cat": categorize_tensor(row.get("tensor_name", "")),
                }
            )
    return rows, meta


def apply_log_filter(
    rows: list[dict[str, Any]],
    log_x: bool,
    log_y: bool,
    log_floor: float | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        x, y = r["x"], r["y"]
        if log_floor is not None:
            if log_x and x <= 0:
                x = log_floor
            if log_y and y <= 0:
                y = log_floor
        else:
            if (log_x and x <= 0) or (log_y and y <= 0):
                continue
        out.append({**r, "x": x, "y": y})
    return out


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    if not sorted_values:
        return 0.0
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def compute_outlier_mask(
    rows_per_csv: list[list[dict[str, Any]]],
    low_pct: float,
    high_pct: float,
) -> tuple[list[list[bool]], dict[str, dict[str, float]]]:
    """
    Mark a point as an outlier if its x or y falls outside the [low_pct, high_pct]
    percentile range computed WITHIN ITS TENSOR CATEGORY across all CSVs.

    Computing thresholds per category (rather than globally) keeps the filter
    meaningful: Q-projection outliers are judged against other Q projections,
    MLP outliers against other MLP weights, etc. - so the categories with
    naturally smaller deltas don't get swamped by the heavy-tailed MLP cloud.

    Returns per-CSV masks plus a per-category thresholds dict
    ``{category: {"x_lo": ..., "x_hi": ..., "y_lo": ..., "y_hi": ..., "n": ...}}``.
    """
    empty: list[list[bool]] = [[False] * len(rows) for rows in rows_per_csv]
    thresholds: dict[str, dict[str, float]] = {}

    if low_pct >= high_pct or (low_pct <= 0 and high_pct >= 100):
        return empty, thresholds

    # Pool x / y values per category across all CSVs.
    by_cat_x: dict[str, list[float]] = {}
    by_cat_y: dict[str, list[float]] = {}
    for rows in rows_per_csv:
        for r in rows:
            cat = r["cat"]
            by_cat_x.setdefault(cat, []).append(r["x"])
            by_cat_y.setdefault(cat, []).append(r["y"])
    if not by_cat_x:
        return empty, thresholds

    for cat, xs in by_cat_x.items():
        sx = sorted(xs)
        sy = sorted(by_cat_y[cat])
        thresholds[cat] = {
            "x_lo": _percentile(sx, low_pct),
            "x_hi": _percentile(sx, high_pct),
            "y_lo": _percentile(sy, low_pct),
            "y_hi": _percentile(sy, high_pct),
            "n": float(len(xs)),
        }

    masks: list[list[bool]] = []
    for rows in rows_per_csv:
        m: list[bool] = []
        for r in rows:
            t = thresholds.get(r["cat"])
            if t is None:
                m.append(False)
                continue
            x, y = r["x"], r["y"]
            m.append(
                (x < t["x_lo"]) or (x > t["x_hi"]) or (y < t["y_lo"]) or (y > t["y_hi"])
            )
        masks.append(m)
    return masks, thresholds


def build_traces(
    rows_per_csv: list[list[dict[str, Any]]],
    outlier_masks: list[list[bool]],
    labels: list[str],
    colors: list[str],
    x_key: str,
    y_key: str,
    marker_size: float,
    alpha: float,
    webgl: bool,
    hide_outliers_initial: bool,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """
    One trace per (model, tensor category). Each trace stores ``meta`` so the
    custom legend's hover/click handlers can match by model or category.
    """
    traces: list[dict[str, Any]] = []
    present_categories: list[str] = []
    for rows, mask, label, color in zip(rows_per_csv, outlier_masks, labels, colors):
        by_cat: dict[str, list[tuple[dict[str, Any], bool]]] = {}
        for r, is_out in zip(rows, mask):
            by_cat.setdefault(r["cat"], []).append((r, is_out))
        for cat in CATEGORY_ORDER:
            if cat not in by_cat:
                continue
            if cat not in present_categories:
                present_categories.append(cat)
            pts = by_cat[cat]
            x = [p[0]["x"] for p in pts]
            y = [p[0]["y"] for p in pts]
            names = [p[0]["name"] for p in pts]
            shapes = [p[0]["shape"] for p in pts]
            outlier_flags = [bool(p[1]) for p in pts]
            # Per-point initial marker opacity: 0 for outliers if they should
            # start hidden, otherwise alpha. JS keeps this array in sync with
            # the outlier + dim-filter state.
            initial_opacity = [
                (0.0 if (hide_outliers_initial and o) else alpha)
                for o in outlier_flags
            ]
            trace = {
                "type": "scattergl" if webgl else "scatter",
                "mode": "markers",
                "x": x,
                "y": y,
                "name": f"{label} \u2013 {CATEGORY_LABEL[cat]}",
                "showlegend": False,  # we render our own legend
                "opacity": 1.0,  # trace-level; reserved for dim filter
                "marker": {
                    "size": marker_size,
                    "color": color,
                    "symbol": PLOTLY_SYMBOL[CATEGORY_MARKER[cat]],
                    "opacity": initial_opacity,
                    "line": {"width": 0},
                },
                "hovertemplate": (
                    f"<b>%{{customdata[0]}}</b><br>"
                    f"shape: %{{customdata[1]}}<br>"
                    f"model: {html.escape(label)}<br>"
                    f"type: {html.escape(CATEGORY_LABEL[cat])}<br>"
                    f"{x_key}: %{{x:.4g}}<br>"
                    f"{y_key}: %{{y:.4g}}<extra></extra>"
                ),
                "customdata": list(zip(names, shapes)),
                "meta": {
                    "model": label,
                    "cat": cat,
                    "outlierMask": outlier_flags,
                },
            }
            traces.append(trace)
    return traces, [l for l in labels], present_categories


_LATEX_TO_UNICODE = {
    r"\rightarrow": "→",
    r"\to": "→",
    r"\leftarrow": "←",
    r"\Delta": "Δ",
    r"\delta": "δ",
    r"\infty": "∞",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\sigma": "σ",
    r"\theta": "θ",
    r"\|": "‖",
    r"\,": " ",
    r"\;": " ",
    r"\:": " ",
}


def latex_to_plain(s: str) -> str:
    """Best-effort strip of LaTeX markup for use in browser-tab titles."""
    # Strip $...$ delimiters but keep the contents.
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    for tex, uni in _LATEX_TO_UNICODE.items():
        s = s.replace(tex, uni)
    # Drop any remaining \cmd tokens.
    s = re.sub(r"\\[A-Za-z]+", "", s)
    # Remove math grouping characters / subscript carets.
    s = re.sub(r"[{}]", "", s)
    s = s.replace("_", "").replace("^", "")
    return re.sub(r"\s+", " ", s).strip()


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__PAGE_TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"></script>
<style>
  :root {
    --bg: #ffffff;
    --fg: #1f2937;
    --muted: #6b7280;
    --border: #e5e7eb;
    --panel: #f9fafb;
    --accent: #2563eb;
    --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 1px 3px rgba(0,0,0,0.06);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f172a;
      --fg: #e5e7eb;
      --muted: #94a3b8;
      --border: #1f2937;
      --panel: #111827;
      --accent: #60a5fa;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 1px 3px rgba(0,0,0,0.4);
    }
  }
  html, body { background: var(--bg); color: var(--fg); }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    line-height: 1.4;
  }
  .wrap {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 280px;
    gap: 16px;
    padding: 16px;
    max-width: 1200px;
    margin: 0 auto;
  }
  @media (max-width: 800px) {
    .wrap { grid-template-columns: 1fr; }
  }
  .title { font-size: 16px; font-weight: 600; margin: 0 0 4px 0; }
  .subtitle { color: var(--muted); margin: 0 0 12px 0; font-size: 12px; }
  #plot { width: 100%; height: 620px; }
  .legend-box {
    border: 1px solid var(--border);
    background: var(--panel);
    border-radius: 8px;
    padding: 12px 14px;
    box-shadow: var(--shadow);
  }
  .legend-box + .legend-box { margin-top: 12px; }
  .legend-title {
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin: 0 0 8px 0;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 6px;
    border-radius: 4px;
    cursor: pointer;
    user-select: none;
    border: 1px solid transparent;
  }
  .legend-item:hover { background: rgba(127,127,127,0.08); }
  .legend-item.active {
    border-color: var(--accent);
    background: rgba(37,99,235,0.08);
  }
  .legend-item.hidden .swatch,
  .legend-item.hidden .legend-label,
  .legend-item.hidden .legend-count {
    opacity: 0.35;
    text-decoration: line-through;
  }
  .swatch { width: 14px; height: 14px; flex: 0 0 14px; }
  .swatch svg { display: block; }
  .legend-label { flex: 1; }
  .legend-count {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    font-size: 12px;
    margin-right: 2px;
  }
  .legend-eye {
    width: 18px; height: 18px;
    flex: 0 0 18px;
    display: flex; align-items: center; justify-content: center;
    color: var(--muted);
    border-radius: 3px;
    cursor: pointer;
  }
  .legend-eye:hover { color: var(--fg); background: rgba(127,127,127,0.12); }
  .legend-eye svg { display: block; }
  .legend-hint { color: var(--muted); font-size: 11px; margin: 8px 0 0 0; }
  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
    margin: 0 0 8px 0;
    font-size: 12px;
    color: var(--muted);
  }
  .controls .group { display: inline-flex; align-items: center; gap: 6px; }
  .controls .label { font-weight: 600; color: var(--muted); }
  .seg {
    display: inline-flex;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    background: var(--panel);
  }
  .seg button {
    appearance: none;
    background: transparent;
    color: var(--fg);
    border: 0;
    padding: 4px 10px;
    font: inherit;
    cursor: pointer;
    line-height: 1.4;
  }
  .seg button + button { border-left: 1px solid var(--border); }
  .seg button:hover { background: rgba(127,127,127,0.10); }
  .seg button.active {
    background: var(--accent);
    color: white;
  }
  .controls .muted {
    color: var(--muted);
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }
</style>
</head>
<body>
<div class="wrap">
  <div>
    <h1 class="title math">__TITLE__</h1>
    <p class="subtitle math">__SUBTITLE__</p>
    <div class="controls">
      <span class="group">
        <span class="label">X axis</span>
        <span class="seg" id="seg-x" role="group" aria-label="X axis scale">
          <button type="button" data-axis="x" data-type="linear">Linear</button>
          <button type="button" data-axis="x" data-type="log">Log</button>
        </span>
      </span>
      <span class="group">
        <span class="label">Y axis</span>
        <span class="seg" id="seg-y" role="group" aria-label="Y axis scale">
          <button type="button" data-axis="y" data-type="linear">Linear</button>
          <button type="button" data-axis="y" data-type="log">Log</button>
        </span>
      </span>
      <span class="group">
        <span class="label">Outliers</span>
        <span class="seg" id="seg-outliers" role="group" aria-label="Outlier filter">
          <button type="button" data-outliers="show">Show</button>
          <button type="button" data-outliers="hide">Hide</button>
        </span>
        <span class="muted" id="outlier-status"></span>
      </span>
    </div>
    <div id="plot"></div>
  </div>
  <div>
    <div class="legend-box">
      <div class="legend-title">Model</div>
      <div id="legend-model"></div>
    </div>
    <div class="legend-box">
      <div class="legend-title">Tensor type</div>
      <div id="legend-type"></div>
      <p class="legend-hint">
        Hover or click row to highlight / pin.
        Click the eye to fully hide.
      </p>
    </div>
  </div>
</div>
<script>
const TRACES = __TRACES__;
const LAYOUT = __LAYOUT__;
const CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
  toImageButtonOptions: { format: 'png', scale: 2 },
};
const MODELS = __MODELS__;
const CATEGORIES = __CATEGORIES__;
const MODEL_COLOR = __MODEL_COLOR__;
const CATEGORY_LABEL = __CATEGORY_LABEL__;
const CATEGORY_SYMBOL = __CATEGORY_SYMBOL__;
const BASE_OPACITY = __BASE_OPACITY__;
const DIM_OPACITY = 0.04;
const INITIAL_AXIS_TYPE = __INITIAL_AXIS_TYPE__;
const INITIAL_HIDE_OUTLIERS = __INITIAL_HIDE_OUTLIERS__;
const OUTLIER_PCTS = __OUTLIER_PCTS__;
const OUTLIER_THRESHOLDS = __OUTLIER_THRESHOLDS__;
const OUTLIER_TOTAL = __OUTLIER_TOTAL__;
const TOTAL_POINTS = __TOTAL_POINTS__;

const plotDiv = document.getElementById('plot');
Plotly.newPlot(plotDiv, TRACES, LAYOUT, CONFIG);

const traceModel = TRACES.map(t => t.meta.model);
const traceCat = TRACES.map(t => t.meta.cat);

let pinnedFilter = null; // { kind: 'model'|'cat', value: string } or null
const hiddenModels = new Set();
const hiddenCats = new Set();
let hideOutliers = !!INITIAL_HIDE_OUTLIERS;

// Pre-extract outlier masks per trace for fast opacity recomputation.
const traceOutlierMask = TRACES.map(t => (t.meta && t.meta.outlierMask) || []);

function isTraceHidden(i) {
  return hiddenModels.has(traceModel[i]) || hiddenCats.has(traceCat[i]);
}

function perTraceFactor(i, filter) {
  // Trace-level dim factor in [DIM_OPACITY/BASE_OPACITY, 1].
  if (!filter) return 1;
  const match = filter.kind === 'model'
    ? traceModel[i] === filter.value
    : traceCat[i] === filter.value;
  return match ? 1 : (DIM_OPACITY / BASE_OPACITY);
}

function computeMarkerOpacityArrays(filter) {
  // Per-point marker.opacity arrays, combining the dim filter with the
  // outlier-hide toggle. We use marker.opacity (per-point) rather than the
  // trace-level opacity so a single restyle call drives both effects.
  return TRACES.map((t, i) => {
    const factor = perTraceFactor(i, filter);
    const base = BASE_OPACITY * factor;
    const mask = traceOutlierMask[i];
    return t.x.map((_, k) => {
      if (hideOutliers && mask[k]) return 0;
      return base;
    });
  });
}

function applyOpacity(filter) {
  const ops = computeMarkerOpacityArrays(filter);
  Plotly.restyle(plotDiv, { 'marker.opacity': ops });
}

function applyVisibility() {
  const vis = TRACES.map((_, i) => isTraceHidden(i) ? false : true);
  Plotly.restyle(plotDiv, { visible: vis });
}

function setPinned(filter) {
  // If the row being clicked is fully hidden, ignore (eye icon is the way to unhide).
  if (filter) {
    if (filter.kind === 'model' && hiddenModels.has(filter.value)) return;
    if (filter.kind === 'cat' && hiddenCats.has(filter.value)) return;
  }
  if (pinnedFilter
      && filter
      && pinnedFilter.kind === filter.kind
      && pinnedFilter.value === filter.value) {
    pinnedFilter = null;
  } else {
    pinnedFilter = filter;
  }
  document.querySelectorAll('.legend-item').forEach(el => {
    const k = el.dataset.kind;
    const v = el.dataset.value;
    const active = pinnedFilter && pinnedFilter.kind === k && pinnedFilter.value === v;
    el.classList.toggle('active', !!active);
  });
  applyOpacity(pinnedFilter);
}

function hoverFilter(filter) {
  if (pinnedFilter) return;
  if (filter) {
    if (filter.kind === 'model' && hiddenModels.has(filter.value)) return;
    if (filter.kind === 'cat' && hiddenCats.has(filter.value)) return;
  }
  applyOpacity(filter);
}

function toggleHidden(kind, value, rowEl) {
  const set = kind === 'model' ? hiddenModels : hiddenCats;
  const willHide = !set.has(value);
  if (willHide) {
    set.add(value);
    // If the row we just hid is currently pinned, clear the pin.
    if (pinnedFilter && pinnedFilter.kind === kind && pinnedFilter.value === value) {
      pinnedFilter = null;
      rowEl.classList.remove('active');
    }
  } else {
    set.delete(value);
  }
  rowEl.classList.toggle('hidden', willHide);
  // Update eye icon glyph.
  const eye = rowEl.querySelector('.legend-eye');
  if (eye) eye.innerHTML = eyeSvg(!willHide);
  applyVisibility();
  // Re-apply opacity (hidden traces still need consistent stored opacity).
  applyOpacity(pinnedFilter);
}

function eyeSvg(visible) {
  if (visible) {
    return `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M1.5 8 C 3.5 4, 6 2.5, 8 2.5 S 12.5 4, 14.5 8 C 12.5 12, 10 13.5, 8 13.5 S 3.5 12, 1.5 8 Z"/><circle cx="8" cy="8" r="2.2"/></svg>`;
  }
  return `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 4 L 13.5 12"/><path d="M3 9 C 4.5 11.5, 6.5 13, 8 13 C 9.6 13, 11.4 12, 12.7 10.3"/><path d="M5.2 5.6 C 6.1 5.1, 7 4.8, 8 4.8 C 10.5 4.8, 12.7 6.5, 14 9"/></svg>`;
}

function svgSwatch(symbol, color) {
  // Approximate Plotly marker shapes as inline SVG glyphs so the legend
  // matches the data points without bundling extra icons.
  const c = color;
  switch (symbol) {
    case 'star':
      return `<svg width="14" height="14" viewBox="0 0 14 14"><polygon points="7,1 8.6,5.4 13.3,5.4 9.5,8.2 11,12.6 7,9.9 3,12.6 4.5,8.2 0.7,5.4 5.4,5.4" fill="${c}"/></svg>`;
    case 'triangle-up':
      return `<svg width="14" height="14" viewBox="0 0 14 14"><polygon points="7,1.5 12.5,12 1.5,12" fill="${c}"/></svg>`;
    case 'triangle-down':
      return `<svg width="14" height="14" viewBox="0 0 14 14"><polygon points="7,12.5 12.5,2 1.5,2" fill="${c}"/></svg>`;
    case 'diamond':
      return `<svg width="14" height="14" viewBox="0 0 14 14"><polygon points="7,1 13,7 7,13 1,7" fill="${c}"/></svg>`;
    case 'circle':
      return `<svg width="14" height="14" viewBox="0 0 14 14"><circle cx="7" cy="7" r="5.5" fill="${c}"/></svg>`;
    case 'cross':
      return `<svg width="14" height="14" viewBox="0 0 14 14"><polygon points="5,1 9,1 9,5 13,5 13,9 9,9 9,13 5,13 5,9 1,9 1,5 5,5" fill="${c}"/></svg>`;
    case 'square':
      return `<svg width="14" height="14" viewBox="0 0 14 14"><rect x="1.5" y="1.5" width="11" height="11" fill="${c}"/></svg>`;
    case 'x':
      return `<svg width="14" height="14" viewBox="0 0 14 14"><path d="M2,2 L12,12 M12,2 L2,12" stroke="${c}" stroke-width="3" stroke-linecap="round"/></svg>`;
    default:
      return `<svg width="14" height="14" viewBox="0 0 14 14"><circle cx="7" cy="7" r="5.5" fill="${c}"/></svg>`;
  }
}

function countPointsFor(filter) {
  let n = 0;
  TRACES.forEach((t, i) => {
    const match = filter.kind === 'model'
      ? traceModel[i] === filter.value
      : traceCat[i] === filter.value;
    if (match) n += t.x.length;
  });
  return n;
}

function buildLegend(container, items) {
  container.innerHTML = '';
  items.forEach(it => {
    const el = document.createElement('div');
    el.className = 'legend-item';
    el.dataset.kind = it.kind;
    el.dataset.value = it.value;
    el.innerHTML =
      `<span class="swatch">${it.swatchHtml}</span>` +
      `<span class="legend-label"></span>` +
      `<span class="legend-count"></span>` +
      `<span class="legend-eye" role="button" title="Show / hide" aria-label="Show or hide">${eyeSvg(true)}</span>`;
    el.querySelector('.legend-label').textContent = it.label;
    el.querySelector('.legend-count').textContent = it.count.toLocaleString();
    el.addEventListener('mouseenter', () => hoverFilter({ kind: it.kind, value: it.value }));
    el.addEventListener('mouseleave', () => hoverFilter(null));
    el.addEventListener('click', (ev) => {
      // Eye icon has its own handler; don't treat clicks inside it as a pin toggle.
      if (ev.target.closest('.legend-eye')) return;
      setPinned({ kind: it.kind, value: it.value });
    });
    el.querySelector('.legend-eye').addEventListener('click', (ev) => {
      ev.stopPropagation();
      toggleHidden(it.kind, it.value, el);
    });
    container.appendChild(el);
  });
}

buildLegend(
  document.getElementById('legend-model'),
  MODELS.map(m => ({
    kind: 'model',
    value: m,
    label: m,
    swatchHtml: svgSwatch('circle', MODEL_COLOR[m]),
    count: countPointsFor({ kind: 'model', value: m }),
  }))
);

buildLegend(
  document.getElementById('legend-type'),
  CATEGORIES.map(c => ({
    kind: 'cat',
    value: c,
    label: CATEGORY_LABEL[c],
    swatchHtml: svgSwatch(CATEGORY_SYMBOL[c], 'currentColor'),
    count: countPointsFor({ kind: 'cat', value: c }),
  }))
);

// --- Axis scale toggle (linear / log) ---------------------------------------
const axisState = { x: INITIAL_AXIS_TYPE.x, y: INITIAL_AXIS_TYPE.y };

function refreshAxisButtons() {
  document.querySelectorAll('.seg button[data-axis]').forEach(btn => {
    const ax = btn.dataset.axis;
    btn.classList.toggle('active', btn.dataset.type === axisState[ax]);
  });
}

function setAxisType(axis, type) {
  if (axisState[axis] === type) return;
  axisState[axis] = type;
  const key = axis === 'x' ? 'xaxis.type' : 'yaxis.type';
  // autorange:true also forces Plotly to recompute the range for the new scale.
  const update = {};
  update[key] = type;
  update[axis === 'x' ? 'xaxis.autorange' : 'yaxis.autorange'] = true;
  Plotly.relayout(plotDiv, update);
  refreshAxisButtons();
}

document.querySelectorAll('.seg button[data-axis]').forEach(btn => {
  btn.addEventListener('click', () => setAxisType(btn.dataset.axis, btn.dataset.type));
});
refreshAxisButtons();

// --- Outlier toggle ---------------------------------------------------------
function refreshOutlierButtons() {
  document.querySelectorAll('.seg button[data-outliers]').forEach(btn => {
    const state = btn.dataset.outliers === 'hide' ? hideOutliers : !hideOutliers;
    btn.classList.toggle('active', state);
  });
}

function refreshOutlierStatus() {
  const el = document.getElementById('outlier-status');
  if (!el) return;
  if (OUTLIER_TOTAL <= 0) {
    el.textContent = '';
    return;
  }
  const lo = OUTLIER_PCTS.low.toString();
  const hi = OUTLIER_PCTS.high.toString();
  if (hideOutliers) {
    el.textContent = `${OUTLIER_TOTAL.toLocaleString()} hidden (outside p${lo}–p${hi})`;
  } else {
    el.textContent = `${OUTLIER_TOTAL.toLocaleString()} flagged (outside p${lo}–p${hi})`;
  }
}

function setHideOutliers(value) {
  const next = !!value;
  if (hideOutliers === next) return;
  hideOutliers = next;
  refreshOutlierButtons();
  refreshOutlierStatus();
  applyOpacity(pinnedFilter);
}

document.querySelectorAll('.seg button[data-outliers]').forEach(btn => {
  btn.addEventListener('click', () => setHideOutliers(btn.dataset.outliers === 'hide'));
});
refreshOutlierButtons();
refreshOutlierStatus();

// --- KaTeX auto-render for the title and subtitle ---------------------------
// Defer-loaded katex + auto-render scripts have executed by the time
// DOMContentLoaded fires, so renderMathInElement is guaranteed to be defined.
function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  const opts = {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
      { left: '\\\\(', right: '\\\\)', display: false },
      { left: '\\\\[', right: '\\\\]', display: true }
    ],
    throwOnError: false,
    strict: 'ignore'
  };
  document.querySelectorAll('.math').forEach(el => renderMathInElement(el, opts));
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', renderMath);
} else {
  renderMath();
}
</script>
</body>
</html>
"""


def render_html(
    traces: list[dict[str, Any]],
    layout: dict[str, Any],
    models: list[str],
    categories: list[str],
    model_color: dict[str, str],
    base_opacity: float,
    title: str,
    subtitle: str,
    initial_axis_type: dict[str, str],
    initial_hide_outliers: bool,
    outlier_pcts: dict[str, float],
    outlier_thresholds: dict[str, dict[str, float]],
    outlier_total: int,
    total_points: int,
) -> str:
    category_label = {k: CATEGORY_LABEL[k] for k in categories}
    category_symbol = {k: PLOTLY_SYMBOL[CATEGORY_MARKER[k]] for k in categories}

    def dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    out = _HTML_TEMPLATE
    out = out.replace("__PAGE_TITLE__", html.escape(latex_to_plain(title)))
    # Title / subtitle keep $...$ so KaTeX can render them client-side.
    # html.escape only escapes < > & " '; $ and \ pass through unchanged,
    # so KaTeX still sees valid math delimiters.
    out = out.replace("__TITLE__", html.escape(title))
    out = out.replace("__SUBTITLE__", html.escape(subtitle))
    out = out.replace("__TRACES__", dumps(traces))
    out = out.replace("__LAYOUT__", dumps(layout))
    out = out.replace("__MODELS__", dumps(models))
    out = out.replace("__CATEGORIES__", dumps(categories))
    out = out.replace("__MODEL_COLOR__", dumps(model_color))
    out = out.replace("__CATEGORY_LABEL__", dumps(category_label))
    out = out.replace("__CATEGORY_SYMBOL__", dumps(category_symbol))
    out = out.replace("__BASE_OPACITY__", dumps(base_opacity))
    out = out.replace("__INITIAL_AXIS_TYPE__", dumps(initial_axis_type))
    out = out.replace("__INITIAL_HIDE_OUTLIERS__", dumps(bool(initial_hide_outliers)))
    out = out.replace("__OUTLIER_PCTS__", dumps(outlier_pcts))
    out = out.replace("__OUTLIER_THRESHOLDS__", dumps(outlier_thresholds))
    out = out.replace("__OUTLIER_TOTAL__", dumps(int(outlier_total)))
    out = out.replace("__TOTAL_POINTS__", dumps(int(total_points)))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generate an interactive HTML scatter plot of per-tensor weight-change "
            "norms. Hover/click the side legend to filter by model or tensor type."
        )
    )
    p.add_argument("--csvs", type=Path, nargs="+", required=True, metavar="CSV")
    p.add_argument("--labels", type=str, nargs="+", default=None, metavar="LABEL")
    p.add_argument("--x-norm", type=str, default="l2")
    p.add_argument("--y-norm", type=str, default="l_inf")
    p.add_argument("--log-x", action="store_true")
    p.add_argument("--log-y", action="store_true")
    p.add_argument(
        "--log-floor",
        type=float,
        default=1e-12,
        help="Replace zero/non-positive values with this floor on log axes. "
        "Set <=0 to drop those points instead.",
    )
    p.add_argument(
        "--outlier-low",
        type=float,
        default=1.0,
        metavar="P",
        help="Lower percentile (0..100) for the outlier filter (default 1).",
    )
    p.add_argument(
        "--outlier-high",
        type=float,
        default=99.0,
        metavar="P",
        help="Upper percentile (0..100) for the outlier filter (default 99). "
        "A point is an outlier if either x or y falls outside [low, high] "
        "across all plotted points.",
    )
    p.add_argument(
        "--hide-outliers",
        action="store_true",
        help="Start with outliers hidden. They can always be toggled back on "
        "with the Outliers button in the controls bar.",
    )
    p.add_argument("--alpha", type=float, default=0.6, help="Base marker opacity (0..1).")
    p.add_argument("--marker-size", type=float, default=7.0, help="Marker size in pixels.")
    p.add_argument(
        "--no-webgl",
        action="store_true",
        help="Use SVG scatter instead of WebGL (slower for many points, more compatible).",
    )
    p.add_argument("--title", type=str, default=None)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Default: weight_change_analysis_output/per_tensor_scatter_<ynorm>_vs_<xnorm>.html",
    )
    args = p.parse_args(argv)

    x_key = canonical_norm(args.x_norm)
    y_key = canonical_norm(args.y_norm)
    x_col = PER_TENSOR_COLUMN[x_key]
    y_col = PER_TENSOR_COLUMN[y_key]

    csv_paths = [c.resolve() for c in args.csvs]
    missing = [p for p in csv_paths if not p.is_file()]
    if missing:
        for m in missing:
            print(f"CSV not found: {m}", file=sys.stderr)
        return 1
    if args.labels is not None and len(args.labels) != len(csv_paths):
        print(
            f"--labels has {len(args.labels)} entries but --csvs has {len(csv_paths)}.",
            file=sys.stderr,
        )
        return 1

    log_floor = args.log_floor if args.log_floor > 0 else None

    rows_per_csv: list[list[dict[str, Any]]] = []
    metas: list[dict[str, str]] = []
    for path in csv_paths:
        rows, meta = load_rows(path, x_col, y_col)
        # Always normalize non-positive values (floor or drop) on BOTH axes so the
        # in-browser axis-scale toggle can switch to log without breaking points.
        rows = apply_log_filter(rows, log_x=True, log_y=True, log_floor=log_floor)
        if not rows:
            print(f"[warn] {path}: no points after filtering", file=sys.stderr)
        rows_per_csv.append(rows)
        metas.append(meta)

    labels = (
        args.labels
        if args.labels is not None
        else [m.get("run_folder") or p.stem for m, p in zip(metas, csv_paths)]
    )
    colors = [MODEL_PALETTE[i % len(MODEL_PALETTE)] for i in range(len(csv_paths))]
    model_color = dict(zip(labels, colors))

    outlier_masks, outlier_thresholds = compute_outlier_mask(
        rows_per_csv, args.outlier_low, args.outlier_high
    )

    traces, models, categories = build_traces(
        rows_per_csv=rows_per_csv,
        outlier_masks=outlier_masks,
        labels=labels,
        colors=colors,
        x_key=x_key,
        y_key=y_key,
        marker_size=args.marker_size,
        alpha=args.alpha,
        webgl=not args.no_webgl,
        hide_outliers_initial=args.hide_outliers,
    )

    total_outliers = sum(sum(1 for f in m if f) for m in outlier_masks)
    total_points = sum(len(m) for m in outlier_masks)

    if not traces:
        print("No data to plot.", file=sys.stderr)
        return 1

    title = args.title or f"Per-tensor weight-change norms: {y_key} vs {x_key}"
    subtitle = (
        f"{sum(len(r) for r in rows_per_csv):,} points across "
        f"{len(csv_paths)} model(s). "
        "Color = model · marker shape = tensor type."
    )

    layout: dict[str, Any] = {
        "xaxis": {
            "title": {"text": AXIS_LABEL[x_key]},
            "type": "log" if args.log_x else "linear",
            "gridcolor": "rgba(127,127,127,0.18)",
            "zerolinecolor": "rgba(127,127,127,0.4)",
            "showspikes": False,
        },
        "yaxis": {
            "title": {"text": AXIS_LABEL[y_key]},
            "type": "log" if args.log_y else "linear",
            "gridcolor": "rgba(127,127,127,0.18)",
            "zerolinecolor": "rgba(127,127,127,0.4)",
            "showspikes": False,
        },
        "margin": {"l": 64, "r": 16, "t": 12, "b": 56},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "hovermode": "closest",
        "showlegend": False,
        "font": {"family": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"},
    }

    html_doc = render_html(
        traces=traces,
        layout=layout,
        models=models,
        categories=categories,
        model_color=model_color,
        base_opacity=args.alpha,
        title=title,
        subtitle=subtitle,
        initial_axis_type={
            "x": "log" if args.log_x else "linear",
            "y": "log" if args.log_y else "linear",
        },
        initial_hide_outliers=args.hide_outliers,
        outlier_pcts={"low": args.outlier_low, "high": args.outlier_high},
        outlier_thresholds=outlier_thresholds,
        outlier_total=total_outliers,
        total_points=total_points,
    )

    out_path = args.output
    if out_path is None:
        default_dir = Path("weight_change_analysis_output")
        default_dir.mkdir(parents=True, exist_ok=True)
        out_path = default_dir / f"per_tensor_scatter_{y_key}_vs_{x_key}.html"
    else:
        out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
