from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

Palette = {
    "blue": "#4c72b0",
    "red": "#c44e52",
    "green": "#55a868",
    "orange": "#dd8452",
    "purple": "#8172b2",
    "gray": "#b0b0b0",
}

# Set a clean default style (seaborn) with smaller text
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
plt.rcParams.update({
    "axes.facecolor": "#f8f9fb",
    "figure.facecolor": "#f8f9fb",
    "axes.edgecolor": "#d0d7de",
    "grid.color": "#d0d7de",
    "grid.alpha": 0.6,
    "axes.titleweight": "bold",
})


def _shade_missing(ax, dates: pd.Series, mask: pd.Series, label: str, alpha: float = 0.12):
    """Shade continuous ranges where mask is True."""
    if mask is None or mask.empty or mask.sum() == 0:
        return
    spans = []
    current_start = None
    current_end = None
    for d, missing in zip(dates, mask):
        if missing and current_start is None:
            current_start = d
            current_end = d
        elif missing:
            current_end = d
        elif current_start is not None:
            spans.append((current_start, current_end))
            current_start = None
            current_end = None
    if current_start is not None:
        spans.append((current_start, current_end))

    for s, e in spans:
        ax.axvspan(s, e, color=Palette["gray"], alpha=alpha, label=label)
        label = None  # only label first span to avoid legend clutter


def _format_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _sig_marker(p: float) -> str:
    if p is None or pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "·"
    return ""


def plot_daily_steps_and_mood(daily_metrics: pd.DataFrame, mood_daily: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if daily_metrics.empty or mood_daily.empty:
        return None
    merged = daily_metrics.merge(mood_daily, on="calendarDate", how="inner").sort_values("calendarDate")
    merged["date"] = pd.to_datetime(merged["calendarDate"])

    # Correlation for annotation
    corr = merged[["steps_total", "mood_mean"]].apply(pd.to_numeric, errors="coerce").corr().iloc[0, 1]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(merged["date"], merged["steps_total"] / 1000, color=Palette["blue"], alpha=0.55, label="Steps (k)")
    ax1.set_ylabel("Steps (thousands)", color=Palette["blue"])
    ax1.tick_params(axis="y", labelcolor=Palette["blue"])
    ax1.grid(True, axis="y", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(merged["date"], merged["mood_mean"], color=Palette["red"], marker="o", linewidth=2.2, label="Mood")
    ax2.set_ylabel("Mean mood (1-5)", color=Palette["red"])
    ax2.tick_params(axis="y", labelcolor=Palette["red"])
    ax2.set_ylim(1, 5)

    _format_date_axis(ax1)
    corr_txt = f"r = {corr:.2f}" if pd.notna(corr) else "r = n/a"
    ax1.set_title(f"Daily steps vs. average mood ({corr_txt})")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    return _save(fig, output_dir, "daily_steps_and_mood")


def plot_rolling_stress_and_mood(daily_metrics: pd.DataFrame, mood_daily: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if daily_metrics.empty or mood_daily.empty:
        return None
    merged = daily_metrics.merge(mood_daily, on="calendarDate", how="inner").sort_values("calendarDate")
    merged["date"] = pd.to_datetime(merged["calendarDate"])
    merged["stress_roll7"] = merged["stress_mean"].rolling(window=7, min_periods=3).mean()
    merged["mood_roll7"] = merged["mood_mean"].rolling(window=7, min_periods=3).mean()

    def zscore(s: pd.Series):
        s = s.astype(float)
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std and not np.isclose(std, 0) else s * 0

    merged["stress_norm"] = zscore(merged["stress_roll7"])
    merged["mood_norm"] = zscore(merged["mood_roll7"])

    # Linear trend on normalized mood
    trend = merged.dropna(subset=["date", "mood_norm"])
    trend_line = None
    if not trend.empty:
        x = mdates.date2num(trend["date"])
        coeffs = np.polyfit(x, trend["mood_norm"], 1)
        trend_line = np.poly1d(coeffs)(mdates.date2num(merged["date"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(merged["date"], merged["stress_norm"], color=Palette["orange"], linewidth=2.0, label="Stress (7d avg, z-score)")
    ax.plot(merged["date"], merged["mood_norm"], color=Palette["green"], linewidth=2.0, label="Mood (7d avg, z-score)")
    ax.fill_between(merged["date"], merged["stress_norm"], color=Palette["orange"], alpha=0.08)
    ax.fill_between(merged["date"], merged["mood_norm"], color=Palette["green"], alpha=0.08)
    if trend_line is not None:
        ax.plot(merged["date"], trend_line, color=Palette["blue"], linestyle="--", linewidth=2.0, label="Mood trend (linear)")

    missing_mask = merged[["stress_norm", "mood_norm"]].isna().any(axis=1)
    _shade_missing(ax, merged["date"], missing_mask, label="Missing data")

    ax.set_title("Stress vs. mood (7-day rolling, normalized)")
    ax.set_ylabel("z-score")
    _format_date_axis(ax)
    ax.legend()

    return _save(fig, output_dir, "rolling_stress_mood")


def plot_body_battery_vs_energy(mood_events: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if mood_events.empty or "body_battery_at_event" not in mood_events.columns:
        return None

    df = mood_events.dropna(subset=["body_battery_at_event", "energy_level"])
    if df.empty:
        return None

    corr = df[["body_battery_at_event", "energy_level"]].apply(pd.to_numeric, errors="coerce").corr().iloc[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x="body_battery_at_event",
        y="energy_level",
        hue="mood",
        palette="viridis",
        s=90,
        edgecolor="white",
        alpha=0.9,
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="body_battery_at_event",
        y="energy_level",
        scatter=False,
        ci=None,
        color=Palette["red"],
        line_kws={"linewidth": 2, "linestyle": "--"},
        ax=ax,
    )

    ax.set_xlabel("Body battery at event")
    ax.set_ylabel("Self-reported energy")
    corr_txt = f"rho = {corr:.2f}" if pd.notna(corr) else "rho = n/a"
    ax.set_title(f"Body battery vs. perceived energy ({corr_txt})")
    ax.legend(loc="lower right", title="Mood")
    sns.despine(ax=ax)

    return _save(fig, output_dir, "body_battery_vs_energy")


def plot_sleep_vs_next_day_mood(
    daily_metrics: pd.DataFrame,
    mood_daily: pd.DataFrame,
    output_dir: Path,
    lag_corr: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[Path]:
    if daily_metrics.empty or mood_daily.empty:
        return None

    mood_shift = mood_daily.copy()
    mood_shift["prev_date"] = (pd.to_datetime(mood_shift["calendarDate"]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    merged = daily_metrics.merge(mood_shift[["prev_date", "mood_mean"]], left_on="calendarDate", right_on="prev_date", how="inner")
    if merged.empty:
        return None

    merged["date"] = pd.to_datetime(merged["calendarDate"])

    perm = _spearman_perm(merged["date"], merged["sleep_hours"], merged["mood_mean"])
    corr = perm.get("rho") if perm else np.nan
    p_val = perm.get("p") if perm else np.nan
    q_val = lag_corr.get("sleep_hours", {}).get("q") if lag_corr else None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(merged["date"], merged["sleep_hours"], color=Palette["purple"], alpha=0.6, label="Sleep (h)")
    ax1.set_ylabel("Sleep duration (hours)", color=Palette["purple"])
    ax1.tick_params(axis="y", labelcolor=Palette["purple"])
    ax1.grid(True, axis="y", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(merged["date"], merged["mood_mean"], color=Palette["green"], marker="o", linewidth=2.2, label="Next-day mood")
    ax2.set_ylabel("Next-day mood", color=Palette["green"])
    ax2.tick_params(axis="y", labelcolor=Palette["green"])
    ax2.set_ylim(1, 5)

    sig_txt = f"q={q_val:.3f}" if q_val is not None else (f"p={p_val:.3f}" if pd.notna(p_val) else "")
    corr_txt = f"rho={corr:.2f} {sig_txt}" if pd.notna(corr) else "rho = n/a"
    ax1.set_title(f"Prior-night sleep vs. next-day mood ({corr_txt})")
    _format_date_axis(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    return _save(fig, output_dir, "sleep_vs_next_day_mood")


def plot_event_stress_vs_mood(mood_events: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if mood_events.empty:
        return None

    df = mood_events.dropna(subset=["mood", "stress_3h_mean"])
    if df.empty:
        return None

    corr = df[["stress_3h_mean", "mood"]].apply(pd.to_numeric, errors="coerce").corr().iloc[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x="stress_3h_mean",
        y="mood",
        size=df["steps_6h_sum"].fillna(0),
        sizes=(40, 260),
        color=Palette["blue"],
        edgecolor="white",
        alpha=0.75,
        legend=False,
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="stress_3h_mean",
        y="mood",
        scatter=False,
        ci=None,
        color=Palette["red"],
        line_kws={"linewidth": 2, "linestyle": "--"},
        ax=ax,
    )
    ax.set_xlabel("Stress (prior 3h avg)")
    ax.set_ylabel("Mood score")
    corr_txt = f"rho = {corr:.2f}" if pd.notna(corr) else "rho = n/a"
    ax.set_title(f"Stress load leading into mood events ({corr_txt})")
    ax.grid(True, alpha=0.6)
    ax.legend(["Trend"], loc="lower right")
    sns.despine(ax=ax)

    return _save(fig, output_dir, "event_stress_vs_mood")


def plot_health_overview(daily_metrics: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if daily_metrics.empty:
        return None

    df = daily_metrics.copy()
    df["date"] = pd.to_datetime(df["calendarDate"])

    metrics = {
        "steps_total": (Palette["blue"], "Steps"),
        "sleep_hours": (Palette["purple"], "Sleep (h)"),
        "stress_mean": (Palette["orange"], "Stress"),
        "body_battery_delta": (Palette["green"], "Body battery Δ"),
        "heart_rate_mean": (Palette["red"], "HR mean"),
    }

    def zscore(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce")
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std and not np.isclose(std, 0) else s * 0

    fig, ax = plt.subplots(figsize=(10, 6))
    composite_z = []
    for col, (color, label) in metrics.items():
        if col not in df.columns:
            continue
        series = zscore(df[col])
        composite_z.append(series)
        ax.plot(df["date"], series, label=f"{label} (z)", color=color, linewidth=2.0)

    if composite_z:
        comp = pd.concat(composite_z, axis=1).abs().mean(axis=1)
        df["composite_z"] = comp
        top = df.nlargest(3, "composite_z")
        ax.scatter(top["date"], [0] * len(top), color=Palette["red"], marker="o", s=60, label="High variance days")
        # Stagger labels to avoid overlap
        offsets = [12, -14, 26]
        for i, d in enumerate(top.itertuples()):
            ax.annotate(
                d.calendarDate,
                (d.date, 0),
                textcoords="offset points",
                xytext=(0, offsets[i % len(offsets)]),
                ha="center",
                fontsize=8,
                color=Palette["red"],
                bbox=dict(boxstyle="round,pad=0.15", fc="#fff9f9", ec=Palette["red"], alpha=0.6),
            )

    missing_mask = df[list(metrics.keys())].isna().any(axis=1)
    _shade_missing(ax, df["date"], missing_mask, label="Missing data")

    ax.set_title("Health overview (normalized)")
    ax.set_ylabel("z-score (per metric)")
    _format_date_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen[l] = True
        uniq_handles.append(h)
        uniq_labels.append(l)
    ax.legend(uniq_handles, uniq_labels, ncol=2)

    return _save(fig, output_dir, "health_overview_normalized")


def plot_correlation_bars(corr: Dict[str, Dict[str, float]], title: str, output_dir: Path, name: str) -> Optional[Path]:
    if not corr:
        return None
    items = sorted(corr.items(), key=lambda kv: kv[1].get("rho", 0), reverse=True)
    labels = [k for k, _ in items]
    rhos = [v.get("rho") for _, v in items]
    pvals = [v.get("p") for _, v in items]
    qvals = [v.get("q") for _, v in items]
    palette = [Palette["green"] if r is not None and r >= 0 else Palette["red"] for r in rhos]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Use hue to avoid seaborn palette deprecation
    sns.barplot(x=rhos, y=labels, hue=labels, palette=palette, ax=ax, orient="h", legend=False)
    xmin, xmax = min(rhos), max(rhos)
    pad = 0.15
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.axvline(0, color=Palette["gray"], linewidth=1.2)
    ax.set_xlabel("Spearman rho (significance by q-value)")
    ax.set_title(title)
    for i, (r, p, q) in enumerate(zip(rhos, pvals, qvals)):
        sig_value = q if q is not None else p
        marker = _sig_marker(sig_value)
        sig_label = f"q={q:.3f}" if q is not None else (f"p={p:.3f}" if p is not None else "")
        x_pos = r + (0.03 if r >= 0 else 0.03)
        ax.text(
            x_pos,
            i,
            f"{r:.2f} {sig_label} {marker}",
            va="center",
            ha="left",
            fontsize=9,
            clip_on=False,
        )
    sns.despine(ax=ax)
    return _save(fig, output_dir, name)


def plot_weekday_mood_bar(mood_daily: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if mood_daily.empty:
        return None
    df = mood_daily.copy()
    df["date"] = pd.to_datetime(df["calendarDate"])
    df["weekday"] = df["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped = df.groupby("weekday")["mood_mean"].mean()
    grouped = grouped.reindex(order).dropna()
    if grouped.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grouped.index, grouped.values, color=Palette["blue"], alpha=0.8)
    ax.set_ylim(1, 5)
    ax.set_ylabel("Average mood")
    ax.set_title("Mood by weekday")
    for i, v in enumerate(grouped.values):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom")
    return _save(fig, output_dir, "mood_by_weekday_bar")


def plot_context_bars(mood_events: pd.DataFrame, context_col: str, title: str, name: str, output_dir: Path) -> Optional[Path]:
    if mood_events.empty or context_col not in mood_events.columns:
        return None
    grouped = mood_events.groupby(context_col).agg(mood_mean=("mood", "mean"), count=("id", "count"), stress_3h_mean=("stress_3h_mean", "mean")).dropna(how="all")
    if grouped.empty:
        return None
    grouped = grouped.sort_values("mood_mean", ascending=False)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(grouped.index, grouped["mood_mean"], color=Palette["green"], alpha=0.8, label="Mood")
    ax1.set_ylabel("Mood")
    ax1.set_ylim(1, 5)
    ax2 = ax1.twinx()
    ax2.plot(grouped.index, grouped["stress_3h_mean"], color=Palette["orange"], marker="o", label="Stress 3h")
    ax2.set_ylabel("Stress (prior 3h)")
    ax1.set_title(title)
    for i, v in enumerate(grouped["mood_mean"]):
        count = grouped.iloc[i]["count"]
        ax1.text(i, v + 0.05, f"{v:.2f} (n={int(count)})", ha="center", va="bottom", fontsize=8)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    return _save(fig, output_dir, name)


def plot_metric_coverage(daily_metrics: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if daily_metrics.empty:
        return None
    cols = [
        "steps_total",
        "sleep_hours",
        "stress_mean",
        "heart_rate_mean",
        "body_battery_delta",
        "moderate_minutes",
        "stress_load",
    ]
    total_days = daily_metrics.shape[0]
    coverage = []
    for col in cols:
        if col not in daily_metrics.columns:
            continue
        have = pd.to_numeric(daily_metrics[col], errors="coerce").notna().sum()
        coverage.append((col, have / total_days * 100, have, total_days))
    if not coverage:
        return None
    coverage.sort(key=lambda x: x[1])
    labels, pct, have, total = zip(*coverage)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, pct, color=Palette["blue"], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coverage (% of days)")
    ax.set_xlim(0, 100)
    ax.set_title("Data coverage by metric")
    for i, p in enumerate(pct):
        ax.text(p + 1, i, f"{p:.0f}% ({have[i]}/{total[i]})", va="center")
    return _save(fig, output_dir, "metric_coverage")


def plot_recent_change_bars(recent_change: Dict[str, Dict[str, float]], metric_labels: Dict[str, str], output_dir: Path) -> Optional[Path]:
    if not recent_change:
        return None
    items = []
    for k, v in recent_change.items():
        delta = v.get("delta")
        ci = v.get("ci")
        if delta is None:
            continue
        items.append((k, delta, ci))
    if not items:
        return None
    items.sort(key=lambda t: t[1])
    labels = [metric_labels.get(k, k) for k, _, _ in items]
    deltas = [d for _, d, _ in items]
    cis = [ci for _, _, ci in items]
    y_pos = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [Palette["green"] if d >= 0 else Palette["red"] for d in deltas]
    xerr = None
    if any(ci is not None for ci in cis):
        lower = [d - ci[0] if ci else 0 for d, ci in zip(deltas, cis)]
        upper = [ci[1] - d if ci else 0 for d, ci in zip(deltas, cis)]
        xerr = [lower, upper]
    ax.barh(y_pos, deltas, xerr=xerr, color=colors, alpha=0.85, ecolor="#444", capsize=6)
    ax.axvline(0, color=Palette["gray"], linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Recent change (last 14d - prior 14d)")
    ax.set_xscale("symlog", linthresh=50, linscale=1.2)
    ax.set_title("Recent change with block-bootstrap CI (symlog scale)")
    sns.despine(ax=ax)
    return _save(fig, output_dir, "recent_change_bars")


def plot_trend_with_ci(daily_metrics: pd.DataFrame, metric: str, label: str, output_dir: Path) -> Optional[Path]:
    if metric not in daily_metrics.columns or daily_metrics.empty:
        return None
    dm = daily_metrics.dropna(subset=[metric]).copy()
    if dm.shape[0] < 5:
        return None
    band = _trend_band(dm["calendarDate"], dm[metric])
    if band is None:
        return None
    dates, fit_line, low, high = band
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.to_datetime(dm["calendarDate"]), dm[metric], color=Palette["blue"], alpha=0.35, label="Daily value")
    ax.plot(dates, fit_line, color=Palette["red"], linewidth=2.0, label="Trend (linear)")
    ax.fill_between(dates, low, high, color=Palette["red"], alpha=0.16, label="Trend CI")
    ax.set_title(f"Trend with CI: {label}")
    ax.set_ylabel(label)
    _format_date_axis(ax)
    ax.legend()
    return _save(fig, output_dir, f"trend_{metric}")


def plot_coverage_heatmap(daily_metrics: pd.DataFrame, metrics: List[str], output_dir: Path) -> Optional[Path]:
    if daily_metrics.empty:
        return None
    df = daily_metrics.copy()
    df["date"] = pd.to_datetime(df["calendarDate"]).dt.normalize()
    presence = {}
    for m in metrics:
        if m not in df.columns:
            continue
        presence[m] = pd.to_numeric(df[m], errors="coerce").notna().astype(int)
    if not presence:
        return None
    mat = pd.DataFrame(presence)
    mat["date"] = df["date"].values
    mat = mat.set_index("date").sort_index()
    mat = mat.groupby(mat.index).first()  # collapse duplicates per day

    fig, ax = plt.subplots(figsize=(12, 4.5 + len(presence) * 0.2))
    cmap = sns.color_palette(["#e5e5e5", Palette["blue"]], as_cmap=True)
    sns.heatmap(mat.T, cmap=cmap, vmin=0, vmax=1, cbar=False, ax=ax, linewidths=0.4, linecolor="#d8dee4")
    ax.set_title("Metric availability by date (1 = present, 0 = missing)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Metric")

    # Reduce x tick density to weekly labels
    dates = mat.index.to_list()
    if dates:
        step = max(1, len(dates) // 10)
        xticks = np.arange(0, len(dates), step)
        ax.set_xticks(xticks + 0.5)
        ax.set_xticklabels([dates[i].strftime("%b %d") for i in xticks], rotation=35, ha="right")
    return _save(fig, output_dir, "metric_coverage_heatmap")


def _block_indices(n: int, block_size: int = 2) -> List[np.ndarray]:
    idx = np.arange(n)
    return [idx[i : i + block_size] for i in range(0, n, block_size)]


def _spearman_perm(dates: pd.Series, x: pd.Series, y: pd.Series, n_perm: int = 1000, block_size: int = 2) -> Optional[Dict[str, float]]:
    df = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce"), "x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if df.shape[0] < 3:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    n = df.shape[0]
    rho_obs, _ = stats.spearmanr(df["x"], df["y"])
    if pd.isna(rho_obs):
        return None
    blocks = _block_indices(n, block_size=block_size)
    perm_stats = 0
    for _ in range(n_perm):
        order = np.random.permutation(len(blocks))
        perm_idx = np.concatenate([blocks[i] for i in order])
        rho_perm, _ = stats.spearmanr(df.loc[perm_idx, "x"], df["y"])
        if pd.notna(rho_perm) and abs(rho_perm) >= abs(rho_obs):
            perm_stats += 1
    p_perm = (perm_stats + 1) / (n_perm + 1)
    return {"rho": float(rho_obs), "p": float(p_perm), "n": int(n)}


def _bh_correction(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    p_items = [(k, v.get("p")) for k, v in results.items() if v and v.get("p") is not None]
    m = len(p_items)
    if m == 0:
        return {}
    p_sorted = sorted(p_items, key=lambda kv: kv[1])
    q_map: Dict[str, float] = {}
    for rank, (k, p) in enumerate(p_sorted, start=1):
        q = p * m / rank
        q_map[k] = min(q, 1.0)
    for i in range(m - 2, -1, -1):
        k, _ = p_sorted[i]
        k_next, _ = p_sorted[i + 1]
        q_map[k] = min(q_map[k], q_map[k_next])
    return q_map


def _block_bootstrap_ci(
    dates: pd.Series,
    values: pd.Series,
    stat_fn,
    n_boot: int = 400,
    block_size: int = 2,
    alpha: float = 0.05,
) -> Optional[Tuple[float, float]]:
    df = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce"), "value": pd.to_numeric(values, errors="coerce")}).dropna()
    if df.shape[0] < block_size + 1:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    n = df.shape[0]
    blocks = _block_indices(n, block_size=block_size)
    samples: List[float] = []
    for _ in range(n_boot):
        order = np.random.randint(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in order])[:n]
        boot_df = df.loc[idx].reset_index(drop=True)
        stat_val = stat_fn(boot_df["date"], boot_df["value"])
        if stat_val is not None and not pd.isna(stat_val):
            samples.append(float(stat_val))
    if len(samples) < max(30, int(0.3 * n_boot)):
        return None
    low, high = np.percentile(samples, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(low), float(high)


def _per_day_slope(dates: pd.Series, values: pd.Series) -> Optional[float]:
    ser = pd.to_numeric(values, errors="coerce")
    ts = pd.to_datetime(dates, errors="coerce")
    mask = ser.notna() & ts.notna()
    if mask.sum() < 3:
        return None
    x = ts[mask].map(pd.Timestamp.toordinal).astype(float)
    y = ser[mask].astype(float)
    try:
        slope = np.polyfit(x, y, 1)[0]
    except Exception:
        return None
    return float(slope)


def _recent_change(dates: pd.Series, values: pd.Series, window_days: int = 14) -> Optional[float]:
    ser = pd.to_numeric(values, errors="coerce")
    ts = pd.to_datetime(dates, errors="coerce")
    mask = ser.notna() & ts.notna()
    if mask.sum() < 6:
        return None
    df = pd.DataFrame({"date": ts[mask], "value": ser[mask]}).sort_values("date")
    max_date = df["date"].max()
    if pd.isna(max_date):
        return None
    recent_start = max_date - pd.Timedelta(days=window_days)
    prior_start = max_date - pd.Timedelta(days=2 * window_days)
    recent = df.loc[df["date"] > recent_start, "value"]
    prior = df.loc[(df["date"] > prior_start) & (df["date"] <= recent_start), "value"]
    if len(recent) < 3 or len(prior) < 3:
        return None
    return float(recent.mean() - prior.mean())


def _trend_band(
    dates: pd.Series,
    values: pd.Series,
    n_boot: int = 400,
    block_size: int = 2,
    alpha: float = 0.05,
) -> Optional[Tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray]]:
    df = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce"), "value": pd.to_numeric(values, errors="coerce")}).dropna()
    if df.shape[0] < 5:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    x = mdates.date2num(df["date"])
    y = df["value"].to_numpy(dtype=float)
    try:
        coeffs = np.polyfit(x, y, 1)
    except Exception:
        return None
    fit_line = np.poly1d(coeffs)(x)
    blocks = _block_indices(len(df), block_size=block_size)
    preds = []
    for _ in range(n_boot):
        order = np.random.randint(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in order])[: len(df)]
        boot = df.iloc[idx].reset_index(drop=True)
        bx = mdates.date2num(boot["date"])
        by = boot["value"].to_numpy(dtype=float)
        if len(np.unique(bx)) < 2:
            continue
        try:
            bcoeffs = np.polyfit(bx, by, 1)
        except Exception:
            continue
        preds.append(np.poly1d(bcoeffs)(x))
    if len(preds) < max(30, int(0.3 * n_boot)):
        return None
    pred_arr = np.vstack(preds)
    low = np.percentile(pred_arr, alpha / 2 * 100, axis=0)
    high = np.percentile(pred_arr, (1 - alpha / 2) * 100, axis=0)
    return df["date"], fit_line, low, high


def _corr_stats(df: pd.DataFrame, cols: List[str], target: str = "mood_mean", date_col: str = "calendarDate") -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col in cols:
        if col not in df.columns or date_col not in df.columns:
            continue
        res = _spearman_perm(df[date_col], df[col], df[target])
        if res:
            out[col] = res
    return out


def _corr_daily(daily_metrics: pd.DataFrame, mood_daily: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if daily_metrics.empty or mood_daily.empty:
        return {}
    merged = daily_metrics.merge(mood_daily, on="calendarDate", how="inner")
    cols = [
        "steps_total",
        "sleep_hours",
        "stress_mean",
        "body_battery_delta",
        "active_minutes",
        "moderate_minutes",
        "stress_load",
    ]
    return _corr_stats(merged, cols, date_col="calendarDate")


def _lagged_corr(daily_metrics: pd.DataFrame, mood_daily: pd.DataFrame, lag_days: int = 1) -> Dict[str, Dict[str, float]]:
    if daily_metrics.empty or mood_daily.empty:
        return {}
    dm = daily_metrics.copy()
    md = mood_daily.copy()
    md["target_date"] = pd.to_datetime(md["calendarDate"])
    dm["target_date"] = pd.to_datetime(dm["calendarDate"]) + pd.Timedelta(days=lag_days)
    merged = md.merge(dm, on="target_date", suffixes=("_mood", ""))
    cols = [
        "steps_total",
        "active_minutes",
        "moderate_minutes",
        "stress_mean",
        "stress_load",
        "sleep_hours",
        "body_battery_am",
        "body_battery_delta",
    ]
    return _corr_stats(merged, cols, date_col="target_date")


def generate_all_plots(
    daily_metrics: pd.DataFrame,
    mood_daily: pd.DataFrame,
    mood_events: pd.DataFrame,
    output_dir: str | Path = "analysis_output/plots",
) -> List[Path]:
    output_dir = Path(output_dir)
    corr = _corr_daily(daily_metrics, mood_daily)
    lag_corr = _lagged_corr(daily_metrics, mood_daily, lag_days=1)
    combined = {f"same_{k}": v for k, v in corr.items()}
    combined.update({f"lag_{k}": v for k, v in lag_corr.items()})
    q_map = _bh_correction(combined)
    for k, v in corr.items():
        if v:
            v["q"] = q_map.get(f"same_{k}")
    for k, v in lag_corr.items():
        if v:
            v["q"] = q_map.get(f"lag_{k}")

    metric_labels = {
        "steps_total": "Total steps per day",
        "sleep_hours": "Sleep duration (h)",
        "stress_mean": "Average daily stress",
        "heart_rate_mean": "Mean daily heart rate",
        "body_battery_delta": "Body battery Δ",
        "active_minutes": "Active minutes",
        "moderate_minutes": "Moderate minutes",
        "stress_load": "Stress load",
        "body_battery_am": "Body battery AM",
        "mood_mean": "Mean mood (1-5)",
    }

    # Recent change and trend for selected metrics
    recent_change: Dict[str, Dict[str, float]] = {}
    trend_metrics = [
        "sleep_hours",
        "stress_mean",
        "steps_total",
        "body_battery_delta",
    ]
    if not daily_metrics.empty:
        dm_sorted = daily_metrics.sort_values("calendarDate")
        for col in trend_metrics:
            if col in dm_sorted:
                delta = _recent_change(dm_sorted["calendarDate"], dm_sorted[col], window_days=14)
                ci_delta = _block_bootstrap_ci(dm_sorted["calendarDate"], dm_sorted[col], lambda d, v: _recent_change(d, v, window_days=14))
                if delta is not None:
                    recent_change[col] = {"delta": delta}
                    if ci_delta:
                        recent_change[col]["ci"] = ci_delta

    # Trend plots use bootstrap bands
    trend_plots = [
        plot_trend_with_ci(daily_metrics, "sleep_hours", metric_labels["sleep_hours"], output_dir),
        plot_trend_with_ci(daily_metrics, "stress_mean", metric_labels["stress_mean"], output_dir),
        plot_trend_with_ci(daily_metrics, "steps_total", metric_labels["steps_total"], output_dir),
    ]

    plots: List[Optional[Path]] = [
        plot_daily_steps_and_mood(daily_metrics, mood_daily, output_dir),
        plot_rolling_stress_and_mood(daily_metrics, mood_daily, output_dir),
        plot_sleep_vs_next_day_mood(daily_metrics, mood_daily, output_dir, lag_corr=lag_corr),
        plot_body_battery_vs_energy(mood_events, output_dir),
        plot_event_stress_vs_mood(mood_events, output_dir),
        plot_health_overview(daily_metrics, output_dir),
        plot_weekday_mood_bar(mood_daily, output_dir),
        plot_correlation_bars(corr, "Mood correlations", output_dir, "mood_correlations"),
        plot_correlation_bars(lag_corr, "Next-day mood correlations", output_dir, "mood_correlations_lag1"),
        plot_metric_coverage(daily_metrics, output_dir),
        plot_context_bars(mood_events, "social_context", "Mood by social context", "mood_by_social", output_dir),
        plot_context_bars(mood_events, "environment_context", "Mood by environment", "mood_by_environment", output_dir),
        plot_recent_change_bars(recent_change, metric_labels, output_dir),
        plot_coverage_heatmap(daily_metrics, list(metric_labels.keys()), output_dir),
    ]
    plots.extend(trend_plots)
    return [p for p in plots if p is not None]
