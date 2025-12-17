from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

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


def plot_sleep_vs_next_day_mood(daily_metrics: pd.DataFrame, mood_daily: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if daily_metrics.empty or mood_daily.empty:
        return None

    mood_shift = mood_daily.copy()
    mood_shift["prev_date"] = (pd.to_datetime(mood_shift["calendarDate"]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    merged = daily_metrics.merge(mood_shift[["prev_date", "mood_mean"]], left_on="calendarDate", right_on="prev_date", how="inner")
    if merged.empty:
        return None

    merged["date"] = pd.to_datetime(merged["calendarDate"])

    corr = merged[["sleep_hours", "mood_mean"]].apply(pd.to_numeric, errors="coerce").corr().iloc[0, 1]

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

    corr_txt = f"r = {corr:.2f}" if pd.notna(corr) else "r = n/a"
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
    palette = [Palette["green"] if r is not None and r >= 0 else Palette["red"] for r in rhos]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Use hue to avoid seaborn palette deprecation
    sns.barplot(x=rhos, y=labels, hue=labels, palette=palette, ax=ax, orient="h", legend=False)
    xmin, xmax = min(rhos), max(rhos)
    pad = 0.08
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.axvline(0, color=Palette["gray"], linewidth=1.2)
    ax.set_xlabel("Spearman rho")
    ax.set_title(title)
    for i, (r, p) in enumerate(zip(rhos, pvals)):
        marker = _sig_marker(p)
        ax.text(
            r + (0.025 if r >= 0 else -0.025),
            i,
            f"{r:.2f} {marker}",
            va="center",
            ha="left" if r >= 0 else "right",
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
        ax1.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
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


def _corr_stats(df: pd.DataFrame, cols: List[str], target: str = "mood_mean") -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(df[target], errors="coerce")
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n < 3:
            continue
        rho, p = stats.spearmanr(x[mask], y[mask])
        if pd.notna(rho) and pd.notna(p):
            out[col] = {"rho": float(rho), "p": float(p), "n": n}
    return out


def _corr_daily(daily_metrics: pd.DataFrame, mood_daily: pd.DataFrame) -> Dict[str, float]:
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
    return _corr_stats(merged, cols)


def _lagged_corr(daily_metrics: pd.DataFrame, mood_daily: pd.DataFrame, lag_days: int = 1) -> Dict[str, float]:
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
    return _corr_stats(merged, cols)


def generate_all_plots(
    daily_metrics: pd.DataFrame,
    mood_daily: pd.DataFrame,
    mood_events: pd.DataFrame,
    output_dir: str | Path = "analysis_output/plots",
) -> List[Path]:
    output_dir = Path(output_dir)
    corr = _corr_daily(daily_metrics, mood_daily)
    lag_corr = _lagged_corr(daily_metrics, mood_daily, lag_days=1)

    plots: List[Optional[Path]] = [
        plot_daily_steps_and_mood(daily_metrics, mood_daily, output_dir),
        plot_rolling_stress_and_mood(daily_metrics, mood_daily, output_dir),
        plot_sleep_vs_next_day_mood(daily_metrics, mood_daily, output_dir),
        plot_body_battery_vs_energy(mood_events, output_dir),
        plot_event_stress_vs_mood(mood_events, output_dir),
        plot_health_overview(daily_metrics, output_dir),
        plot_weekday_mood_bar(mood_daily, output_dir),
        plot_correlation_bars(corr, "Mood correlations", output_dir, "mood_correlations"),
        plot_correlation_bars(lag_corr, "Next-day mood correlations", output_dir, "mood_correlations_lag1"),
        plot_metric_coverage(daily_metrics, output_dir),
        plot_context_bars(mood_events, "social_context", "Mood by social context", "mood_by_social", output_dir),
        plot_context_bars(mood_events, "environment_context", "Mood by environment", "mood_by_environment", output_dir),
    ]
    return [p for p in plots if p is not None]
