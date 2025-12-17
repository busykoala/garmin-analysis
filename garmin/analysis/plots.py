from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Palette = {
    "blue": "#4c72b0",
    "red": "#c44e52",
    "green": "#55a868",
    "orange": "#dd8452",
    "purple": "#8172b2",
    "gray": "#b0b0b0",
}


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


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_daily_steps_and_mood(daily_metrics: pd.DataFrame, mood_daily: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if daily_metrics.empty or mood_daily.empty:
        return None
    merged = daily_metrics.merge(mood_daily, on="calendarDate", how="inner").sort_values("calendarDate")
    merged["date"] = pd.to_datetime(merged["calendarDate"])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(merged["date"], merged["steps_total"] / 1000, color=Palette["blue"], alpha=0.6, label="Steps (k)")
    ax1.set_ylabel("Steps (thousands)", color=Palette["blue"])
    ax1.tick_params(axis="y", labelcolor=Palette["blue"])

    ax2 = ax1.twinx()
    ax2.plot(merged["date"], merged["mood_mean"], color=Palette["red"], marker="o", label="Mood")
    ax2.set_ylabel("Mean mood (1-5)", color=Palette["red"])
    ax2.tick_params(axis="y", labelcolor=Palette["red"])

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    ax1.set_title("Daily steps vs. average mood")

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
    ax.plot(merged["date"], merged["stress_norm"], color=Palette["orange"], label="Stress (7d avg, z-score)")
    ax.plot(merged["date"], merged["mood_norm"], color=Palette["green"], label="Mood (7d avg, z-score)")
    if trend_line is not None:
        ax.plot(merged["date"], trend_line, color=Palette["blue"], linestyle="--", label="Mood trend (linear)")

    missing_mask = merged[["stress_norm", "mood_norm"]].isna().any(axis=1)
    _shade_missing(ax, merged["date"], missing_mask, label="Missing data")

    ax.set_title("Stress vs. mood (7-day rolling, normalized)")
    ax.set_ylabel("z-score")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    ax.legend()

    return _save(fig, output_dir, "rolling_stress_mood")


def plot_body_battery_vs_energy(mood_events: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if mood_events.empty or "body_battery_at_event" not in mood_events.columns:
        return None

    df = mood_events.dropna(subset=["body_battery_at_event", "energy_level"])
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df["body_battery_at_event"],
        df["energy_level"],
        c=df["mood"],
        cmap="viridis",
        s=80,
        edgecolor="white",
        alpha=0.9,
    )
    cbar = fig.colorbar(sc, ax=ax, label="Mood")
    cbar.set_ticks(sorted(df["mood"].unique()))

    ax.set_xlabel("Body battery at event")
    ax.set_ylabel("Self-reported energy")
    ax.set_title("Body battery vs. perceived energy")

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

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(merged["date"], merged["sleep_hours"], color=Palette["purple"], alpha=0.6, label="Sleep (h)")
    ax1.set_ylabel("Sleep duration (hours)", color=Palette["purple"])
    ax1.tick_params(axis="y", labelcolor=Palette["purple"])

    ax2 = ax1.twinx()
    ax2.plot(merged["date"], merged["mood_mean"], color=Palette["green"], marker="o", label="Next-day mood")
    ax2.set_ylabel("Next-day mood", color=Palette["green"])
    ax2.tick_params(axis="y", labelcolor=Palette["green"])

    ax1.set_title("Prior-night sleep vs. next-day mood")
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

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

    fig, ax = plt.subplots(figsize=(7, 6))
    denom = df["steps_6h_sum"].fillna(0).max()
    denom = 1 if pd.isna(denom) or denom <= 0 else denom
    sizes = (df["steps_6h_sum"].fillna(0) / denom) * 220 + 40
    ax.scatter(
        df["stress_3h_mean"],
        df["mood"],
        s=sizes,
        color=Palette["blue"],
        alpha=0.7,
        edgecolor="white",
    )
    ax.set_xlabel("Stress (prior 3h avg)")
    ax.set_ylabel("Mood score")
    ax.set_title("Stress load leading into mood events")

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
        ax.plot(df["date"], series, label=f"{label} (z)", color=color, linewidth=1.6)

    if composite_z:
        comp = pd.concat(composite_z, axis=1).abs().mean(axis=1)
        df["composite_z"] = comp
        top = df.nlargest(3, "composite_z")
        ax.scatter(top["date"], [0] * len(top), color=Palette["red"], marker="o", s=70, label="High variance days")

    missing_mask = df[list(metrics.keys())].isna().any(axis=1)
    _shade_missing(ax, df["date"], missing_mask, label="Missing data")

    ax.set_title("Health overview (normalized)")
    ax.set_ylabel("z-score (per metric)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    ax.legend(ncol=2)

    return _save(fig, output_dir, "health_overview_normalized")


def generate_all_plots(
    daily_metrics: pd.DataFrame,
    mood_daily: pd.DataFrame,
    mood_events: pd.DataFrame,
    output_dir: str | Path = "analysis_output/plots",
) -> List[Path]:
    output_dir = Path(output_dir)
    plots: List[Optional[Path]] = [
        plot_daily_steps_and_mood(daily_metrics, mood_daily, output_dir),
        plot_rolling_stress_and_mood(daily_metrics, mood_daily, output_dir),
        plot_sleep_vs_next_day_mood(daily_metrics, mood_daily, output_dir),
        plot_body_battery_vs_energy(mood_events, output_dir),
        plot_event_stress_vs_mood(mood_events, output_dir),
        plot_health_overview(daily_metrics, output_dir),
    ]
    return [p for p in plots if p is not None]
