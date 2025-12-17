from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .data import load_mood, load_timeseries
from .features import aggregate_mood_daily, compute_daily_metrics, enrich_mood_events
from .plots import generate_all_plots


def _correlations(daily_metrics: pd.DataFrame, mood_daily: pd.DataFrame) -> Dict[str, float]:
    merged = daily_metrics.merge(mood_daily, on="calendarDate", how="inner")
    if merged.empty:
        return {}
    cols = [
        "steps_total",
        "sleep_hours",
        "stress_mean",
        "body_battery_delta",
        "active_minutes",
    ]
    subset = merged[["mood_mean"] + cols].apply(pd.to_numeric, errors="coerce")
    corr = subset.corr().get("mood_mean")
    if corr is None:
        return {}
    corr = corr.drop(index="mood_mean", errors="ignore")
    return {k: v for k, v in corr.items() if pd.notna(v)}


def _weekday_profile(mood_daily: pd.DataFrame) -> Optional[pd.Series]:
    if mood_daily.empty:
        return None
    mood_daily = mood_daily.copy()
    mood_daily["weekday"] = pd.to_datetime(mood_daily["calendarDate"]).dt.day_name()
    return mood_daily.groupby("weekday")["mood_mean"].mean().sort_index()


def run_analysis(
    export_path: str | Path = "garmin_export",
    mood_path: str | Path = "garmin_export/mood_tracker.csv",
    output_dir: str | Path = "analysis_output",
    last_n_days: Optional[int] = None,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_ts, daily_raw = load_timeseries(export_path=export_path, last_n_days=last_n_days)
    target_tz = getattr(df_ts.index, "tz", None) if not df_ts.empty else None
    mood_df = load_mood(mood_path=mood_path, target_tz=target_tz)

    daily_metrics = compute_daily_metrics(df_ts)

    # Use sleepingSeconds from daily summaries when present to avoid undercounted sleep
    if not daily_raw.empty and "sleepingSeconds" in daily_raw.columns:
        sleep_map = daily_raw["sleepingSeconds"].dropna() / 3600
        if not sleep_map.empty:
            daily_metrics = daily_metrics.set_index("calendarDate")
            daily_metrics["sleep_hours"] = daily_metrics["sleep_hours"].combine_first(sleep_map)
            daily_metrics["sleep_minutes"] = daily_metrics["sleep_hours"] * 60
            daily_metrics = daily_metrics.reset_index()

    sleep_by_date = {row.calendarDate: row.sleep_hours for row in daily_metrics.itertuples()}
    mood_daily = aggregate_mood_daily(mood_df)
    mood_events = enrich_mood_events(df_ts, mood_df, sleep_by_date=sleep_by_date)
    mood_days = daily_metrics.merge(mood_daily, on="calendarDate", how="inner").shape[0]

    plot_dir = output_dir / "plots"
    plot_paths = generate_all_plots(daily_metrics, mood_daily, mood_events, output_dir=plot_dir)

    stats = {}
    key_cols = [
        "steps_total",
        "sleep_hours",
        "stress_mean",
        "heart_rate_mean",
        "body_battery_delta",
    ]
    coverage = {}
    for col in key_cols:
        if col in daily_metrics.columns and not daily_metrics.empty:
            s = pd.to_numeric(daily_metrics[col], errors="coerce")
            stats[col] = {
                "mean": float(s.mean()),
                "min": float(s.min()),
                "max": float(s.max()),
            }
            coverage[col] = (int(s.notna().sum()), int(daily_metrics.shape[0]))

    # Identify top variance days used in health overview
    composite = None
    if not daily_metrics.empty:
        zcols = []
        for col in key_cols:
            if col in daily_metrics:
                s = pd.to_numeric(daily_metrics[col], errors="coerce")
                std = s.std(ddof=0)
                z = (s - s.mean()) / std if std and not pd.isna(std) and std != 0 else s * 0
                zcols.append(z.abs())
        if zcols:
            composite = pd.concat(zcols, axis=1).mean(axis=1)
            top_idx = composite.nlargest(3).index
            variance_days = daily_metrics.loc[top_idx, "calendarDate"].tolist()
        else:
            variance_days = []
    else:
        variance_days = []

    weekday_profile = _weekday_profile(mood_daily)
    period_start = str(df_ts.index.min().date()) if not df_ts.empty else "n/a"
    period_end = str(df_ts.index.max().date()) if not df_ts.empty else "n/a"
    insight = {
        "rows": len(df_ts),
        "days": daily_metrics.shape[0],
        "mood_entries": mood_df.shape[0],
        "plots": [str(p) for p in plot_paths],
        "correlations": _correlations(daily_metrics, mood_daily),
        "weekday_profile": weekday_profile.to_dict() if weekday_profile is not None else {},
        "stats": stats,
        "variance_days": variance_days,
        "period": (period_start, period_end),
        "mood_days": mood_days,
        "coverage": coverage,
    }

    metric_labels = {
        "steps_total": "total steps per day",
        "sleep_hours": "nightly sleep duration (Garmin sleepingSeconds when available)",
        "stress_mean": "average daily stress score",
        "heart_rate_mean": "mean daily heart rate",
        "body_battery_delta": "change from first to last body battery reading that day (positive = recharged)",
        "active_minutes": "minutes with >=30 steps per minute",
    }

    summary_lines: List[str] = [
        f"# Garmin + mood summary ({period_start} to {period_end})",
        "",
        "## Overview",
        "",
        "Daily aggregates aligned with mood check-ins to show overall coverage and sample sizes.",
        "",
        f"- Rows in minute-level data: {insight['rows']}",
        f"- Days covered: {insight['days']}",
        f"- Mood entries: {insight['mood_entries']}",
        f"- Days with mood + metrics: {mood_days}",
        "- All metrics are per-day aggregates; stats span the full period.",
        "",
        "## Key stats (mean / min / max)",
        "",
        "Per-day distributions of core health metrics (missing days are excluded from the means).",
        "",
    ]

    if stats:
        for col, vals in stats.items():
            label = metric_labels.get(col, col)
            summary_lines.append(f"- {col}: {vals['mean']:.2f} / {vals['min']:.2f} / {vals['max']:.2f} ({label})")

    if insight["correlations"]:
        summary_lines.append("")
        summary_lines.append("## Mood correlations")
        summary_lines.append("")
        summary_lines.append("Relationships between daily metrics and mood on the days with mood entries.")
        summary_lines.append("")
        summary_lines.append(f"Based on {mood_days} days with mood entries.")
        summary_lines.append("Positive: higher metric aligns with better mood; negative: higher metric aligns with worse mood.")
        summary_lines.append("Magnitude guide: ~0.1 weak, ~0.3 moderate, >0.5 strong (small sample; interpret cautiously).")
        summary_lines.append("")
        for k, v in insight["correlations"].items():
            label = metric_labels.get(k, k)
            summary_lines.append(f"- {k}: {v:.2f} ({label})")

    if insight["weekday_profile"]:
        summary_lines.append("")
        summary_lines.append("## Mood by weekday (mean)")
        summary_lines.append("")
        summary_lines.append("Average mood per weekday to highlight weekly patterns (1 = low mood, 5 = high mood).")
        summary_lines.append("")
        for k, v in insight["weekday_profile"].items():
            summary_lines.append(f"- {k}: {v:.2f}")

    if coverage:
        summary_lines.append("")
        summary_lines.append("## Data coverage (non-missing days / total days)")
        summary_lines.append("")
        summary_lines.append("Completeness of each metric across the analysis window.")
        summary_lines.append("")
        for col, (have, total) in coverage.items():
            label = metric_labels.get(col, col)
            summary_lines.append(f"- {col}: {have}/{total} ({label})")

    if variance_days:
        summary_lines.append("")
        summary_lines.append("## High variance days across metrics")
        summary_lines.append("")
        summary_lines.append("Days where normalized metrics diverged most (potential outliers or notable events).")
        summary_lines.append("")
        for d in variance_days:
            summary_lines.append(f"- {d}")

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines))

    return insight


if __name__ == "__main__":
    info = run_analysis()
    print("Analysis complete. See summary:")
    print((Path("analysis_output") / "summary.md").read_text())
