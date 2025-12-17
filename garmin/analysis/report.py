from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .data import load_mood, load_timeseries
from .features import aggregate_mood_daily, compute_daily_metrics, enrich_mood_events
from .plots import generate_all_plots


def _corr_stats(df: pd.DataFrame, cols: List[str], target: str = "mood_mean", date_col: str = "calendarDate") -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col in cols:
        if col not in df.columns or date_col not in df.columns:
            continue
        res = _spearman_perm(df[date_col], df[col], df[target])
        if res:
            out[col] = res
    return out


def _correlations(daily_metrics: pd.DataFrame, mood_daily: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    merged = daily_metrics.merge(mood_daily, on="calendarDate", how="inner")
    if merged.empty:
        return {}
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


def _weekday_profile(mood_daily: pd.DataFrame) -> Optional[pd.Series]:
    if mood_daily.empty:
        return None
    mood_daily = mood_daily.copy()
    mood_daily["weekday"] = pd.to_datetime(mood_daily["calendarDate"]).dt.day_name()
    return mood_daily.groupby("weekday")["mood_mean"].mean().sort_index()


def _lagged_correlations(daily_metrics: pd.DataFrame, mood_daily: pd.DataFrame, lag_days: int = 1) -> Dict[str, Dict[str, float]]:
    if daily_metrics.empty or mood_daily.empty:
        return {}
    dm = daily_metrics.copy()
    md = mood_daily.copy()
    md["target_date"] = pd.to_datetime(md["calendarDate"])  # mood day
    dm["target_date"] = pd.to_datetime(dm["calendarDate"]) + pd.Timedelta(days=lag_days)  # metrics shifted forward so they match next-day mood
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


def _per_day_slope(dates: pd.Series, values: pd.Series) -> Optional[float]:
    # Linear slope of value per day; small sample guard
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
    # Mean difference between last window_days and the prior window_days
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
    # ensure monotonicity from largest p
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
    stats_samples: List[float] = []
    for _ in range(n_boot):
        order = np.random.randint(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in order])[:n]
        boot_df = df.loc[idx].reset_index(drop=True)
        stat_val = stat_fn(boot_df["date"], boot_df["value"])
        if stat_val is not None and not pd.isna(stat_val):
            stats_samples.append(float(stat_val))
    if len(stats_samples) < max(30, int(0.3 * n_boot)):
        return None
    low, high = np.percentile(stats_samples, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(low), float(high)


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

    # Add sleep stage percentages when present in daily summary
    if not daily_raw.empty and {"deepSleepSeconds", "lightSleepSeconds", "remSleepSeconds"}.issubset(set(daily_raw.columns)):
        stages = daily_raw.set_index("calendarDate")[["deepSleepSeconds", "lightSleepSeconds", "remSleepSeconds", "sleepingSeconds"]]
        stages = stages.replace({0: pd.NA})
        for col in ["deepSleepSeconds", "lightSleepSeconds", "remSleepSeconds", "sleepingSeconds"]:
            stages[col] = pd.to_numeric(stages[col], errors="coerce")
        stages["deep_pct"] = stages["deepSleepSeconds"] / stages["sleepingSeconds"] * 100
        stages["rem_pct"] = stages["remSleepSeconds"] / stages["sleepingSeconds"] * 100
        stages["light_pct"] = stages["lightSleepSeconds"] / stages["sleepingSeconds"] * 100
        daily_metrics = daily_metrics.set_index("calendarDate").join(stages[["deep_pct", "rem_pct", "light_pct"]], how="left").reset_index()

    sleep_by_date = {row.calendarDate: row.sleep_hours for row in daily_metrics.itertuples()}
    mood_daily = aggregate_mood_daily(mood_df)
    mood_events = enrich_mood_events(df_ts, mood_df, sleep_by_date=sleep_by_date)
    mood_days = daily_metrics.merge(mood_daily, on="calendarDate", how="inner").shape[0]
    corr_same_day = _correlations(daily_metrics, mood_daily)
    lag_corr = _lagged_correlations(daily_metrics, mood_daily, lag_days=1)

    # Multiple-comparison correction (BH-FDR) across all mood-metric correlations
    combined = {f"same_{k}": v for k, v in corr_same_day.items()}
    combined.update({f"lag_{k}": v for k, v in lag_corr.items()})
    q_map = _bh_correction(combined)
    for k, v in corr_same_day.items():
        if v:
            v["q"] = q_map.get(f"same_{k}")
    for k, v in lag_corr.items():
        if v:
            v["q"] = q_map.get(f"lag_{k}")

    # Context slices from mood events
    social_summary = {}
    env_summary = {}
    if not mood_events.empty:
        social_summary = (
            mood_events.groupby("social_context")
            .agg(mood_mean=("mood", "mean"), stress_3h_mean=("stress_3h_mean", "mean"), n=("id", "count"))
            .dropna(how="all")
            .to_dict(orient="index")
        )
        env_summary = (
            mood_events.groupby("environment_context")
            .agg(mood_mean=("mood", "mean"), stress_3h_mean=("stress_3h_mean", "mean"), n=("id", "count"))
            .dropna(how="all")
            .to_dict(orient="index")
        )

    plot_dir = output_dir / "plots"
    plot_paths = generate_all_plots(daily_metrics, mood_daily, mood_events, output_dir=plot_dir)

    stats = {}
    key_cols = [
        "steps_total",
        "sleep_hours",
        "stress_mean",
        "heart_rate_mean",
        "body_battery_delta",
        "moderate_minutes",
        "stress_load",
        "deep_pct",
        "rem_pct",
        "light_pct",
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
    recent_window = 14

    trend_slopes: Dict[str, Dict[str, float]] = {}
    recent_change: Dict[str, Dict[str, float]] = {}

    if not daily_metrics.empty:
        dm_sorted = daily_metrics.sort_values("calendarDate")
        for col in [
            "steps_total",
            "sleep_hours",
            "stress_mean",
            "stress_load",
            "moderate_minutes",
            "body_battery_delta",
        ]:
            if col in dm_sorted:
                slope = _per_day_slope(dm_sorted["calendarDate"], dm_sorted[col])
                ci_slope = _block_bootstrap_ci(dm_sorted["calendarDate"], dm_sorted[col], _per_day_slope)
                if slope is not None:
                    trend_slopes[col] = {"slope": slope}
                    if ci_slope:
                        trend_slopes[col]["ci"] = ci_slope
                delta = _recent_change(dm_sorted["calendarDate"], dm_sorted[col], window_days=recent_window)
                ci_delta = _block_bootstrap_ci(
                    dm_sorted["calendarDate"], dm_sorted[col], lambda d, v: _recent_change(d, v, window_days=recent_window)
                )
                if delta is not None:
                    recent_change[col] = {"delta": delta}
                    if ci_delta:
                        recent_change[col]["ci"] = ci_delta

    if not mood_daily.empty:
        slope = _per_day_slope(mood_daily["calendarDate"], mood_daily["mood_mean"])
        ci_slope = _block_bootstrap_ci(mood_daily["calendarDate"], mood_daily["mood_mean"], _per_day_slope)
        if slope is not None:
            trend_slopes["mood_mean"] = {"slope": slope}
            if ci_slope:
                trend_slopes["mood_mean"]["ci"] = ci_slope
        delta = _recent_change(mood_daily["calendarDate"], mood_daily["mood_mean"], window_days=recent_window)
        ci_delta = _block_bootstrap_ci(
            mood_daily["calendarDate"], mood_daily["mood_mean"], lambda d, v: _recent_change(d, v, window_days=recent_window)
        )
        if delta is not None:
            recent_change["mood_mean"] = {"delta": delta}
            if ci_delta:
                recent_change["mood_mean"]["ci"] = ci_delta
    insight = {
        "rows": len(df_ts),
        "days": daily_metrics.shape[0],
        "mood_entries": mood_df.shape[0],
        "plots": [str(p) for p in plot_paths],
        "correlations": corr_same_day,
        "lagged_corr": lag_corr,
        "weekday_profile": weekday_profile.to_dict() if weekday_profile is not None else {},
        "stats": stats,
        "variance_days": variance_days,
        "period": (period_start, period_end),
        "mood_days": mood_days,
        "coverage": coverage,
        "social_summary": social_summary,
        "env_summary": env_summary,
        "trend_slopes": trend_slopes,
        "recent_change": recent_change,
        "recent_window": recent_window,
    }

    metric_labels = {
        "steps_total": "total steps per day",
        "sleep_hours": "nightly sleep duration (Garmin sleepingSeconds when available)",
        "stress_mean": "average daily stress score",
        "heart_rate_mean": "mean daily heart rate",
        "body_battery_delta": "change from first to last body battery reading that day (positive = recharged)",
        "active_minutes": "minutes with >=30 steps per minute",
        "moderate_minutes": "minutes with 60-99 steps per minute",
        "stress_load": "sum of stress above 25 (per-day load)",
        "body_battery_am": "first body battery reading of the day",
        "deep_pct": "deep sleep as % of sleep",
        "rem_pct": "REM sleep as % of sleep",
        "light_pct": "light sleep as % of sleep",
        "mood_mean": "average mood (1-5)",
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
        summary_lines.append(
            "Spearman rho with block-permutation p-values (block=2 days, 1000 draws) and BH-FDR q-values across all correlations. Positive: higher metric aligns with better mood; negative: higher metric aligns with worse mood. Magnitude guide: ~0.1 weak, ~0.3 moderate, >0.5 strong (small sample; interpret cautiously)."
        )
        summary_lines.append("")
        for k, vals in insight["correlations"].items():
            label = metric_labels.get(k, k)
            rho = vals.get("rho")
            p = vals.get("p")
            q = vals.get("q")
            n = vals.get("n")
            q_part = f", q={q:.3f}" if q is not None else ""
            summary_lines.append(f"- {k}: rho={rho:.2f}, p={p:.3f}{q_part}, n={n} ({label})")

    if insight.get("lagged_corr"):
        summary_lines.append("")
        summary_lines.append("## Next-day mood correlations (lagged 1 day)")
        summary_lines.append("")
        summary_lines.append(
            "How yesterday's metrics relate to today's mood (same magnitude guidance; small sample). Spearman rho with block-permutation p-values and BH-FDR q-values (shared correction across all correlations)."
        )
        summary_lines.append("")
        for k, vals in insight["lagged_corr"].items():
            label = metric_labels.get(k, k)
            rho = vals.get("rho")
            p = vals.get("p")
            q = vals.get("q")
            n = vals.get("n")
            q_part = f", q={q:.3f}" if q is not None else ""
            summary_lines.append(f"- {k}: rho={rho:.2f}, p={p:.3f}{q_part}, n={n} ({label})")

    if insight.get("recent_change"):
        window_days = insight.get("recent_window", 14)
        summary_lines.append("")
        summary_lines.append(f"## Recent change (last {window_days}d vs prior {window_days}d)")
        summary_lines.append("")
        summary_lines.append("Mean difference between the most recent window and the preceding window (positive = higher recently). Block-bootstrap 95% CI (block=2 days, 400 reps) when available.")
        summary_lines.append("")
        for k, v in insight["recent_change"].items():
            label = metric_labels.get(k, k)
            delta = v.get("delta")
            ci = v.get("ci")
            ci_part = f" (CI [{ci[0]:+.2f}, {ci[1]:+.2f}])" if ci else ""
            summary_lines.append(f"- {k}: {delta:+.2f}{ci_part} ({label})")

    if insight.get("trend_slopes"):
        summary_lines.append("")
        summary_lines.append("## Trends (per-day slope)")
        summary_lines.append("")
        summary_lines.append("Linear slopes across the full period (positive = increasing over time). Block-bootstrap 95% CI (block=2 days, 400 reps) when available.")
        summary_lines.append("")
        for k, v in insight["trend_slopes"].items():
            label = metric_labels.get(k, k)
            slope = v.get("slope")
            ci = v.get("ci")
            ci_part = f" (CI [{ci[0]:+.4f}, {ci[1]:+.4f}])" if ci else ""
            summary_lines.append(f"- {k}: {slope:+.4f} per day{ci_part} ({label})")

    if insight.get("social_summary"):
        summary_lines.append("")
        summary_lines.append("## Mood by social context")
        summary_lines.append("")
        summary_lines.append("Average mood and prior 3h stress grouped by social context at the time of the check-in.")
        summary_lines.append("")
        for ctx, vals in insight["social_summary"].items():
            mood_v = vals.get("mood_mean")
            stress_v = vals.get("stress_3h_mean")
            n = vals.get("n")
            summary_lines.append(f"- {ctx}: mood {mood_v:.2f} | stress_3h {stress_v:.2f} | n={n}")

    if insight.get("env_summary"):
        summary_lines.append("")
        summary_lines.append("## Mood by environment")
        summary_lines.append("")
        summary_lines.append("Average mood and prior 3h stress grouped by environment context.")
        summary_lines.append("")
        for ctx, vals in insight["env_summary"].items():
            mood_v = vals.get("mood_mean")
            stress_v = vals.get("stress_3h_mean")
            n = vals.get("n")
            summary_lines.append(f"- {ctx}: mood {mood_v:.2f} | stress_3h {stress_v:.2f} | n={n}")

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
