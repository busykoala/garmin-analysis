from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def _first_valid(series: pd.Series):
    dropna = series.dropna()
    return dropna.iloc[0] if not dropna.empty else pd.NA


def _last_valid(series: pd.Series):
    dropna = series.dropna()
    return dropna.iloc[-1] if not dropna.empty else pd.NA


def compute_daily_metrics(df_ts: pd.DataFrame) -> pd.DataFrame:
    if df_ts.empty:
        return pd.DataFrame()

    df = df_ts.copy()
    for col in ["steps_per_min", "heart_rate", "stress_level", "body_battery", "sleep_movement"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Prevent negative stress artifacts
    if "stress_level" in df.columns:
        df["stress_level"] = df["stress_level"].clip(lower=0)

    df["calendarDate"] = df["calendarDate"].astype(str)
    grouped = df.groupby("calendarDate")

    sleep_minutes = grouped["sleep_present"].agg(
        lambda s: s.fillna(False).astype(bool).sum() if s.notna().any() else pd.NA
    )
    steps_sum = grouped["steps_per_min"].sum(min_count=1)
    steps_obs = grouped["steps_present"].sum(min_count=1)

    daily = pd.DataFrame({
        "steps_total": steps_sum,
        "active_minutes": grouped["steps_per_min"].agg(lambda s: (s.fillna(0) >= 30).sum()),
        "vigorous_minutes": grouped["steps_per_min"].agg(lambda s: (s.fillna(0) >= 100).sum()),
        "heart_rate_mean": grouped["heart_rate"].mean(),
        "heart_rate_min": grouped["heart_rate"].min(),
        "heart_rate_max": grouped["heart_rate"].max(),
        "stress_mean": grouped["stress_level"].mean(),
        "body_battery_am": grouped["body_battery"].apply(_first_valid),
        "body_battery_pm": grouped["body_battery"].apply(_last_valid),
        "sleep_minutes": sleep_minutes,
    })

    daily["sleep_hours"] = daily["sleep_minutes"] / 60
    daily.loc[daily["sleep_minutes"] == 0, ["sleep_minutes", "sleep_hours"]] = pd.NA

    # If no step observations that day, treat steps and derived minutes as missing
    no_steps = steps_obs == 0
    daily.loc[no_steps, ["steps_total", "active_minutes", "vigorous_minutes"]] = pd.NA

    # Body battery delta only when both endpoints exist
    daily["body_battery_delta"] = daily["body_battery_pm"] - daily["body_battery_am"]
    daily.loc[daily[["body_battery_pm", "body_battery_am"]].isna().any(axis=1), "body_battery_delta"] = pd.NA
    daily = daily.reset_index()
    return daily


def aggregate_mood_daily(mood_df: pd.DataFrame) -> pd.DataFrame:
    if mood_df.empty:
        return pd.DataFrame()

    grouped = mood_df.groupby("calendarDate")
    daily = grouped.agg(
        mood_mean=("mood", "mean"),
        mood_median=("mood", "median"),
        entries=("id", "count"),
        stress_feeling_mean=("stress_feeling", "mean"),
        energy_level_mean=("energy_level", "mean"),
        motivation_level_mean=("motivation_level", "mean"),
        focus_level_mean=("focus_level", "mean"),
        physical_tension_mean=("physical_tension", "mean"),
        sleepiness_mean=("sleepiness", "mean"),
    ).reset_index()
    return daily


def enrich_mood_events(
    df_ts: pd.DataFrame,
    mood_df: pd.DataFrame,
    sleep_by_date: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if df_ts.empty or mood_df.empty:
        return pd.DataFrame()

    ts = df_ts.sort_index()
    mood_events = []

    if sleep_by_date is None:
        sleep_by_date = {}

    for _, row in mood_df.iterrows():
        t = row["timestamp"]
        day = row["calendarDate"]

        window_30m = ts.loc[(t - pd.Timedelta(minutes=30)) : t]
        window_3h = ts.loc[(t - pd.Timedelta(hours=3)) : t]
        window_6h = ts.loc[(t - pd.Timedelta(hours=6)) : t]
        window_12h = ts.loc[(t - pd.Timedelta(hours=12)) : t]

        def nearest(series: pd.Series, tolerance_minutes: int = 30):
            if series.empty:
                return pd.NA
            try:
                val = series.reindex(index=[t], method="nearest", tolerance=pd.Timedelta(minutes=tolerance_minutes)).iloc[0]
                return val
            except Exception:
                return pd.NA

        entry = {
            "id": row.get("id"),
            "timestamp": t,
            "calendarDate": day,
            "mood": row.get("mood"),
            "stress_feeling": row.get("stress_feeling"),
            "energy_level": row.get("energy_level"),
            "motivation_level": row.get("motivation_level"),
            "focus_level": row.get("focus_level"),
            "physical_tension": row.get("physical_tension"),
            "sleepiness": row.get("sleepiness"),
            "social_context": row.get("social_context"),
            "environment_context": row.get("environment_context"),
            "body_battery_at_event": nearest(ts["body_battery"]),
            "heart_rate_30m_mean": window_30m["heart_rate"].mean(),
            "stress_3h_mean": window_3h["stress_level"].mean(),
            "steps_6h_sum": window_6h["steps_per_min"].sum(min_count=1),
            "steps_12h_sum": window_12h["steps_per_min"].sum(min_count=1),
            "active_6h_minutes": (window_6h["steps_per_min"].fillna(0) >= 30).sum(),
            "body_battery_3h_delta": (window_3h["body_battery"].iloc[-1] - window_3h["body_battery"].iloc[0]) if len(window_3h) > 1 else pd.NA,
            "sleep_prev_night_hours": sleep_by_date.get(day, pd.NA),
        }
        mood_events.append(entry)

    mood_events_df = pd.DataFrame(mood_events)
    mood_events_df = mood_events_df.sort_values("timestamp").reset_index(drop=True)
    return mood_events_df
