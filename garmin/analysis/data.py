from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from garmin.loader import structure_data


def _normalize_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    daily = df.copy()
    if "calendarDate" not in daily.columns:
        daily = daily.reset_index().rename(columns={"index": "calendarDate"})
    daily["calendarDate"] = daily["calendarDate"].astype(str)
    return daily


def load_timeseries(
    export_path: str | Path = "garmin_export",
    freq: str = "1min",
    interpolate_gaps: bool = True,
    max_interp_minutes: int = 5,
    last_n_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-minute Garmin data with a timezone-aware timestamp index."""
    df_all, daily_summary = structure_data(
        export_path=str(export_path),
        freq=freq,
        interpolate_gaps=interpolate_gaps,
        max_interp_minutes=max_interp_minutes,
        last_n_days=last_n_days,
    )
    if df_all.empty:
        return df_all, _normalize_daily_summary(daily_summary)

    df = df_all.copy()
    df.index.name = "timestamp"
    df["timestamp"] = df.index
    df["calendarDate"] = df["calendarDate"].astype(str)

    daily = _normalize_daily_summary(daily_summary)
    return df, daily


def load_mood(
    mood_path: str | Path = "garmin_export/mood_tracker.csv",
    target_tz=None,
) -> pd.DataFrame:
    """Load subjective mood entries and align them to the target timezone."""
    path = Path(mood_path)
    if not path.exists():
        return pd.DataFrame()

    mood_df = pd.read_csv(path, parse_dates=["timestamp"])
    mood_df["timestamp"] = pd.to_datetime(mood_df["timestamp"], utc=True)
    if target_tz is not None:
        mood_df["timestamp"] = mood_df["timestamp"].dt.tz_convert(target_tz)
    mood_df = mood_df.sort_values("timestamp").reset_index(drop=True)
    mood_df["calendarDate"] = mood_df["timestamp"].dt.strftime("%Y-%m-%d")
    return mood_df
