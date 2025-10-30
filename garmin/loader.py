import json
from pathlib import Path
from dateutil import tz
import pandas as pd
from typing import Tuple, Optional


def _infer_tz_from_summary(summary: dict):
    # prefer wellnessStartTimeLocal / wellnessStartTimeGmt pair if available
    local = summary.get("wellnessStartTimeLocal")
    gmt = summary.get("wellnessStartTimeGmt")
    if local and gmt:
        try:
            t_local = pd.to_datetime(local)
            t_gmt = pd.to_datetime(gmt)
            offset_seconds = int((t_local - t_gmt).total_seconds())
            return tz.tzoffset(None, offset_seconds)
        except Exception:
            pass
    # try timezoneOffset in nested events (ms)
    events = summary.get("bodyBatteryActivityEventList") or summary.get("bodyBatteryActivityEvent")
    if events and isinstance(events, list) and len(events) > 0:
        ev = events[0]
        if "timezoneOffset" in ev:
            offset_ms = int(ev.get("timezoneOffset") or 0)
            return tz.tzoffset(None, int(offset_ms / 1000))
    # fallback to UTC
    return tz.tzutc()


def _read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        txt = f.read()
    # some files include // comments at top; remove //... lines
    lines = [l for l in txt.splitlines() if not l.strip().startswith("//")]
    return json.loads("\n".join(lines))


def _parse_steps(path: Path, local_tz, freq="1min") -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    arr = _read_json(path)
    parts = []
    for item in arr:
        start = pd.to_datetime(item["startGMT"]) if isinstance(item["startGMT"], str) else pd.to_datetime(item.get("startGMT"))
        end = pd.to_datetime(item["endGMT"]) if isinstance(item["endGMT"], str) else pd.to_datetime(item.get("endGMT"))
        # convert to aware UTC then to local
        start_utc = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end_utc = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        start_local = start_utc.tz_convert(local_tz)
        end_local = end_utc.tz_convert(local_tz)
        steps = item.get("steps", 0) or 0
        # number of whole minutes in interval
        minutes = int((end_local - start_local).total_seconds() / 60)
        if minutes <= 0:
            continue
        per_min = steps / minutes
        idx = pd.date_range(start_local, periods=minutes, freq=freq, tz=local_tz)
        s = pd.Series(per_min, index=idx)
        parts.append(s)
    if not parts:
        return pd.Series(dtype=float)
    res = pd.concat(parts).sort_index()
    # ensure Series (concat can return DataFrame if shapes differ)
    if isinstance(res, pd.DataFrame):
        res = res.squeeze()
    return res


def _parse_point_series(path: Path, array_key: str, local_tz, freq="1min") -> Tuple[pd.Series, pd.Series]:
    # returns (resampled_mean_series, presence_mask_series_before_interp)
    if not path.exists():
        return pd.Series(dtype=float), pd.Series(dtype=bool)
    data = _read_json(path)
    values = data.get(array_key) or data.get("heartRateValues") or data.get("stressValuesArray") or data
    if not isinstance(values, list):
        return pd.Series(dtype=float), pd.Series(dtype=bool)
    times = []
    vals = []
    for v in values:
        if not v:
            continue
        ts = int(v[0])
        value = v[1]
        t = pd.to_datetime(ts, unit="ms", utc=True).tz_convert(local_tz)
        times.append(t)
        vals.append(value)
    if not times:
        return pd.Series(dtype=float), pd.Series(dtype=bool)
    sr = pd.Series(vals, index=pd.DatetimeIndex(times))
    # group into minute bins (mean)
    res = sr.resample(freq).mean()
    presence = (sr.resample(freq).count() > 0).astype("boolean")
    return res, presence


def _parse_sleep(path: Path, local_tz, freq="1min") -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    data = _read_json(path)
    parts = []
    # sleepMovement is an array of minute buckets
    sm = data.get("sleepMovement") or []
    for item in sm:
        start = pd.to_datetime(item.get("startGMT"))
        start_utc = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        start_local = start_utc.tz_convert(local_tz)
        minutes = int((pd.to_datetime(item.get("endGMT")).tz_localize("UTC").tz_convert(local_tz) - start_local).total_seconds() / 60)
        if minutes <= 0:
            continue
        idx = pd.date_range(start_local, periods=minutes, freq=freq, tz=local_tz)
        s = pd.Series(item.get("activityLevel"), index=idx)
        parts.append(s)
    if not parts:
        return pd.Series(dtype=float)
    res = pd.concat(parts).sort_index()
    if isinstance(res, pd.DataFrame):
        res = res.squeeze()
    return res


def _parse_body_battery(path: Path, local_tz, freq="1min") -> Tuple[pd.Series, pd.Series]:
    if not path.exists():
        return pd.Series(dtype=float), pd.Series(dtype="boolean")
    arr = _read_json(path)
    if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict):
        arr = arr[0].get("bodyBatteryValuesArray") or []
    if not isinstance(arr, list):
        return pd.Series(dtype=float), pd.Series(dtype="boolean")
    times = []
    vals = []
    for v in arr:
        ts = int(v[0])
        val = v[1]
        t = pd.to_datetime(ts, unit="ms", utc=True).tz_convert(local_tz)
        times.append(t)
        vals.append(val)
    if not times:
        return pd.Series(dtype=float), pd.Series(dtype="boolean")
    sr = pd.Series(vals, index=pd.DatetimeIndex(times))
    # upsample to minute index via nearest forward-fill
    res = sr.resample(freq).nearest(limit=1)
    # ensure numeric dtype to avoid object-dtype downcasting warnings on ffill
    res = pd.to_numeric(res, errors="coerce")
    # then forward fill
    res = res.ffill()
    presence = (sr.resample(freq).count() > 0).astype("boolean")
    return res, presence


def structure_data(export_path: str = "garmin_export", freq: str = "1min", interpolate_gaps: bool = True, max_interp_minutes: int = 5, last_n_days: Optional[int] = 3):
    """Load recent garmin_export and build a combined per-minute multivariate time-series across all days.

    Returns (df_all_days, daily_summary_df)
    - df_all_days: index = timezone-aware timestamps (local), columns = channels + presence masks + calendarDate
    - daily_summary_df: one row per calendarDate with fields from summary.json

    Args:
    - export_path: base path to garmin_export folder
    - freq: resampling frequency (default: 1min)
    - interpolate_gaps: whether to interpolate gaps in heart rate and stress data (default: True)
    - max_interp_minutes: maximum gap size in minutes to interpolate (default: 5)
    - last_n_days: number of most recent days to include (default: 3). If None, include all days.
    """
    root = Path(export_path)
    activities_dir = root / "activities"
    if not activities_dir.exists():
        raise FileNotFoundError(f"{activities_dir} not found")

    summary_files = sorted(activities_dir.glob("*_summary.json"))
    # allow loading only the most recent N days; if last_n_days is None load all
    if last_n_days is not None and isinstance(last_n_days, int) and last_n_days > 0:
        summary_files = summary_files[-last_n_days:]

    day_frames = []
    daily_rows = []

    # run processing
    for sf in summary_files:
        try:
            summary = _read_json(sf)
        except Exception:
            continue
        calendar_date = summary.get("calendarDate") or summary.get("date")
        if not calendar_date:
            continue
        local_tz = _infer_tz_from_summary(summary)
        # Determine day's window
        try:
            start_gmt = summary.get("wellnessStartTimeGmt") or summary.get("startTimestampGMT")
            end_gmt = summary.get("wellnessEndTimeGmt") or summary.get("endTimestampGMT")
            if start_gmt and end_gmt:
                start = pd.to_datetime(start_gmt).tz_localize("UTC").tz_convert(local_tz)
                end = pd.to_datetime(end_gmt).tz_localize("UTC").tz_convert(local_tz)
            else:
                # full local day
                start = pd.to_datetime(calendar_date + "T00:00:00").tz_localize(local_tz)
                end = start + pd.Timedelta(days=1)
        except Exception:
            start = pd.to_datetime(calendar_date + "T00:00:00").tz_localize(local_tz)
            end = start + pd.Timedelta(days=1)

        # make end exclusive by subtracting one minute
        minute_index = pd.date_range(start, end - pd.Timedelta(minutes=1), freq=freq, tz=local_tz)

        # parse source files
        base = activities_dir
        steps_path = base / f"{calendar_date}_steps.json"
        hr_path = root / "heart_rate" / f"{calendar_date}_hr.json"
        stress_path = root / "stress" / f"{calendar_date}_stress.json"
        sleep_path = root / "sleep" / f"{calendar_date}_sleep.json"
        body_batt_path = root / "body_battery" / f"{calendar_date}_battery.json"

        steps_series = _parse_steps(steps_path, local_tz, freq)
        hr_series, hr_presence = _parse_point_series(hr_path, "heartRateValues", local_tz, freq)
        stress_series, stress_presence = _parse_point_series(stress_path, "stressValuesArray", local_tz, freq)
        sleep_series = _parse_sleep(sleep_path, local_tz, freq)
        try:
            body_series, body_presence = _parse_body_battery(body_batt_path, local_tz, freq)
        except Exception:
            body_series = pd.Series(dtype=float)
            body_presence = pd.Series(dtype=bool)

        # reindex to minute_index
        df = pd.DataFrame(index=minute_index)
        df["steps_per_min"] = steps_series.reindex(minute_index).astype(float)
        df["heart_rate"] = hr_series.reindex(minute_index).astype(float)
        df["stress_level"] = stress_series.reindex(minute_index).astype(float)
        df["sleep_movement"] = sleep_series.reindex(minute_index).astype(float)
        df["body_battery"] = body_series.reindex(minute_index).astype(float)

        # presence masks
        df["steps_present"] = ~df["steps_per_min"].isna()
        # reindex presence series using fill_value=False to avoid creating object-dtype NaNs
        hrp_reindexed = hr_presence.reindex(minute_index, fill_value=False)
        df["heart_rate_present"] = pd.Series(hrp_reindexed.astype("boolean"), index=minute_index)

        sp_reindexed = stress_presence.reindex(minute_index, fill_value=False)
        df["stress_present"] = pd.Series(sp_reindexed.astype("boolean"), index=minute_index)

        df["sleep_present"] = ~df["sleep_movement"].isna()
        try:
            bp_reindexed = body_presence.reindex(minute_index, fill_value=False)
            df["body_battery_present"] = pd.Series(bp_reindexed.astype("boolean"), index=minute_index)
        except Exception:
            df["body_battery_present"] = ~df["body_battery"].isna()

        # activity level from steps file (categorical)
        # read raw steps file again to get activity labels per bucket
        activity_levels = []
        if steps_path.exists():
            arr = _read_json(steps_path)
            for item in arr:
                start = pd.to_datetime(item["startGMT"]).tz_localize("UTC").tz_convert(local_tz)
                end = pd.to_datetime(item["endGMT"]).tz_localize("UTC").tz_convert(local_tz)
                minutes = int((end - start).total_seconds() / 60)
                if minutes <= 0:
                    continue
                idx = pd.date_range(start, periods=minutes, freq=freq, tz=local_tz)
                s = pd.Series(item.get("primaryActivityLevel"), index=idx)
                activity_levels.append(s)
        if activity_levels:
            activity = pd.concat(activity_levels).reindex(minute_index)
        else:
            activity = pd.Series(index=minute_index, dtype=object)
        df["activity_level"] = activity

        # interpolation for HR and stress (limit gaps)
        if interpolate_gaps:
            df["heart_rate"] = df["heart_rate"].interpolate(method="time", limit=max_interp_minutes)
            df["stress_level"] = df["stress_level"].interpolate(method="time", limit=max_interp_minutes)

        # calendarDate column (string)
        df["calendarDate"] = calendar_date

        # attach selected daily summary fields (minimal metadata)
        daily_meta = {k: summary.get(k) for k in [
            "totalSteps",
            "totalKilocalories",
            "activeKilocalories",
            "totalDistanceMeters",
            "restingHeartRate",
            "restingHeartRate",
            "minHeartRate",
            "maxHeartRate",
            "averageStressLevel",
            "bodyBatteryAtWakeTime",
            "bodyBatteryMostRecentValue",
            "sleepingSeconds",
            "deepSleepSeconds",
            "lightSleepSeconds",
            "remSleepSeconds",
            "sleepScore"
        ] if summary.get(k) is not None}
        daily_meta["calendarDate"] = calendar_date
        daily_rows.append(daily_meta)

        day_frames.append(df)

    if not day_frames:
        return pd.DataFrame(), pd.DataFrame()

    df_all = pd.concat(day_frames).sort_index()

    daily_summary_df = pd.DataFrame(daily_rows).drop_duplicates(subset=["calendarDate"]).set_index("calendarDate")

    return df_all, daily_summary_df
