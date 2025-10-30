from datetime import date, timedelta
from pathlib import Path

from .api import safe_api_call
from .utils import EXPORT_ROOT, save_json


def _exists(path: Path) -> bool:
    """Return True if path exists (file already downloaded)."""
    return path.exists()


def export_body_battery(api, day):
    """Try multiple method names for body battery export.

    Skip downloading if the target file already exists.
    """
    target = EXPORT_ROOT / "body_battery" / f"{day}_battery.json"
    if _exists(target):
        print(f"✅ Skipping body battery for {day} (already exists)")
        return

    methods = [
        "get_body_battery_data",
        "get_body_battery",
        "get_body_battery_json",
        "get_wellness",
        "get_stats",
    ]
    for name in methods:
        if hasattr(api, name):
            print(f"🔍 Trying {name}() for body battery...")
            success, data, err = safe_api_call(getattr(api, name), day)
            if success and data:
                save_json(data, target)
                print(f"✅ Body battery data fetched using {name}()")
                return
            elif err:
                print(f"⚠️ {name} failed: {err}")
    print("⚠️ No working body battery method found.")


def export_user_profile(api):
    user_path = EXPORT_ROOT / "user_profile.json"
    if not _exists(user_path):
        success, data, err = safe_api_call(api.get_full_name)
        if success:
            save_json({"full_name": data}, user_path)
        else:
            print(f"⚠️ Could not get user profile: {err}")
    else:
        print("✅ Skipping user profile (already exists)")

    devices_path = EXPORT_ROOT / "devices.json"
    if not _exists(devices_path):
        success, device_info, err = safe_api_call(api.get_device_last_used)
        if success:
            save_json(device_info, devices_path)
        else:
            print(f"⚠️ Could not get device info: {err}")
    else:
        print("✅ Skipping devices info (already exists)")


def export_activity_data(api, days_back=30):
    """Fetch daily summaries and activities for last N days.

    Skips any files that are already present on disk.
    """
    for i in range(days_back):
        d = date.today() - timedelta(days=i)
        day = d.isoformat()

        summary_path = EXPORT_ROOT / "activities" / f"{day}_summary.json"
        if not _exists(summary_path):
            success, summary, err = safe_api_call(api.get_user_summary, day)
            if success:
                save_json(summary, summary_path)
            else:
                print(f"⚠️ Summary {day}: {err}")
        else:
            print(f"✅ Skipping summary {day} (already exists)")

        steps_path = EXPORT_ROOT / "activities" / f"{day}_steps.json"
        if not _exists(steps_path):
            success, steps, err = safe_api_call(api.get_steps_data, day)
            if success:
                save_json(steps, steps_path)
        else:
            print(f"✅ Skipping steps {day} (already exists)")

        sleep_path = EXPORT_ROOT / "sleep" / f"{day}_sleep.json"
        if not _exists(sleep_path):
            success, sleep, err = safe_api_call(api.get_sleep_data, day)
            if success:
                save_json(sleep, sleep_path)
        else:
            print(f"✅ Skipping sleep {day} (already exists)")

        stress_path = EXPORT_ROOT / "stress" / f"{day}_stress.json"
        if not _exists(stress_path):
            success, stress, err = safe_api_call(api.get_stress_data, day)
            if success:
                save_json(stress, stress_path)
        else:
            print(f"✅ Skipping stress {day} (already exists)")

        # Updated adaptive body battery handling
        export_body_battery(api, day)

        hydration_path = EXPORT_ROOT / "hydration" / f"{day}_hydration.json"
        if not _exists(hydration_path):
            success, hydration, err = safe_api_call(api.get_hydration_data, day)
            if success:
                save_json(hydration, hydration_path)
        else:
            print(f"✅ Skipping hydration {day} (already exists)")

        hr_path = EXPORT_ROOT / "heart_rate" / f"{day}_hr.json"
        if not _exists(hr_path):
            success, hr_data, err = safe_api_call(api.get_heart_rates, day)
            if success:
                save_json(hr_data, hr_path)
        else:
            print(f"✅ Skipping heart rate {day} (already exists)")


def export_body_data(api):
    """Export long-term measurements like weight, stress, etc."""
    body_comp_path = EXPORT_ROOT / "body" / "body_composition.json"
    if not _exists(body_comp_path):
        success, weight, err = safe_api_call(api.get_body_composition)
        if success:
            save_json(weight, body_comp_path)
    else:
        print("✅ Skipping body composition (already exists)")

    stress_today_path = EXPORT_ROOT / "body" / "stress_today.json"
    if not _exists(stress_today_path):
        success, stress, err = safe_api_call(api.get_stress_data, date.today().isoformat())
        if success:
            save_json(stress, stress_today_path)
    else:
        print("✅ Skipping stress_today (already exists)")

    hrv_path = EXPORT_ROOT / "body" / "hrv_today.json"
    if hasattr(api, "get_hrv_data"):
        if not _exists(hrv_path):
            success, hrv, err = safe_api_call(api.get_hrv_data, date.today().isoformat())
            if success:
                save_json(hrv, hrv_path)
        else:
            print("✅ Skipping hrv_today (already exists)")
    else:
        print("⚠️ Skipping HRV export (method not available)")


def export_activities_list(api):
    """Fetch all recorded activities metadata."""
    target = EXPORT_ROOT / "activities" / "activities_list.json"
    if _exists(target):
        print("✅ Skipping activities_list (already exists)")
        return

    print("📦 Fetching list of recorded activities...")
    success, activities, err = safe_api_call(api.get_activities, 0, 100)
    if success:
        save_json(activities, target)
        print(f"✅ Retrieved {len(activities)} activities")
    else:
        print(f"⚠️ Could not get activities list: {err}")

