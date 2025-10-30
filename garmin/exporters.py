from datetime import date, timedelta
from pathlib import Path

from .api import safe_api_call
from .utils import EXPORT_ROOT, save_json


def export_body_battery(api, day):
    """Try multiple method names for body battery export."""
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
                save_json(data, EXPORT_ROOT / "body_battery" / f"{day}_battery.json")
                print(f"✅ Body battery data fetched using {name}()")
                return
            elif err:
                print(f"⚠️ {name} failed: {err}")
    print("⚠️ No working body battery method found.")


def export_user_profile(api):
    success, data, err = safe_api_call(api.get_full_name)
    if success:
        save_json({"full_name": data}, EXPORT_ROOT / "user_profile.json")
    else:
        print(f"⚠️ Could not get user profile: {err}")

    success, device_info, err = safe_api_call(api.get_device_last_used)
    if success:
        save_json(device_info, EXPORT_ROOT / "devices.json")
    else:
        print(f"⚠️ Could not get device info: {err}")


def export_activity_data(api, days_back=30):
    """Fetch daily summaries and activities for last N days."""
    for i in range(days_back):
        d = date.today() - timedelta(days=i)
        day = d.isoformat()

        success, summary, err = safe_api_call(api.get_user_summary, day)
        if success:
            save_json(summary, EXPORT_ROOT / "activities" / f"{day}_summary.json")
        else:
            print(f"⚠️ Summary {day}: {err}")

        success, steps, err = safe_api_call(api.get_steps_data, day)
        if success:
            save_json(steps, EXPORT_ROOT / "activities" / f"{day}_steps.json")

        success, sleep, err = safe_api_call(api.get_sleep_data, day)
        if success:
            save_json(sleep, EXPORT_ROOT / "sleep" / f"{day}_sleep.json")

        success, stress, err = safe_api_call(api.get_stress_data, day)
        if success:
            save_json(stress, EXPORT_ROOT / "stress" / f"{day}_stress.json")

        # Updated adaptive body battery handling
        export_body_battery(api, day)

        success, hydration, err = safe_api_call(api.get_hydration_data, day)
        if success:
            save_json(hydration, EXPORT_ROOT / "hydration" / f"{day}_hydration.json")

        success, hr_data, err = safe_api_call(api.get_heart_rates, day)
        if success:
            save_json(hr_data, EXPORT_ROOT / "heart_rate" / f"{day}_hr.json")


def export_body_data(api):
    """Export long-term measurements like weight, stress, etc."""
    success, weight, err = safe_api_call(api.get_body_composition)
    if success:
        save_json(weight, EXPORT_ROOT / "body" / "body_composition.json")

    success, stress, err = safe_api_call(api.get_stress_data, date.today().isoformat())
    if success:
        save_json(stress, EXPORT_ROOT / "body" / "stress_today.json")

    if hasattr(api, "get_hrv_data"):
        success, hrv, err = safe_api_call(api.get_hrv_data, date.today().isoformat())
        if success:
            save_json(hrv, EXPORT_ROOT / "body" / "hrv_today.json")
    else:
        print("⚠️ Skipping HRV export (method not available)")


def export_activities_list(api):
    """Fetch all recorded activities metadata."""
    print("📦 Fetching list of recorded activities...")
    success, activities, err = safe_api_call(api.get_activities, 0, 100)
    if success:
        save_json(activities, EXPORT_ROOT / "activities" / "activities_list.json")
        print(f"✅ Retrieved {len(activities)} activities")
    else:
        print(f"⚠️ Could not get activities list: {err}")

