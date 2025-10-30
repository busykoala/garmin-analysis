from garmin import (
    init_api,
    export_user_profile,
    export_activity_data,
    export_body_data,
    export_activities_list,
    EXPORT_ROOT,
)
import sys

# Number of days to export (default 30). Update this constant to change the range.
DEFAULT_DAYS_BACK = 30


def main():
    print("🏃‍♂️ Garmin Full Data Export")
    print("=" * 60)

    api = init_api()
    if not api:
        print("❌ Could not initialize Garmin API")
        sys.exit(1)

    export_user_profile(api)
    export_activity_data(api, days_back=DEFAULT_DAYS_BACK)
    export_body_data(api)
    export_activities_list(api)

    print("\n🎉 Export completed!")
    print(f"📁 Data saved in: {EXPORT_ROOT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🚪 Exiting. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
