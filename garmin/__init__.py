"""Package exports for the refactored modules.

Guarded imports make importing this package safe during static analysis or
partial installs (they'll fall back to None placeholders).
"""

try:
    from .api import init_api, safe_api_call
    from .utils import EXPORT_ROOT, save_json
    from .exporters import (
        export_user_profile,
        export_activity_data,
        export_body_data,
        export_activities_list,
    )
except Exception:
    # Provide placeholders to keep attribute access safe in consumers/tests.
    init_api = None
    safe_api_call = None
    EXPORT_ROOT = None
    save_json = None
    export_user_profile = None
    export_activity_data = None
    export_body_data = None
    export_activities_list = None

__all__ = [
    "init_api",
    "safe_api_call",
    "EXPORT_ROOT",
    "save_json",
    "export_user_profile",
    "export_activity_data",
    "export_body_data",
    "export_activities_list",
]
