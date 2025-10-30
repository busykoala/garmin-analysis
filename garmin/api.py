from pathlib import Path
import os
from getpass import getpass

from garth.exc import GarthHTTPError
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)


def safe_api_call(api_method, *args, **kwargs):
    """Safe API call wrapper with robust error handling."""
    try:
        result = api_method(*args, **kwargs)
        return True, result, None
    except (
        GarthHTTPError,
        GarminConnectAuthenticationError,
        GarminConnectConnectionError,
        GarminConnectTooManyRequestsError,
        FileNotFoundError,
        Exception,
    ) as e:
        return False, None, str(e)


def init_api():
    """Authenticate with Garmin and handle MFA if needed."""
    tokenstore = Path(os.getenv("GARMINTOKENS", "~/.garminconnect")).expanduser()
    print(f"🔐 Token storage: {tokenstore}")

    try:
        garmin = Garmin()
        garmin.login(str(tokenstore))
        print("✅ Logged in using saved tokens")
        return garmin
    except Exception:
        print("🔑 No valid tokens found. Starting interactive login...")

    email = os.getenv("EMAIL") or input("Login email: ")
    password = os.getenv("PASSWORD") or getpass("Enter password: ")

    garmin = Garmin(email=email, password=password, is_cn=False, return_on_mfa=True)
    result1, result2 = garmin.login()

    if result1 == "needs_mfa":
        mfa = input("Please enter your MFA code: ")
        garmin.resume_login(result2, mfa)
        print("✅ MFA successful!")

    garmin.garth.dump(str(tokenstore))
    print(f"💾 Tokens saved to {tokenstore}")
    return garmin
