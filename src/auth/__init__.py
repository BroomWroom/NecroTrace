"""
NecroTrace Authentication Package.
"""

from .firebase_auth import (
    get_firebase_config,
    is_firebase_configured,
    check_email_registered_in_firebase,
    sign_in_officer,
    register_officer,
    generate_release_passcode,
    verify_release_passcode,
    save_firebase_config,
    test_firebase_connection,
    DEMO_REGISTERED_OFFICERS,
)

__all__ = [
    "get_firebase_config",
    "is_firebase_configured",
    "check_email_registered_in_firebase",
    "sign_in_officer",
    "register_officer",
    "generate_release_passcode",
    "verify_release_passcode",
    "save_firebase_config",
    "test_firebase_connection",
    "DEMO_REGISTERED_OFFICERS",
]
