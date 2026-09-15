"""
NecroTrace Authentication Package.
"""

from .firebase_auth import (
    get_firebase_config,
    is_firebase_configured,
    check_email_registered_in_firebase,
    sign_in_officer,
    register_officer,
    fetch_officer_from_firestore,
    save_officer_to_firestore,
    parse_firestore_doc,
    build_firestore_fields,
    get_active_officer_session,
    set_active_officer_session,
    clear_active_officer_session,
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
    "fetch_officer_from_firestore",
    "save_officer_to_firestore",
    "parse_firestore_doc",
    "build_firestore_fields",
    "get_active_officer_session",
    "set_active_officer_session",
    "clear_active_officer_session",
    "generate_release_passcode",
    "verify_release_passcode",
    "save_firebase_config",
    "test_firebase_connection",
    "DEMO_REGISTERED_OFFICERS",
]

