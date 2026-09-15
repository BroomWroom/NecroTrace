"""
NecroTrace Forensic Authentication Module
Connects to Firebase Authentication via the Identity Toolkit REST API.
Operates with zero heavy dependencies (pure Python urllib.request & json).
"""

import os
import sys
import json
import urllib.request
import urllib.error
import hashlib
import time
from typing import Dict, Any, Tuple, Optional

# Authorized default demonstration personnel for sandbox / offline evaluation mode
DEMO_REGISTERED_OFFICERS = {
    "coroner@necrotrace.gov": {
        "password": "necrotrace2026",
        "name": "Dr. Tanish Walture",
        "role": "Chief Forensic Pathologist",
        "badge": "CFS-9042",
        "station": "Central Forensic Science Laboratory",
    },
    "examiner@police.gov": {
        "password": "investigation",
        "name": "Insp. V. K. Sharma",
        "role": "Senior Investigating Officer",
        "badge": "IPS-4482",
        "station": "State Police Crime Branch",
    },
    "doctor@forensics.org": {
        "password": "evidence",
        "name": "Dr. A. Sen, M.D.",
        "role": "Forensic Medical Examiner",
        "badge": "WBMC-45826",
        "station": "Department of Forensic Medicine",
    },
}


def get_firebase_config() -> Dict[str, str]:
    """
    Retrieve Firebase Web API credentials safely from Streamlit secrets or environment.
    Never raises an exception if secrets are missing.
    """
    api_key = ""
    project_id = ""

    # Try reading from streamlit.secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "FIREBASE_WEB_API_KEY" in st.secrets:
                api_key = str(st.secrets["FIREBASE_WEB_API_KEY"]).strip()
            elif "firebase" in st.secrets and "web_api_key" in st.secrets["firebase"]:
                api_key = str(st.secrets["firebase"]["web_api_key"]).strip()

            if "FIREBASE_PROJECT_ID" in st.secrets:
                project_id = str(st.secrets["FIREBASE_PROJECT_ID"]).strip()
            elif "firebase" in st.secrets and "project_id" in st.secrets["firebase"]:
                project_id = str(st.secrets["firebase"]["project_id"]).strip()
    except Exception:
        pass

    # Fallback to os.environ
    if not api_key:
        api_key = os.environ.get("FIREBASE_WEB_API_KEY", "").strip()
    if not project_id:
        project_id = os.environ.get("FIREBASE_PROJECT_ID", "").strip()

    return {
        "api_key": api_key,
        "project_id": project_id,
    }


def save_firebase_config(api_key: str, project_id: str) -> Tuple[bool, str]:
    """
    Save Firebase configuration to .streamlit/secrets.toml and update current runtime.
    """
    clean_key = str(api_key).strip()
    clean_proj = str(project_id).strip()

    os.environ["FIREBASE_WEB_API_KEY"] = clean_key
    os.environ["FIREBASE_PROJECT_ID"] = clean_proj

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            st.secrets["FIREBASE_WEB_API_KEY"] = clean_key
            st.secrets["FIREBASE_PROJECT_ID"] = clean_proj
    except Exception:
        pass

    try:
        secrets_dir = os.path.join(os.getcwd(), ".streamlit")
        if not os.path.exists(secrets_dir):
            os.makedirs(secrets_dir, exist_ok=True)
        secrets_path = os.path.join(secrets_dir, "secrets.toml")
        content = (
            "# ==============================================================================\n"
            "# NecroTrace // Local Secrets Configuration (Git-Ignored)\n"
            "# ==============================================================================\n"
            f'FIREBASE_WEB_API_KEY = "{clean_key}"\n'
            f'FIREBASE_PROJECT_ID = "{clean_proj}"\n'
        )
        with open(secrets_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True, "Secrets saved successfully to .streamlit/secrets.toml."
    except Exception as e:
        return False, f"Could not write to .streamlit/secrets.toml: {str(e)}"


def test_firebase_connection(api_key: str) -> Tuple[bool, str]:
    """
    Test live connectivity to Firebase Auth Identity Toolkit with a provided API key.
    """
    clean_key = str(api_key).strip()
    if not clean_key:
        return False, "API key is empty."
    if len(clean_key) < 15:
        return False, "API key appears too short to be a valid Firebase Web API Key."

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:createAuthUri?key={clean_key}"
    payload = {
        "identifier": "test-connectivity-probe@necrotrace.gov",
        "continueUri": "https://necrotrace.streamlit.app",
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True, "Connection verified! Firebase Identity Toolkit authenticated the request successfully."
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            msg = err_body.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        if "API_KEY_INVALID" in msg or "BAD_API_KEY" in msg:
            return False, "Firebase rejected the key: API_KEY_INVALID."
        return True, f"Firebase reached successfully (Response: {msg})."
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def is_firebase_configured() -> bool:
    """Return True if a valid-format Firebase Web API Key is present."""
    cfg = get_firebase_config()
    key = cfg.get("api_key", "")
    return bool(key and len(key) >= 15 and not key.startswith("your-"))


def check_email_registered_in_firebase(email: str) -> Tuple[bool, str]:
    """
    Check whether an email is registered in Firebase Auth using createAuthUri.
    Returns (is_registered, message).
    """
    cfg = get_firebase_config()
    api_key = cfg.get("api_key", "")

    if not is_firebase_configured():
        # Evaluation Sandbox Check
        clean_email = email.strip().lower()
        if clean_email in DEMO_REGISTERED_OFFICERS:
            return True, "Email found in Departmental Medical Examiner Registry (Sandbox Mode)."
        return False, "Email not found in Departmental Medical Examiner Registry."

    # Live Firebase Identity Toolkit API
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:createAuthUri?key={api_key}"
    payload = {
        "identifier": email.strip().lower(),
        "continueUri": "https://necrotrace.streamlit.app",
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            is_reg = bool(data.get("registered", False))
            if is_reg:
                return True, "Email registered in Firebase Forensic Directory."
            else:
                return False, "Email is NOT registered in Firebase Forensic Directory."
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            msg = err_body.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        return False, f"Firebase API Error: {msg}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"


def sign_in_officer(email: str, password: str) -> Dict[str, Any]:
    """
    Authenticate a forensic officer via Firebase Authentication (signInWithPassword).
    Returns a dict with 'success', 'email', 'message', 'token', and 'officer_info'.
    """
    clean_email = email.strip().lower()
    cfg = get_firebase_config()
    api_key = cfg.get("api_key", "")

    # 1. Check Sandbox / Demo Fallback Mode
    if not is_firebase_configured():
        if clean_email in DEMO_REGISTERED_OFFICERS:
            user_data = DEMO_REGISTERED_OFFICERS[clean_email]
            if user_data["password"] == password:
                return {
                    "success": True,
                    "email": clean_email,
                    "message": "Authentication successful (Sandbox Mode).",
                    "mode": "sandbox",
                    "officer_info": {
                        "name": user_data["name"],
                        "role": user_data["role"],
                        "badge": user_data["badge"],
                        "station": user_data["station"],
                    },
                }
            else:
                return {
                    "success": False,
                    "email": clean_email,
                    "message": "Invalid password for registered medical examiner.",
                    "code": "INVALID_PASSWORD",
                }
        else:
            return {
                "success": False,
                "email": clean_email,
                "message": f"Officer email '{clean_email}' is not registered in the Forensic Registry.",
                "code": "EMAIL_NOT_FOUND",
            }

    # 2. Live Firebase Identity Toolkit Authentication
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {
        "email": clean_email,
        "password": password,
        "returnSecureToken": True,
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "success": True,
                "email": data.get("email", clean_email),
                "id_token": data.get("idToken"),
                "local_id": data.get("localId"),
                "mode": "live_firebase",
                "message": "Officer credentials verified against Firebase Auth.",
                "officer_info": {
                    "name": data.get("displayName") or clean_email.split("@")[0].title(),
                    "role": "Authorized Medical Examiner",
                    "badge": f"AUTH-{data.get('localId', '0000')[:6].upper()}",
                    "station": "State Forensic Medical Service",
                },
            }
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            raw_msg = err_body.get("error", {}).get("message", "AUTHENTICATION_FAILED")
        except Exception:
            raw_msg = str(e)

        if "EMAIL_NOT_FOUND" in raw_msg:
            msg = f"Email '{clean_email}' is NOT registered in Firebase Forensic Directory."
            code = "EMAIL_NOT_FOUND"
        elif "INVALID_PASSWORD" in raw_msg or "INVALID_LOGIN_CREDENTIALS" in raw_msg:
            msg = "Incorrect officer credentials/password."
            code = "INVALID_PASSWORD"
        elif "USER_DISABLED" in raw_msg:
            msg = "Officer account has been administratively suspended."
            code = "USER_DISABLED"
        else:
            msg = f"Authentication rejected: {raw_msg}"
            code = raw_msg

        return {
            "success": False,
            "email": clean_email,
            "message": msg,
            "code": code,
        }
    except Exception as e:
        return {
            "success": False,
            "email": clean_email,
            "message": f"Network error during authentication: {str(e)}",
            "code": "NETWORK_ERROR",
        }


def register_officer(email: str, password: str, display_name: str = "") -> Dict[str, Any]:
    """
    Register a new forensic officer in Firebase Authentication (signUp).
    """
    clean_email = email.strip().lower()
    cfg = get_firebase_config()
    api_key = cfg.get("api_key", "")

    if not is_firebase_configured():
        if clean_email in DEMO_REGISTERED_OFFICERS:
            return {
                "success": False,
                "message": "Email already exists in the Forensic Registry.",
                "code": "EMAIL_EXISTS",
            }
        DEMO_REGISTERED_OFFICERS[clean_email] = {
            "password": password,
            "name": display_name or clean_email.split("@")[0].title(),
            "role": "Registered Forensic Examiner",
            "badge": f"MED-{hashlib.md5(clean_email.encode()).hexdigest()[:5].upper()}",
            "station": "Central Medico-Legal Service",
        }
        return {
            "success": True,
            "email": clean_email,
            "message": "Officer registered successfully (Sandbox Mode).",
            "mode": "sandbox",
        }

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"
    payload = {
        "email": clean_email,
        "password": password,
        "returnSecureToken": True,
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "success": True,
                "email": data.get("email", clean_email),
                "id_token": data.get("idToken"),
                "local_id": data.get("localId"),
                "message": "Officer account registered successfully in Firebase.",
            }
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            raw_msg = err_body.get("error", {}).get("message", "REGISTRATION_FAILED")
        except Exception:
            raw_msg = str(e)

        if "EMAIL_EXISTS" in raw_msg:
            msg = "This email is already registered. Please proceed to Sign In."
        elif "WEAK_PASSWORD" in raw_msg:
            msg = "Password is too weak. Please use at least 6 characters."
        else:
            msg = f"Registration rejected: {raw_msg}"

        return {
            "success": False,
            "email": clean_email,
            "message": msg,
            "code": raw_msg,
        }
    except Exception as e:
        return {
            "success": False,
            "email": clean_email,
            "message": f"Network error during registration: {str(e)}",
            "code": "NETWORK_ERROR",
        }


def generate_release_passcode(case_id: str) -> str:
    """
    Generate a human-readable 6-digit release passcode (e.g. NC-8492)
    deterministically linking the mobile verification session to the desktop terminal.
    Valid for the day's case verification.
    """
    clean_case = str(case_id).strip().upper().replace("/", "").replace("-", "")
    day_bucket = time.strftime("%Y%m%d")
    salt = "NECRO_FORENSIC_RELEASE_V1"
    digest = hashlib.sha256(f"{clean_case}|{day_bucket}|{salt}".encode("utf-8")).hexdigest()
    num_part = str(int(digest[:6], 16))[-4:].zfill(4)
    return f"NC-{num_part}"


def verify_release_passcode(case_id: str, entered_code: str) -> bool:
    """Validate whether an entered passcode matches the expected case release passcode."""
    expected = generate_release_passcode(case_id)
    norm_entered = str(entered_code).strip().upper().replace(" ", "")
    if norm_entered == expected:
        return True
    if norm_entered == expected.replace("NC-", ""):
        return True
    return False
