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
    Retrieves full officer profile from Firebase Auth and Firestore.
    """
    clean_email = email.strip().lower()
    cfg = get_firebase_config()
    api_key = cfg.get("api_key", "")
    project_id = cfg.get("project_id", "")

    # 1. Fallback if not configured
    if not is_firebase_configured():
        if clean_email in DEMO_REGISTERED_OFFICERS:
            user_data = DEMO_REGISTERED_OFFICERS[clean_email]
            if user_data["password"] == password:
                return {
                    "success": True,
                    "email": clean_email,
                    "message": "Authentication successful.",
                    "officer_info": {
                        "name": user_data["name"],
                        "role": user_data["role"],
                        "badge": user_data["badge"],
                        "station": user_data["station"],
                        "email": clean_email,
                    },
                }
            return {
                "success": False,
                "email": clean_email,
                "message": "Invalid password for registered medical examiner.",
                "code": "INVALID_PASSWORD",
            }
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
            local_id = data.get("localId", "")
            id_token = data.get("idToken", "")

            # Parse officer details from displayName
            raw_disp = data.get("displayName") or ""
            name = clean_email.split("@")[0].title()
            badge = f"CFS-{local_id[:5].upper()}" if local_id else "CFS-9042"
            station = "Central Forensic Science Laboratory"
            role = "Forensic Medical Examiner"

            if raw_disp:
                parts = [p.strip() for p in raw_disp.split("|")]
                if len(parts) >= 1 and parts[0]:
                    name = parts[0]
                if len(parts) >= 2 and parts[1]:
                    badge = parts[1]
                if len(parts) >= 3 and parts[2]:
                    station = parts[2]
                if len(parts) >= 4 and parts[3]:
                    role = parts[3]

            # Attempt to enrich from Firestore if available
            if project_id and local_id:
                try:
                    fs_url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/officers/{local_id}?key={api_key}"
                    fs_req = urllib.request.Request(
                        fs_url,
                        headers={"Authorization": f"Bearer {id_token}"}
                    )
                    with urllib.request.urlopen(fs_req, timeout=4) as fs_resp:
                        fs_data = json.loads(fs_resp.read().decode("utf-8"))
                        fields = fs_data.get("fields", {})
                        if "name" in fields and fields["name"].get("stringValue"):
                            name = fields["name"]["stringValue"]
                        if "badge" in fields and fields["badge"].get("stringValue"):
                            badge = fields["badge"]["stringValue"]
                        if "station" in fields and fields["station"].get("stringValue"):
                            station = fields["station"]["stringValue"]
                        if "role" in fields and fields["role"].get("stringValue"):
                            role = fields["role"]["stringValue"]
                except Exception:
                    pass

            officer_info = {
                "name": name,
                "role": role,
                "badge": badge,
                "station": station,
                "email": clean_email,
                "local_id": local_id,
            }

            return {
                "success": True,
                "email": clean_email,
                "id_token": id_token,
                "local_id": local_id,
                "mode": "live_firebase",
                "message": f"Officer credentials verified. Welcome, {name}.",
                "officer_info": officer_info,
            }

    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            raw_msg = err_body.get("error", {}).get("message", "AUTHENTICATION_FAILED")
        except Exception:
            raw_msg = str(e)

        if "EMAIL_NOT_FOUND" in raw_msg:
            msg = f"Email '{clean_email}' is not registered. Please enroll first in the Sign-Up tab."
            code = "EMAIL_NOT_FOUND"
        elif "INVALID_PASSWORD" in raw_msg or "INVALID_LOGIN_CREDENTIALS" in raw_msg:
            msg = "Incorrect security password for registered examiner."
            code = "INVALID_PASSWORD"
        elif "USER_DISABLED" in raw_msg:
            msg = "Examiner account has been administratively suspended."
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


def register_officer(
    email: str,
    password: str,
    full_name: str = "",
    badge: str = "",
    station: str = "",
    role: str = "",
    name: str = "",
    **kwargs
) -> Dict[str, Any]:
    """
    Register a new forensic officer in Firebase Authentication (signUp)
    and record officer profile metadata into Firebase.
    """
    clean_email = str(email).strip().lower()
    raw_name = full_name or name or kwargs.get("name", "") or kwargs.get("fullName", "")
    clean_name = str(raw_name).strip() or clean_email.split("@")[0].title()
    raw_badge = badge or kwargs.get("badge_number", "") or kwargs.get("badgeNumber", "")
    clean_badge = str(raw_badge).strip() or "CFS-OFFICER"
    raw_station = station or kwargs.get("police_station", "") or kwargs.get("policeStation", "")
    clean_station = str(raw_station).strip() or "Central Forensic Science Laboratory"
    raw_role = role or kwargs.get("role_title", "")
    clean_role = str(raw_role).strip() or "Forensic Medical Examiner"

    cfg = get_firebase_config()
    api_key = cfg.get("api_key", "")
    project_id = cfg.get("project_id", "")

    # Fallback if not configured
    if not is_firebase_configured():
        if clean_email in DEMO_REGISTERED_OFFICERS:
            return {
                "success": False,
                "message": "Email already exists in the Forensic Registry.",
                "code": "EMAIL_EXISTS",
            }
        officer_info = {
            "name": clean_name,
            "badge": clean_badge,
            "station": clean_station,
            "role": clean_role,
            "email": clean_email,
        }
        DEMO_REGISTERED_OFFICERS[clean_email] = {
            "password": password,
            **officer_info
        }
        return {
            "success": True,
            "email": clean_email,
            "message": "Officer registered successfully.",
            "officer_info": officer_info,
        }

    # 1. Create account in Firebase Auth
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
            local_id = data.get("localId", "")
            id_token = data.get("idToken", "")

        # 2. Update display name in Firebase Auth
        structured_disp = f"{clean_name} | {clean_badge} | {clean_station} | {clean_role}"
        try:
            up_url = f"https://identitytoolkit.googleapis.com/v1/accounts:update?key={api_key}"
            up_payload = {
                "idToken": id_token,
                "displayName": structured_disp,
                "returnSecureToken": True,
            }
            up_req = urllib.request.Request(
                up_url,
                data=json.dumps(up_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(up_req, timeout=8):
                pass
        except Exception:
            pass

        # 3. Try saving structured officer document to Firestore
        if project_id and local_id:
            try:
                fs_url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/officers/{local_id}?key={api_key}"
                fs_payload = {
                    "fields": {
                        "name": {"stringValue": clean_name},
                        "badge": {"stringValue": clean_badge},
                        "station": {"stringValue": clean_station},
                        "role": {"stringValue": clean_role},
                        "email": {"stringValue": clean_email},
                        "enrolled_at": {"stringValue": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                    }
                }
                fs_req = urllib.request.Request(
                    fs_url,
                    data=json.dumps(fs_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {id_token}"},
                    method="PATCH",
                )
                with urllib.request.urlopen(fs_req, timeout=5):
                    pass
            except Exception:
                pass

        officer_info = {
            "name": clean_name,
            "badge": clean_badge,
            "station": clean_station,
            "role": clean_role,
            "email": clean_email,
            "local_id": local_id,
        }

        return {
            "success": True,
            "email": clean_email,
            "id_token": id_token,
            "local_id": local_id,
            "message": "Officer account registered successfully in Firebase.",
            "officer_info": officer_info,
        }

    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            raw_msg = err_body.get("error", {}).get("message", "REGISTRATION_FAILED")
        except Exception:
            raw_msg = str(e)

        if "EMAIL_EXISTS" in raw_msg:
            msg = "This email is already registered. Please proceed to Examiner Sign-In."
        elif "WEAK_PASSWORD" in raw_msg:
            msg = "Password is too weak. Please use at least 6 characters."
        elif "INVALID_EMAIL" in raw_msg:
            msg = "Invalid email format. Please provide an authentic departmental email."
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
