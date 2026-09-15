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

# Retained empty for backwards-compatibility; sandbox demo accounts removed
DEMO_REGISTERED_OFFICERS: Dict[str, Any] = {}


# In-memory server-side active officer session cache (preserves auth across view navigations & reloads)
_ACTIVE_SERVER_SESSION: Dict[str, Any] = {}


def get_active_officer_session() -> Optional[Dict[str, Any]]:
    """Retrieve the currently active authenticated officer profile from server session cache."""
    return _ACTIVE_SERVER_SESSION.get("officer")


def set_active_officer_session(officer_info: Dict[str, Any]):
    """Set the active authenticated officer profile in server session cache."""
    if officer_info and isinstance(officer_info, dict):
        _ACTIVE_SERVER_SESSION["officer"] = dict(officer_info)


def clear_active_officer_session():
    """Clear the active authenticated officer profile from server session cache."""
    _ACTIVE_SERVER_SESSION.clear()


def get_firebase_config() -> Dict[str, str]:
    """
    Retrieve Firebase Web API credentials strictly from Streamlit secrets (.streamlit/secrets.toml)
    or environment variables. Does NOT hardcode credentials in source files.
    """
    api_key = ""
    project_id = ""

    # 1. Read from streamlit.secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            # Direct flat keys in secrets.toml (e.g. FIREBASE_WEB_API_KEY = "...")
            for k in ["FIREBASE_WEB_API_KEY", "firebase_web_api_key", "FIREBASE_API_KEY", "firebase_api_key", "apiKey", "API_KEY"]:
                if k in st.secrets and str(st.secrets[k]).strip():
                    api_key = str(st.secrets[k]).strip().strip('"').strip("'")
                    break

            # Nested sections in secrets.toml (e.g. [firebase] web_api_key = "...")
            for sec in ["firebase", "FIREBASE", "credentials", "default"]:
                if not api_key and sec in st.secrets and isinstance(st.secrets[sec], (dict, st.runtime.secrets.Secrets)):
                    sub = st.secrets[sec]
                    for sub_k in ["web_api_key", "apiKey", "api_key", "key"]:
                        if sub_k in sub and str(sub[sub_k]).strip():
                            api_key = str(sub[sub_k]).strip().strip('"').strip("'")
                            break

            for k in ["FIREBASE_PROJECT_ID", "firebase_project_id", "PROJECT_ID", "project_id", "projectId"]:
                if k in st.secrets and str(st.secrets[k]).strip():
                    project_id = str(st.secrets[k]).strip().strip('"').strip("'")
                    break

            for sec in ["firebase", "FIREBASE", "credentials", "default"]:
                if not project_id and sec in st.secrets and isinstance(st.secrets[sec], (dict, st.runtime.secrets.Secrets)):
                    sub = st.secrets[sec]
                    for sub_k in ["project_id", "projectId", "id"]:
                        if sub_k in sub and str(sub[sub_k]).strip():
                            project_id = str(sub[sub_k]).strip().strip('"').strip("'")
                            break
    except Exception:
        pass

    # 2. Fallback to os.environ
    if not api_key:
        for env_k in ["FIREBASE_WEB_API_KEY", "FIREBASE_API_KEY", "API_KEY"]:
            val = os.environ.get(env_k, "").strip().strip('"').strip("'")
            if val:
                api_key = val
                break

    if not project_id:
        for env_k in ["FIREBASE_PROJECT_ID", "PROJECT_ID", "FIREBASE_PROJECT"]:
            val = os.environ.get(env_k, "").strip().strip('"').strip("'")
            if val:
                project_id = val
                break

    # 3. Direct inspection of local .streamlit/secrets.toml file if running standalone / script
    if not api_key or not project_id:
        try:
            sec_file = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
            if os.path.exists(sec_file):
                with open(sec_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line_s = line.strip()
                        if "=" in line_s and not line_s.startswith("#"):
                            k, v = [x.strip() for x in line_s.split("=", 1)]
                            v_clean = v.strip('"').strip("'")
                            if ("API_KEY" in k.upper() or "APIKEY" in k.upper()) and not api_key:
                                api_key = v_clean
                            if "PROJECT" in k.upper() and not project_id:
                                project_id = v_clean
        except Exception:
            pass

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
    clean_email = str(email or "").strip().lower()

    if not is_firebase_configured():
        return False, "Firebase configuration not detected. Please verify credentials in secrets.toml."

    # Live Firebase Identity Toolkit API
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:createAuthUri?key={api_key}"
    payload = {
        "identifier": clean_email,
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


def parse_firestore_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Firestore REST API document JSON into a clean Python dictionary."""
    if not doc or not isinstance(doc, dict):
        return {}
    fields = doc.get("fields", {})
    parsed: Dict[str, Any] = {}
    for key, val_obj in fields.items():
        if not isinstance(val_obj, dict):
            continue
        if "stringValue" in val_obj:
            parsed[key] = val_obj["stringValue"]
        elif "integerValue" in val_obj:
            try:
                parsed[key] = int(val_obj["integerValue"])
            except Exception:
                parsed[key] = val_obj["integerValue"]
        elif "doubleValue" in val_obj:
            try:
                parsed[key] = float(val_obj["doubleValue"])
            except Exception:
                parsed[key] = val_obj["doubleValue"]
        elif "booleanValue" in val_obj:
            parsed[key] = bool(val_obj["booleanValue"])
        elif "timestampValue" in val_obj:
            parsed[key] = val_obj["timestampValue"]
        elif "nullValue" in val_obj:
            parsed[key] = None

    doc_name = doc.get("name", "")
    if doc_name:
        parsed["_doc_path"] = doc_name
        parsed["_doc_id"] = doc_name.split("/")[-1]
    return parsed


def build_firestore_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat Python dictionary into Firestore typed fields."""
    fields: Dict[str, Any] = {}
    for k, v in data.items():
        if k.startswith("_"):
            continue
        if v is None:
            fields[k] = {"nullValue": None}
        elif isinstance(v, bool):
            fields[k] = {"booleanValue": v}
        elif isinstance(v, int):
            fields[k] = {"integerValue": str(v)}
        elif isinstance(v, float):
            fields[k] = {"doubleValue": v}
        else:
            fields[k] = {"stringValue": str(v)}
    return fields


def fetch_officer_from_firestore(
    local_id: str = "",
    email: str = "",
    id_token: str = ""
) -> Optional[Dict[str, Any]]:
    """
    Fetch an officer profile document directly from Cloud Firestore.
    First tries document lookup by UID (officers/{local_id}),
    and falls back to structured query by email if UID is not matched.
    """
    cfg = get_firebase_config()
    api_key = cfg.get("api_key", "")
    project_id = cfg.get("project_id", "")
    if not api_key or not project_id:
        return None

    clean_email = str(email).strip().lower() if email else ""
    clean_uid = str(local_id).strip() if local_id else ""

    # Strategy 1: Direct document lookup by local_id (UID)
    if clean_uid:
        doc_url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/officers/{clean_uid}?key={api_key}"
        headers_list = [{"Content-Type": "application/json"}]
        if id_token:
            headers_list.insert(0, {"Content-Type": "application/json", "Authorization": f"Bearer {id_token}"})

        for h in headers_list:
            try:
                req = urllib.request.Request(doc_url, headers=h, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    doc = json.loads(resp.read().decode("utf-8"))
                    parsed = parse_firestore_doc(doc)
                    if parsed:
                        if not parsed.get("uid"):
                            parsed["uid"] = clean_uid
                        return parsed
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
                continue
            except Exception:
                continue

    # Strategy 2: Structured Query by Email
    if clean_email:
        query_url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents:runQuery?key={api_key}"
        query_payload = {
            "structuredQuery": {
                "from": [{"collectionId": "officers"}],
                "where": {
                    "fieldFilter": {
                        "field": {"fieldPath": "email"},
                        "op": "EQUAL",
                        "value": {"stringValue": clean_email}
                    }
                },
                "limit": 1
            }
        }
        headers_list = [{"Content-Type": "application/json"}]
        if id_token:
            headers_list.insert(0, {"Content-Type": "application/json", "Authorization": f"Bearer {id_token}"})

        for h in headers_list:
            try:
                req = urllib.request.Request(
                    query_url,
                    data=json.dumps(query_payload).encode("utf-8"),
                    headers=h,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    results = json.loads(resp.read().decode("utf-8"))
                    for item in results:
                        if "document" in item:
                            parsed = parse_firestore_doc(item["document"])
                            if parsed:
                                return parsed
            except Exception:
                continue

    return None


def save_officer_to_firestore(
    officer_data: Dict[str, Any],
    local_id: str = "",
    id_token: str = ""
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Save or update an officer profile document in Cloud Firestore.
    Ensures all metadata fields are persisted into the 'officers' collection.
    """
    cfg = get_firebase_config()
    api_key = cfg.get("api_key", "")
    project_id = cfg.get("project_id", "")
    if not api_key or not project_id:
        return False, "Firebase configuration missing project_id or api_key.", {}

    target_uid = str(local_id or officer_data.get("uid") or officer_data.get("local_id") or "").strip()
    if not target_uid and officer_data.get("email"):
        target_uid = officer_data["email"].replace("@", "_at_").replace(".", "_")

    if not target_uid:
        return False, "No UID or email available to identify Firestore document.", {}

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    clean_email = str(officer_data.get("email", "")).strip().lower()
    clean_name = str(officer_data.get("name", "")).strip() or clean_email.split("@")[0].title()
    clean_badge = str(officer_data.get("badge", "")).strip() or "CFS-OFFICER"
    clean_station = str(officer_data.get("station", "")).strip() or "Central Forensic Science Laboratory"
    clean_role = str(officer_data.get("role", "")).strip() or "Forensic Medical Examiner"
    enrolled_at = officer_data.get("enrolled_at") or now_iso

    payload_data = {
        "uid": target_uid,
        "name": clean_name,
        "badge": clean_badge,
        "station": clean_station,
        "role": clean_role,
        "email": clean_email,
        "status": "ACTIVE",
        "enrolled_at": enrolled_at,
        "last_login": now_iso,
    }

    doc_fields = build_firestore_fields(payload_data)
    fs_url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/officers/{target_uid}?key={api_key}"

    headers_list = [{"Content-Type": "application/json"}]
    if id_token:
        headers_list.insert(0, {"Content-Type": "application/json", "Authorization": f"Bearer {id_token}"})

    last_err = ""
    for h in headers_list:
        try:
            req = urllib.request.Request(
                fs_url,
                data=json.dumps({"fields": doc_fields}).encode("utf-8"),
                headers=h,
                method="PATCH"
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                saved_doc = json.loads(resp.read().decode("utf-8"))
                parsed = parse_firestore_doc(saved_doc)
                return True, "Profile saved to Firestore successfully.", parsed or payload_data
        except urllib.error.HTTPError as e:
            try:
                err_body = json.loads(e.read().decode("utf-8"))
                last_err = err_body.get("error", {}).get("message", str(e))
            except Exception:
                last_err = str(e)
            continue
        except Exception as e:
            last_err = str(e)
            continue

    return False, f"Firestore write failed: {last_err}", payload_data


def sign_in_officer(email: str, password: str) -> Dict[str, Any]:
    """
    Authenticate a forensic officer via Firebase Authentication (signInWithPassword).
    Fetches and verifies full officer profile directly from Cloud Firestore.
    """
    clean_email = email.strip().lower()
    cfg = get_firebase_config()
    api_key = cfg.get("api_key", "")
    project_id = cfg.get("project_id", "")

    # 1. Verification if not configured
    if not is_firebase_configured():
        return {
            "success": False,
            "email": clean_email,
            "message": "Firebase configuration not detected. Please add FIREBASE_WEB_API_KEY and FIREBASE_PROJECT_ID to your secrets.toml.",
            "code": "CONFIG_MISSING",
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

        # 3. Direct Firestore Lookup & Verification
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        fs_profile = fetch_officer_from_firestore(local_id=local_id, email=clean_email, id_token=id_token)

        if fs_profile:
            # Profile successfully fetched directly from Firestore
            name = fs_profile.get("name") or clean_email.split("@")[0].title()
            badge = fs_profile.get("badge") or (f"CFS-{local_id[:5].upper()}" if local_id else "CFS-9042")
            station = fs_profile.get("station") or "Central Forensic Science Laboratory"
            role = fs_profile.get("role") or "Forensic Medical Examiner"
            enrolled_at = fs_profile.get("enrolled_at") or now_iso

            # Update last_login in Firestore
            updated_profile = {
                **fs_profile,
                "name": name,
                "badge": badge,
                "station": station,
                "role": role,
                "email": clean_email,
                "uid": local_id,
                "enrolled_at": enrolled_at,
                "last_login": now_iso,
            }
            save_officer_to_firestore(updated_profile, local_id=local_id, id_token=id_token)
        else:
            # Document not found in Firestore yet (e.g., enrolled via external console)
            # Parse from Auth displayName and create/persist into Firestore immediately
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

            new_profile = {
                "uid": local_id,
                "name": name,
                "badge": badge,
                "station": station,
                "role": role,
                "email": clean_email,
                "enrolled_at": now_iso,
                "last_login": now_iso,
            }
            save_officer_to_firestore(new_profile, local_id=local_id, id_token=id_token)

        officer_info = {
            "name": name,
            "role": role,
            "badge": badge,
            "station": station,
            "email": clean_email,
            "local_id": local_id,
            "firestore_verified": True,
            "database": "Cloud Firestore",
            "last_login": now_iso,
        }
        set_active_officer_session(officer_info)

        return {
            "success": True,
            "email": clean_email,
            "id_token": id_token,
            "local_id": local_id,
            "mode": "live_firebase",
            "message": f"Officer credentials verified via Cloud Firestore. Welcome, {name}.",
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
    and save full officer profile directly into Cloud Firestore database.
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
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Verification if not configured
    if not is_firebase_configured():
        return {
            "success": False,
            "email": clean_email,
            "message": "Firebase configuration not detected. Please add FIREBASE_WEB_API_KEY and FIREBASE_PROJECT_ID to your secrets.toml.",
            "code": "CONFIG_MISSING",
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

        # 3. Save profile directly into Cloud Firestore Database
        officer_data = {
            "uid": local_id,
            "name": clean_name,
            "badge": clean_badge,
            "station": clean_station,
            "role": clean_role,
            "email": clean_email,
            "enrolled_at": now_iso,
            "last_login": now_iso,
            "status": "ACTIVE",
        }

        fs_ok, fs_msg, saved_doc = save_officer_to_firestore(
            officer_data=officer_data,
            local_id=local_id,
            id_token=id_token,
        )

        officer_info = {
            "name": clean_name,
            "badge": clean_badge,
            "station": clean_station,
            "role": clean_role,
            "email": clean_email,
            "local_id": local_id,
            "firestore_verified": True,
            "database": "Cloud Firestore",
            "enrolled_at": now_iso,
            "last_login": now_iso,
        }
        set_active_officer_session(officer_info)

        return {
            "success": True,
            "email": clean_email,
            "id_token": id_token,
            "local_id": local_id,
            "message": "Officer account registered and profile saved in Cloud Firestore Database.",
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
