import os
import logging
import re
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from twilio.base.exceptions import TwilioException
from twilio.rest import Client

ENV_PATH = Path(__file__).resolve().parent / ".env"
logger = logging.getLogger(__name__)
ACCOUNT_SID_PATTERN = re.compile(r"^AC[0-9a-fA-F]{32}$")
E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")
FAILED_STATUSES = {"failed", "undelivered", "canceled"}
TRIAL_SMS_TEMPLATE = "sms_customer_support"

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
ACCOUNT_SID_IN_TEXT = re.compile(r"AC[0-9a-fA-F]{32}")

TWILIO_ERROR_HINTS = {
    20003: (
        "Twilio reports this Account SID is unauthorized or belongs to an expired trial. "
        "In Console, open the account whose SID fingerprint matches the one below, then copy "
        "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER from that same active account. "
        "Stop Streamlit completely and start it again after saving .env."
    ),
    20001: "Twilio rejected the request because the Account SID or Auth Token is invalid.",
    21608: "The destination number is not verified for this Twilio trial account.",
    21211: "The destination phone number is invalid.",
    21606: "The sender number is not a valid Twilio number for this account.",
    572006: (
        "This Twilio trial account only accepts predefined SMS templates. "
        f"Use the approved {TRIAL_SMS_TEMPLATE} template."
    ),
}


def _fingerprint_sid(account_sid):
    sid = (account_sid or "").strip()
    if not sid:
        return "missing"
    if len(sid) < 8:
        return "invalid"
    return f"{sid[:4]}...{sid[-4:]}"


def _config_value(file_values, key):
    """Prefer the project-root .env value, including an explicitly empty value."""
    if key in file_values:
        return str(file_values.get(key) or "").strip()
    return os.getenv(key, "").strip()


def _load_sms_config():
    """Reload the project-local .env so a long-running app cannot keep stale values."""
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    file_values = dotenv_values(ENV_PATH)
    config = {
        "account_sid": _config_value(file_values, "TWILIO_ACCOUNT_SID"),
        "auth_token": _config_value(file_values, "TWILIO_AUTH_TOKEN"),
        "from_number": _config_value(file_values, "TWILIO_PHONE_NUMBER"),
        "to_number": _config_value(file_values, "ALERT_PHONE_NUMBER"),
    }
    logger.info(
        "Twilio configuration loaded from %s; Account SID fingerprint: %s",
        ENV_PATH,
        _fingerprint_sid(config["account_sid"]),
    )
    return config


def _redact_error(message, config):
    text = ANSI_PATTERN.sub("", str(message or ""))
    for value in config.values():
        if value:
            text = text.replace(value, "[REDACTED]")
    text = ACCOUNT_SID_IN_TEXT.sub(lambda match: _fingerprint_sid(match.group(0)), text)
    return " ".join(text.split())


def _exception_code(exc):
    code = getattr(exc, "code", None)
    if code:
        try:
            return int(code)
        except (TypeError, ValueError):
            pass
    text = f"{getattr(exc, 'msg', '')} {exc}"
    match = re.search(r"\b(20003|20001|21608|21211|21606|572006)\b", text)
    return int(match.group(1)) if match else None


def _safe_twilio_detail(exc, config):
    code = _exception_code(exc)
    raw_msg = getattr(exc, "msg", None) or getattr(exc, "detail", None) or ""
    error_text = _redact_error(raw_msg, config)
    fingerprint = _fingerprint_sid(config.get("account_sid"))
    hint = TWILIO_ERROR_HINTS.get(code)
    if code == 20003 or "trial account has expired" in error_text.lower():
        return (
            f"Twilio API error 20003: trial expired or this Account SID cannot send SMS. "
            f"Request used Account SID fingerprint {fingerprint}. {TWILIO_ERROR_HINTS[20003]}"
        )
    detail = f"Twilio API error {code or 'unknown'}"
    if error_text:
        detail = f"{detail}: {error_text}"
    if hint:
        detail = f"{detail} {hint}"
    return detail


def get_sms_config_status():
    """Return safe configuration status without exposing environment values."""
    config = _load_sms_config()
    return {
        "TWILIO_ACCOUNT_SID": bool(config["account_sid"]),
        "TWILIO_AUTH_TOKEN": bool(config["auth_token"]),
        "TWILIO_PHONE_NUMBER": bool(config["from_number"]),
        "ALERT_PHONE_NUMBER": bool(config["to_number"]),
        "account_sid_fingerprint": _fingerprint_sid(config["account_sid"]),
        "from_configured": bool(config["from_number"]),
        "to_configured": bool(config["to_number"]),
    }


def send_sms_alert(vehicle_number, ocr_confidence):
    """Send the trial-approved template after registry acceptance.

    Keep vehicle_number for API compatibility; trial SMS templates cannot include it.
    """
    del vehicle_number
    config = _load_sms_config()
    missing = [
        name
        for name, key in (
            ("TWILIO_ACCOUNT_SID", "account_sid"),
            ("TWILIO_AUTH_TOKEN", "auth_token"),
            ("TWILIO_PHONE_NUMBER", "from_number"),
            ("ALERT_PHONE_NUMBER", "to_number"),
        )
        if not config[key]
    ]
    if missing:
        return False, "not_configured", "Missing Twilio configuration: " + ", ".join(missing)

    if not ACCOUNT_SID_PATTERN.fullmatch(config["account_sid"]):
        return False, "invalid_configuration", "TWILIO_ACCOUNT_SID must be a valid Account SID beginning with AC."
    if not E164_PATTERN.fullmatch(config["from_number"]):
        return False, "invalid_configuration", "TWILIO_PHONE_NUMBER must be in E.164 format (for example, +15551234567)."
    if not E164_PATTERN.fullmatch(config["to_number"]):
        return False, "invalid_configuration", "ALERT_PHONE_NUMBER must be in E.164 format (for example, +15551234567)."

    try:
        confidence = float(ocr_confidence)
        if confidence <= 1:
            confidence *= 100
    except (TypeError, ValueError):
        return False, "invalid_configuration", "OCR confidence must be numeric."

    try:
        client = Client(
            username=config["account_sid"],
            password=config["auth_token"],
            account_sid=config["account_sid"],
            environment={},
        )
        message = client.messages.create(
            body=TRIAL_SMS_TEMPLATE,
            from_=config["from_number"],
            to=config["to_number"],
        )
        message_sid = getattr(message, "sid", None)
        message_status = (getattr(message, "status", None) or "").lower()
        if not message_sid or message_status in FAILED_STATUSES:
            return False, "failed", f"Twilio did not accept the SMS request (status: {message_status or 'unknown'})."
        return True, "accepted", message_sid
    except TwilioException as exc:
        return False, "failed", _safe_twilio_detail(exc, config)
    except Exception as exc:
        error_text = _redact_error(str(exc), config)
        if "trial account has expired" in error_text.lower() or "20003" in error_text:
            return False, "failed", _safe_twilio_detail(exc, config)
        return False, "failed", f"SMS request failed: {error_text}"


def send_sms(vehicle_number, ocr_confidence=0.0):
    """Backward-compatible wrapper for the app's SMS integration."""
    sent, _, detail = send_sms_alert(vehicle_number, ocr_confidence)
    return sent, detail
