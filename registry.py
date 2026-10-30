import csv
import os
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
REGISTRY_PATH = os.path.join(DATA_DIR, "violation_registry.csv")

FIELDS = ["Date", "Time", "Vehicle_Number", "Violation"]
OCR_CONFIDENCE_THRESHOLD = 0.80


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def _normalize_row(row):
    return {
        "Date": row.get("Date", ""),
        "Time": row.get("Time", ""),
        "Vehicle_Number": row.get("Vehicle_Number", ""),
        "Violation": row.get("Violation", "No Helmet"),
    }


def _migrate_registry():
    """Keep registry rows limited to the four violation record fields."""
    _ensure_dirs()
    if not os.path.exists(REGISTRY_PATH):
        return

    with open(REGISTRY_PATH, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = [_normalize_row(row) for row in reader]

    if fieldnames == FIELDS:
        return

    with open(REGISTRY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_violation(result):
    """Save a verified no-helmet violation, rejecting duplicate same-second events."""
    _migrate_registry()
    _ensure_dirs()

    if not result.get("violation", False):
        return False, "No violation detected."

    vehicle_number = str(result.get("vehicle_number", "")).strip().upper()
    if vehicle_number in {"", "UNKNOWN", "NOT DETECTED"}:
        return False, "Vehicle number is invalid. Violation not saved."
    if not result.get("plate_valid", False):
        return False, "Indian plate validation failed. Violation not saved."

    try:
        confidence = float(result.get("confidence"))
    except (TypeError, ValueError):
        return False, "OCR confidence is invalid. Violation not saved."
    if confidence > 1:
        confidence /= 100
    if confidence < OCR_CONFIDENCE_THRESHOLD:
        return False, "OCR confidence is below the required threshold. Violation not saved."

    now = datetime.now()
    date_value = now.strftime("%Y-%m-%d")
    time_value = now.strftime("%H:%M:%S")
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_vehicle = str(row.get("Vehicle_Number", "")).strip().upper()
                if (
                    existing_vehicle == vehicle_number
                    and row.get("Date") == date_value
                    and row.get("Time") == time_value
                    and row.get("Violation", "").strip().lower() == "no helmet"
                ):
                    return False, f"This violation for {vehicle_number} already exists in the registry."

    file_exists = os.path.exists(REGISTRY_PATH)
    with open(REGISTRY_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(FIELDS)
        writer.writerow([date_value, time_value, vehicle_number, "No Helmet"])

    return True, f"Violation saved for {vehicle_number}"


def get_registry():
    """Read all violation records, newest first for display helpers."""
    _migrate_registry()
    if not os.path.exists(REGISTRY_PATH):
        return []

    with open(REGISTRY_PATH, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [_normalize_row(row) for row in reader]
