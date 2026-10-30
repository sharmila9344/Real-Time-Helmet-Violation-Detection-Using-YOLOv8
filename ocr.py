from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import os


# ============================================================
# PADDLE OCR
# ============================================================

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def disambiguate_plate_text(text):
    """
    Correct common OCR character misidentifications for license plates:
    - Trailing letters in number segments (e.g. O -> 0, I -> 1, Z -> 2, S -> 5, B -> 8)
    """
    if len(text) < 4:
        return text

    chars = list(text)

    # Trailing 1-4 characters of a license plate are always digits!
    for i in range(len(chars) - 1, max(len(chars) - 5, 1), -1):
        if chars[i] in ('O', 'D', 'Q'):
            chars[i] = '0'
        elif chars[i] in ('I', 'L'):
            chars[i] = '1'
        elif chars[i] == 'Z':
            chars[i] = '2'
        elif chars[i] == 'S':
            chars[i] = '5'
        elif chars[i] == 'B':
            chars[i] = '8'
        elif chars[i].isdigit():
            continue
        else:
            break

    return "".join(chars)


def clean_plate_text(text):
    """
    Remove spaces, special characters, and disambiguate characters.
    """

    text = text.upper()

    text = re.sub(
        r'[^A-Z0-9]',
        '',
        text
    )

    return disambiguate_plate_text(text)


# ============================================================
# VALIDATE INDIAN VEHICLE NUMBER
# ============================================================

def validate_indian_plate(text):
    """
    Validates Indian vehicle registration formats:
    - Standard RTO: MH12AB1234, KA01AB1234, KL01CA9274
    - Special/Series: MHVAAA0060, MHVAAA0000, DL1A1234
    - Bharat Series: 22BH1234AA
    """

    patterns = [
        r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$',  # Standard: MH12AB1234
        r'^[A-Z]{2}[A-Z0-9]{1,4}\d{1,4}$',     # Series/Custom: MHVAAA0060
        r'^\d{2}BH\d{1,4}[A-Z]{1,2}$'           # Bharat Series: 22BH1234AA
    ]

    return any(bool(re.match(p, text)) for p in patterns)


# ============================================================
# OCR ONE IMAGE
# ============================================================

def run_ocr(image):
    """
    Run PaddleOCR on an image.
    """

    temp_path = "temp_plate.jpg"

    cv2.imwrite(
        temp_path,
        image
    )

    result = ocr.predict(
        temp_path
    )

    texts = []
    scores = []

    for res in result:

        if 'rec_texts' in res:

            texts.extend(
                res['rec_texts']
            )

        if 'rec_scores' in res:

            scores.extend(
                res['rec_scores']
            )

    if not texts:

        return "", 0.0

    raw_text = "".join(texts)

    confidence = (
        float(np.mean(scores))
        if scores
        else 0.0
    )

    return raw_text, confidence


# ============================================================
# PREPROCESS PLATE
# ============================================================

def create_variants(plate_crop):
    """
    Create multiple versions of the plate
    to improve OCR accuracy.
    """

    variants = []


    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    variants.append(
        plate_crop
    )


    # --------------------------------------------------------
    # Upscaled image
    # --------------------------------------------------------

    upscaled = cv2.resize(
        plate_crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    variants.append(
        upscaled
    )


    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )

    variants.append(
        gray
    )


    # --------------------------------------------------------
    # Contrast enhancement
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    variants.append(
        enhanced
    )


    # --------------------------------------------------------
    # Otsu threshold
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    variants.append(
        otsu
    )


    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    variants.append(
        adaptive
    )


    return variants


# ============================================================
# READ LICENSE PLATE
# ============================================================

def read_plate(plate_crop):
    """
    Read license plate using multiple OCR
    preprocessing variants.
    """

    if plate_crop is None:
        return {
            "plate_number": "UNKNOWN",
            "confidence": 0.0,
            "raw_text": "",
            "valid": False
        }


    # --------------------------------------------------------
    # Generate variants
    # --------------------------------------------------------

    variants = create_variants(
        plate_crop
    )


    results = []


    # --------------------------------------------------------
    # Run OCR on every variant
    # --------------------------------------------------------

    for image in variants:

        raw_text, confidence = run_ocr(
            image
        )

        cleaned = clean_plate_text(
            raw_text
        )

        if cleaned:

            results.append({
                "text": cleaned,
                "confidence": confidence,
                "raw": raw_text
            })


    # --------------------------------------------------------
    # Display/debug information in terminal
    # --------------------------------------------------------

    print("\n========== OCR RESULTS ==========")

    for result in results:

        print(
            f"Text: {result['text']} | "
            f"Confidence: {result['confidence']:.3f}"
        )

    print("=================================\n")


    # --------------------------------------------------------
    # Find valid plate results
    # --------------------------------------------------------

    valid_results = [

        result
        for result in results

        if validate_indian_plate(
            result["text"]
        )
    ]


    # --------------------------------------------------------
    # If valid plate found
    # --------------------------------------------------------

    if valid_results:

        best = max(
            valid_results,
            key=lambda x: x["confidence"]
        )

        return {
            "plate_number": best["text"],
            "confidence": best["confidence"],
            "raw_text": best["raw"],
            "valid": True
        }


    # --------------------------------------------------------
    # No valid plate
    # --------------------------------------------------------

    if results:

        best = max(
            results,
            key=lambda x: x["confidence"]
        )

        return {
            "plate_number": "UNKNOWN",
            "confidence": best["confidence"],
            "raw_text": best["raw"],
            "valid": False
        }


    return {
        "plate_number": "UNKNOWN",
        "confidence": 0.0,
        "raw_text": "",
        "valid": False
    }