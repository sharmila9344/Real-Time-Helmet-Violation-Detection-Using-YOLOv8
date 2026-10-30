# Real-Time Helmet Violation Detection System Using YOLOv8

## Project Overview

The **Real-Time Helmet Violation Detection System** is a computer vision-based application designed to detect motorcycle riders who are not wearing helmets, recognize vehicle registration numbers, maintain violation records, and send SMS notifications.

The system integrates deep learning, image processing, Optical Character Recognition (OCR), and a web-based interface to support automated traffic violation monitoring.

## Objectives

- Detect motorcycle riders wearing helmets and those without helmets.
- Identify vehicle registration plates from traffic images.
- Extract vehicle registration numbers using OCR.
- Validate extracted registration numbers using a configured Indian vehicle-number format.
- Store validated violation details in a CSV registry.
- Send SMS notifications using Twilio.

## System Architecture

The system follows a sequential detection and recognition pipeline:

1. **Image Input:** The user uploads a traffic image through the Streamlit application.
2. **Helmet Detection:** YOLOv8n detects helmet and no-helmet conditions.
3. **Conditional Processing:** If a no-helmet violation is detected, the system proceeds to license plate detection.
4. **License Plate Detection:** A YOLO-based license plate detector identifies the number plate.
5. **Image Cropping:** OpenCV crops the detected number plate.
6. **Text Recognition:** PaddleOCR extracts the vehicle registration number.
7. **Text Validation:** The extracted text is normalized and checked against the configured Indian registration-number pattern.
8. **Violation Registry:** Validated violation details are saved in a CSV file.
9. **SMS Notification:** Twilio sends an SMS notification after a validated violation is successfully saved.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLOv8n | Helmet detection |
| YOLO-based plate detector | License plate detection |
| OpenCV | Image processing and plate cropping |
| PaddleOCR | Vehicle number recognition |
| Streamlit | Web application interface |
| CSV | Violation record storage |
| Twilio | SMS notification |

## Project Structure

```text
Real-Time-Helmet-Violation-Detection-Using-YOLOv8/
│
├── app.py
├── helmet_detection.py
├── plate_detection.py
├── ocr.py
├── registry.py
├── sms_alert.py
├── requirements.txt
│
├── data.yaml
├── train/
├── valid/
├── test/
│
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sharmila9344/Real-Time-Helmet-Violation-Detection-Using-YOLOv8.git
```

### 2. Navigate to the Project Directory

```bash
cd Real-Time-Helmet-Violation-Detection-Using-YOLOv8
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create a `.env` file locally and add the Twilio credentials required by the application.

**Do not upload the `.env` file or expose API credentials on GitHub.**

Make sure the required model-weight files are downloaded and placed in the locations expected by the detection modules.

### 7. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Application Workflow

- Upload a traffic image.
- Detect the helmet status.
- If a no-helmet violation is detected, locate the vehicle number plate.
- Crop the plate and extract its text using PaddleOCR.
- Normalize and validate the extracted registration number.
- Save the validated violation details in the CSV registry.
- Trigger the Twilio SMS notification.

## SMS Notification

The project integrates the Twilio API to send SMS notifications after a validated violation is successfully recorded.

The SMS functionality was tested successfully using a Twilio trial account and its supported predefined SMS template.

Trial account restrictions may limit the message content.

## Testing and Observations

The prototype was functionally tested for:

- Helmet and no-helmet detection.
- License plate localization and cropping.
- OCR-based vehicle registration number extraction.
- Registration number normalization and format validation.
- CSV violation record creation.
- Twilio SMS request and successful message receipt.

These functional checks demonstrate the integration of the system components. They do not represent a complete statistical evaluation of detection or OCR accuracy.

## Future Enhancements

- Support for live CCTV and video-stream processing.
- Improved detection under low-light and challenging weather conditions.
- Enhanced OCR performance for blurred and partially visible number plates.
- Integration with secure and authorized vehicle registration databases.
- Improved dashboard analytics and reporting.

## Team Members

**Sharmila S.** – 24BRS1419  
**Dinesh H.** – 24BRS1266

Department of Computer Science and Engineering  
Vellore Institute of Technology (VIT), Chennai, India

## Repository

GitHub: https://github.com/sharmila9344/Real-Time-Helmet-Violation-Detection-Using-YOLOv8

---

**Note:** This project is developed for academic and research purposes. Automated detections should be reviewed by a human before any enforcement action.
