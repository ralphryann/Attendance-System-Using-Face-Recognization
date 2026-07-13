# Attendance System Using Face Recognition

A Flask-based attendance system that uses OpenCV + `face_recognition` to identify students and mark attendance after blink detection.

## Features

- Face-recognition attendance marking
- Blink-based confirmation before marking a student present
- CSV attendance export by batch/month/subject
- Attendance report view in the web UI
- Monthly analytics charts (single-subject or all-subject)

## Tech Stack

- Flask
- OpenCV
- face_recognition / dlib
- NumPy
- Matplotlib

## Project Structure

```text
Attendance-System-Using-Face-Recognization/
├── app.py
├── register_students.py
├── requirements.txt
├── README.md
├── static/
│   ├── attendance/
│   ├── *.gif / *.png
│   └── css/style.css
└── templates/
    ├── base.html
    ├── index.html
    ├── select_batch_subject.html
    ├── attendance_report.html
    ├── view_analytics.html
    └── analytics_result.html
```

## Prerequisites

- Python 3.10+ (3.10/3.11 recommended)
- Webcam connected to the machine running the Flask app
- Build tools needed by `dlib` on your OS

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add student images to a `Photos/` folder in the project root.
4. Register students (extracts and stores an encrypted face encoding per student in the database):

```bash
python register_students.py --photos-dir Photos --batch <name>
```

   (Students can also be added, edited, or removed later from the web UI under **Students**, admin-only.)

5. Run the app:

```bash
python app.py
```

6. Open `http://127.0.0.1:5000`.

## Usage

1. Click **Take Attendance**.
2. Select a batch and subject.
3. The camera window opens (OpenCV). Students blink to confirm identity.
4. Press `q` or `Esc` to stop and save attendance.
5. Review the generated report.
6. Use **View Analytics** to generate monthly charts.

## Attendance File Location

Attendance CSV files are stored at:

```text
static/attendance/<Batch>/<Month>/<Subject>_<YYYY-MM-DD>.csv
```

## Notes

- Default batches: `BatchA`, `BatchB`
- Default subjects: `Math`, `Science`, `English`
- Student names are currently defined in `app.py` (`known_face_names`)

## Troubleshooting

- **`ModuleNotFoundError: face_recognition` / `dlib` build errors**  
  Use Python 3.10/3.11 and reinstall dependencies in a fresh virtual environment.
- **No face recognized**  
  Ensure clear student images in `Photos/`, then re-run `register_students.py` (or re-upload the photo via the Students admin page).
- **No analytics data**  
  First generate attendance files for the selected batch/month.
