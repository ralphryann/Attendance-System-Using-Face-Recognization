from flask import Flask, render_template, request, redirect, url_for, flash, abort, jsonify, send_file
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
from flask_wtf import CSRFProtect
from datetime import datetime
from urllib.parse import urlparse
from cryptography.fernet import Fernet, InvalidToken
from openpyxl import Workbook
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import csv
import re
import base64
import secrets
import time
import random
import math
from functools import wraps
import numpy as np

from models import db, User, Batch, Student


app = Flask(__name__)
_secret_key_env = os.environ.get("SECRET_KEY")
if not _secret_key_env:
    _secret_key_env = secrets.token_urlsafe(32)
    print(
        "WARNING: SECRET_KEY is not set. Generated a temporary key for this process only -- "
        "all login sessions will be invalidated on restart. Set SECRET_KEY as an environment "
        "variable to persist sessions across restarts."
    )
app.secret_key = _secret_key_env
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB cap on uploaded student photos

# CSRF protection for all state-changing form POSTs. The JSON /api/* endpoints are exempted
# individually (below) -- they authenticate with an unguessable per-session attendance token
# that an attacker can't forge, which already defeats cross-site request forgery for them.
csrf = CSRFProtect(app)

_face_key_env = os.environ.get("FACE_VECTOR_ENCRYPTION_KEY")
if not _face_key_env:
    _face_key_env = Fernet.generate_key().decode()
    print(
        "WARNING: FACE_VECTOR_ENCRYPTION_KEY is not set. Generated a temporary key for this "
        "process only -- any face data registered now will NOT be readable after a restart. "
        "Set this exact value as an environment variable to persist it:\n"
        f"  FACE_VECTOR_ENCRYPTION_KEY={_face_key_env}"
    )
_face_cipher = Fernet(_face_key_env.encode() if isinstance(_face_key_env, str) else _face_key_env)

_geofence_lat_env = os.environ.get("SCHOOL_LATITUDE")
_geofence_lon_env = os.environ.get("SCHOOL_LONGITUDE")
GEOFENCE_ENABLED = _geofence_lat_env is not None and _geofence_lon_env is not None
GEOFENCE_LATITUDE = float(_geofence_lat_env) if GEOFENCE_ENABLED else None
GEOFENCE_LONGITUDE = float(_geofence_lon_env) if GEOFENCE_ENABLED else None
GEOFENCE_RADIUS_METERS = float(os.environ.get("GEOFENCE_RADIUS_METERS", "150"))

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "attendance.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Directory to store attendance CSVs
ATTENDANCE_ROOT = os.path.join('static', 'attendance')
os.makedirs(ATTENDANCE_ROOT, exist_ok=True)

# Batches and subjects
BATCHES = ['BatchA', 'BatchB']
SUBJECTS = ['Math', 'Science', 'English']

MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# Simple pattern for safe file-path components (alphanumeric, dash, underscore)
SAFE_NAME_RE = re.compile(r'^[A-Za-z0-9_-]+$')

# In-memory store of live attendance sessions, keyed by an unguessable token.
# Fine for a single-process kiosk deployment; not shared across worker processes.
ACTIVE_SESSIONS = {}
SESSION_TTL_SECONDS = 15 * 60
FRAME_INTERVAL_MS = 700

BLINK_THRESHOLD = 0.20
BLINK_FRAMES = 3

# 1:N face match: a live frame's encoding is compared against every registered student's
# encoding (system-wide, not just the current batch); the closest one wins if it's within
# this Euclidean-distance tolerance. Lower = stricter (fewer false matches, more false rejects).
FACE_MATCH_TOLERANCE = 0.55

# Active liveness: each student is assigned one of these challenges at random the
# first time they're seen in a session, so an attacker can't know in advance which
# action a pre-recorded photo/video would need to perform.
LIVENESS_CHALLENGES = ("blink", "smile", "turn_head")
# Mouth width relative to inter-eye distance (a stable reference that doesn't change with
# expression), not mouth width/height alone -- a neutral closed mouth already has near-zero
# height, so a height-based ratio can't tell "closed" apart from "smiling".
SMILE_RATIO_THRESHOLD = 1.15
SMILE_FRAMES = 3              # consecutive above-threshold frames required to count as a smile

# A moderate head turn, using the same eye/nose skew ratio as the pose quality-gate below.
# Deliberately kept under POSE_RATIO_THRESHOLD so a genuine turn still passes the quality
# gate and gets recognized -- turning far enough to trip "bad_angle" just means turn back
# a little. We don't claim a specific left-vs-right direction (see note on _pose_skew_ratio).
TURN_RATIO_THRESHOLD = 1.4
TURN_FRAMES = 3

LIVENESS_VERBS = {
    "blink": ("blink", "blinked"),
    "smile": ("smile", "smiled"),
    "turn_head": ("turn your head slightly to either side", "turned their head"),
}

# Frame-quality gates, checked on the detected face crop before recognition runs.
# These are heuristic thresholds tuned for a typical laptop webcam; adjust if your
# camera runs consistently darker/noisier or the checks feel too strict/lax.
BLUR_VARIANCE_THRESHOLD = 60.0   # variance of Laplacian; lower = blurrier
DARKNESS_THRESHOLD = 60.0        # mean grayscale intensity (0-255); lower = darker
POSE_RATIO_THRESHOLD = 1.8       # left/right eye-to-nose distance skew; higher = sharper angle

QUALITY_MESSAGES = {
    "too_blurry": "Image is too blurry. Hold the camera steady and try again.",
    "too_dark": "Lighting is too dark. Move to a brighter area.",
    "bad_angle": "Face the camera directly (angle is too sharp).",
    "no_face_crop": "Could not get a clear view of your face.",
}


def _check_frame_quality(cv2, frame_bgr, location):
    """Reject a face crop that's too blurry or too dark. Returns a QUALITY_MESSAGES key or None."""
    top, right, bottom, left = location
    crop = frame_bgr[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
    if crop.size == 0:
        return "no_face_crop"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if gray.mean() < DARKNESS_THRESHOLD:
        return "too_dark"

    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_VARIANCE_THRESHOLD:
        return "too_blurry"

    return None


def _pose_skew_ratio(landmarks):
    """How far the nose sits from centered between the eyes: 1.0 = straight-on, higher = more
    turned to one side. Returns None if there isn't enough landmark data to judge.

    Deliberately direction-agnostic (doesn't report *which* side). face_recognition's
    left_eye/right_eye keys follow dlib's convention, and without a real camera to test
    against we can't be confident whether that's the subject's own left/right or the image's
    left/right as seen by the camera -- getting it backwards would mean a "turn left" prompt
    that actually requires turning right. Safer to only claim "turned", not a direction.
    """
    nose_points = landmarks.get("nose_tip") or landmarks.get("nose_bridge")
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")
    if not nose_points or not left_eye or not right_eye:
        return None

    nose_x = np.mean([p[0] for p in nose_points])
    left_eye_x = np.mean([p[0] for p in left_eye])
    right_eye_x = np.mean([p[0] for p in right_eye])

    left_dist = abs(nose_x - left_eye_x)
    right_dist = abs(nose_x - right_eye_x)
    smaller = min(left_dist, right_dist)
    larger = max(left_dist, right_dist)

    if smaller < 1e-3:
        return float("inf")
    return larger / smaller


def _check_face_pose(landmarks):
    """Reject a face turned too far from the camera for reliable recognition."""
    ratio = _pose_skew_ratio(landmarks)
    if ratio is not None and ratio > POSE_RATIO_THRESHOLD:
        return "bad_angle"
    return None


def _haversine_distance_meters(lat1, lon1, lat2, lon2):
    earth_radius_m = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * earth_radius_m * math.asin(math.sqrt(a))


def _prune_expired_sessions():
    now = time.time()
    expired = [token for token, s in ACTIVE_SESSIONS.items() if now - s["last_active"] > SESSION_TTL_SECONDS]
    for token in expired:
        ACTIVE_SESSIONS.pop(token, None)


def _get_owned_session(token):
    """Look up a session, pruning expired ones and enforcing that it belongs to the caller."""
    _prune_expired_sessions()
    session = ACTIVE_SESSIONS.get(token)
    if session is None or session["user_id"] != current_user.id:
        return None
    return session


def _eye_aspect_ratio(eye):
    a = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    b = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    c = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (a + b) / (2.0 * c)


def _smile_ratio(landmarks):
    left_eye_center = np.mean(landmarks["left_eye"], axis=0)
    right_eye_center = np.mean(landmarks["right_eye"], axis=0)
    eye_distance = np.linalg.norm(left_eye_center - right_eye_center)

    mouth_xs = [p[0] for p in landmarks["top_lip"]] + [p[0] for p in landmarks["bottom_lip"]]
    mouth_width = max(mouth_xs) - min(mouth_xs)

    return mouth_width / max(eye_distance, 1e-6)


def _new_liveness_state():
    return {
        "challenge": random.choice(LIVENESS_CHALLENGES),
        "eyes_closed_frames": 0,
        "blink_detected": False,
        "smile_frames": 0,
        "smile_detected": False,
        "turn_frames": 0,
        "turn_detected": False,
        "attendance_marked": False,
    }


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def initialize_database():
    """Create tables and seed default batches/subjects/admin user if missing."""
    with app.app_context():
        db.create_all()

        for batch_name in BATCHES:
            get_or_create_batch(batch_name)

        if not db.session.execute(db.select(User)).first():
            admin = User(username="admin", role="Admin")
            admin.set_password(os.environ.get("ADMIN_DEFAULT_PASSWORD", "admin123"))
            db.session.add(admin)

        db.session.commit()


def get_or_create_batch(name):
    batch = db.session.execute(db.select(Batch).where(Batch.name == name)).scalar_one_or_none()
    if batch is None:
        batch = Batch(name=name)
        db.session.add(batch)
        db.session.commit()
    return batch


def serialize_face_encoding(encoding):
    """Store only the raw numeric vector (never a picklable object) and encrypt it at rest."""
    raw = np.asarray(encoding, dtype=np.float64).tobytes()
    return _face_cipher.encrypt(raw)


def deserialize_face_encoding(blob):
    raw = _face_cipher.decrypt(bytes(blob))
    return np.frombuffer(raw, dtype=np.float64)


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if current_user.role != 'Admin':
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def _extract_face_encoding_from_upload(file_storage):
    """Read an uploaded photo in memory (never written to disk) and return
    (encoding_blob, error_message) -- exactly one of which will be set."""
    import face_recognition

    try:
        image = face_recognition.load_image_file(file_storage)
    except Exception:
        return None, "Could not read that file as an image."

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if len(face_encodings) == 0:
        return None, "No face was detected in that photo."
    if len(face_encodings) > 1:
        return None, "Multiple faces were detected; upload a photo with exactly one face."

    return serialize_face_encoding(face_encodings[0]), None


def _resolve_attendance_file(file_path):
    """Resolve a query-string file path safely inside ATTENDANCE_ROOT, or return None."""
    abs_file = os.path.abspath(file_path)
    abs_root = os.path.abspath(ATTENDANCE_ROOT)
    if not (abs_file == abs_root or abs_file.startswith(abs_root + os.sep)) or not os.path.isfile(abs_file):
        return None
    return abs_file


def _sanitize_cell(value):
    """Guard against CSV/spreadsheet formula injection. A cell whose text begins with a
    formula-trigger character gets a leading single quote so Excel/Sheets/LibreOffice treat
    it as literal text instead of executing it (e.g. a student named '=HYPERLINK(...)')."""
    text = "" if value is None else str(value)
    if text[:1] in ('=', '+', '-', '@', '\t', '\r'):
        return "'" + text
    return text


def _rows_to_xlsx(rows, sheet_title="Attendance"):
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_title
    for row in rows:
        ws.append([_sanitize_cell(cell) for cell in row])
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer


def _rows_to_csv_buffer(rows):
    text_buffer = io.StringIO()
    csv.writer(text_buffer).writerows([_sanitize_cell(cell) for cell in row] for row in rows)
    return io.BytesIO(text_buffer.getvalue().encode("utf-8"))


XLSX_MIMETYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _is_safe_redirect_target(target):
    """Only allow redirects to same-site relative paths, to prevent open-redirect phishing.
    Rejects absolute URLs, protocol-relative URLs (//evil.com), and backslash tricks."""
    if not target:
        return False
    # Normalize backslashes so browsers can't reinterpret \\evil.com as //evil.com
    target = target.replace('\\', '/')
    parsed = urlparse(target)
    return not parsed.scheme and not parsed.netloc and target.startswith('/') and not target.startswith('//')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = db.session.execute(db.select(User).where(User.username == username)).scalar_one_or_none()

        if user is None or not user.check_password(password):
            flash("Invalid username or password.", "error")
            return redirect(url_for('login'))

        login_user(user)
        next_target = request.form.get('next', '')
        if _is_safe_redirect_target(next_target):
            return redirect(next_target)
        return redirect(url_for('home'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not current_user.check_password(current_password):
            flash("Current password is incorrect.", "error")
        elif new_password != confirm_password:
            flash("New passwords do not match.", "error")
        elif len(new_password) < 8:
            flash("New password must be at least 8 characters.", "error")
        else:
            current_user.set_password(new_password)
            db.session.commit()
            flash("Password updated successfully.", "success")
            return redirect(url_for('home'))

    return render_template('change_password.html')


@app.route('/take_attendance', methods=['POST', 'GET'])
@login_required
def take_attendance():
    return render_template('select_batch_subject.html', batches=BATCHES, subjects=SUBJECTS)

@app.route('/submit_attendance', methods=['POST'])
@login_required
def submit_attendance():
    batch = request.form.get('batch', '')
    subject = request.form.get('subject', '')

    if batch not in BATCHES or subject not in SUBJECTS:
        flash("Please select a valid batch and subject.", "error")
        return redirect(url_for('take_attendance'))

    batch_obj = db.session.execute(db.select(Batch).where(Batch.name == batch)).scalar_one_or_none()
    students = batch_obj.students if batch_obj else []

    if not students:
        flash(f"No registered students found for {batch}. Run register_students.py first.", "error")
        return redirect(url_for('take_attendance'))

    current_month_number = datetime.today().month
    batch_path = os.path.join(ATTENDANCE_ROOT, batch, MONTHS[current_month_number])
    os.makedirs(batch_path, exist_ok=True)
    file_name = f"{batch_path}/{subject}_{datetime.today().date()}.csv"

    known_face_names = [s.name for s in students]

    # 1:N match pool: every registered student system-wide, not just this batch, so a
    # recognized face always resolves to one specific student ID before we check whether
    # that person is actually enrolled in this batch.
    all_students = db.session.execute(db.select(Student)).scalars().all()

    match_pool_ids, match_pool_names, match_pool_batch_ids, match_pool_encodings = [], [], [], []
    for s in all_students:
        try:
            match_pool_encodings.append(deserialize_face_encoding(s.face_encoding))
        except InvalidToken:
            # Encrypted with a different FACE_VECTOR_ENCRYPTION_KEY than is currently set;
            # skip this student rather than fail the whole session for everyone else.
            print(f"WARNING: could not decrypt face data for student id={s.id} ({s.name}); skipping.")
            continue
        match_pool_ids.append(s.id)
        match_pool_names.append(s.name)
        match_pool_batch_ids.append(s.batch_id)

    token = secrets.token_urlsafe(32)
    _prune_expired_sessions()
    ACTIVE_SESSIONS[token] = {
        "user_id": current_user.id,
        "batch": batch,
        "batch_id": batch_obj.id,
        "subject": subject,
        "file_name": file_name,
        "known_face_names": known_face_names,
        "student_list": known_face_names.copy(),
        "present_students": [],
        "liveness_status": {name: _new_liveness_state() for name in known_face_names},
        "match_pool_ids": match_pool_ids,
        "match_pool_names": match_pool_names,
        "match_pool_batch_ids": match_pool_batch_ids,
        "match_pool_encodings": match_pool_encodings,
        "location_verified": not GEOFENCE_ENABLED,
        "last_active": time.time(),
    }

    return render_template(
        'attendance_session.html',
        batch_name=batch,
        subject_name=subject,
        attendance_token=token,
        frame_interval_ms=FRAME_INTERVAL_MS,
        total_students=len(known_face_names),
        geofence_enabled=GEOFENCE_ENABLED,
    )


@app.route('/api/verify_location', methods=['POST'])
@csrf.exempt
@login_required
def api_verify_location():
    payload = request.get_json(silent=True) or {}
    token = payload.get('attendance_token', '')

    session = _get_owned_session(token)
    if session is None:
        return jsonify(status="error", message="This attendance session has expired. Please restart."), 404

    if not GEOFENCE_ENABLED:
        session["location_verified"] = True
        return jsonify(status="success", message="Location check not required.")

    try:
        latitude = float(payload.get('latitude'))
        longitude = float(payload.get('longitude'))
    except (TypeError, ValueError):
        return jsonify(status="error", message="Could not read your device location."), 400

    distance = _haversine_distance_meters(latitude, longitude, GEOFENCE_LATITUDE, GEOFENCE_LONGITUDE)

    if distance > GEOFENCE_RADIUS_METERS:
        session["location_verified"] = False
        return jsonify(
            status="error",
            message=(
                f"You appear to be about {int(distance)}m from the allowed location "
                f"(must be within {int(GEOFENCE_RADIUS_METERS)}m). Attendance can't be taken here."
            ),
        ), 403

    session["location_verified"] = True
    return jsonify(status="success", message="Location verified.")


@app.route('/api/process_frame', methods=['POST'])
@csrf.exempt
@login_required
def api_process_frame():
    import cv2
    import face_recognition

    payload = request.get_json(silent=True) or {}
    token = payload.get('attendance_token', '')
    image_data = payload.get('image_data', '')

    session = _get_owned_session(token)
    if session is None:
        return jsonify(status="error", message="This attendance session has expired. Please restart."), 404

    if not session.get("location_verified"):
        return jsonify(status="error", message="Location not verified for this session yet."), 403

    if not image_data:
        return jsonify(status="error", message="No image received."), 400

    try:
        encoded = image_data.split(',', 1)[1] if ',' in image_data else image_data
        raw_bytes = base64.b64decode(encoded)
        frame_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return jsonify(status="error", message="Received an invalid image frame."), 400

    session["last_active"] = time.time()

    match_pool_encodings = session["match_pool_encodings"]
    match_pool_ids = session["match_pool_ids"]
    match_pool_names = session["match_pool_names"]
    match_pool_batch_ids = session["match_pool_batch_ids"]
    liveness_status = session["liveness_status"]
    student_list = session["student_list"]
    present_students = session["present_students"]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    recognized_student = None
    recognized_student_id = None
    message = "No face detected."

    for location, face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
        quality_issue = _check_frame_quality(cv2, frame, location) or _check_face_pose(landmarks)
        if quality_issue:
            message = QUALITY_MESSAGES[quality_issue]
            continue

        # 1:N search: closest encoding across every registered student, gated by an
        # explicit distance tolerance rather than face_recognition's default threshold.
        name = "Unknown"
        best_match_index = None
        if match_pool_encodings:
            distances = face_recognition.face_distance(match_pool_encodings, face_encoding)
            best_match_index = int(np.argmin(distances))
            if distances[best_match_index] < FACE_MATCH_TOLERANCE:
                name = match_pool_names[best_match_index]

        if name == "Unknown":
            message = "Face detected but not recognized."
            continue

        recognized_student = name
        recognized_student_id = match_pool_ids[best_match_index]

        if match_pool_batch_ids[best_match_index] != session["batch_id"]:
            message = f"Recognized {name}, but they are not enrolled in this batch."
            continue

        if name not in liveness_status:
            liveness_status[name] = _new_liveness_state()

        state = liveness_status[name]
        challenge = state["challenge"]
        action_verb, action_past = LIVENESS_VERBS[challenge]

        if challenge == "blink":
            avg_ear = (
                _eye_aspect_ratio(landmarks["left_eye"]) + _eye_aspect_ratio(landmarks["right_eye"])
            ) / 2.0

            if avg_ear < BLINK_THRESHOLD:
                state["eyes_closed_frames"] += 1
            else:
                if state["eyes_closed_frames"] >= BLINK_FRAMES:
                    state["blink_detected"] = True
                state["eyes_closed_frames"] = 0

            liveness_confirmed = state["blink_detected"]
        elif challenge == "smile":
            smile_ratio = _smile_ratio(landmarks)

            if smile_ratio > SMILE_RATIO_THRESHOLD:
                state["smile_frames"] += 1
                if state["smile_frames"] >= SMILE_FRAMES:
                    state["smile_detected"] = True
            else:
                state["smile_frames"] = 0

            liveness_confirmed = state["smile_detected"]
        else:  # "turn_head"
            skew_ratio = _pose_skew_ratio(landmarks)

            # Below TURN_RATIO_THRESHOLD: not turned enough. Above POSE_RATIO_THRESHOLD: the
            # quality gate already rejected this frame before we got here, so this branch only
            # ever sees the "moderately turned" window in between.
            if skew_ratio is not None and skew_ratio > TURN_RATIO_THRESHOLD:
                state["turn_frames"] += 1
                if state["turn_frames"] >= TURN_FRAMES:
                    state["turn_detected"] = True
            else:
                state["turn_frames"] = 0

            liveness_confirmed = state["turn_detected"]

        if liveness_confirmed and not state["attendance_marked"] and name in student_list:
            present_students.append((name, datetime.now().strftime("%H:%M:%S")))
            student_list.remove(name)
            state["blink_detected"] = False
            state["smile_detected"] = False
            state["turn_detected"] = False
            state["attendance_marked"] = True
            message = f"{name} {action_past}! Attendance marked."
        elif state["attendance_marked"]:
            message = f"{name} is already marked present."
        else:
            message = f"Recognized {name}. Please {action_verb} to confirm."

    return jsonify(
        status="success",
        message=message,
        recognized_student=recognized_student,
        recognized_student_id=recognized_student_id,
        remaining_students=len(student_list),
    )


@app.route('/api/finalize_attendance', methods=['POST'])
@csrf.exempt
@login_required
def api_finalize_attendance():
    payload = request.get_json(silent=True) or {}
    token = payload.get('attendance_token', '')

    session = _get_owned_session(token)
    if session is None:
        return jsonify(status="error", message="This attendance session has expired. Please restart."), 404

    if not session.get("location_verified"):
        return jsonify(status="error", message="Location not verified for this session yet."), 403

    file_name = session["file_name"]
    known_face_names = session["known_face_names"]
    present_students = session["present_students"]
    student_list = session["student_list"]

    try:
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Status", "Time"])
            for name, time_marked in present_students:
                writer.writerow([name, "Present", time_marked])
            for student in student_list:
                writer.writerow([student, "Absent", "--"])
            writer.writerow([])
            writer.writerow(["Total Students", "Present Students", "Absent Students"])
            writer.writerow([len(known_face_names), len(present_students), len(student_list)])
    except Exception as e:
        return jsonify(status="error", message=f"Failed to save attendance: {e}"), 500

    ACTIVE_SESSIONS.pop(token, None)

    return jsonify(status="success", report_url=url_for('attendance_report', file=file_name))


@app.route("/attendance_report")
@login_required
def attendance_report():
    file_path = request.args.get("file", "")
    abs_file = _resolve_attendance_file(file_path)
    if abs_file is None:
        abort(404)

    with open(abs_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # The last 3 rows are always [blank separator, totals header, totals values] (see writer above)
    filtered_rows = rows[:-3] if len(rows) > 3 else rows
    header = filtered_rows[0] if filtered_rows else []
    body_rows = filtered_rows[1:] if len(filtered_rows) > 1 else []

    # Calculate the summary
    total_students = len(body_rows)
    present_count = sum(1 for row in body_rows if 'Present' in row)
    absent_count = sum(1 for row in body_rows if 'Absent' in row)

    summary = {
        'Total': total_students,
        'Present': present_count,
        'Absent': absent_count
    }

    return render_template(
        'attendance_report.html', header=header, rows=body_rows, summary=summary, file=file_path
    )


@app.route("/attendance_report/download")
@login_required
def download_attendance_report():
    file_path = request.args.get("file", "")
    fmt = request.args.get("format", "csv")
    abs_file = _resolve_attendance_file(file_path)
    if abs_file is None:
        abort(404)

    base_name = os.path.splitext(os.path.basename(abs_file))[0]

    with open(abs_file, newline='') as csvfile:
        rows = list(csv.reader(csvfile))

    if fmt == "xlsx":
        buffer = _rows_to_xlsx(rows)
        return send_file(
            buffer, as_attachment=True, download_name=f"{base_name}.xlsx", mimetype=XLSX_MIMETYPE
        )

    if fmt == "csv":
        # Re-serialized through _rows_to_csv_buffer (not streamed raw) so the formula-injection
        # guard is applied to the downloaded file.
        buffer = _rows_to_csv_buffer(rows)
        return send_file(
            buffer, as_attachment=True, download_name=f"{base_name}.csv", mimetype="text/csv"
        )

    abort(400)

@app.route('/view_analytics', methods=['GET', 'POST'])
@admin_required
def view_analytics():
    if request.method == 'POST':
        batch = request.form.get('batch', '').strip()
        choice = request.form.get('choice', '')
        subject = request.form.get('subject', '').strip()

        # Validate month
        try:
            month_number = int(request.form['month'])
            if month_number < 1 or month_number > 12:
                raise ValueError
        except (ValueError, KeyError):
            flash("Please enter a valid month number (1–12).", "error")
            return redirect('/view_analytics')

        # Validate batch name
        if not batch or not SAFE_NAME_RE.match(batch):
            flash("Please enter a valid batch name (letters, numbers, dashes, underscores).", "error")
            return redirect('/view_analytics')

        # Validate subject when single-subject mode
        if choice == '1' and (not subject or not SAFE_NAME_RE.match(subject)):
            flash("Please enter a valid subject name.", "error")
            return redirect('/view_analytics')

        batch_path = os.path.join(ATTENDANCE_ROOT, batch, MONTHS[month_number])
        if not os.path.exists(batch_path):
            flash("Batch folder for the given month does not exist.", "error")
            return redirect('/view_analytics')

        try:
            if choice == '1':  # Single Subject
                total_students = total_present = total_absent = 0
                for file in os.listdir(batch_path):
                    # Match on the "{subject}_" boundary so subject "Math" doesn't also
                    # pick up files for a subject named "Mathematics".
                    if file.split("_")[0] == subject:
                        with open(os.path.join(batch_path, file), "r") as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                last_line = lines[-1].strip().split(",")
                                total_students = int(last_line[0])
                                total_present += int(last_line[1])
                                total_absent += int(last_line[2])

                if total_students == 0:
                    flash("No attendance records found for the given month/subject.", "error")
                    return redirect('/view_analytics')

                labels = ["Present", "Absent"]
                sizes = [total_present, total_absent]
                colors = ["green", "red"]

                fig = plt.figure(figsize=(6, 6))
                plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
                plt.title(f"Attendance for {subject} ({MONTHS[month_number]})")
                img_path = f"static/analytics_{batch}_{subject}.png"
                fig.savefig(img_path)
                plt.close()

                return render_template("analytics_result.html", image=img_path)

            elif choice == '2':  # All Subjects
                subject_attendance = {}
                for file in os.listdir(batch_path):
                    subject = file.split("_")[0]
                    with open(os.path.join(batch_path, file), "r") as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            last_line = lines[-1].strip().split(",")
                            present = int(last_line[1])
                            subject_attendance[subject] = subject_attendance.get(subject, 0) + present

                if not subject_attendance:
                    flash("No attendance records found.", "error")
                    return redirect('/view_analytics')

                fig = plt.figure(figsize=(8, 8))
                plt.pie(subject_attendance.values(), labels=subject_attendance.keys(), autopct="%1.1f%%", startangle=140)
                plt.title(f"Attendance for All Subjects ({MONTHS[month_number]})")
                img_path = f"static/analytics_{batch}_all.png"
                fig.savefig(img_path)
                plt.close()

                return render_template("analytics_result.html", image=img_path)

            else:
                flash("Invalid choice.", "error")
                return redirect('/view_analytics')

        except Exception as e:
            flash("Error while processing: " + str(e), "error")
            return redirect('/view_analytics')

    return render_template("view_analytics.html", batches=BATCHES, subjects=SUBJECTS)


@app.route('/export_attendance')
@admin_required
def export_attendance():
    batch = request.args.get('batch', '').strip()
    subject = request.args.get('subject', '').strip()
    choice = request.args.get('choice', '')
    fmt = request.args.get('format', 'csv')

    try:
        month_number = int(request.args['month'])
        if month_number < 1 or month_number > 12:
            raise ValueError
    except (ValueError, KeyError):
        flash("Please enter a valid month number (1–12).", "error")
        return redirect(url_for('view_analytics'))

    if not batch or not SAFE_NAME_RE.match(batch):
        flash("Please enter a valid batch name.", "error")
        return redirect(url_for('view_analytics'))

    if choice == '1' and (not subject or not SAFE_NAME_RE.match(subject)):
        flash("Please enter a valid subject name.", "error")
        return redirect(url_for('view_analytics'))

    batch_path = os.path.join(ATTENDANCE_ROOT, batch, MONTHS[month_number])
    if not os.path.exists(batch_path):
        flash("Batch folder for the given month does not exist.", "error")
        return redirect(url_for('view_analytics'))

    combined_rows = [["Date", "Subject", "Student", "Status", "Time"]]
    for filename in sorted(os.listdir(batch_path)):
        file_subject = filename.split("_")[0]
        if choice == '1' and file_subject != subject:
            continue

        date_part = filename[len(file_subject) + 1:].rsplit(".", 1)[0]

        with open(os.path.join(batch_path, filename), newline='') as f:
            file_rows = list(csv.reader(f))

        body_rows = file_rows[1:-3] if len(file_rows) > 3 else file_rows[1:]
        for row in body_rows:
            if len(row) >= 3:
                combined_rows.append([date_part, file_subject, row[0], row[1], row[2]])

    if len(combined_rows) == 1:
        flash("No attendance records found for the given filters.", "error")
        return redirect(url_for('view_analytics'))

    # Default sort: by date, then student name. Any spreadsheet app can re-sort from there.
    combined_rows[1:] = sorted(combined_rows[1:], key=lambda r: (r[0], r[2]))

    scope_label = subject if choice == '1' else "all_subjects"
    base_name = f"{batch}_{scope_label}_{MONTHS[month_number]}"

    if fmt == 'xlsx':
        buffer = _rows_to_xlsx(combined_rows)
        return send_file(
            buffer, as_attachment=True, download_name=f"{base_name}.xlsx", mimetype=XLSX_MIMETYPE
        )

    buffer = _rows_to_csv_buffer(combined_rows)
    return send_file(
        buffer, as_attachment=True, download_name=f"{base_name}.csv", mimetype="text/csv"
    )


@app.route('/students')
@admin_required
def list_students():
    students = db.session.execute(
        db.select(Student).join(Batch).order_by(Batch.name, Student.name)
    ).scalars().all()
    return render_template('students.html', students=students)


@app.route('/students/new', methods=['GET', 'POST'])
@admin_required
def new_student():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        batch_name = request.form.get('batch', '')
        photo = request.files.get('photo')

        if not name:
            flash("Please enter a student name.", "error")
            return redirect(url_for('new_student'))
        if batch_name not in BATCHES:
            flash("Please select a valid batch.", "error")
            return redirect(url_for('new_student'))
        if not photo or not photo.filename:
            flash("Please upload a photo.", "error")
            return redirect(url_for('new_student'))

        encoding_blob, error = _extract_face_encoding_from_upload(photo)
        if error:
            flash(error, "error")
            return redirect(url_for('new_student'))

        batch_obj = get_or_create_batch(batch_name)

        duplicate = db.session.execute(
            db.select(Student).where(Student.name == name, Student.batch_id == batch_obj.id)
        ).scalar_one_or_none()
        if duplicate:
            flash(f"A student named '{name}' already exists in {batch_name}.", "error")
            return redirect(url_for('new_student'))

        db.session.add(Student(name=name, batch_id=batch_obj.id, face_encoding=encoding_blob))
        db.session.commit()
        flash(f"Added {name} to {batch_name}.", "success")
        return redirect(url_for('list_students'))

    return render_template('student_form.html', student=None, batches=BATCHES)


@app.route('/students/<int:student_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_student(student_id):
    student = db.session.get(Student, student_id)
    if student is None:
        abort(404)

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        batch_name = request.form.get('batch', '')
        photo = request.files.get('photo')

        if not name:
            flash("Please enter a student name.", "error")
            return redirect(url_for('edit_student', student_id=student_id))
        if batch_name not in BATCHES:
            flash("Please select a valid batch.", "error")
            return redirect(url_for('edit_student', student_id=student_id))

        batch_obj = get_or_create_batch(batch_name)

        duplicate = db.session.execute(
            db.select(Student).where(
                Student.name == name, Student.batch_id == batch_obj.id, Student.id != student.id
            )
        ).scalar_one_or_none()
        if duplicate:
            flash(f"A student named '{name}' already exists in {batch_name}.", "error")
            return redirect(url_for('edit_student', student_id=student_id))

        if photo and photo.filename:
            encoding_blob, error = _extract_face_encoding_from_upload(photo)
            if error:
                flash(error, "error")
                return redirect(url_for('edit_student', student_id=student_id))
            student.face_encoding = encoding_blob

        student.name = name
        student.batch_id = batch_obj.id
        db.session.commit()
        flash(f"Updated {name}.", "success")
        return redirect(url_for('list_students'))

    return render_template('student_form.html', student=student, batches=BATCHES)


@app.route('/students/<int:student_id>/delete', methods=['POST'])
@admin_required
def delete_student(student_id):
    student = db.session.get(Student, student_id)
    if student is None:
        abort(404)
    name = student.name
    db.session.delete(student)
    db.session.commit()
    flash(f"Deleted {name}.", "success")
    return redirect(url_for('list_students'))


if __name__ == '__main__':
    # Debug mode exposes the Werkzeug interactive debugger (a remote code execution console
    # if the app is ever reachable off-localhost). Off by default; opt in with FLASK_DEBUG=1.
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(debug=debug_mode)
