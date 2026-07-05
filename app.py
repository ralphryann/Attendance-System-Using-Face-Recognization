from flask import Flask, render_template, request, redirect, url_for, flash, abort
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless server environments
import matplotlib.pyplot as plt
import os
import csv
import re


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/take_attendance', methods=['POST', 'GET'])
def take_attendance():
    return render_template('select_batch_subject.html', batches=BATCHES, subjects=SUBJECTS)

@app.route('/submit_attendance', methods=['POST'])
def submit_attendance():
    import cv2
    import face_recognition
    import numpy as np
    import os, csv, time
    from datetime import datetime
    import pickle   
    batch = request.form['batch']
    subject = request.form['subject']

    try:
        # Colour.change_color("bo,u")
        heading = True

        current_month_number = datetime.today().month
        batch_path = os.path.join(ATTENDANCE_ROOT, batch, MONTHS[current_month_number])
        os.makedirs(batch_path, exist_ok=True)
        file_name = f"{batch_path}/{subject}_{datetime.today().date()}.csv"

        # Dummy encoding loader
        with open("face_encodings.pkl", "rb") as f:
               known_face_encodings =pickle.load(f)
        
        # known_face_encodings = np.load("encodings.npy")  # Replace with actual path
        known_face_names = ["Divy", "Elon", "Manasvi", "Meet", "SRK"]
        student_list = known_face_names.copy()
        present_students = []

        blink_status = {
            name: {"eyes_closed_frames": 0, "blink_detected": False, "attendance_marked": False}
            for name in known_face_names
        }

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

            for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                if name not in blink_status:
                    blink_status[name] = {"eyes_closed_frames": 0, "blink_detected": False, "attendance_marked": False}

                def eye_aspect_ratio(eye):
                    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
                    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
                    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
                    return (A + B) / (2.0 * C)

                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                BLINK_THRESHOLD = 0.20
                BLINK_FRAMES = 3

                if avg_ear < BLINK_THRESHOLD:
                    blink_status[name]["eyes_closed_frames"] += 1
                else:
                    if blink_status[name]["eyes_closed_frames"] >= BLINK_FRAMES:
                        blink_status[name]["blink_detected"] = True
                    blink_status[name]["eyes_closed_frames"] = 0

                if blink_status[name]["blink_detected"] and not blink_status[name]["attendance_marked"] and name in student_list:
                    print(f"{name} blinked! ✅ Marking attendance.")
                    present_students.append((name, datetime.now().strftime("%H:%M:%S")))
                    student_list.remove(name)
                    blink_status[name]["blink_detected"] = False
                    blink_status[name]["attendance_marked"] = True

                color = (0, 255, 0) if blink_status[name]["attendance_marked"] else (0, 255, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                msg = "Attendance Marked" if blink_status[name]["attendance_marked"] else "Blink to mark attendance"
                cv2.putText(frame, msg, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, "Press 'q' or ESC to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()
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
            print("⚠️ Failed to write attendance file:", e)

        print(f"✅ Attendance saved to {file_name}")
        abs_path = os.path.abspath(file_name)
        # os.startfile(abs_path)

        view_url = url_for('view_attendance', file=file_name)
        return redirect(view_url)

    except Exception as e:
        print("⚠️ ERROR:", e)
        flash(f"Error during attendance: {e}", "error")
        return redirect(url_for('home'))
    

@app.route("/view_attendance")
def view_attendance():
    file_path = request.args.get("file", "")

    # Prevent path traversal: resolve and verify the path stays inside the project
    abs_file = os.path.abspath(file_path)
    abs_root = os.path.abspath(ATTENDANCE_ROOT)
    if not abs_file.startswith(abs_root) or not os.path.isfile(abs_file):
        abort(404)

    rows = []
    with open(abs_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    summary = {}
    filtered_rows = []

    # If last two rows are the summary, extract them
    if len(rows) >= 2 and rows[-2][0] == "Total":
        header = rows[-2]
        values = rows[-1]
        summary = {header[i]: values[i] for i in range(min(len(header), len(values)))}
        filtered_rows = rows[:-2]  # exclude the last two rows
    else:
        filtered_rows = rows

    header = filtered_rows[0] if filtered_rows else []  # assuming first row is header
    body_rows = filtered_rows[1:-3] if len(filtered_rows) > 1 else []  # rest are the data rows

    # Calculate the summary
    total_students = len(body_rows)
    present_count = sum(1 for row in body_rows if 'Present' in row)
    absent_count = sum(1 for row in body_rows if 'Absent' in row)

    summary = {
        'Total': total_students,
        'Present': present_count,
        'Absent': absent_count
    }

    return render_template('attendance_report.html', header=header, rows=body_rows, summary=summary)

@app.route('/view_analytics', methods=['GET', 'POST'])
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
                    if file.startswith(subject):
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

if __name__ == '__main__':
    app.run(debug=True)
