from __future__ import annotations

import argparse
from pathlib import Path

import face_recognition

from app import app, get_or_create_batch, initialize_database, serialize_face_encoding
from models import Student, db


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register students from the Photos folder into the database.")
    parser.add_argument("--photos-dir", default="Photos", help="Path to the folder containing student photos.")
    parser.add_argument(
        "--batch",
        help="Batch name to use for images stored directly inside Photos/. Subfolders use their folder name as the batch name.",
    )
    return parser.parse_args()


def discover_photo_jobs(photos_dir: Path, fallback_batch: str | None) -> list[tuple[Path, str]]:
    jobs: list[tuple[Path, str]] = []

    for image_path in sorted(path for path in photos_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
        relative_parts = image_path.relative_to(photos_dir).parts

        if len(relative_parts) == 1:
            if not fallback_batch:
                raise ValueError("Images in the root Photos folder require --batch <name>.")
            batch_name = fallback_batch
        else:
            batch_name = relative_parts[0]

        jobs.append((image_path, batch_name))

    return jobs


def register_students(photos_dir: Path, fallback_batch: str | None) -> None:
    if not photos_dir.exists():
        raise FileNotFoundError(f"Photos directory not found: {photos_dir}")

    jobs = discover_photo_jobs(photos_dir, fallback_batch)
    if not jobs:
        raise ValueError("No supported image files were found in the Photos directory.")

    initialize_database()

    with app.app_context():
        created_count = 0
        updated_count = 0
        skipped_count = 0

        for image_path, batch_name in jobs:
            student_name = image_path.stem
            print(f"Processing {image_path.name} for batch {batch_name}...")

            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if len(face_encodings) != 1:
                print(f"Skipping {image_path.name}: expected 1 face, found {len(face_encodings)}.")
                skipped_count += 1
                continue

            batch = get_or_create_batch(batch_name)
            face_blob = serialize_face_encoding(face_encodings[0])

            student = db.session.execute(
                db.select(Student).where(Student.name == student_name, Student.batch_id == batch.id)
            ).scalar_one_or_none()

            if student is None:
                db.session.add(Student(name=student_name, batch_id=batch.id, face_encoding=face_blob))
                created_count += 1
            else:
                student.face_encoding = face_blob
                updated_count += 1

        db.session.commit()
        print(f"Student registration complete. Created: {created_count}, Updated: {updated_count}, Skipped: {skipped_count}")


def main() -> None:
    args = parse_args()
    register_students(Path(args.photos_dir), args.batch)


if __name__ == "__main__":
    main()