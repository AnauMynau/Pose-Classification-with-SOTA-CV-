import os
import cv2

def extract_frames_by_time(video_path, output_dir, start_sec, end_sec, step=6):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть файл '{video_path}'. Проверь название и путь.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    saved_count = 0

    print(f"Нарезка интервала {start_sec}c - {end_sec}c в '{output_dir}'...")

    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (current_frame - start_frame) % step == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        current_frame += 1

    cap.release()
    print(f"-> Ready: saved {saved_count} photos in '{output_dir}'\n")


if __name__ == "__main__":
    VIDEO_FILE = "workout.mp4"

    # 1. Standing: 0 - 26 s
    extract_frames_by_time(
        video_path=VIDEO_FILE,
        output_dir="./raw_data/standing",
        start_sec=0,
        end_sec=26,
        step=5
    )

    # 2. Pushup: 26 - 49 s
    extract_frames_by_time(
        video_path=VIDEO_FILE,
        output_dir="./raw_data/pushup",
        start_sec=26,
        end_sec=49,
        step=5
    )

    # 3. Squat: 49 - 100 s
    extract_frames_by_time(
        video_path=VIDEO_FILE,
        output_dir="./raw_data/squat",
        start_sec=49,
        end_sec=100,
        step=10
    )