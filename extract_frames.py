import cv2
import os

def extract_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 3:
        print("Video is too short.")
        return

    # Determine the frame indices
    frame_indices = [0, total_frames // 2, total_frames - 1]
    frame_names = ["first", "middle", "last"]

    for idx, name in zip(frame_indices, frame_names):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f"{name}_frame.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
        else:
            print(f"Failed to read frame at index {idx}")

    cap.release()

# Example usage
video_file = "./corn_production_line.mp4"
output_folder = "extracted_frames"
extract_frames(video_file, output_folder)
