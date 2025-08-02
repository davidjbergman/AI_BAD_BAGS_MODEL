import cv2
import os
from skimage.metrics import structural_similarity as ssim

MAX_FRAMES = 10000
target_size = (128, 128)  # (width, height)

video_path = "DJI_0824.MP4"
output_dir = "frames"
frame_interval = 30  # Save 1 frame every 10 frames

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0
prev_frame_gray = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        if count // frame_interval > MAX_FRAMES:
            break
        resized_frame = cv2.resize(frame, target_size)

        # Optional Gaussian blur to reduce noise
        # resized_frame = cv2.GaussianBlur(resized_frame, (3, 3), 0)

        # Optional crop to focus on center region (adjust as needed)
        # h, w = resized_frame.shape[:2]
        # cropped_frame = resized_frame[h//4:3*h//4, w//4:3*w//4]
        # resized_frame = cv2.resize(cropped_frame, target_size)

        # Convert to grayscale and apply histogram equalization
        current_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.equalizeHist(current_gray)

        if prev_frame_gray is not None:
            similarity = ssim(prev_frame_gray, current_gray)
            if similarity > 0.99:
                print(f"Skipped duplicate frame at count {count} (SSIM: {similarity:.4f})")
                count += 1
                continue
        prev_frame_gray = current_gray
        lap_var = cv2.Laplacian(current_gray, cv2.CV_64F).var()
        if lap_var < 100.0:
            print(f"Skipped blurry frame at count {count} (Laplacian Var: {lap_var:.2f})")
            count += 1
            continue
        filename = f"{output_dir}/frame_{count:05d}.png"
        cv2.imwrite(filename, resized_frame)
        print(f"Saved {filename}")
    
    count += 1

cap.release()