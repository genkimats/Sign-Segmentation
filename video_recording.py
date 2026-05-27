import cv2
import os

# --- Configuration ---
base_name = ""
output_dir = "raw_video"
countdown_time = 5 

# Screen and Window dimensions
screen_w, screen_h = 1470, 956
window_w, window_h = 1280, 720

# Create directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Automatic Increment Logic ---
counter = 1
while True:
    filename = f"{base_name}_{counter}.mp4"
    save_path = os.path.join(output_dir, filename)
    if not os.path.exists(save_path):
        break  # Found a filename that doesn't exist yet
    counter += 1

# Calculate center position
pos_x = int((screen_w - window_w) / 2)
pos_y = int((screen_h - window_h) / 2)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Setup and Center the Window
window_name = 'Webcam Feed'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, window_w, window_h)
cv2.moveWindow(window_name, pos_x, pos_y)

print(f"Target file: {filename}")
print(f"Press 'r' to start {countdown_time}s countdown. Press 'q' to quit.")

recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF

    # Trigger Recording
    if key == ord('r') and not recording:
        for i in range(countdown_time, 0, -1):
            ret, frame = cap.read()
            # Visual countdown
            cv2.putText(frame, str(i), (int(frame.shape[1]/2)-50, int(frame.shape[0]/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1000)
        
        # Define codec and writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))
        
        recording = True
        print(f"Now recording to {filename}...")

    if recording:
        out.write(frame)
        # Red recording dot
        cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
        cv2.imshow(window_name, frame)
        
        # Press 's' to stop recording
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    if key == ord('q'):
        break

# Cleanup
cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
print("Done.")