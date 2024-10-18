import cv2
import numpy as np
import pytesseract
import tensorflow as tf  # or import torch if using PyTorch
import time

# Constants
NO_PARKING_LABEL = "No Parking"
SPEED_LIMIT_LABEL = "Speed Limit"

# Load your trained model (replace 'model_path' with your actual model path)
# For TensorFlow
# model = tf.keras.models.load_model('model_path')
# For PyTorch
# model = torch.load('model_path')
# model.eval()

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change '0' to your camera index

def detect_signs(frame):
    """
    Detect No Parking and Speed Limit signs in the frame.
    This function should implement the object detection logic.
    Replace with your actual detection model code.
    """
    # Placeholder for detection logic
    detected_signs = []

    # Example of a detected sign (you should implement your detection logic)
    # detected_signs.append((x, y, w, h, "No Parking"))  # (x, y, width, height, label)
    
    return detected_signs

def extract_speed_limit(roi):
    """
    Use OCR to extract speed limit text from the region of interest (ROI).
    """
    text = pytesseract.image_to_string(roi)
    return text.strip()  # Clean the extracted text

def alert_driver(sign_type):
    """
    Trigger alerts based on detected signs.
    """
    if sign_type == NO_PARKING_LABEL:
        print("Alert: No Parking Zone!")
        # You can add audio alerts here if needed
    elif sign_type == SPEED_LIMIT_LABEL:
        print("Alert: Speed Limit Exceeded!")

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Pre-process the frame (if necessary)
        # Example: Convert to RGB, resize, etc.
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect signs in the frame
        detected_signs = detect_signs(frame)

        # Process detected signs
        for (x, y, w, h, label) in detected_signs:
            # Draw bounding box around detected sign
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if label == SPEED_LIMIT_LABEL:
                # Extract the ROI for OCR
                roi = frame[y:y + h, x:x + w]
                speed_limit_text = extract_speed_limit(roi)

                # Display the detected speed limit on the frame
                cv2.putText(frame, f"Speed Limit: {speed_limit_text}", (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check speed limit (you'll need to implement your speed detection logic)
                current_speed = 60  # Placeholder for actual speed reading
                if current_speed > int(speed_limit_text):
                    alert_driver(SPEED_LIMIT_LABEL)

            if label == NO_PARKING_LABEL:
                alert_driver(NO_PARKING_LABEL)

        # Show the frame with detections
        cv2.imshow('Video Feed', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
