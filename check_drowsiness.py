import cv2
import dlib
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
import time

# --- Configuration ---
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
MODEL_DIR = "models"
YAWN_MODEL_FILENAME = "yawn_detection_model.pth"
EYE_MODEL_FILENAME = "eye_detection_model.pth"  # New eye model filename
YAWN_MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, YAWN_MODEL_FILENAME)
EYE_MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, EYE_MODEL_FILENAME)  # Path for eye model

# Yawn detection parameters
YAWN_CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence to consider as yawning
YAWN_FRAME_THRESHOLD = 6          # Number of consecutive frames needed to confirm a yawn
MOUTH_START_INDEX = 48
MOUTH_END_INDEX = 67
YAWN_NUM_CLASSES = 2              # Yawning, Not Yawning
YAWN_CLASS_NAMES = ["Not Yawning", "Yawning"]

# Eye detection parameters
EYE_CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence for eye state
EYE_FRAME_THRESHOLD = 5           # Number of consecutive frames needed to confirm eye state
LEFT_EYE_START_INDEX = 36
LEFT_EYE_END_INDEX = 41
RIGHT_EYE_START_INDEX = 42
RIGHT_EYE_END_INDEX = 47
EYE_NUM_CLASSES = 2               # Open_Eyes, Closed_Eyes
EYE_CLASS_NAMES = ["Open_Eyes", "Closed_Eyes"]

# --- Initialization ---
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# --- PyTorch Setup ---
print("[INFO] Setting up PyTorch...")
device = torch.device("cpu")
print(f"[INFO] Using device: {device}")

# --- Define Preprocessing ---
preprocess_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_frame_roi(roi_image_np, device):
    """Preprocess ROI for the model"""
    try:
        img_rgb = cv2.cvtColor(roi_image_np, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess_transform(img_rgb)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        return img_tensor
    except Exception as e:
        print(f"[WARNING] Error processing ROI image: {e}")
        return None

# --- Load PyTorch Models ---
# Load Yawn Detection Model
print(f"[INFO] Loading yawn detection model from: {YAWN_MODEL_WEIGHTS_PATH}")
yawn_model = models.resnet18(weights=None)
num_ftrs = yawn_model.fc.in_features
yawn_model.fc = torch.nn.Linear(num_ftrs, YAWN_NUM_CLASSES)
yawn_model.load_state_dict(torch.load(YAWN_MODEL_WEIGHTS_PATH, map_location=device))
yawn_model.to(device)
yawn_model.eval()

# Load Eye Detection Model
print(f"[INFO] Loading eye detection model from: {EYE_MODEL_WEIGHTS_PATH}")
eye_model = models.resnet18(weights=None)
num_ftrs = eye_model.fc.in_features
eye_model.fc = torch.nn.Linear(num_ftrs, EYE_NUM_CLASSES)
eye_model.load_state_dict(torch.load(EYE_MODEL_WEIGHTS_PATH, map_location=device))
eye_model.to(device)
eye_model.eval()

# --- Tracking ---
yawn_counter = 0       # Count consecutive frames where yawning is detected
is_yawning = False     # Current yawn state
eyes_closed_counter = 0  # Count consecutive frames where eyes are closed
eyes_closed = False    # Current eye state

# --- Video Stream ---
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open video device")
    exit()

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame")
        break

    # Make a copy for drawing
    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rects = detector(gray, 1)
    
    # Process each detected face (we'll focus on the first one for simplicity)
    for (i, rect) in enumerate(rects):
        # Get landmarks
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype="int")
        for j in range(0, 68):
            coords[j] = (shape.part(j).x, shape.part(j).y)

        # --- MOUTH PROCESSING ---
        # Extract mouth bounding box
        mouth_coords = coords[MOUTH_START_INDEX:MOUTH_END_INDEX + 1]
        if len(mouth_coords) > 0:
            (x_min, y_min) = np.min(mouth_coords, axis=0)
            (x_max, y_max) = np.max(mouth_coords, axis=0)
            padding = 5

            # Ensure coordinates are within frame boundaries
            roi_x_start = max(0, x_min - padding)
            roi_y_start = max(0, y_min - padding)
            roi_x_end = min(frame.shape[1], x_max + padding)
            roi_y_end = min(frame.shape[0], y_max + padding)

            # Draw rectangle around mouth
            cv2.rectangle(display_frame, (roi_x_start, roi_y_start), 
                        (roi_x_end, roi_y_end), (0, 255, 0), 2)

            # Extract and process mouth ROI
            mouth_roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            
            if mouth_roi.shape[0] > 0 and mouth_roi.shape[1] > 0:
                input_tensor = preprocess_frame_roi(mouth_roi, device)
                
                if input_tensor is not None:
                    try:
                        # Perform inference for yawn detection
                        with torch.no_grad():
                            output = yawn_model(input_tensor)
                        
                        # Get prediction and confidence
                        probabilities = torch.softmax(output, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                        predicted_class = YAWN_CLASS_NAMES[predicted_idx.item()]
                        confidence_score = confidence.item()
                        
                        # Check if yawning with sufficient confidence
                        current_frame_yawning = (predicted_idx.item() == 1 and 
                                                confidence_score >= YAWN_CONFIDENCE_THRESHOLD)
                        
                        # Update yawn counter
                        if current_frame_yawning:
                            yawn_counter += 1
                        else:
                            yawn_counter = 0
                        
                        # Determine yawn state based on consecutive frame threshold
                        if yawn_counter >= YAWN_FRAME_THRESHOLD:
                            is_yawning = True
                        elif yawn_counter == 0:
                            is_yawning = False
                        
                        # Display appropriate message
                        if is_yawning:
                            message = f"YAWNING! ({yawn_counter} frames)"
                            color = (0, 0, 255)  # Red for confirmed yawn
                        elif yawn_counter > 0:
                            message = f"Possible yawn ({yawn_counter}/{YAWN_FRAME_THRESHOLD})"
                            color = (0, 165, 255)  # Orange for potential yawn
                        else:
                            message = f"{predicted_class} ({confidence_score:.2f})"
                            color = (0, 255, 0)  # Green for not yawning
                        
                        # Draw status text
                        cv2.putText(display_frame, message, 
                                    (roi_x_start, roi_y_start - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        print(f"[WARNING] Error during yawn inference: {e}")

        # --- EYE PROCESSING ---
        # Extract eyes bounding boxes
        # Left eye
        left_eye_coords = coords[LEFT_EYE_START_INDEX:LEFT_EYE_END_INDEX + 1]
        right_eye_coords = coords[RIGHT_EYE_START_INDEX:RIGHT_EYE_END_INDEX + 1]
        
        if len(left_eye_coords) > 0 and len(right_eye_coords) > 0:
            # Process both eyes together
            eye_coords = np.vstack((left_eye_coords, right_eye_coords))
            (ex_min, ey_min) = np.min(eye_coords, axis=0)
            (ex_max, ey_max) = np.max(eye_coords, axis=0)
            padding = 10
            
            # Ensure coordinates are within frame boundaries
            eye_roi_x_start = max(0, ex_min - padding)
            eye_roi_y_start = max(0, ey_min - padding)
            eye_roi_x_end = min(frame.shape[1], ex_max + padding)
            eye_roi_y_end = min(frame.shape[0], ey_max + padding)
            
            # Draw rectangle around eyes
            cv2.rectangle(display_frame, (eye_roi_x_start, eye_roi_y_start), 
                        (eye_roi_x_end, eye_roi_y_end), (255, 0, 0), 2)
            
            # Extract and process eyes ROI
            eyes_roi = frame[eye_roi_y_start:eye_roi_y_end, eye_roi_x_start:eye_roi_x_end]
            
            if eyes_roi.shape[0] > 0 and eyes_roi.shape[1] > 0:
                input_tensor = preprocess_frame_roi(eyes_roi, device)
                
                if input_tensor is not None:
                    try:
                        # Perform inference for eye detection
                        with torch.no_grad():
                            output = eye_model(input_tensor)
                        
                        # Get prediction and confidence
                        probabilities = torch.softmax(output, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                        predicted_class = EYE_CLASS_NAMES[predicted_idx.item()]
                        confidence_score = confidence.item()
                        
                        # Check if eyes are closed with sufficient confidence
                        current_frame_eyes_closed = (predicted_idx.item() == 1 and 
                                                    confidence_score >= EYE_CONFIDENCE_THRESHOLD)
                        
                        # Update eye state counter
                        if current_frame_eyes_closed:
                            eyes_closed_counter += 1
                        else:
                            eyes_closed_counter = 0
                        
                        # Determine eye state based on consecutive frame threshold
                        if eyes_closed_counter >= EYE_FRAME_THRESHOLD:
                            eyes_closed = True
                        elif eyes_closed_counter == 0:
                            eyes_closed = False
                        
                        # Display appropriate message
                        if eyes_closed:
                            eye_message = f"EYES CLOSED! ({eyes_closed_counter} frames)"
                            eye_color = (0, 0, 255)  # Red for closed eyes
                        elif eyes_closed_counter > 0:
                            eye_message = f"Closing eyes ({eyes_closed_counter}/{EYE_FRAME_THRESHOLD})"
                            eye_color = (0, 165, 255)  # Orange for potentially closing eyes
                        else:
                            eye_message = f"{predicted_class} ({confidence_score:.2f})"
                            eye_color = (255, 0, 0)  # Blue for open eyes
                        
                        # Draw status text
                        cv2.putText(display_frame, eye_message, 
                                    (eye_roi_x_start, eye_roi_y_start - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
                    except Exception as e:
                        print(f"[WARNING] Error during eye inference: {e}")
    
    # Add threshold indicators
    cv2.putText(display_frame, 
                f"Yawn Threshold: {YAWN_FRAME_THRESHOLD} frames", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(display_frame, 
                f"Eye Threshold: {EYE_FRAME_THRESHOLD} frames", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow("Yawn & Eye Detection - Press 'q' to quit", display_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()