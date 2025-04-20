import cv2
import dlib
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os # To build the model path reliably

# --- Configuration ---
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
# Build path relative to the script directory
MODEL_DIR = "models"
MODEL_FILENAME = "yawn_detection_model.pth"
# Use os.path.join for cross-platform compatibility
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
# --- PyTorch Setup ---
print("[INFO] Setting up PyTorch...")
# Use CPU as requested
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

MOUTH_START_INDEX = 48
MOUTH_END_INDEX = 67
NUM_CLASSES = 2 # Yawning, Not Yawning
CLASS_NAMES = ["Not Yawning", "Yawning"] # Assign labels to output indices

# --- Initialization ---
print("[INFO] Loading facial landmark predictor...")
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError as e:
    print(f"[ERROR] Failed to load Dlib predictor model: {e}")
    print(f"[ERROR] Make sure '{PREDICTOR_PATH}' exists.")
    exit()
except Exception as e:
    print(f"[ERROR] An unexpected error occurred loading Dlib components: {e}")
    exit()


# --- PyTorch Setup ---
print("[INFO] Setting up PyTorch...")
# Use CPU as requested
device = torch.device("cpu")
print(f"[INFO] Using device: {device}")

# --- Define Preprocessing ---
# Define the transformations sequence based on the provided function
# Input size 224x224, Grayscale (3 channels), Tensor, Normalize
preprocess_transform = transforms.Compose([
    transforms.ToPILImage(), # Convert NumPy array (HWC) to PIL Image first
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), # Convert to grayscale but replicate to 3 channels
    transforms.ToTensor(), # Converts PIL image [0, 255] to Tensor (CHW) [0.0, 1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
])

def preprocess_frame_roi(roi_image_np, device):
    """
    Preprocesses a NumPy image array (ROI) using the defined transform sequence.

    Args:
        roi_image_np: The cropped mouth region (NumPy array, BGR format from OpenCV).
        device: The torch device (CPU or CUDA).

    Returns:
        A preprocessed image tensor ready for the model (batch dim added), moved to the device.
        Returns None on error.
    """
    try:
        # OpenCV captures in BGR, ToPILImage expects RGB or Grayscale. Convert BGR -> RGB.
        img_rgb = cv2.cvtColor(roi_image_np, cv2.COLOR_BGR2RGB)

        # Apply the transformations pipeline
        img_tensor = preprocess_transform(img_rgb)

        # Add batch dimension [C, H, W] -> [B, C, H, W] and move to device
        img_tensor = img_tensor.unsqueeze(0).to(device)
        return img_tensor
    except Exception as e:
        # Print error during preprocessing a specific frame, but don't exit the program
        print(f"[WARNING] Error processing ROI image: {e}")
        return None

# --- Load PyTorch Model (Assuming ResNet18) ---
print(f"[INFO] Loading yawn detection model from: {MODEL_WEIGHTS_PATH}")
try:
    # 1. Load the ResNet18 base architecture. We'll load weights later.
    #    Set 'weights=None' as we are loading our own fine-tuned weights.
    model = models.resnet18(weights=None) # Or potentially models.resnet18(pretrained=False) in older torchvision

    # 2. Modify the final fully connected layer for NUM_CLASSES outputs
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

    # 3. Load your saved weights
    if not os.path.exists(MODEL_WEIGHTS_PATH):
         raise FileNotFoundError(f"Model file not found at {MODEL_WEIGHTS_PATH}. Make sure it's in a 'models' subfolder.")

    # Load the state dictionary onto the CPU
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))

    # 4. Move model structure to the specified device (CPU)
    model.to(device)

    # 5. Set to evaluation mode (important!)
    model.eval()
    print("[INFO] Yawn detection model loaded successfully.")

except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    exit()
except Exception as e:
    print(f"[ERROR] Failed to load PyTorch model: {e}")
    # Consider printing more details or traceback for debugging complex model load issues
    exit()


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

    # Make a copy for drawing to avoid modifying the original fed to detectors/models if needed later
    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rects = detector(gray, 1)

    # Process each detected face
    for (i, rect) in enumerate(rects):
        # Get landmarks
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype="int")
        for j in range(0, 68):
            coords[j] = (shape.part(j).x, shape.part(j).y)

        # Extract mouth bounding box
        mouth_coords = coords[MOUTH_START_INDEX : MOUTH_END_INDEX + 1]
        if len(mouth_coords) < 1: # Check if mouth coords were found
            continue

        (x_min, y_min) = np.min(mouth_coords, axis=0)
        (x_max, y_max) = np.max(mouth_coords, axis=0)
        padding = 5 # Add some padding around the exact landmarks

        # Ensure coordinates are within frame boundaries BEFORE cropping
        roi_x_start = max(0, x_min - padding)
        roi_y_start = max(0, y_min - padding)
        roi_x_end = min(frame.shape[1], x_max + padding) # frame.shape[1] is width
        roi_y_end = min(frame.shape[0], y_max + padding) # frame.shape[0] is height

        # Draw rectangle on the *display* frame
        cv2.rectangle(display_frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2) # Green rectangle

        # --- Yawn Detection Inference ---
        # 1. Extract Mouth ROI from the *original* frame
        mouth_roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # 2. Check if ROI is valid (has non-zero dimensions)
        if mouth_roi.shape[0] > 0 and mouth_roi.shape[1] > 0:
            # 3. Preprocess the ROI
            input_tensor = preprocess_frame_roi(mouth_roi, device)

            prediction_text = "Processing..." # Default text

            if input_tensor is not None:
                try:
                    # 4. Perform Inference
                    with torch.no_grad(): # Disable gradient calculation for inference
                        output = model(input_tensor)

                    # 5. Post-process Output (Get prediction)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    predicted_class = CLASS_NAMES[predicted_idx.item()]
                    confidence_score = confidence.item()

                    prediction_text = f"{predicted_class} ({confidence_score:.2f})"

                except Exception as e:
                    print(f"[WARNING] Error during model inference: {e}")
                    prediction_text = "Inference Error"
            else:
                 prediction_text = "Preproc Error"


            # 6. Display prediction on the *display* frame
            text_pos = (roi_x_start, roi_y_start - 10 if roi_y_start > 10 else roi_y_start + 10)
            cv2.putText(display_frame, prediction_text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # Red text


    # --- Display Frame ---
    cv2.imshow("Yawn Detection - Press 'q' to quit", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[INFO] Quitting...")
        break

# --- Cleanup ---
print("[INFO] Releasing resources...")
cap.release()
cv2.destroyAllWindows()