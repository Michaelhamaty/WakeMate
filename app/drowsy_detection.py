import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import dlib
import math
from chime_manager import ChimeManager
from config.model_config import model_config as k

def __init_models():
    face_detector = dlib.get_frontal_face_detector()
    facial_features_detector = dlib.shape_predictor(k.PREDICTOR_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    yawn_model = models.resnet18(weights=None)
    yawn_model.fc = torch.nn.Linear(yawn_model.fc.in_features, k.YAWN_NUM_CLASSES)
    yawn_model.load_state_dict(torch.load(k.YAWN_MODEL_WEIGHTS_PATH, map_location=device))
    yawn_model.to(device)
    yawn_model.eval()

    eye_model = models.resnet18(weights=None)
    eye_model.fc = torch.nn.Linear(eye_model.fc.in_features, k.EYE_NUM_CLASSES)
    eye_model.load_state_dict(torch.load(k.EYE_MODEL_WEIGHTS_PATH, map_location=device))
    eye_model.to(device) 
    eye_model.eval()

    print(device)

    return face_detector, facial_features_detector, device, preprocess_transform, yawn_model, eye_model

def __get_roi(frame, pts, scale_factor=2):
    """
    Extracts a square region of interest (ROI) centered around a set of landmark points,
    with size scaled relative to facial geometry to maintain consistency across varying face depths.

    The core idea:
    - Face proximity to the camera affects absolute pixel distances.
    - We compute a reference width using key facial points (e.g., eye corners or mouth width).
    - A square crop is created around the center of the landmarks, scaled by a factor of this width.
    
    Math:
    - Center of ROI = mean of all (x, y) coordinates
    - Reference width = Euclidean distance between two anchor points (typically farthest apart)
    - Crop size = reference width * scale_factor (e.g., 2.5x the feature width)
    - ROI bounds are clamped to frame dimensions

    This ensures that the extracted region captures the same relative area of the face
    regardless of distance from the camera, minimizing variability during model inference.
    """
    if not pts:
        return None, None, None, None, None

    # Compute center of region
    cx = int(sum(p[0] for p in pts) / len(pts))
    cy = int(sum(p[1] for p in pts) / len(pts))

    # Use bounding box width as base size reference
    width = math.dist(pts[0], pts[len(pts) // 2])  # e.g. eye: pt0 to pt3, mouth: pt48 to pt54
    crop_size = int(width * scale_factor)

    x_min = max(cx - crop_size // 2, 0)
    y_min = max(cy - crop_size // 2, 0)
    x_max = min(cx + crop_size // 2, frame.shape[1])
    y_max = min(cy + crop_size // 2, frame.shape[0])

    roi = frame[y_min:y_max, x_min:x_max]
    return x_min, y_min, x_max, y_max, roi

    
def __preprocess_roi(roi, preprocess_transform):
    return preprocess_transform(roi).unsqueeze(0) if roi is not None else None

def start_tracking(cap: cv2.VideoCapture, chime_manager: ChimeManager):
    face_detector, facial_faetures_detector, device, preprocess_transform, yawn_model, eye_model = __init_models()
    yawn_counter = 0
    is_yawning = False
    closed_eyes_counter = 0
    is_eyes_closed = False

    if not cap.isOpened():
        print("[ERROR] Could not open video capture.")
        return
    
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame from video capture.")
            break 

        display_frame = frame.copy()

        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(grey_frame, 1)

        for face in faces:
            features = facial_faetures_detector(grey_frame, face)
            # using a 68-point facial landmark model
            facial_features_pts = [(features.part(i).x, features.part(i).y) for i in range(68)]

            left_eye_idx = list(range(k.LEFT_EYE_START_INDEX, k.LEFT_EYE_END_INDEX + 1))
            right_eye_idx = list(range(k.RIGHT_EYE_START_INDEX, k.RIGHT_EYE_END_INDEX + 1))
            mouth_idx = list(range(k.MOUTH_START_INDEX, k.MOUTH_END_INDEX + 1))

            # Extract coordinates for left eye, right eye, and mouth
            left_eye_pts = [facial_features_pts[i] for i in left_eye_idx]
            right_eye_pts = [facial_features_pts[i] for i in right_eye_idx]
            mouth_pts = [facial_features_pts[i] for i in mouth_idx]

            # Get bounding boxes for each region of interest
            left_eye_x_min, left_eye_y_min, _, _, left_eye_roi = __get_roi(frame, left_eye_pts, 1.75)
            _, _, right_eye_x_max, right_eye_y_max, right_eye_roi = __get_roi(frame, right_eye_pts, 1.75)
            mouth_x_min, mouth_y_min, mouth_x_max, mouth_y_max, mouth_roi = __get_roi(frame, mouth_pts, 2.5)


            # Convert to tensors for model input
            left_eye_tensor = __preprocess_roi(left_eye_roi, preprocess_transform)
            right_eye_tensor = __preprocess_roi(right_eye_roi, preprocess_transform)
            mouth_tensor = __preprocess_roi(mouth_roi, preprocess_transform)

            if left_eye_tensor is not None and right_eye_tensor is not None and mouth_tensor is not None:
                try:
                    eyes_conf = None
                    eyes_class = None
                    yawn_conf = None
                    yawn_class = None
                    with torch.no_grad():
                        # Model returns probabilities for each class in raw logits
                        left_eye_output = eye_model(left_eye_tensor.to(device))
                        right_eye_output = eye_model(right_eye_tensor.to(device))
                        yawn_output = yawn_model(mouth_tensor.to(device))

                        # Convert raw logits to probabilities between 0 and 1
                        left_eye_prob = torch.softmax(left_eye_output, dim=1)
                        right_eye_prob = torch.softmax(right_eye_output, dim=1)
                        yawn_prob = torch.softmax(yawn_output, dim=1)

                        # Take max probability and class index
                        left_eye_conf, left_eye_class = torch.max(left_eye_prob, dim=1)
                        right_eye_conf, right_eye_class = torch.max(right_eye_prob, dim=1)
                        yawn_conf, yawn_class = torch.max(yawn_prob, dim=1)
                        
                        left_eye_conf, left_eye_class = left_eye_conf.item(), left_eye_class.item()
                        right_eye_conf, right_eye_class = right_eye_conf.item(), right_eye_class.item()
                        yawn_conf, yawn_class = yawn_conf.item(), yawn_class.item()

                        eyes_conf = (left_eye_conf + right_eye_conf) / 2.0
                        eyes_class = 1 if left_eye_class == 1 and right_eye_class == 1 else 0

                    # Closed Eyes Detection
                    if eyes_class == 1 and eyes_conf >= k.EYE_CONFIDENCE_THRESHOLD:
                        closed_eyes_counter += 1
                    else:
                        closed_eyes_counter = 0
                        is_eyes_closed = False

                    if closed_eyes_counter >= k.EYE_FRAME_THRESHOLD and not is_eyes_closed:
                        is_eyes_closed = True
                        chime_manager.record_eye_close()
                        closed_eyes_counter = 0
                    
                    # Yawning Detection
                    if yawn_class == 1 and yawn_conf >= k.YAWN_CONFIDENCE_THRESHOLD:
                        yawn_counter += 1
                    else:
                        yawn_counter = 0
                        is_yawning = False
                        
                    if yawn_counter >= k.YAWN_FRAME_THRESHOLD and not is_yawning:
                        is_yawning = True
                        chime_manager.record_yawn()
                        yawn_counter = 0
                    
                    eyes_message, yawn_message = "", ""
                    eyes_color, yawn_color = (0, 255, 0), (0, 255, 0)
                    
                    # Eyes Status Display
                    if not is_eyes_closed and closed_eyes_counter > 0:
                        eyes_message = f"Eyes Closing: {yawn_conf:.2f}"
                        eyes_color = (0, 165, 255)
                    else:
                        eyes_message = f"{k.EYE_CLASS_NAMES[eyes_class]}: {eyes_conf:.2f}"
                        eyes_color = (0, 255, 0) if eyes_class == 0 else (0, 0, 255)
                    
                    # Yawning Status Display
                    if not is_yawning and yawn_counter > 0:
                        yawn_message += f"Possibly Yawning: {yawn_conf:.2f}"
                        yawn_color = (0, 165, 255)
                    else:
                        yawn_message += f"{k.YAWN_CLASS_NAMES[yawn_class]}: {yawn_conf:.2f}"
                        yawn_color = (0, 255, 0) if yawn_class == 0 else (0, 0, 255)
                    
                    # Draw Eyes Bounding Boxes and Messages
                    cv2.putText(display_frame, eyes_message, 
                                (left_eye_x_min, left_eye_y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, eyes_color, 2)
                    cv2.rectangle(display_frame, 
                                (left_eye_x_min, left_eye_y_min), 
                                (right_eye_x_max, right_eye_y_max), 
                                eyes_color, 2)
                    
                    # Draw Yawn Bounding Box and Message
                    cv2.putText(display_frame, yawn_message, 
                                (mouth_x_min, mouth_y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, yawn_color, 2)
                    cv2.rectangle(display_frame, 
                                (mouth_x_min, mouth_y_min), 
                                (mouth_x_max, mouth_y_max), 
                                yawn_color, 2)
                except Exception as e:
                    print(f"[ERROR] Model inference failed: {e}")
                    continue


        # Display total Yawns and Eyes Closed Frames
        cv2.putText(display_frame,
                    f"Yawns: {chime_manager.get_yawn_count()}  Eyes Closed: {chime_manager.get_eye_close_count()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', display_frame)
        display_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + display_frame + b'\r\n')
        frame_counter += 1