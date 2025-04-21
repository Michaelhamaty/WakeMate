import cv2
import dlib
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import threading  # added for async API calls
from chime_manager import ChimeManager
from config.model_config import model_config as k

def __init_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(k.PREDICTOR_PATH)
    device = torch.device("cpu")
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

    return detector, predictor, device, preprocess_transform, yawn_model, eye_model

def _preprocess_frame_roi(roi_image_np, device, preprocess_transform):
    """Preprocess ROI for the model"""
    try:
        img_rgb = cv2.cvtColor(roi_image_np, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess_transform(img_rgb)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        return img_tensor
    except Exception as e:
        print(f"[WARNING] Error processing ROI image: {e}")
        return None

def start_tracking(cap: cv2.VideoCapture, chime_manager: ChimeManager):
    detector, predictor, device, preprocess_transform, yawn_model, eye_model = __init_models()
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
            print("[INFO] No frames to read from video capture.")
            break
        
        display_frame = frame.copy()
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(grey_frame, 1)

        for (i, face) in enumerate(faces):
            face_shape = predictor(grey_frame, face)
            face_coords = np.zeros((68, 2), dtype="int")
            for j in range(0, 68):
                face_coords[j] = (face_shape.part(j).x, face_shape.part(j).y)
            
            mouth_coords = face_coords[k.MOUTH_START_INDEX:k.MOUTH_END_INDEX + 1]
            left_eye_coords = face_coords[k.LEFT_EYE_START_INDEX:k.LEFT_EYE_END_INDEX + 1]
            right_eye_coords = face_coords[k.RIGHT_EYE_START_INDEX:k.RIGHT_EYE_END_INDEX + 1]
            if len(mouth_coords) > 0 and len(left_eye_coords) > 0 and len(right_eye_coords) > 0:
                (mouth_x_min, mouth_y_min), (mouth_x_max, mouth_y_max) = np.min(mouth_coords, axis=0), np.max(mouth_coords, axis=0)
                eye_coords = np.vstack((left_eye_coords, right_eye_coords))
                # Calculate min/max for left eye
                (left_eye_x_min, left_eye_y_min), (left_eye_x_max, left_eye_y_max)= np.min(left_eye_coords, axis=0), np.max(left_eye_coords, axis=0)

                # Calculate min/max for right eye
                (right_eye_x_min, right_eye_y_min), (right_eye_x_max, right_eye_y_max)= np.min(right_eye_coords, axis=0), np.max(right_eye_coords, axis=0)

                mouth_padding = 5
                eye_padding = 15

                mouth_roi_x_start = max(0, mouth_x_min - mouth_padding)
                mouth_roi_y_start = max(0, mouth_y_min - mouth_padding)
                mouth_roi_x_end = min(frame.shape[1], mouth_x_max + mouth_padding)
                mouth_roi_y_end = min(frame.shape[0], mouth_y_max + mouth_padding)


                left_eye_roi_x_start = max(0, left_eye_x_min - eye_padding)
                left_eye_roi_y_start = max(0, left_eye_y_min - eye_padding)
                left_eye_roi_x_end = min(frame.shape[1], left_eye_x_max + eye_padding)
                left_eye_roi_y_end = min(frame.shape[0], left_eye_y_max + eye_padding)

                # Calculate bounding box for the right eye
                right_eye_roi_x_start = max(0, right_eye_x_min - eye_padding)
                right_eye_roi_y_start = max(0, right_eye_y_min - eye_padding)
                right_eye_roi_x_end = min(frame.shape[1], right_eye_x_max + eye_padding)
                right_eye_roi_y_end = min(frame.shape[0], right_eye_y_max + eye_padding)

                # Combine left and right eye ROIs into a single eye ROI
                eye_roi_x_start = min(left_eye_roi_x_start, right_eye_roi_x_start)
                eye_roi_y_start = min(left_eye_roi_y_start, right_eye_roi_y_start)
                eye_roi_x_end = max(left_eye_roi_x_end, right_eye_roi_x_end)
                eye_roi_y_end = max(left_eye_roi_y_end, right_eye_roi_y_end) 

                cv2.rectangle(display_frame, (mouth_roi_x_start, mouth_roi_y_start), (mouth_roi_x_end, mouth_roi_y_end), (0, 255, 0), 2)
                cv2.rectangle(display_frame, (eye_roi_x_start, eye_roi_y_start), (eye_roi_x_end, eye_roi_y_end), (255, 0, 0), 2)


                mouth_roi = frame[mouth_roi_y_start:mouth_roi_y_end, mouth_roi_x_start:mouth_roi_x_end]
                left_eye_roi = frame[left_eye_roi_y_start:left_eye_roi_y_end, left_eye_roi_x_start:left_eye_roi_x_end]
                right_eye_roi = frame[right_eye_roi_y_start:right_eye_roi_y_end, right_eye_roi_x_start:right_eye_roi_x_end]

                if mouth_roi.shape[0] > 0 and mouth_roi.shape[1] > 0:
                    mouth_tensor = _preprocess_frame_roi(mouth_roi, device, preprocess_transform)
                
                if left_eye_roi.shape[0] > 0 and left_eye_roi.shape[1] > 0:
                    left_eye_tensor = _preprocess_frame_roi(left_eye_roi, device, preprocess_transform)
                
                if right_eye_roi.shape[0] > 0 and right_eye_roi.shape[1] > 0:
                    right_eye_tensor = _preprocess_frame_roi(right_eye_roi, device, preprocess_transform)
                
                if mouth_tensor is not None and left_eye_tensor is not None and right_eye_tensor is not None:
                    try:
                        with torch.no_grad():
                            yawn_output = yawn_model(mouth_tensor)
                            left_eye_output = eye_model(left_eye_tensor)
                            right_eye_output = eye_model(right_eye_tensor)

                            yawn_probs = torch.softmax(yawn_output, dim=1)
                            left_eye_probs = torch.softmax(left_eye_output, dim=1)
                            right_eye_probs = torch.softmax(right_eye_output, dim=1)

                            yawn_confidence, yawn_class = torch.max(yawn_probs, dim=1)
                            left_eye_confidence, left_eye_class = torch.max(left_eye_probs, dim=1)
                            right_eye_confidence, right_eye_class = torch.max(right_eye_probs, dim=1)

                            yawn_confidence, yawn_class = yawn_confidence.item(), yawn_class.item()
                            left_eye_confidence, left_eye_class = left_eye_confidence.item(), left_eye_class.item()
                            right_eye_confidence, right_eye_class = right_eye_confidence.item(), right_eye_class.item()

                            eye_confidence = (left_eye_confidence + right_eye_confidence) / 2.0
                            eye_class = 1 if (left_eye_class == 1 and right_eye_class == 1) else 0

                            # YAWN DETECTION
                            if yawn_class and yawn_confidence >= k.YAWN_CONFIDENCE_THRESHOLD :
                                yawn_counter += 1
                            else:
                                yawn_counter = 0
                                is_yawning = False
                            if yawn_counter >= k.YAWN_FRAME_THRESHOLD and not is_yawning:
                                is_yawning = True
                                chime_manager.record_yawn()
                                yawn_counter = 0

                            # EYE-CLOSURE DETECTION
                            if eye_class == 1 and eye_confidence >= k.EYE_CONFIDENCE_THRESHOLD:
                                closed_eyes_counter += 1
                            else:
                                closed_eyes_counter = 0
                                is_eyes_closed = False
                            if closed_eyes_counter >= k.EYE_FRAME_THRESHOLD and not is_eyes_closed:
                                is_eyes_closed = True
                                chime_manager.record_eye_close()
                                closed_eyes_counter = 0

                            # Yawn status
                            if is_yawning:
                                # Confirmed yawn - show class name
                                message = f"{k.YAWN_CLASS_NAMES[yawn_class]}" # Assumes index 1 is 'Yawning'
                                color = (0, 0, 255)  # Red for confirmed yawn
                            elif yawn_counter > 0:
                                # Potential yawn
                                message = f"Possible yawn ({yawn_counter}/{k.YAWN_FRAME_THRESHOLD})"
                                color = (0, 165, 255)  # Orange for potential yawn
                            else:
                                # Default - show class name and confidence
                                message = f"{k.YAWN_CLASS_NAMES[yawn_class]} ({yawn_confidence:.2f})"
                                color = (0, 255, 0) if yawn_class == 0 else (0, 165, 255) # Green if not yawning, orange otherwise (edge case)

                            cv2.putText(display_frame, message,
                                        (mouth_roi_x_start, mouth_roi_y_start - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            # Eye status
                            if is_eyes_closed:
                                # Confirmed closed - show class name
                                eye_message = f"{k.EYE_CLASS_NAMES[eye_class]}" # Assumes index 1 is 'Closed'
                                eye_color = (0, 0, 255)  # Red for confirmed closed eyes
                            elif closed_eyes_counter > 0:
                                # Potential closure
                                eye_message = f"Closing eyes ({closed_eyes_counter}/{k.EYE_FRAME_THRESHOLD})"
                                eye_color = (0, 165, 255)  # Orange for potentially closing eyes
                            else:
                                # Default - show class name and confidence
                                eye_message = f"{k.EYE_CLASS_NAMES[eye_class]} ({eye_confidence:.2f})"
                                eye_color = (255, 0, 0) 

                            # Draw eye status text
                            cv2.putText(display_frame, eye_message,
                                        (eye_roi_x_start, eye_roi_y_start - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
                    except Exception as e:
                        print(f"[WARNING] Error during model inference or drawing: {e}")
        # overlay live counts
        cv2.putText(display_frame,
                    f"Yawns: {chime_manager.get_yawn_count()}  Eyes Closed: {chime_manager.get_eye_close_count()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        display_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + display_frame + b'\r\n')
        frame_counter += 1