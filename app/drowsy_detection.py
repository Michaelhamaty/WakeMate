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
    eye_model.to(device)        # ← correctly send eye_model to device
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
                (ex_min, ey_min), (ex_max, ey_max) = np.min(eye_coords, axis=0), np.max(eye_coords, axis=0)

                mouth_padding = 5
                eye_padding = 10

                mouth_roi_x_start = max(0, mouth_x_min - mouth_padding)
                mouth_roi_y_start = max(0, mouth_y_min - mouth_padding)
                mouth_roi_x_end = min(frame.shape[1], mouth_x_max + mouth_padding)
                mouth_roi_y_end = min(frame.shape[0], mouth_y_max + mouth_padding)

                eye_roi_x_start = max(0, ex_min - eye_padding)
                eye_roi_y_start = max(0, ey_min - eye_padding)
                eye_roi_x_end = min(frame.shape[1], ex_max + eye_padding)
                eye_roi_y_end = min(frame.shape[0], ey_max + eye_padding)

                cv2.rectangle(display_frame, (mouth_roi_x_start, mouth_roi_y_start), (mouth_roi_x_end, mouth_roi_y_end), (0, 255, 0), 2)
                cv2.rectangle(display_frame, (eye_roi_x_start, eye_roi_y_start), (eye_roi_x_end, eye_roi_y_end), (255, 0, 0), 2)


                mouth_roi = frame[mouth_roi_y_start:mouth_roi_y_end, mouth_roi_x_start:mouth_roi_x_end]
                eye_roi = frame[eye_roi_y_start:eye_roi_y_end, eye_roi_x_start:eye_roi_x_end]

                if mouth_roi.shape[0] > 0 and mouth_roi.shape[1] > 0:
                    mouth_tensor = _preprocess_frame_roi(mouth_roi, device, preprocess_transform)
                
                if eye_roi.shape[0] > 0 and eye_roi.shape[1] > 0:
                    eye_tensor = _preprocess_frame_roi(eye_roi, device, preprocess_transform)
                
                if mouth_tensor is not None and eye_tensor is not None:
                    try:
                        with torch.no_grad():
                            yawn_output = yawn_model(mouth_tensor)
                            eye_output = eye_model(eye_tensor)

                            yawn_probs = torch.softmax(yawn_output, dim=1)
                            eye_probs = torch.softmax(eye_output, dim=1)

                            yawn_confidence, yawn_class = torch.max(yawn_probs, dim=1)
                            eye_confidence, eye_class = torch.max(eye_probs, dim=1)

                            yawn_confidence = yawn_confidence.item()
                            eye_confidence = eye_confidence.item()

                            # YAWN DETECTION
                            if yawn_class.item() == 1 and yawn_confidence >= k.YAWN_CONFIDENCE_THRESHOLD:
                                yawn_counter += 1
                            else:
                                yawn_counter = 0
                                is_yawning = False
                            if yawn_counter >= k.YAWN_FRAME_THRESHOLD and not is_yawning:
                                is_yawning = True
                                chime_manager.record_yawn()
                                yawn_counter = 0

                            # EYE-CLOSURE DETECTION
                            if eye_class.item() == 1 and eye_confidence >= k.EYE_CONFIDENCE_THRESHOLD:
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
                                message = f"{k.YAWN_CLASS_NAMES[1]}" # Assumes index 1 is 'Yawning'
                                color = (0, 0, 255)  # Red for confirmed yawn
                            elif yawn_counter > 0:
                                # Potential yawn
                                message = f"Possible yawn ({yawn_counter}/{k.YAWN_FRAME_THRESHOLD})"
                                color = (0, 165, 255)  # Orange for potential yawn
                            else:
                                # Default - show class name and confidence
                                message = f"{k.YAWN_CLASS_NAMES[yawn_class.item()]} ({yawn_confidence:.2f})"
                                color = (0, 255, 0) if yawn_class.item() == 0 else (0, 165, 255) # Green if not yawning, orange otherwise (edge case)

                            # Draw yawn status text
                            cv2.putText(display_frame, message,
                                        (mouth_roi_x_start, mouth_roi_y_start - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            # Eye status
                            if is_eyes_closed:
                                # Confirmed closed - show class name
                                eye_message = f"{k.EYE_CLASS_NAMES[1]}" # Assumes index 1 is 'Closed'
                                eye_color = (0, 0, 255)  # Red for confirmed closed eyes
                            elif closed_eyes_counter > 0:
                                # Potential closure
                                eye_message = f"Closing eyes ({closed_eyes_counter}/{k.EYE_FRAME_THRESHOLD})"
                                eye_color = (0, 165, 255)  # Orange for potentially closing eyes
                            else:
                                # Default - show class name and confidence
                                eye_message = f"{k.EYE_CLASS_NAMES[eye_class.item()]} ({eye_confidence:.2f})"
                                eye_color = (255, 0, 0) if eye_class.item() == 0 else (0, 165, 255) # Blue if open, orange otherwise (edge case)

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