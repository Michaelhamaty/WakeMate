import os

class model_config:
    """
    Configuration class for model paths and parameters.
    """
    MODEL_DIR = "models"
    PREDICTOR_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
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