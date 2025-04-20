import cv2
from flask import Flask, render_template, Response
import drowsy_detection
from chime_manager import ChimeManager

app = Flask(__name__)

camera = None
chime_manager = ChimeManager()


def gen_frames():
    global camera
    # this returns a generator that yields JPEG multipart frames
    for frame in drowsy_detection.start_tracking(camera, chime_manager):
        yield frame

def initialize_camera():
    """Initializes the global camera object."""
    global camera
    global chime_manager
    chime_manager.reset_counts()
    if camera is None: # Or check if not camera.isOpened() if it might exist but be closed
        camera = cv2.VideoCapture(0) # Use appropriate camera index or source
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
        if not camera.isOpened():
            print("Error: Could not open camera.")
            camera = None # Reset if opening failed
        else:
            print("Camera initialized.")

def release_camera():
    """Releases the global camera object."""
    global camera
    global chime_manager
    chime_manager.reset_counts()  # Reset counts when releasing camera
    if camera is not None:
        camera.release()
        camera = None
        print("Camera released.")

@app.route('/')
def home():
    release_camera()
    return render_template('index.html')

@app.route('/about')
def about():
    release_camera()
    return render_template('about.html')

@app.route('/demo')
def demo():
    initialize_camera()
    return render_template('demo.html')

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)