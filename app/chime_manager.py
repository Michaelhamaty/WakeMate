import time
import threading                    # added for async alerts
from gemini_chime import generate_wake_up_message
from elevenlabs_agent import elevenlabs_tts

class ChimeManager:
    """
    Keeps track of yawn and eye closure events and triggers voice messages.
    """
    def __init__(self):
        self.yawn_count = 0
        self.eye_close_count = 0
        self.last_trigger_time = 0.0       # timestamp of last voice alert
        self.cooldown_sec = 20.0            # minimum seconds between alerts

    def _check_and_trigger_message(self):
        """Checks counts and triggers Gemini/ElevenLabs message if threshold met."""
        now = time.time()
        if now - self.last_trigger_time < self.cooldown_sec:
            return                         # still in cooldown, skip

        total_chimes = self.yawn_count + self.eye_close_count
        gemini_message = generate_wake_up_message(total_chimes)
        if gemini_message and total_chimes:
            self.last_trigger_time = now   # ← mark when we spoke
            elevenlabs_tts(gemini_message)

    def record_yawn(self):
        """Increments the yawn count and checks for message trigger."""
        self.yawn_count += 1
        print(f"Yawn recorded. Total chimes: {self.yawn_count + self.eye_close_count}")
        # offload to background thread
        threading.Thread(target=self._check_and_trigger_message, daemon=True).start()

    def record_eye_close(self):
        """Increments the eye close count and checks for message trigger."""
        self.eye_close_count += 1
        print(f"Eye close recorded. Total chimes: {self.yawn_count + self.eye_close_count}")
        # offload to background thread
        threading.Thread(target=self._check_and_trigger_message, daemon=True).start()

    def get_yawn_count(self):
        """Returns the current yawn count."""
        return self.yawn_count

    def get_eye_close_count(self):
        """Returns the current eye close count."""
        return self.eye_close_count

    def reset_counts(self):
        """Resets counts to 0."""
        self.yawn_count = 0
        self.eye_close_count = 0
        self.last_trigger_time = 0.0
