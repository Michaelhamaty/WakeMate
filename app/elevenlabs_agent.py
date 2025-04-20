import os
from elevenlabs.client import ElevenLabs
from elevenlabs import ElevenLabs, play
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize the ElevenLabs client
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def elevenlabs_tts(text_to_speak):
    """Converts the given text to speech using ElevenLabs."""
    if not text_to_speak:
        print("No text provided for TTS.")
        return

    try:
        # Generate audio from text
        audio = client.generate(
            text=text_to_speak,
            voice="21m00Tcm4TlvDq8ikWAM", # Example Voice ID, replace if needed
            model="eleven_turbo_v2" # Or another suitable model
        )
        # Play the audio (requires the 'elevenlabs' library's play function or similar)
        # from elevenlabs import play
        play(audio)
        print(f"ElevenLabs generated audio for: '{text_to_speak}' (Playback/Save needs implementation)")

    except Exception as e:
        print(f"Error during ElevenLabs TTS generation: {e}")
