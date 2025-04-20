import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_wake_up_message(chime_count):
    """
    Generates a prompt based on the chime count and calls the Gemini API.

    Args:
        chime_count: The number of chimes detected.

    Returns:
        The response object from the Gemini API call, or None if no prompt
        is generated for the given chime_count or if an error occurs.
    """
    prompt = None
    if chime_count == 6:
        prompt = "Generate a short, friendly, and engaging statement (1-2 sentences) to gently alert a driver who seems a bit drowsy."
    elif chime_count == 12:
        prompt = "Generate a slightly more serious statement (2-3 sentences) cautioning a drowsy driver about the importance of staying alert. Offer to play some upbeat music."
    elif chime_count == 18:
        prompt = "Generate a firm and direct statement (2-3 sentences) strongly advising a very drowsy driver to pull over at the nearest rest stop or gas station immediately for safety."

    if prompt:
        prompt += " Keep it 50 words or less."
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating content: {e}")
            # Return None or handle the error appropriately
            return None
    else:
        # Return None if the chime_count doesn't trigger a prompt
        return None