import re

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from src.modules.image.img_to_text import ImageToText
from src.modules.image.text_to_img import TextToImage
from src.modules.speech.tts import TextToSpeech
from src.modules.speech.stt import SpeechToText
from src.settings import settings
from dotenv import load_dotenv 
load_dotenv()

def get_chat_model(temperature: float = 0.7):
    return ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name=settings.TEXT_MODEL_NAME,
        temperature=temperature,
    )

def get_text_to_speech_module():
    return TextToSpeech()

def get_text_to_image_module():
    return TextToImage()

def get_image_to_text_module():
    return ImageToText()

def get_speech_to_text_module():
    return SpeechToText()

def remove_asterisk_content(text: str) -> str:
    """Remove content between asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text):
        return remove_asterisk_content(super().parse(text))