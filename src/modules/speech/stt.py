import os
import tempfile
from typing import Optional
from groq import Groq
from src.settings import Settings
from src.core.exceptions import SpeechToTextError