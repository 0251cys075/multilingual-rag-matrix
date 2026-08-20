import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()

def transcribe_audio(audio_path: str, lang_code: str = "hi-IN") -> str:
    """Transcribes audio using Sarvam Saaras STT."""
    if not SARVAM_API_KEY:
        return "कॉरपोरेशन क्या है?"
        
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": SARVAM_API_KEY}
    
    try:
        with open(audio_path, "rb") as f:
            files = {"file": f}
            data = {"language_code": lang_code, "model": "saaras:v3"}
            res = requests.post(url, headers=headers, files=files, data=data, timeout=10)
            
        if res.status_code == 200:
            return res.json().get("transcript", "")
        else:
            print(f"[STT Error {res.status_code}] {res.text}")
    except Exception as e:
        print(f"[STT Exception] {e}")
        
    return ""

def synthesize_speech(text: str, target_lang: str = "hi-IN") -> str:
    """Synthesizes text into base64 audio via Sarvam Bulbul TTS."""
    if not SARVAM_API_KEY or not text.strip():
        return ""
        
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Clean text to prevent TTS rejection
    clean_text = text.replace('"', '').replace("'", "").strip()[:500]
    
    payload = {
        "inputs": [clean_text],
        "target_language_code": target_lang,
        "speaker": "meera",
        "pace": 1.1,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            audios = data.get("audios", [])
            if audios and len(audios) > 0:
                return audios[0]
        else:
            print(f"[TTS Error {res.status_code}] {res.text}")
    except Exception as e:
        print(f"[TTS Exception] {e}")
        
    return ""