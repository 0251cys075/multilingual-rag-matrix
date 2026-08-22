import re
import unicodedata

def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = value.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"\s+", " ", value.casefold()).strip()

def evaluate_safety_guardrail(value: str) -> dict:
    q = _normalize(value)
    
    cyber_patterns = (
        r"\bhow\s+to\s+(hack|break\s+into|bypass|steal)\b",
        r"\b(hack|attack)\s+(someone|a\s+person|a\s+server|an?\s+account)\b",
        r"\b(steal|dump|exfiltrate)\s+(password|credentials|data|a\s+database)\b",
        r"\b(bypass|evade|disable)\s+(security|authentication|firewall|detection)\b",
        r"\b(make|create|deploy|write|build)\s+(malware|ransomware|keylogger|trojan|botnet)\b",
        r"\b(sql\s+injection|ddos|denial\s+of\s+service)\s+(attack|someone|a\s+server)\b",
        r"(?:\bhack\b\s*){2,}",
        r"सिस्टम में घुस|पासवर्ड चुर|सुरक्षा को बायपास|मैलवेयर बन",
        r"سسٹم میں گھس|پاس ورڈ چور|سیکیورٹی بائی پاس|مالویئر بن",
    )
    if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in cyber_patterns):
        return {"blocked": True, "label": "CYBER_ATTACK_PREVENTED"}

    harm_patterns = (
        r"\b(how\s+to|ways?\s+to|method(?:s)?\s+to|steps?\s+to|make|build)\b.*\b(kill|murder|bomb|shoot|poison|suicide|self[- ]?harm)\b",
        r"\b(kill|murder|shoot|poison)\s+(someone|a\s+person|myself)\b",
        r"\b(harm|hurt)\s+myself\b",
        r"\b(make|build)\s+(a\s+)?bomb\b",
        r"आत्महत्या\s*(कैसे|का तरीका|के तरीके|करें|करना)",
        r"खुदकुशी\s*(कैसे|का तरीका|करें|करना)",
        r"(किसी को|किसी व्यक्ति को)\s*(कैसे|जा[ऩ] से)?\s*मार",
        r"बम\s*कैसे\s*बन",
        r"خودکشی\s*(کیسے|کا طریقہ|کریں|کرنا)",
        r"خودکشی\s*(کے طریقے|کی ترکیب)",
        r"(کسی کو|کسی شخص کو)\s*(کیسے|جان سے)?\s*مار",
        r"بم\s*کیسے\s*بنا",
    )
    if any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in harm_patterns):
        return {"blocked": True, "label": "VIOLENCE_OR_SELF_HARM"}

    return {"blocked": False, "label": "CLEAR"}
