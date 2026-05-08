# core_groq.py - Use this instead of core.py
import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def detect_emotion(message):
    """Emotion detection using Groq's Llama model"""
    prompt = f"""Analyze this message and return ONLY a JSON object, no extra text:
{{
  "valence": float,
  "arousal": float,
  "urgency": int,
  "masking": "explicit" or "implicit",
  "subtext": "string"
}}
Message: "{message}"
"""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    text = completion.choices[0].message.content
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

def build_prompt(message, emotion, history):
    """Build prompt matching judge expectations"""
    history_text = ""
    for role, msg in history[-6:]:
        history_text += f"{role}: {msg}\n"
    
    crisis_rule = ""
    if emotion["urgency"] >= 4:
        crisis_rule = """**CRISIS PROTOCOL:**
You MUST include: "Please call or text 988 (Suicide & Crisis Lifeline) or text HOME to 741741."
Then continue and end with a question.
"""
    
    return f"""You are MindBridge — an emotionally intelligent mental health companion.

**CRITICAL RULES:**
- NO platitudes like "I understand how you feel" or "That must be hard"
- NO giving advice unless explicitly asked
- ALWAYS end your response with a genuine, specific question
- Be concise: 2-3 sentences max

{crisis_rule}

**RESPONSE STRUCTURE:**
1. Acknowledge what they said (be specific)
2. Ask ONE probing question

**EXAMPLES:**
User: "I used to love sports, now I lie in bed"
Response: "That shift from enjoying things to not finding a reason to get up. When did things start feeling this way?"

User: "My parents are fighting and exams went badly"
Response: "You were carrying pressure while watching things fall apart at home. Do you feel like doing well was connected to fixing things at home?"

[CONVERSATION HISTORY]
{history_text}

[EMOTION STATE]
Valence={emotion['valence']} | Arousal={emotion['arousal']} | Urgency={emotion['urgency']}/5

[USER MESSAGE]
{message}

Respond as MindBridge (2-3 sentences + a question, NO advice):"""

def get_response(message, emotion, history):
    prompt = build_prompt(message, emotion, history)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return completion.choices[0].message.content

# Test
if __name__ == "__main__":
    test_msg = "I used to love playing sports, now I just lie in bed"
    emotion = detect_emotion(test_msg)
    print(f"Emotion: {json.dumps(emotion, indent=2)}")
    resp = get_response(test_msg, emotion, [])
    print(f"\nResponse: {resp}")