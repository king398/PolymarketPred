import io
import time
import base64
import os
import requests
import keyboard
import pyautogui
import mss
from PIL import Image

# ---------------- CONFIG ----------------
HOTKEY = "ctrl+shift+z"

MODEL = "anthropic/claude-sonnet-4.5"
# Other good options:
# "openai/gpt-4o"
# "anthropic/claude-3.5-sonnet"
# "google/gemini-2.0-flash-001"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SITE_URL = "http://localhost"      # optional
APP_NAME = "VisionHotkeySolver"    # optional
# --------------------------------------

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")


def take_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ask_llm(image_bytes):
    image_b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/png;base64,{image_b64}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": APP_NAME,
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Solve the question shown in the image.\n"
                            "Return ONLY the final answer wrapped exactly as:\n"
                            "\\boxed{answer}\n"
                            "No explanation. No extra text."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"].strip()


def solve_and_type():
    time.sleep(0.25)  # allow hotkey release
    screenshot = take_screenshot()
    print("🖼️ Screenshot taken. Asking LLM...")
    #answer = ask_llm(screenshot)

    #pyautogui.write(answer, interval=0.01)


print(f"🔥 Ready. Press {HOTKEY} to capture → solve → type.")
keyboard.add_hotkey(HOTKEY, solve_and_type)
keyboard.wait()
