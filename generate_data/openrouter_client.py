import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_model(prompt, model, temperature, max_tokens):
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost", 
        "X-Title": "grayswan_assessment"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    r = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)

    if r.status_code != 200:
        print("STATUS:", r.status_code)
        print("BODY:", r.text) 
        r.raise_for_status()

    data = r.json()
    return data["choices"][0]["message"]["content"]

if __name__ == "__main__":
    print(call_model("Write a short sentence about machine learning."))