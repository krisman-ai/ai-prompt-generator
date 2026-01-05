import os
import google.generativeai as genai

def refine_prompt_with_ai(base_prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY belum diisi")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(base_prompt)

    return response.text
