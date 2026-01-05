import os
import google.generativeai as genai

def refine_prompt_with_ai(prompt: str, model_name: str = "gemini-1.5-pro"):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY belum diisi di Streamlit Secrets")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)

    return response.text
