from __future__ import annotations

from dataclasses import dataclass
import os
import google.generativeai as genai


# 1) Ini yang diminta app.py
@dataclass
class PromptInputs:
    role: str
    task: str
    goal: str
    audience: str
    tone: str
    language: str
    context: str = ""
    constraints: str = ""
    output_format: str = ""


# 2) Ini yang diminta app.py
def build_base_prompt(p: PromptInputs) -> str:
    parts = [
        f"ROLE:\n{p.role}",
        f"TASK:\n{p.task}",
        f"GOAL:\n{p.goal}",
        f"AUDIENCE:\n{p.audience}",
        f"TONE:\n{p.tone}",
        f"LANGUAGE:\n{p.language}",
    ]
    if p.context.strip():
        parts.append(f"CONTEXT:\n{p.context}")
    if p.constraints.strip():
        parts.append(f"CONSTRAINTS:\n{p.constraints}")
    if p.output_format.strip():
        parts.append(f"OUTPUT FORMAT:\n{p.output_format}")

    parts.append(
        "\nINSTRUCTIONS:\n"
        "Write a high-quality prompt that is clear, specific, and ready to paste into ChatGPT.\n"
        "Include variables/placeholders if needed.\n"
    )

    return "\n\n".join(parts)


# 3) Ini yang diminta app.py
def refine_with_ai(base_prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY belum diisi di Streamlit Secrets")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(base_prompt)

    # Aman kalau kosong
    return (getattr(resp, "text", None) or "").strip() or "Refine berhasil tapi output kosong."
