from __future__ import annotations

from dataclasses import dataclass
import os

from google import genai


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
        "Rewrite this into a single high-quality prompt that is clear, specific, and ready to paste into ChatGPT.\n"
        "Keep it concise but complete. Use bullet points if helpful.\n"
    )
    return "\n\n".join(parts)


def refine_with_ai(base_prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY belum diisi di Streamlit Secrets")

    client = genai.Client(api_key=api_key)

    # fallback biar jarang error kalau satu model lagi “ngambek”
    candidates = [model_name, "gemini-1.5-pro", "gemini-1.0-pro"]

    last_err = None
    for m in candidates:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=base_prompt,
            )
            text = (getattr(resp, "text", None) or "").strip()
            if text:
                return text
            return "Refine berhasil tapi output kosong."
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Gagal refine di semua model. Error terakhir: {last_err}")
