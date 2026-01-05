from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

from google import genai


# =========================
# Data Model
# =========================
@dataclass
class PromptInputs:
    role: str
    task: str
    goal: str
    audience: str
    tone: str
    language: str = "Indonesian"
    context: str = ""
    constraints: str = ""
    output_format: str = "Prompt siap pakai (jelas, terstruktur, bisa langsung ditempel ke ChatGPT)."


# =========================
# Base Prompt Builder
# =========================
def build_base_prompt(p: PromptInputs) -> str:
    """
    Membuat base prompt template yang bagus tanpa API.
    """
    # amanin None / kosong
    role = (p.role or "").strip()
    task = (p.task or "").strip()
    goal = (p.goal or "").strip()
    audience = (p.audience or "").strip()
    tone = (p.tone or "").strip()
    language = (p.language or "Indonesian").strip()
    context = (p.context or "").strip()
    constraints = (p.constraints or "").strip()
    output_format = (p.output_format or "").strip()

    parts = []
    parts.append("You are an expert prompt engineer. You write precise, testable prompts.")
    parts.append("")
    parts.append("ROLE:")
    parts.append(role if role else "(not specified)")
    parts.append("")
    parts.append("TASK:")
    parts.append(task if task else "(not specified)")
    parts.append("")
    parts.append("GOAL:")
    parts.append(goal if goal else "(not specified)")
    parts.append("")
    parts.append("AUDIENCE:")
    parts.append(audience if audience else "(not specified)")
    parts.append("")
    parts.append("TONE:")
    parts.append(tone if tone else "(not specified)")
    parts.append("")
    parts.append("LANGUAGE:")
    parts.append(language)

    if context:
        parts.append("")
        parts.append("CONTEXT (optional):")
        parts.append(context)

    if constraints:
        parts.append("")
        parts.append("CONSTRAINTS (optional):")
        parts.append(constraints)

    if output_format:
        parts.append("")
        parts.append("OUTPUT FORMAT:")
        parts.append(output_format)

    parts.append("")
    parts.append("Now produce the BEST possible final prompt based on the details above.")
    parts.append("Make it clear, structured, and ready to paste into an AI chat.")
    return "\n".join(parts)


# =========================
# Gemini Refiner
# =========================
def refine_with_ai(
    base_prompt: str,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.7,
    max_output_tokens: int = 800,
) -> str:
    """
    Refinement menggunakan Google Gemini API via SDK 'google-genai'.
    - Ambil API key dari env var: GOOGLE_API_KEY
    - Fallback model jika model pilihan error
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY belum diisi. Isi di Streamlit Secrets: GOOGLE_API_KEY=\"...\"")

    client = genai.Client(api_key=api_key)

    # Jangan pakai gemini-1.0-pro (sering bikin 404 di beberapa endpoint/versi)
    candidates = [model_name, "gemini-1.5-pro", "gemini-1.5-flash"]

    last_err: Optional[Exception] = None

    # Prompt pengarah biar hasil lebih “prompt final”, bukan jawaban random
    system_hint = (
        "You are a senior prompt engineer. Your job is to rewrite and improve prompts.\n"
        "Return ONLY the final prompt (no explanations, no markdown fences)."
    )

    user_content = (
        "Improve this prompt into a final, high-quality prompt.\n"
        "Keep it structured, clear, and actionable.\n\n"
        "PROMPT TO IMPROVE:\n"
        f"{base_prompt}"
    )

    for m in candidates:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=[
                    {"role": "user", "parts": [{"text": system_hint + "\n\n" + user_content}]}
                ],
                config={
                    "temperature": float(temperature),
                    "max_output_tokens": int(max_output_tokens),
                },
            )

            # google-genai biasanya mengembalikan .text
            text = (getattr(resp, "text", None) or "").strip()
            if not text:
                raise RuntimeError("Respon kosong dari model. Coba model lain / cek kuota API.")
            return text

        except Exception as e:
            last_err = e
            continue

    # Kalau semua model gagal
    raise RuntimeError(f"Gagal refine di semua model. Error terakhir: {last_err}")
