from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

# Pakai SDK resmi baru: google-genai
# Install: google-genai
from google import genai


# =========================
# Data Model (Inputs)
# =========================
@dataclass
class PromptInputs:
    role: str
    task: str
    goal: str
    audience: str
    tone: str
    context: str = ""
    constraints: str = ""
    output_format: str = ""
    language: str = "Indonesian"


# =========================
# Base Prompt Builder
# =========================
def build_base_prompt(p: PromptInputs) -> str:
    """Template prompt dasar (tanpa AI)."""
    parts = []
    parts.append("You are an expert prompt engineer. You write precise, testable prompts.")
    parts.append("")
    parts.append(f"LANGUAGE: {p.language}")
    parts.append("")
    parts.append("ROLE:")
    parts.append(p.role.strip())
    parts.append("")
    parts.append("TASK:")
    parts.append(p.task.strip())
    parts.append("")
    parts.append("GOAL:")
    parts.append(p.goal.strip())
    parts.append("")
    parts.append("AUDIENCE:")
    parts.append(p.audience.strip())
    parts.append("")
    parts.append("TONE:")
    parts.append(p.tone.strip())

    if p.context and p.context.strip():
        parts.append("")
        parts.append("CONTEXT (optional):")
        parts.append(p.context.strip())

    if p.constraints and p.constraints.strip():
        parts.append("")
        parts.append("CONSTRAINTS (optional):")
        parts.append(p.constraints.strip())

    if p.output_format and p.output_format.strip():
        parts.append("")
        parts.append("OUTPUT FORMAT (optional):")
        parts.append(p.output_format.strip())

    parts.append("")
    parts.append("Now write the BEST possible prompt for the user based on the above information.")
    parts.append("Make it structured, clear, and ready to copy-paste into an AI chat.")
    return "\n".join(parts)


# =========================
# Gemini Model Normalizer
# =========================
def _normalize_model(model: str) -> str:
    """
    Banyak error kamu muncul karena model name tidak cocok.
    Jadi kita normalisasi + fallback otomatis.
    """
    if not model:
        return "gemini-2.0-flash"

    m = model.strip()

    # Kalau user memilih model 1.5 (sering error 404 di beberapa setup),
    # kita fallback ke model yang umumnya tersedia:
    if m.startswith("gemini-1.5-"):
        return "gemini-2.0-flash"

    # Fallback umum kalau ada typo / kosong
    return m


# =========================
# AI Refine (Gemini)
# =========================
def refine_with_ai(
    base_prompt: str,
    model: str,
    temperature: float = 0.5,
    max_output_tokens: int = 700,
) -> str:
    """
    Refine prompt pakai Gemini API (Google AI Studio key).
    Ambil key dari environment: GOOGLE_API_KEY
    (Streamlit Secrets otomatis jadi environment variable)
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY belum diisi di Streamlit Secrets.")

    safe_model = _normalize_model(model)

    # Client Gemini (Developer API via API key)
    client = genai.Client(api_key=api_key)

    system_instruction = (
        "You are a senior prompt engineer. Improve the given prompt. "
        "Keep it aligned to the user's goal, add structure, constraints, and clarity. "
        "Output ONLY the final improved prompt."
    )

    try:
        # SDK google-genai: models.generate_content(...)
        resp = client.models.generate_content(
            model=safe_model,
            contents=[
                {"role": "user", "parts": [{"text": base_prompt}]}
            ],
            config={
                "system_instruction": system_instruction,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )

        # resp.text biasanya tersedia
        text = getattr(resp, "text", None)
        if text and text.strip():
            return text.strip()

        # fallback kalau struktur berbeda
        if hasattr(resp, "candidates") and resp.candidates:
            cand0 = resp.candidates[0]
            if hasattr(cand0, "content") and cand0.content and hasattr(cand0.content, "parts"):
                parts = cand0.content.parts or []
                joined = "\n".join([getattr(pt, "text", "") for pt in parts if getattr(pt, "text", "")])
                if joined.strip():
                    return joined.strip()

        raise RuntimeError("Respon Gemini kosong. Coba ulangi lagi.")
    except Exception as e:
        # kasih error yang enak dibaca
        raise RuntimeError(f"Gagal refine dengan Gemini. Detail: {e}")
