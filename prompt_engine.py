from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

import google.generativeai as genai


# ===== Data input (tetap sama) =====
@dataclass
class PromptInputs:
    role: str
    task: str
    goal: str
    audience: str
    tone: str
    context: str
    constraints: str
    output_format: str
    language: str


# ===== Base prompt template =====
def build_base_prompt(inputs: PromptInputs) -> str:
    parts = [
        f"ROLE:\n{inputs.role}",
        f"TASK:\n{inputs.task}",
        f"GOAL:\n{inputs.goal}",
        f"AUDIENCE:\n{inputs.audience}",
        f"TONE:\n{inputs.tone}",
    ]

    if inputs.context.strip():
        parts.append(f"CONTEXT:\n{inputs.context}")

    if inputs.constraints.strip():
        parts.append(f"CONSTRAINTS:\n{inputs.constraints}")

    if inputs.output_format.strip():
        parts.append(f"OUTPUT FORMAT:\n{inputs.output_format}")

    if inputs.language.strip():
        parts.append(f"LANGUAGE:\n{inputs.language}")

    parts.append(
        "\nINSTRUCTIONS:\n"
        "Write a single high-quality prompt that the user can paste into an AI tool.\n"
        "Make it clear, structured, and actionable.\n"
        "Add placeholders like {PRODUCT}, {OFFER}, {DATE} if helpful.\n"
    )

    return "\n\n".join(parts)


# ===== Gemini refine =====
def refine_with_ai(base_prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it in Streamlit Secrets.")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        [
            "You are a helpful assistant that improves prompts. "
            "Rewrite the prompt to be clearer, more specific, and more effective.",
            base_prompt,
        ]
    )

    # response.text biasanya sudah ada
    text = getattr(response, "text", None)
    if not text:
        # fallback kalau format response berubah
        text = str(response)

    return text.strip()
