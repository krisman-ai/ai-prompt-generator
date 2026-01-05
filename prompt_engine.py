from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

from google import genai


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


def build_base_prompt(p: PromptInputs) -> str:
    parts = []
    parts.append("You are an expert prompt engineer. You write precise, testable prompts.\n")

    parts.append(f"LANGUAGE: {p.language}\n")
    parts.append(f"ROLE:\n{p.role}\n")
    parts.append(f"TASK:\n{p.task}\n")
    parts.append(f"GOAL:\n{p.goal}\n")
    parts.append(f"AUDIENCE:\n{p.audience}\n")
    parts.append(f"TONE:\n{p.tone}\n")

    if p.context.strip():
        parts.append(f"CONTEXT:\n{p.context.strip()}\n")

    if p.constraints.strip():
        parts.append(f"CONSTRAINTS:\n{p.constraints.strip()}\n")

    if p.output_format.strip():
        parts.append(f"OUTPUT FORMAT:\n{p.output_format.strip()}\n")

    parts.append(
        "\nINSTRUCTIONS:\n"
        "- Create a single final prompt that the user can paste into ChatGPT.\n"
        "- Make it structured (sections + bullet points).\n"
        "- Include any necessary assumptions.\n"
        "- Add a short checklist for quality.\n"
        "- Output ONLY the final prompt text.\n"
    )

    return "\n".join(parts).strip()


def _get_google_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY belum diisi di Streamlit Secrets.")
    return key


def refine_with_ai(
    base_prompt: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.5,
    max_output_tokens: int = 900,
) -> str:
    api_key = _get_google_api_key()
    client = genai.Client(api_key=api_key)

    instruction = (
        "Refine the following prompt to be clearer, more actionable, and higher quality.\n"
        "Return ONLY the improved final prompt, no explanations.\n\n"
        f"PROMPT TO REFINE:\n{base_prompt}"
    )

    resp = client.models.generate_content(
        model=model,
        contents=instruction,
        config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
    )

    text = getattr(resp, "text", None)
    if not text or not text.strip():
        raise RuntimeError("Gemini tidak mengembalikan output. Coba lagi.")
    return text.strip()
