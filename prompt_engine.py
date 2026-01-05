from __future__ import annotations

from dataclasses import dataclass
import os
import google.generativeai as genai


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
        "Write ONE final prompt that the user can copy-paste into an AI tool.\n"
        "Make it specific, structured, and actionable.\n"
        "Include placeholders like {PRODUCT}, {OFFER}, {DATE} if helpful.\n"
    )

    return "\n\n".join(parts)


def refine_with_ai(base_prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY belum diisi (set di Streamlit Secrets).")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        [
            "You are a prompt engineering assistant. Improve the prompt below.",
            "Rewrite it to be clearer, more specific, and more effective.",
            base_prompt,
        ]
    )

    text = getattr(response, "text", None)
    if not text:
        text = str(response)

    return text.strip()
