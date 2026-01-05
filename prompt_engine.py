from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import os

from openai import OpenAI


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


ROLE_SYSTEM: Dict[str, str] = {
    "Marketer": "You are a senior performance marketer and conversion copywriter.",
    "Designer": "You are a brand designer and creative director who writes clear design briefs.",
    "ChatGPT User": "You are an expert prompt engineer who writes precise, testable prompts.",
    "Student": "You are an academic writing coach who explains step by step.",
    "Business": "You are a business strategist who turns ideas into execution plans.",
}

TASK_HINTS: Dict[str, str] = {
    "Write Caption": "Write 3 strong captions with hooks, value, and CTA.",
    "Generate Content Ideas": "Generate 15 content ideas with angles and formats.",
    "Landing Page Copy": "Write landing page copy (headline, benefits, CTA).",
    "Design Brief": "Create a complete design brief.",
    "Email": "Write a structured email with CTA.",
    "Prompt Improvement": "Rewrite the prompt to be clearer and higher quality.",
}


def build_base_prompt(p: PromptInputs) -> str:
    system_role = ROLE_SYSTEM.get(p.role, ROLE_SYSTEM["ChatGPT User"])
    task_hint = TASK_HINTS.get(p.task, p.task)

    return f"""
{system_role}

TASK:
{task_hint}

GOAL:
{p.goal}

AUDIENCE:
{p.audience}

TONE:
{p.tone}

CONTEXT:
{p.context}

CONSTRAINTS:
{p.constraints}

OUTPUT FORMAT:
{p.output_format}

LANGUAGE:
{p.language}

RULES:
- Be specific
- Avoid generic advice
- Ask clarifying questions if needed
""".strip()


def refine_prompt_with_openai(
    base_prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4.1-mini",
) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY belum diisi")

    client = OpenAI(api_key=key)

    response = client.responses.create(
        model=model,
        input=f"Improve this prompt:\n\n{base_prompt}",
    )

    return response.output_text
