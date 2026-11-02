#!/usr/bin/env python3
"""Real Qwen3 GENERATIVE project-requirement decomposition for Sherlocking.

Only the query's `abstract` is sent to Qwen. No faculty data, papers, or graph
are loaded here. Invalid model JSON fails closed; never falls back to regex.
"""
from __future__ import annotations

import json
import math
from typing import Any

SYSTEM_PROMPT = """You are a scientific research-project REQUIREMENTS DECOMPOSER, not a recommender.

The user message consists ONLY of a research project abstract. Extract the
independently assessable capabilities, technical tasks, or evaluation needs
EXPLICITLY stated in that abstract. The output will later be matched against
publication abstracts as evidence of research expertise.

Rules:
1. Use ONLY facts explicitly stated in the supplied abstract. Do not invent
   methods, datasets, models, tools, equipment, application areas, metrics,
   collaborators, or experimental designs.
2. A single sentence may express SEVERAL distinct requirements; several
   sentences may describe the SAME requirement. Decompose by independently
   assessable scientific need, NOT by sentence boundaries.
3. Separate explicitly mentioned design/implementation, scientific methods,
   domain knowledge, data/equipment, privacy/security, and validation or
   reproducibility needs when they require different evidence of expertise.
4. Use clear self-contained task/capability descriptions grounded in the
   abstract. Preserve technical terminology. Avoid generic labels like
   'machine learning' if the abstract explicitly calls for a narrower task.
5. Do not turn general motivation or background into a requirement unless it
   explicitly specifies something the project needs to accomplish.
6. Do not split trivial synonyms or invent extra specificity. If a need is
   genuinely too vague to decompose, keep it as one general requirement.
7. Do NOT rank, recommend, or discuss researchers, papers, coauthors, networks,
   or graphs. You are not given these materials.
8. Every requirement has weight 1.0. Do not infer project priorities.
9. Return EXACTLY one valid JSON object, with exactly one top-level key:
   "requirements". Its value is a nonempty array. Each array entry has
   EXACTLY two keys: "description" (nonempty string) and "weight" (number 1.0).
   No markdown, code fences, comments, explanation, or extra fields.

Required JSON shape:
{"requirements":[{"description":"Specific task from abstract","weight":1.0}]}
"""


def validate_model_output(raw: str) -> list[dict[str, Any]]:
    """Reject prose, fences, extra keys, invalid numbers, duplicates and empties."""
    def reject_constant(value: str):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    try:
        result = json.loads(raw, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Qwen returned invalid JSON: {exc}") from exc
    if not isinstance(result, dict) or set(result) != {"requirements"}:
        raise ValueError("Qwen output must have exactly the top-level key 'requirements'")
    items = result["requirements"]
    if not isinstance(items, list) or not items:
        raise ValueError("Qwen requirements must be a nonempty array")

    checked = []
    seen = set()
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict) or set(item) != {"description", "weight"}:
            raise ValueError(f"Requirement #{i} must contain exactly 'description' and 'weight'")
        description = item["description"]
        weight = item["weight"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Requirement #{i} has an empty or nonstring description")
        if (isinstance(weight, bool) or not isinstance(weight, (float, int))
                or not math.isfinite(weight) or weight != 1.0):
            raise ValueError(f"Requirement #{i} must have numeric weight 1.0")
        description = description.strip()
        if description.casefold() in seen:
            raise ValueError(f"Duplicate requirement: {description}")
        seen.add(description.casefold())
        checked.append({"description": description, "weight": 1.0})
    return checked


def generate_requirements(abstract: str, model_name: str = "Qwen/Qwen3-4B") -> list[dict[str, Any]]:
    """Actually invoke a generative Qwen3 model, then validate its raw JSON."""
    if not isinstance(abstract, str) or not abstract.strip():
        raise ValueError("A nonempty project abstract is required")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": abstract.strip()},
    ]
    # Qwen3 (original/hybrid) models support the non-thinking switch.
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([formatted], return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[-1]

    failures = []
    for attempt in range(3):
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=True,
                temperature=(0.4, 0.55, 0.7)[attempt],
                top_p=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode ONLY new tokens, not the system prompt or abstract.
        raw = tokenizer.decode(
            generated[0][prompt_length:], skip_special_tokens=True
        ).strip()
        try:
            return validate_model_output(raw)
        except ValueError as exc:
            failures.append(f"attempt {attempt + 1}: {exc}; output: {raw[:500]!r}")

    raise RuntimeError(
        "Qwen could not produce schema-valid requirements after 3 attempts. "
        "No sentence-splitting fallback was used.\n" + "\n".join(failures)
    )


def decompose_query(query: dict, model_name: str = "Qwen/Qwen3-4B") -> dict:
    """Preserve original query fields, replace requirements with actual Qwen output."""
    if not isinstance(query, dict):
        raise ValueError("Input proposal must be a JSON object")
    abstract = query.get("abstract")
    if not isinstance(abstract, str) or not abstract.strip():
        raise ValueError("Input proposal requires a nonempty 'abstract' string")
    reqs = generate_requirements(abstract, model_name=model_name)
    return {**query, "requirements": reqs}
