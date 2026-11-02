#!/usr/bin/env python3
"""Real Qwen3 GENERATIVE project-requirement decomposition for Sherlocking.

Only the query's `abstract` is sent to Qwen. No faculty data, papers, or graph
are loaded here. Invalid model JSON fails closed; never falls back to regex.
"""
from __future__ import annotations

import json
import math
from typing import Any

SYSTEM_PROMPT = """You are a scientific research-project EXPERTISE REQUIREMENTS DECOMPOSER, not a recommender.

You receive ONLY a project abstract. Extract independently assessable research
expertise that could be supported by researchers' publication abstracts. Do not
merely summarize sentences or turn every project action into a requirement.

Analyze the abstract through TWO MAIN DIMENSIONS:

1. DOMAIN KNOWLEDGE / EXPERTISE (what scientific or application area must be understood)
   Identify the explicitly stated scientific disciplines, phenomena, subject
   matter, materials, data types, specialized procedures, and application or
   benchmark workflows that require substantive domain knowledge. A domain
   matters even if it appears only in evaluation, a case study, or a dataset.
   When multiple named domains/workflows call for different expertise, preserve
   them as distinct requirements, even when listed together in one sentence.
   Retain their technical names. Do NOT claim the project develops new techniques
   in a domain if it only uses that domain for evaluation.

2. METHODS / SYSTEMS / EVALUATION (how the research is carried out and assessed)
   Identify the project's explicitly proposed research approach, distinguishing:
   - METHODS: algorithms, models, inference, optimization, analysis, or other
     technical techniques that the research uses or develops.
   - SYSTEMS: architectures, components, integrations, implementations,
     operating procedures, and meaningful technical constraints.
   - EVALUATION: experimental designs, benchmarks, measured outcomes,
     comparisons, validation, robustness, and generalization analyses.
   These are sub-aspects of ONE dimension, not mandatory output categories.
   Keep a method, system, or evaluation need separate only when it demands
   distinct expertise or different publication evidence. Avoid duplicating a
   single contribution as an overall objective plus several near-paraphrases.

EXTRACTION RULES:
1. Use ONLY information stated in the abstract. Do not invent technical details,
   scientific capabilities, datasets, tools, equipment, metrics, or priorities.
2. Consider BOTH dimensions before writing the final output. Neither is
   inherently more important. There is NO fixed requirement count or quota:
   the actual abstract decides what appears and how many items each needs.
   If no substantive domain is stated, do not manufacture one.
3. Decompose by INDEPENDENTLY EVIDENCEABLE EXPERTISE, not sentence boundaries.
   Separate distinct, technically specialized domains where combining them
   would make publication retrieval less targeted. Combine synonymous or tightly
   coupled items when one body of research evidence would support both.
4. Make each requirement concise, self-contained, technically specific, and
   suitable as a semantic-search query against publication abstracts. Preserve
   specific scientific terminology; avoid vague umbrella descriptions.
5. Preserve the distinction between WHAT the project does and WHERE it is
   applied or evaluated. Domain-evaluation requirements may mention evaluation
   in the domain, without pretending to develop domain-specific methods.
6. Do not create requirements solely from background motivation, generic claims
   of improved performance, or repeated formulations of the same objective
   unless they add independently assessable expertise or evaluation needs.
7. Do not recommend scientists or discuss faculty, papers, coauthors, or
   collaboration graphs. Only decompose the supplied abstract.
8. Every requirement must have weight 1.0. Return exactly ONE valid JSON object
   with ONLY the top-level key "requirements", containing a nonempty array.
   Each entry has EXACTLY "description" (nonempty string) and "weight" (1.0).
   No labels, category fields, reasoning, markdown, or additional text.

BEFORE RETURNING, silently verify: distinct domain expertise is represented
(including evaluation settings); methods/systems/evaluations are not overlooked;
no near-duplicates, unsupported claims, or artificial balancing have appeared.

Required JSON shape:
{"requirements":[{"description":"Distinct research expertise grounded in abstract","weight":1.0}]}
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


def validate_with_narrow_repair(raw: str) -> tuple[list[dict[str, Any]], str]:
    """Repair only the two observed closing-delimiter mistakes, then validate.

    Requirement content and schema are never repaired. Every candidate still
    passes through validate_model_output(), so invalid fields, values, prose,
    duplicates, and other malformed output continue to fail closed.
    """
    candidates = [("original", raw)]

    # Qwen produced ...}]}} instead of ...}]} (one extra closing brace).
    if raw.endswith("}]}}"):
        candidates.append(("removed one extra closing brace", raw[:-1]))
    # Qwen produced ...}} instead of ...}]} (missing the array bracket).
    elif raw.startswith('{"requirements":[') and raw.endswith("}}"):
        candidates.append(("inserted missing requirements-array bracket",
                           raw[:-1] + "]}"))

    errors = []
    for repair_note, candidate in candidates:
        try:
            return validate_model_output(candidate), repair_note
        except ValueError as exc:
            errors.append(f"{repair_note}: {exc}")
    raise ValueError("; ".join(errors))


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
                max_new_tokens=1500,
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
            requirements, repair_note = validate_with_narrow_repair(raw)
            if repair_note != "original":
                print(f"Warning: repaired Qwen JSON: {repair_note}")
            return requirements
        except ValueError as exc:
            failures.append(f"attempt {attempt + 1}: {exc}; output: {raw[:]!r}")

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
