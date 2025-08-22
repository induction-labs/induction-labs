#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Expected files vLLM may look for (models differ: some won't have merges/vocab; SP models use tokenizer.model or spiece.model)
EXPECTED_NAMES: set[str] = {
    "added_tokens.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "chat_template.json",
    # "tokenizer.model",  # SentencePiece (e.g., Llama/T5 variants often use spm)
    # "spiece.model",
}


def patch_config_architectures(config_path: Path) -> bool:
    """
    Open config.json, and if any entry in `architectures` starts with 'FSDP',
    strip that prefix (e.g., 'FSDPQwen...' -> 'Qwen...'). Save back if changed.

    Returns:
        True iff a modification was made and saved.
    """
    if not config_path.exists():
        return False

    try:
        data: dict[str, Any] = json.loads(config_path.read_text())
    except Exception:
        return False

    arch = data.get("architectures")
    if not isinstance(arch, list):
        return False

    changed: bool = False
    new_arch: list[str] = []
    for a in arch:
        if isinstance(a, str) and a.startswith("FSDP"):
            new_arch.append(a[4:])  # drop the 'FSDP' prefix
            changed = True
        else:
            new_arch.append(a)

    if changed:
        data["architectures"] = new_arch
        config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return changed


def write_chat_template_if_present(tokenizer: Any, out_dir: Path) -> str | None:
    """If tokenizer has a chat template and no chat_template.json exists, write it and return the filename."""
    chat_template: str | None = getattr(tokenizer, "chat_template", None)
    path = out_dir / "chat_template.json"
    if chat_template and not path.exists():
        payload: dict[str, str] = {"chat_template": chat_template}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return path.name
    return None


def list_present_expected_files(out_dir: Path) -> list[str]:
    """Return the subset of EXPECTED_NAMES present in out_dir."""
    present: list[str] = []
    for name in EXPECTED_NAMES:
        if (out_dir / name).exists():
            present.append(name)
    return sorted(present)


def download_tokenizer_and_preprocessor(
    model_id: str,
    out_dir: str,
    token: str | None = None,
    trust_remote_code: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Download/save tokenizer + (optional) processor files for `model_id` into `out_dir`.

    Returns:
        (present_after, missing_after): lists of filenames from EXPECTED_NAMES.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Import inside function so the file can be dropped in any environment easily.
    from transformers import AutoTokenizer

    # Some versions of transformers use 'use_auth_token', newer accept 'token' via HF env.
    # We pass via 'use_auth_token' for broad compatibility.
    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        use_auth_token=token,
    )
    # Save tokenizer files (tokenizer.json, tokenizer_config.json, special_tokens_map.json, vocab/merges or tokenizer.model, added_tokens.json)
    tok.save_pretrained(out_path)

    # Write a standalone chat_template.json if available
    write_chat_template_if_present(tok, out_path)

    # Try to grab preprocessor_config.json (for models that define processors, e.g., multi-modal)
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_auth_token=token,
        )
        proc.save_pretrained(out_path)
    except Exception:
        # Many text-only models won't have a processor; that's fine.
        pass
    patch_config_architectures(out_path / "config.json")

    present_after = list_present_expected_files(out_path)
    missing_after = sorted(EXPECTED_NAMES - set(present_after))
    return present_after, missing_after
