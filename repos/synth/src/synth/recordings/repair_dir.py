#!/usr/bin/env python3
# repair_dir.py  — replace troublesome \uXXXX sequences in every file, now with a tqdm bar
from __future__ import annotations

import codecs
import concurrent.futures
import pathlib
import sys

from tqdm import tqdm

# -------- lookup table ---------
subs = {
    r"\u2212": "-",  # −  minus
    r"\u2010": "-",  # ‐  hyphen
    r"\u2012": "-",  # ‒  figure dash
    r"\u2013": "-",  # –  en dash
    r"\u2014": "--",  # —  em dash
    r"\u2018": "'",  # ‘
    r"\u2019": "'",  # ’
    r"\u201c": '\\"',  # “
    r"\u201d": '\\"',  # ”
    r"\u2026": "...",  # …
    r"\u2022": "*",  # •
    r"\u203a": ">",  # ›
    r"\u2082": "2",  # ₂
    r"\u2192": "->",  # →
    r"\u2193": "v",  # ↓
    r"\u21b5": "<CR>",  # ↵
    r"\u2248": "~",  # ≈
    r"\u2261": "==",  # ≡
    r"\u2265": ">=",  # ≥
    r"\u25b6": ">",  # ►
    r"\u25b8": ">",  # ▸
    r"\u25ba": ">",  # ►
    r"\u00ad": "",  # ­  soft hyphen
    r"\u00b0": " deg",  # °
    r"\u00b6": "¶",  # ¶
    r"\u00d7": "x",  # ×
    r"\u00e9": "e",  # é
    r"\u00ed": "i",  # í
    r"\u0142": "l",  # ł
    r"\u1e6d": "t",  # ṭ
    r"\u20ac": "EUR",  # €
    r"\u03a3": "S",  # Σ
    r"\u2003": " ",  # EM space
    r"\u200a": " ",  # hair space
    r"\u200b": "",  # zero-width space
    r"\uff0c": ",",  # full-width comma
    r"\uff1b": ";",  # full-width semicolon
    r"\u3002": ".",  # ideographic full stop
}


def fix_string(s: str) -> str:
    """Replace troublesome unicode sequences in a string."""
    for k, v in subs.items():
        esc_char = codecs.decode(k, "unicode_escape")
        s = s.replace(esc_char, v)
        s = s.replace(r"\"", '"')
    return s


# -------- worker ---------
def fix_file(path: str):
    path = pathlib.Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return  # unreadable ⇒ skip

    for k, v in subs.items():
        text = text.replace(k, v)
        text = text.replace(r'\\"', '"')

    try:
        path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(e)
        # leave file unchanged if write fails

    print("Fixed", path)


# -------- main ---------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python repair_dir_mp.py <directory> [workers]")

    root_dir = pathlib.Path(sys.argv[1]).expanduser()
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    files = [str(p) for p in root_dir.rglob("*") if p.is_file()]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        # submit everything up-front and update tqdm as each finishes
        futures = [pool.submit(fix_file, f) for f in files]
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            unit="file",
            desc=f"Fixing with {n_workers} procs",
        ):
            pass
