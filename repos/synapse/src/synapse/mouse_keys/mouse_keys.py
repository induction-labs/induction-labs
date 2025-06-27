from __future__ import annotations

import pandas as pd

# Read into a flat DataFrame
df = pd.read_json("path/to/your_file.jsonl", lines=True)

# Normalize the nested 'action' dict into its own columns
df_flat = pd.json_normalize(df["action"])
df_final = pd.concat([df_flat, df["timestamp"]], axis=1)
