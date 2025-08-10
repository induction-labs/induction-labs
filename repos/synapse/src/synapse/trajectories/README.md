# Computer Annotation UI-Tars style

Trajectores that are just images and text;

## Structure

```
<folder>/
├── samples.jsonl          # Trajectory indexes
└── metadata/
    └── <attempt_id>.json  # Trajectory steps (~25MB each)
```

## Format

### samples.jsonl (Trajectory Indexes)
Each line contains:
```json
{
  "attempt_id": "uuid",           # References metadata/<attempt_id>.json
  "instruction": "string",
  ** other wildcard metadata (optional)
}
```

### metadata/<attempt_id>.json (Trajectory Steps)
JSON array of trajectory steps:
```json
[
  {
    "step": 0,                    # Required: Step number
    "image": "<base64_image>",    # Required: Screenshot as base64
    "text": "string",             # Required: Agent's reasoning
    ** other wildcard metadata (optiona)
  }
]
```
