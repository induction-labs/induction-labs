#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import timedelta

from google.cloud import storage

### ------------- CONFIG ------------- ###
GCS_URI_PREFIX = (
    "gs://induction-labs-data-ext-passive-mangodesk/action_capture/"  # <-- CHANGE ME
)
CONCURRENCY = 8  # threads for download
# -------------------------------------- #


def humanize(seconds: int) -> str:
    return str(timedelta(seconds=seconds))


def iter_blobs(client, bucket_name, prefix):
    bucket = client.bucket(bucket_name)
    yield from client.list_blobs(bucket, prefix=prefix)


def main():
    if not GCS_URI_PREFIX.startswith("gs://"):
        sys.exit("GCS_URI_PREFIX must start with gs://")

    bucket_name, prefix = GCS_URI_PREFIX[5:].split("/", 1)
    storage_client = storage.Client()
    user_to_buckets = defaultdict(set)  # {user: {bucket_id, …}}

    print("Scanning GCS…", file=sys.stderr)
    for blob in iter_blobs(storage_client, bucket_name, prefix):
        if not blob.name.endswith(".jsonl"):  # skip stray files
            continue

        # Extract <user>/... from "<prefix>/<user>/<folder>/file"
        rel_path = blob.name[len(prefix) :].lstrip("/")
        user = rel_path.split("/", 1)[0]

        # Stream the blob
        for line in blob.open("r"):
            try:
                ts = json.loads(line)["timestamp"]
            except Exception:  # malformed line
                continue
            bucket_id = int(ts // 30)
            user_to_buckets[user].add(bucket_id)

    # ----------  Report  ----------
    grand_total = 0
    widths = (max(map(len, user_to_buckets)) + 2, 12)
    print("\nUser".ljust(widths[0]), "Active time")
    print("-" * sum(widths))
    for user, buckets in sorted(user_to_buckets.items(), key=lambda kv: -len(kv[1])):
        active_sec = len(buckets) * 30
        grand_total += active_sec
        print(user.ljust(widths[0]), humanize(active_sec))

    print("-" * sum(widths))
    print("TOTAL".ljust(widths[0]), humanize(grand_total))


if __name__ == "__main__":
    main()
