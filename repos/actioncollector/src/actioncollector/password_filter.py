"""Password filtering functionality for action capture logs."""

from __future__ import annotations

import base64
import json
import os
from typing import Any


def load_passwords_lowercase(passwords_file: str = ".passwords") -> list[str]:
    """Load base64 encoded passwords from file."""
    passwords = []

    if not os.path.exists(passwords_file):
        return passwords

    try:
        with open(passwords_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        # Decode base64 password
                        password = base64.b64decode(line).decode("utf-8")
                        passwords.append(password.lower())
                    except Exception:
                        # Skip invalid base64 entries
                        continue
    except Exception:
        # If file can't be read, return empty list
        pass

    return passwords


def contains_password_sequence(
    actions: list[dict[str, Any]], passwords: list[str]
) -> bool:
    """Check if actions contain any password sequence."""
    if not passwords:
        return False

    keystrokes = []

    for action in actions:
        if (
            action.get("action", {}).get("action") == "key_button"
            and action.get("action", {}).get("is_down") is True
        ):
            key = action.get("action", {}).get("key", "")

            # Skip modifier keys
            if len(key) == 1:
                keystrokes.append(key)

            if key == "space":
                # Treat space as a word separator
                keystrokes.append(" ")

    keystroke_text = "".join(keystrokes).lower()

    # Check if any password appears in the keystroke sequence
    return any(password.lower() in keystroke_text for password in passwords)


def filter_password_actions(
    actions: list[dict[str, Any]], passwords: list[str]
) -> list[dict[str, Any]]:
    """Remove actions that are password keystrokes."""
    if not passwords:
        return actions

    filtered = []
    i = 0

    while i < len(actions):
        action = actions[i]

        # Check if this could start a password sequence
        if (
            action.get("action", {}).get("action") == "key_button"
            and action.get("action", {}).get("is_down") is True
        ):
            key = action.get("action", {}).get("key", "").lower()

            # Check if this key could start any password
            password_match = False
            for password in passwords:
                if password and key == password[0].lower():
                    # Try to match the full password sequence
                    match_count = 0
                    j = i

                    while j < len(actions) and match_count < len(password):
                        curr_action = actions[j]
                        if (
                            curr_action.get("action", {}).get("action") == "key_button"
                            and curr_action.get("action", {}).get("is_down") is True
                        ):
                            curr_key = (
                                curr_action.get("action", {}).get("key", "").lower()
                            )
                            if len(curr_key) == 1 or curr_key == "space":
                                curr_key_text = curr_key if curr_key != "space" else " "
                                if curr_key_text == password[match_count].lower():
                                    match_count += 1
                                else:
                                    break
                        j += 1

                    if match_count == len(password):
                        # Skip this password sequence
                        i = j
                        password_match = True
                        break

            if password_match:
                continue

        filtered.append(action)
        i += 1

    return filtered


def filter_actions_file(
    input_file: str, output_file: str, passwords: list[str]
) -> bool:
    """Filter a JSONL actions file and return True if filtering was applied."""
    if not passwords:
        # No passwords to filter, just copy the file
        with open(input_file) as inf, open(output_file, "w") as outf:
            outf.write(inf.read())
        return False

    # Read actions
    actions = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                actions.append(json.loads(line))

    # Check if passwords exist
    has_passwords = contains_password_sequence(actions, passwords)

    if has_passwords:
        # Filter out password actions
        filtered_actions = filter_password_actions(actions, passwords)

        # Write filtered actions
        with open(output_file, "w") as f:
            for action in filtered_actions:
                f.write(json.dumps(action) + "\n")

        return True
    else:
        # No passwords found, just copy the file
        with open(input_file) as inf, open(output_file, "w") as outf:
            outf.write(inf.read())

        return False
