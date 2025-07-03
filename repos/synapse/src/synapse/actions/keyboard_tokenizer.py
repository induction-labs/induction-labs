from __future__ import annotations

import enum
import string

from pydantic import BaseModel
from synapse.utils import read_from_path, write_to_path

mod_keys = """alt
backspace
caps_lock
cmd
win
ctrl
delete
down
end
enter
esc
f1
f2
f3
f4
f5
f6
f7
f8
f9
f10
f11
f12
f13
f14
f15
f16
f17
f18
f19
f20
f21
f22
f23
f24
home
left
page_down
page_up
right
shift
space
tab
up
media_play_pause
media_stop
media_volume_mute
media_volume_down
media_volume_up
media_previous
media_next
media_eject
insert
menu
num_lock
pause
print_screen
scroll_lock""".strip()


class KeyActionType(enum.Enum):
    CLICK = "click"
    DOWN = "down"
    UP = "up"


TokenizerType = dict[str, dict[KeyActionType, int] | int]


class Tokenizer(BaseModel):
    """
    A tokenizer for keyboard actions, mapping keys to unique integer identifiers.
    """

    mappings: TokenizerType

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        return cls.model_validate_json(read_from_path(path))

    def debug_reverse_mapping(self, token_id: int) -> str:
        for key, value in self.mappings.items():
            if isinstance(value, dict):
                for action, number in value.items():
                    if number == token_id:
                        return f"{key} ({action})"
            elif value == token_id:
                return key

        return "Unknown Token ID"


def build_tokenizer() -> Tokenizer:
    strings = (
        string.ascii_lowercase
        + string.ascii_uppercase
        + string.digits
        + string.punctuation
    )
    special_tokens = ["[pad]", "[wait]", "[end]"]

    modifiers = mod_keys.splitlines()
    characters = list(strings)
    tokenizer: dict[str, dict[KeyActionType, int] | int] = {
        special: idx for idx, special in enumerate(special_tokens)
    }

    current_tokenizer_idx = len(tokenizer)
    for modifier in modifiers:
        tokenizer[modifier] = {
            KeyActionType.CLICK: current_tokenizer_idx,
            KeyActionType.DOWN: current_tokenizer_idx + 1,
            KeyActionType.UP: current_tokenizer_idx + 2,
        }
        current_tokenizer_idx += 3

    for char in characters:
        tokenizer[char] = {
            KeyActionType.CLICK: current_tokenizer_idx,
            KeyActionType.DOWN: current_tokenizer_idx + 1,
            KeyActionType.UP: current_tokenizer_idx + 2,
        }
        current_tokenizer_idx += 3

    return Tokenizer(mappings=tokenizer)


def save_tokenizer(path: str, tokenizer: Tokenizer) -> None:
    write_to_path(path, tokenizer.model_dump_json())


def test_tokenizer():
    # make sure all numbers are unique
    tokenizer = build_tokenizer()
    all_numbers = set()
    for key, value in tokenizer.mappings.items():
        if isinstance(value, dict):
            for action, number in value.items():
                assert number not in all_numbers, (
                    f"Duplicate number found for {key} {action}: {number}"
                )
                all_numbers.add(number)
        else:
            assert value not in all_numbers, (
                f"Duplicate number found for {key}: {value}"
            )
            all_numbers.add(value)


test_tokenizer()

if __name__ == "__main__":
    save_tokenizer(
        "gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json", build_tokenizer()
    )
    print(Tokenizer.load("gs://induction-labs/common/keyboard_tokenizer_v0.0.1.json"))
