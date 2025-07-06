from __future__ import annotations

import time

from pynput.keyboard import Controller, Key

keyboard = Controller()

# Press and release space

# sends Bb
# keyboard.press("B")
# keyboard.release("B")
# keyboard.press("b")
# keyboard.release("b")

# sends B
# keyboard.press(Key.shift)
# keyboard.press("b")
# keyboard.release(Key.shift)
# keyboard.release("b")

# sends bb
# keyboard.press("b")
# keyboard.press("b")
# keyboard.release("b")

# sends 3
# keyboard.press(Key.shift)
# keyboard.press("3")
# keyboard.release(Key.shift)
# keyboard.release("3")

# sends $
# keyboard.press(Key.shift)
# keyboard.press("$")
# keyboard.release(Key.shift)
# keyboard.release("$")

# sends }
# keyboard.press(Key.shift)
# keyboard.press("]")
# keyboard.release(Key.shift)
# keyboard.release("]")

# sends `
# keyboard.press(Key.shift)
# keyboard.press("`")
# keyboard.release(Key.shift)
# keyboard.release("`")

# sends -
# keyboard.press(Key.shift)
# keyboard.press("-")
# keyboard.release(Key.shift)
# keyboard.release("-")

# sends =
# keyboard.press(Key.shift)
# keyboard.press("=")
# keyboard.release(Key.shift)
# keyboard.release("=")

# sends "
# keyboard.press(Key.shift)
# keyboard.press("'")
# keyboard.release(Key.shift)
# keyboard.release("'")

time.sleep(5)
shift_will_translate = r"`-=[]\;',./"
typed_result = '~-={}|:"<./'
# changed chars: `[]\;',
# unchanged chars: -=./
for shift_char in shift_will_translate:
    keyboard.press(Key.shift)
    keyboard.press(shift_char)
    keyboard.release(Key.shift)
    keyboard.release(shift_char)
