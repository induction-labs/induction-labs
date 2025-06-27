from __future__ import annotations

from pathlib import Path

from env import Action, ScreenshotEnv
from generators import (
    CursorPathConfig,
    TypingConfig,
    cursor_path_generator,
    typing_generator,
)
from recorder import Recorder

FPS = 60
DURATION = 5  # seconds

bg = Path("assets/screenshot.png")  # put a PNG or JPG here
font = Path("assets/arial.ttf")  # any ttf

env = ScreenshotEnv(bg, font)
typer = typing_generator(TypingConfig("hello world this is synthetic!", 100))
cursor = cursor_path_generator(
    CursorPathConfig([(100, 200), (400, 220), (600, 300), (800, 180)], DURATION),
    fps=FPS,
)

actions = sorted(list(typer) + list(cursor), key=lambda a: a.ts)

rec = Recorder("out.mp4", "out.jsonl", fps=FPS)
env.reset()
frame_ts = 0.0
frame_dt = 1 / FPS

for action in actions:
    # emit idle frames until it's time for the next action
    while frame_ts + 1e-6 < action.ts:
        rec.capture(env.step(Action(ts=frame_ts)), Action(ts=frame_ts))
        frame_ts += frame_dt
    rec.capture(env.step(action), action)
    frame_ts = max(frame_ts, action.ts)

# pad to full duration
while frame_ts < DURATION:
    rec.capture(env.step(Action(ts=frame_ts)), Action(ts=frame_ts))
    frame_ts += frame_dt

rec.close()
print("done → out.mp4 / out.jsonl")
