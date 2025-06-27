#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from pynput import keyboard, mouse
from synapse.actions.models import Action, KeyButton, MouseButton, MouseMove, Scroll

from actioncollector.password_filter import (
    filter_actions_file,
    load_passwords_lowercase,
)
from actioncollector.utils import upload_to_gcs_and_delete


class ActionRecorder:
    def __init__(
        self,
        output_dir_template: str,
        gs_file_path: str,
        thread_pool: ThreadPoolExecutor,
        chunk_size: int = 10000,
        uploaded_callback=None,
        passwords_file: str = ".passwords",
    ):
        self.event_queue: Queue[Action] = Queue()

        # e.g., "tmp/output_{i:06d}.jsonl"
        self.output_dir_template = output_dir_template
        self.chunk_size = chunk_size

        self.keyboard_listener = None
        self.mouse_listener = None
        self.writer_thread = None

        self.thread_pool = thread_pool
        self.gs_file_path = gs_file_path

        self.uploaded_callback = uploaded_callback

        # Load passwords for filtering
        self.passwords = load_passwords_lowercase(passwords_file)
        if self.passwords:
            print(f"[info] loaded {len(self.passwords)} passwords for filtering")

    def on_move(self, x, y):
        action = Action.from_action_type(MouseMove(x=int(x), y=int(y)))
        self.event_queue.put(action)

    def on_click(self, x, y, button, pressed):
        action = Action.from_action_type(
            MouseButton(x=int(x), y=int(y), button=button.name, is_down=pressed)
        )
        self.event_queue.put(action)

    def on_scroll(self, x, y, dx, dy):
        action = Action.from_action_type(
            Scroll(delta_x=int(dx), delta_y=int(dy), x=int(x), y=int(y))
        )
        self.event_queue.put(action)

    def on_press(self, key, is_down: bool):
        if isinstance(key, keyboard.KeyCode):
            if key.char is None:
                # this is so troll, special keys that don't have a char representation
                # result in None
                # if we don't do this then the keyboard logger will crash when you press keys that
                # don't have a char representation, e.g. Fn on a macbook keyboard
                return

            action = Action.from_action_type(KeyButton(key=key.char, is_down=is_down))
            self.event_queue.put(action)
        else:
            action = Action.from_action_type(KeyButton(key=key.name, is_down=is_down))
            self.event_queue.put(action)

    def finished_writing_file(self, filename: str):
        def process_finished_file(filename: str):
            # Filter passwords before uploading
            filtered_filename = filename.replace(".jsonl", "_filtered.jsonl")

            was_filtered = filter_actions_file(
                filename, filtered_filename, self.passwords
            )

            if was_filtered:
                print(f"[info] filtered passwords from {filename.split('/')[-1]}")

            # Upload the filtered file and delete both original and filtered
            upload_to_gcs_and_delete(
                filtered_filename, self.gs_file_path + filename.split("/")[-1]
            )

            # Clean up original file if it still exists
            import os

            if os.path.exists(filename):
                os.remove(filename)

            if self.uploaded_callback:
                self.uploaded_callback()

        self.thread_pool.submit(process_finished_file, filename)

    def writer(self):
        file_chunk = 0
        event_no = 0
        while True:
            action = self.event_queue.get()
            event_no += 1

            if action is None:
                self.finished_writing_file(
                    self.output_dir_template.format(i=file_chunk)
                )
                break

            if event_no % self.chunk_size == 0:
                self.finished_writing_file(
                    self.output_dir_template.format(i=file_chunk)
                )
                file_chunk += 1

            with open(self.output_dir_template.format(i=file_chunk), "a") as f:
                json.dump(action.model_dump(), f)
                f.write("\n")

    def start(self):
        writer = threading.Thread(target=self.writer)
        writer.daemon = True
        writer.start()

        mouse_listener = mouse.Listener(
            on_move=self.on_move, on_click=self.on_click, on_scroll=self.on_scroll
        )
        keyboard_listener = keyboard.Listener(
            on_press=lambda x: self.on_press(x, is_down=True),
            on_release=lambda x: self.on_press(x, is_down=False),
        )
        keyboard_listener.start()
        mouse_listener.start()

        self.keyboard_listener = keyboard_listener
        self.mouse_listener = mouse_listener
        self.writer_thread = writer

    def stop(self):
        assert self.keyboard_listener is not None
        assert self.mouse_listener is not None
        assert self.writer_thread is not None
        print("[info] stopping listeners...")
        self.keyboard_listener.stop()
        self.mouse_listener.stop()

        print("[info] stopping writer thread...")
        # @MonliH: why do we need to put None in the queue?
        self.event_queue.put(None)
        self.writer_thread.join()
        print("[done] all actions recorded.")
