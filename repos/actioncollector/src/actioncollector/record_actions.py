#!/usr/bin/env python3
import json
from pynput import mouse, keyboard
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from actioncollector.models import Action, MouseMove, MouseButton, Scroll, KeyButton
from actioncollector.utils import upload_to_gcs_and_delete


class ActionRecorder:
    def __init__(
        self,
        output_dir_template: str,
        gs_file_path: str,
        thread_pool: ThreadPoolExecutor,
        chunk_size: int = 2500,
        uploaded_callback=None,
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
            # This is where you can add any post-processing logic
            upload_to_gcs_and_delete(
                filename, self.gs_file_path + filename.split("/")[-1]
            )
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
        self.keyboard_listener.stop()
        self.mouse_listener.stop()

        print("[info] stopping writer thread...")
        self.event_queue.put(None)
        self.writer_thread.join()
        print("[done] all actions recorded.")
