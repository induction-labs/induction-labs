from __future__ import annotations

import time

from synapse.elapsed_timer.elapsed_timer import elapsed_timer


def main():
    # You can add your code here to run the module.
    # This is a placeholder for the main function.
    with elapsed_timer("Example Timer") as global_timer:
        # Sleep for 1s
        time.sleep(0.2)
        with elapsed_timer("Nested Timer"):
            # Sleep for 0.5s
            time.sleep(0.5)

    global_timer.print_timing_tree()


if __name__ == "__main__":
    main()
    # You can also run tests or other functions here.
    # For example, you can call a test function if it exists.
    # test_elapsed_timer()  # Uncomment if you have a test function defined.
