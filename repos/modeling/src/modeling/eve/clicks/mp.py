from __future__ import annotations

import multiprocessing
from collections.abc import Callable

from colorama import Fore, Style, init
from tqdm import tqdm

init(autoreset=True)


def run_mp[T, O](
    items: list[T],
    process_func: Callable[[T], O],
    output_cls: type[O],
    num_workers: int = 1,
) -> list[O]:
    results: list[O] = []

    print(f"Using {num_workers} concurrent workers for processing")

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        print(f"{Fore.RED}Multiprocessing start method already set.{Style.RESET_ALL}")

    results = []

    try:
        with (
            tqdm(total=len(items), desc="Evaluating", unit="item") as pbar,
            multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool,
        ):
            for result in pool.imap_unordered(process_func, items):
                assert isinstance(result, output_cls)
                results.append(result)
                pbar.update(1)
    except Exception as e:
        print(f"{Fore.RED}Error in multiprocessing: {e}{Style.RESET_ALL}")
        raise e

    return results
