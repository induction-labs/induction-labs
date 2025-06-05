from __future__ import annotations

import importlib
from typing import Any


def import_from_string(import_string: str) -> Any:
    """
    Import a class, function, or module from a string.

    Examples:
        import_from_string("os.path.join")
        import_from_istring("datetime.datetime")
        import_from_string("mypackage.MyClass")
    """
    try:
        # Try to import as module first
        return importlib.import_module(import_string)
    except ImportError:
        # Split into module and attribute
        module_path, class_name = import_string.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


# # Usage
# DateTime = import_from_string("datetime.datetime")
# now = DateTime.now()

# PathJoin = import_from_string("os.path.join")
# path = PathJoin("/home", "user", "file.txt")
