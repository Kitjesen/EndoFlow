"""
File I/O, directory management, and timing utilities.
"""

import os
import sys
import time


def file_check(file_name):
    """Resolve file path, searching multiple locations. Raises FileNotFoundError."""
    import neupan
    root_path = os.path.dirname(os.path.dirname(neupan.__file__))

    if file_name is None:
        return None

    candidates = [
        file_name,
        os.path.join(sys.path[0], file_name),
        os.path.join(os.getcwd(), file_name),
    ]
    if root_path:
        candidates.append(os.path.join(root_path, file_name))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"File not found: {file_name}")


def repeat_mk_dirs(path, max_num=100):
    """Create directory; append _N suffix if it already contains files."""
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    if len(os.listdir(path)) == 0:
        return path
    for i in range(1, max_num):
        new_path = f"{path}_{i}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
    raise RuntimeError(f"Could not create unique directory from {path}")


def time_it(name="Function"):
    """Decorator: measure and optionally print function execution time."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            wrapper.count += 1
            start = time.time()
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start
            wrapper.func_count += 1
            from neupan.utils import transforms
            if transforms.time_print:
                print(f"{name} execute time {elapsed:.6f} seconds")
            return result
        wrapper.count = 0
        wrapper.func_count = 0
        return wrapper
    return decorator
