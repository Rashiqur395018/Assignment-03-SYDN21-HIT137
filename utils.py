import time
import functools

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        print(f"[TIMED] {func.__qualname__} took {elapsed:.3f} s")
        return res
    return wrapper

def logged(tag=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.__qualname__
            t = tag or name
            print(f"[LOG] ({t}) Calling {name} with args={args[1:] if len(args)>1 else '[]'} kwargs={kwargs}")
            result = func(*args, **kwargs)
            print(f"[LOG] ({t}) {name} finished")
            return result
        return wrapper
    return decorator
