import os
import pickle

import sys


def cached(cache_name):
    def real_decorator(func):
        def new_func(*args, **kwargs):
            pickle_path = "tmp/" + cache_name
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    return pickle.load(f)
            else:
                print("Cache (%s) not found. Rebuilding..." % cache_name, file=sys.stderr)
                res = func(*args, **kwargs)
                with open(pickle_path, "wb") as f:
                    pickle.dump(res, f)
                return res
        return new_func
    return real_decorator
