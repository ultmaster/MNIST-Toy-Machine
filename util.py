import time
import sys


class Timer(object):

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        print('%s start...' % self.name.capitalize())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = time.time()
        print('Finished in %.3fs\n' % (self.t1 - self.t0))
