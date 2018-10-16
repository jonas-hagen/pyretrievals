import time
import numpy as np


def var_allan(y, axis=0):
    """
    Compute Allan variance of `y` along `axis`.
    """
    var = np.mean(np.square(np.diff(y, axis=axis)), axis=axis) / 2
    return var


class Timer:
    def __init__(self, clock='all', quiet=False):
        self.wall_start = 0
        self.wall_end = 0
        self.wall_interval = 0
        self.proc_start = 0
        self.proc_end = 0
        self.pro_interval = 0
        self.quiet = quiet

        if clock == 'wall':
            self.show_wall = True
            self.show_proc = False
        elif clock == 'proc':
            self.show_wall = False
            self.show_proc = True
        elif clock == 'all':
            self.show_wall = True
            self.show_proc = True
        else:
            raise LookupError('Invalid clock "{}". Use "wall", "proc" or "all".'.format(clock))

    def __enter__(self):
        self.wall_start = time.time()
        self.proc_start = time.process_time()
        return self

    def __exit__(self, *args):
        self.wall_end = time.time()
        self.proc_end = time.process_time()
        self.wall_interval = self.wall_end - self.wall_start
        self.proc_interval = self.proc_end - self.proc_start
        if not self.quiet:
            self.talk()

    def talk(self):
        if self.show_wall:
            print('Elapsed wall clock time is {:f} s.'.format(self.wall_interval))
        if self.show_proc:
            print('Elapsed process time is {:f} s'.format(self.proc_interval))