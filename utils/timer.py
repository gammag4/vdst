import time


class Timer:
    def __init__(self, total_steps = None):
        self._n_deltas = 0
        self.delta = 0.0
        self.avg_delta = 0.0
        self.total = 0.0
        self._last_time = time.perf_counter()
        self.total_steps = total_steps
        self.skipped = False
    
    @property
    def eta(self):
        if self.total_steps is None:
            return None
        
        return (self.total_steps - self._n_deltas) * self.avg_delta
    
    def update(self, increment_size=1):
        t = time.perf_counter()
        self.delta = t - self._last_time
        self._last_time = t
        self.total += self.delta
        self._n_deltas += increment_size
        self.avg_delta = self.total / self._n_deltas
        
        # Resets after first three steps bc it compiles kernels in the first step
        if self._n_deltas >= 3 and not self.skipped:
            self.total = 0
            self._n_deltas = 0
            self.skipped = True
    
    def state_dict(self):
        return {
            'n_deltas': self._n_deltas,
            'delta': self.delta,
            'avg_delta': self.avg_delta,
            'total': self.total,
            'last_time': self._last_time,
            'total_steps': self.total_steps,
            'skipped': self.skipped,
        }
    
    def load_state_dict(self, state_dict):
        self._n_deltas = state_dict['n_deltas']
        self.delta = state_dict['delta']
        self.avg_delta = state_dict['avg_delta']
        self.total = state_dict['total']
        self._last_time = state_dict['last_time']
        self.total_steps = state_dict['total_steps']
        self.skipped = state_dict['skipped']
