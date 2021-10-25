from joblib import Parallel, delayed
from tqdm.auto import tqdm


class ProgressParallel(Parallel):
    #https://stackoverflow.com/a/61027781/3541756

    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self, *args, **kwargs):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
        super().print_progress(*args, **kwargs)
