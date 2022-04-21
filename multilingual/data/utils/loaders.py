# Copyright 2021 TUNiB Inc.


class InfiniteDataLoader:
    """
    Make dataloader to have infinite iterator

    This is copy of ``deepspeed.runtime.dataloader.RepeatingLoader``
    """

    def __init__(self, loader):
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch
