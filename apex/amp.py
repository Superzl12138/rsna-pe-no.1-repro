from contextlib import contextmanager


def initialize(model, optimizer=None, *args, **kwargs):
    if optimizer is None:
        return model
    return model, optimizer


@contextmanager
def scale_loss(loss, optimizer, *args, **kwargs):
    yield loss
