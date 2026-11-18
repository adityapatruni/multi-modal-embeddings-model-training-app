from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def autocast_if(enabled: bool):
    with torch.cuda.amp.autocast(enabled=enabled):
        yield
