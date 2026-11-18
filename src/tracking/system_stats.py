from __future__ import annotations

import time
from typing import Dict

import psutil


def get_system_stats() -> Dict[str, float]:
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": psutil.virtual_memory().percent,
    }
