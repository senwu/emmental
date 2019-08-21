from emmental.schedulers.mixed_scheduler import MixedScheduler
from emmental.schedulers.round_robin_scheduler import RoundRobinScheduler
from emmental.schedulers.sequential_scheduler import SequentialScheduler

SCHEDULERS = {
    "mixed": MixedScheduler,
    "round_robin": RoundRobinScheduler,
    "sequential": SequentialScheduler,
}

__all__ = ["mixed", "round_robin", "sequential"]
