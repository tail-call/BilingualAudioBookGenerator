from typing import Protocol


class Worker(Protocol):
    async def dispatch(self): ...


class Runnable(Protocol):
    """
    Used in workers to get the job orchestrator's running state so
    the workers would stop whenever the job orchestrator does.
    """
    @property
    def running(self) -> bool: ...
