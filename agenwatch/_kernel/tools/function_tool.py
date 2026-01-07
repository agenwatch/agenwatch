from typing import Callable, Any, Dict


class FunctionTool:
    """
    Lightweight wrapper to adapt plain Python functions
    into AgenWatch tools.
    """

    def __init__(self, name: str, fn: Callable[..., Any]):
        self.name = name
        self.fn = fn
        self.schema = None  # optional, used by registry

    async def run(self, **kwargs) -> Any:
        result = self.fn(**kwargs)
        if hasattr(result, "__await__"):
            return await result
        return result

__INTERNAL__ = True
