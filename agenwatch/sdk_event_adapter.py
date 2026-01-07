from agenwatch.types import StreamEvent

def adapt_kernel_event(event_type: str, data: dict) -> StreamEvent:
    """
    Convert kernel event → SDK StreamEvent
    """
    return StreamEvent(
        type=event_type,
        payload=dict(data) if data else {},
    )



