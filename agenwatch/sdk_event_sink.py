from queue import Queue
from agenwatch.sdk_event_adapter import adapt_kernel_event

class SDKEventSink:
    def __init__(self):
        self.queue = Queue()

    def __call__(self, event_type: str, data: dict):
        event = adapt_kernel_event(event_type, data)
        self.queue.put(event)

    def next(self):
        return self.queue.get()


