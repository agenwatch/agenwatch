import asyncio

class Scheduler:
    pass

class AgentScheduler:
    def __init__(self, agent):
        import asyncio
        self.agent = agent
        self.queue = asyncio.Queue()
        self.results = []   # <<< ADD THIS

    async def submit(self, tool_call):
        await self.queue.put(tool_call)

        result = await self.agent._execute_single_tool(
            tool_call
        )

        self.results.append(result)
        self.queue.task_done()
        return result




class ToolScheduler:
    def __init__(self, max_concurrent: int = 3, max_queue: int = 50):
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = asyncio.Queue(max_queue)
        self.running = False

    async def submit(self, tool_call):
        if self.queue.full():
            raise RuntimeError("scheduler_queue_full")
        await self.queue.put(tool_call)

    async def start(self, execute_fn):
        if self.running:
            return
        self.running = True

        async def worker():
            while self.running:
                tc = await self.queue.get()
                async with self.semaphore:
                    try:
                        await execute_fn(tc)
                    finally:
                        self.queue.task_done()

        for _ in range(self.max_concurrent):
            asyncio.create_task(worker())

    async def stop(self):
        self.running = False

__INTERNAL__ = True


