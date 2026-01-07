class NullMemorySystem:
    async def recall(self, *args, **kwargs):
        return []

    async def add_memory(self, *args, **kwargs):
        return None

__INTERNAL__ = True


