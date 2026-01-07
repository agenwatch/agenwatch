import os
import asyncio
from openai import OpenAI

class EmbeddingService:
    def __init__(self, model="text-embedding-3-large"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing")
        self.client = OpenAI(api_key=api_key)

    def embed_sync(self, text):
        result = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return result.data[0].embedding

    async def embed(self, text):
        return await asyncio.to_thread(self.embed_sync, text)

__INTERNAL__ = True
