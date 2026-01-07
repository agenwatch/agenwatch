import os
import asyncio
from typing import List, Optional

from .providers.gemini_provider import GeminiEmbeddingProvider
from .providers.local_provider import LocalEmbeddingProvider
from .providers.tfidf_provider import TfidfEmbeddingProvider


class EmbeddingManager:
    """
    Universal embedding manager with 3-tier fallback:
    1. Google Gemini embeddings (FREE)
    2. Local sentence-transformers (optional)
    3. TF-IDF fallback (always available)
    """

    def __init__(self):
        self.providers = []
        self.default_provider = None

        # Provider 1: Gemini (free)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            self.providers.append(GeminiEmbeddingProvider(gemini_key))

        # Provider 2: Local sentence-transformers
        try:
            self.providers.append(LocalEmbeddingProvider())
        except Exception:
            pass

        # Provider 3: Always available fallback
        self.fallback = TfidfEmbeddingProvider()

        # Choose first available provider
        self.default_provider = self.providers[0] if self.providers else self.fallback

    async def embed(self, text: str) -> List[float]:
        """
        Try providers in priority order.
        Always returns a valid embedding.
        """

        # 1. Try premium providers
        for provider in self.providers:
            try:
                return await provider.embed(text)
            except Exception:
                continue

        # 2. Fallback always works
        return await self.fallback.embed(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embedding with fallback.
        """

        for provider in self.providers:
            try:
                return await provider.embed_batch(texts)
            except Exception:
                continue

        return await self.fallback.embed_batch(texts)

__INTERNAL__ = True


