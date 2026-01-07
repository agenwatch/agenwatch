"""
AgenWatch Memory System - CLEAN + COMPATIBLE WITH NEW MEMORY TYPES
===================================================================
Memory types supported:
- EPISODIC     (short term, transient)
- DECLARATIVE  (persistent long-term)
- REPAIR       (repair patterns stored as declarative facts)
- LONG_TERM    (compressed episodic)
- WORKING      (session-local working memory)
"""

import json
import os
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from agenwatch._kernel.memory_store.vector_store import SimpleVectorStore
from agenwatch._kernel.memory_types import MemoryType


# ======================================================================
# MEMORY ENTRY
# ======================================================================
@dataclass
class Memory:
    id: str
    content: str
    memory_type: MemoryType
    user_id: str
    session_id: str
    timestamp: str
    embedding: Optional[List[float]] = None
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    score: float = 0.0
    source: str = "unknown"


    def __post_init__(self):
        self.tags = self.tags or []
        self.metadata = self.metadata or {}
        self.last_accessed = self.last_accessed or self.timestamp

    def to_dict(self):
        d = asdict(self)
        d["memory_type"] = self.memory_type.value
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        data_copy = data.copy()
        data_copy["memory_type"] = MemoryType(data_copy["memory_type"])
        score = data_copy.pop("score", 0.0)
        m = Memory(**data_copy)
        if score:
            m.metadata["score"] = score
        return m

    def calculate_priority(self) -> float:
        """Calculate priority for memory retention."""
        now = datetime.now()
        last = datetime.fromisoformat(self.last_accessed)
        hours = (now - last).total_seconds() / 3600
        recency = max(0, 1 - (hours / (30 * 24)))  # 30 days decay
        return (self.importance * 2) + (self.access_count * 0.1) + recency


# ======================================================================
# MEMORY SYSTEM
# ======================================================================
class MemorySystem:
    """
    Memory system with multiple storage backends:
    - DECLARATIVE → persistent vector store
    - EPISODIC    → recent short-term memory (RAM)
    - REPAIR      → tool repair patterns (conflict-aware)
    - LONG_TERM   → compressed episodic memories
    - WORKING     → session-local working memory
    """

    def __init__(
        self,
        embedding_fn=None,
        storage_path="agenwatch_memory.json",
        max_memories=1000,
        max_episodic=100,
    ):
        self.embedding_fn = embedding_fn
        self.storage_path = storage_path
        self.max_memories = max_memories
        self.max_episodic = max_episodic

        # Storage backends
        self.declarative = SimpleVectorStore()  # DECLARATIVE memories
        self.long_term = SimpleVectorStore()    # LONG_TERM memories
        
        # RAM-only stores
        self.episodic: List[Memory] = []        # EPISODIC memories
        self.repair: List[Memory] = []          # REPAIR memories
        self.working: List[Memory] = []         # WORKING memories

        # Episodic indexing (token → list of memory IDs)
        self.episodic_index: Dict[str, List[str]] = {}

        self._load()

    # =====================================================================
    # ADD MEMORY
    # =====================================================================
    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        user_id: str,
        session_id="default",
        importance=0.5,
        tags=None,
        metadata=None,
        auto_embed=True,
        source="unknown"
    ) -> Memory:
        """Add a memory to the system."""
        
        memory_id = f"mem_{int(time.time() * 1000000)}"
        timestamp = datetime.now().isoformat()

        # Generate embedding if needed (skip for REPAIR, EPISODIC, WORKING)
        embedding = None
        if (auto_embed and self.embedding_fn and 
            memory_type in [MemoryType.DECLARATIVE, MemoryType.LONG_TERM]):
            try:
                if asyncio.iscoroutinefunction(self.embedding_fn):
                    embedding = await self.embedding_fn(content)
                else:
                    embedding = self.embedding_fn(content)
            except Exception as e:
                print(f"⚠️ Embedding failed: {e}")

        mem = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            session_id=session_id,
            timestamp=timestamp,
            embedding=embedding,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            source=source
        )

        self._store_memory(mem)
        self._prune()
        self._save()
        return mem

    # =====================================================================
    # STORE MEMORY
    # =====================================================================
    def _store_memory(self, mem: Memory):
        """Store memory in appropriate backend based on type."""
        
        # REPAIR memory → own storage (conflict-aware)
        if mem.memory_type == MemoryType.REPAIR:
            if not self._resolve_repair_conflict(mem):
                return
            self.repair.append(mem)
            if mem.metadata.get("success_count", 0) >= 2:
                self._promote_repair_to_global(mem)
            self._dedupe_repair_memories()
            return

        # DECLARATIVE → vector store (requires embedding)
        if mem.memory_type == MemoryType.DECLARATIVE:
            if mem.embedding is None:
                raise ValueError("DECLARATIVE memory requires embedding")
            self.declarative.add(mem.to_dict())
            return

        # EPISODIC → RAM list with indexing
        if mem.memory_type == MemoryType.EPISODIC:
            self.episodic.append(mem)
            self._index_episodic(mem)
            return

        # WORKING → RAM list (session-local)
        if mem.memory_type == MemoryType.WORKING:
            self.working.append(mem)
            return

        # LONG_TERM → vector store (requires embedding)
        if mem.memory_type == MemoryType.LONG_TERM:
            if mem.embedding is None:
                raise ValueError("LONG_TERM memory requires embedding")
            self.long_term.add(mem.to_dict())
            return

    def _index_episodic(self, mem: Memory):
        """Build lightweight keyword index for episodic memory."""
        tokens = set()

        # Index words from content (skip short words)
        for word in mem.content.lower().split():
            if len(word) > 2:
                tokens.add(word)

        # Index tags
        for t in mem.tags or []:
            tokens.add(t.lower())

        # Add to index
        for token in tokens:
            self.episodic_index.setdefault(token, []).append(mem.id)

    def _search_episodic_index(self, query: str, user_id: str) -> List[Memory]:
        """Search episodic memory using keyword index."""
        tokens = query.lower().split()
        hits = set()

        for t in tokens:
            if t in self.episodic_index:
                hits.update(self.episodic_index[t])

        # Return Memory objects
        results = []
        for mem in self.episodic:
            if mem.id in hits and mem.user_id == user_id:
                results.append(mem)

        return results

    def _promote_repair_to_global(self, mem: Memory):
        """Promote successful repair memory to global scope."""
        mem.session_id = "GLOBAL"
        mem.importance = min(1.0, mem.importance + 0.2)
        mem.tags = list(set((mem.tags or []) + ["global"]))

    def _resolve_repair_conflict(self, new_mem: Memory) -> bool:
        """
        Check if a new repair memory conflicts with existing ones.
        Keep the more successful repair pattern.
        Returns: True if new memory should be stored, False otherwise.
        """
        for old in self.repair:
            # Check if same tool with same broken args
            if (old.user_id == new_mem.user_id and
                old.metadata.get("tool") == new_mem.metadata.get("tool") and
                old.metadata.get("broken") == new_mem.metadata.get("broken")):
                
                # Compare: (success_count, access_count, recency)
                old_score = (
                    old.metadata.get("success_count", 0),
                    old.access_count,
                    old.timestamp
                )
                new_score = (
                    new_mem.metadata.get("success_count", 0),
                    new_mem.access_count,
                    new_mem.timestamp
                )
                
                # Keep old if it's better
                if old_score >= new_score:
                    return False
                
                # Remove old, keep new
                self.repair.remove(old)

        return True

    def _dedupe_repair_memories(self):
        """Remove duplicate repair memories, keeping best performers."""
        seen = {}
        
        for mem in list(self.repair):
            key = (
                mem.user_id,
                mem.metadata.get("tool"),
                json.dumps(mem.metadata.get("broken", {}), sort_keys=True)
            )

            if key in seen:
                existing = seen[key]
                # Keep the one with higher success count
                if mem.metadata.get("success_count", 0) > existing.metadata.get("success_count", 0):
                    if existing in self.repair:
                        self.repair.remove(existing)
                    seen[key] = mem
                else:
                    if mem in self.repair:
                        self.repair.remove(mem)
            else:
                seen[key] = mem

    # =====================================================================
    # SEMANTIC RECALL
    # =====================================================================
    async def recall(
        self,
        query: str,
        user_id: str,
        session_id=None,
        memory_types=None,
        top_k=5,
        min_score=0.3,
        tags=None
    ) -> List[Memory]:
        """Recall memories using semantic search + keyword index."""

        # Default memory types
        if memory_types is None:
            memory_types = [
                MemoryType.DECLARATIVE,
                MemoryType.LONG_TERM,
                MemoryType.REPAIR
            ]

        results = []

        # =====================================================================
        # SEMANTIC VECTOR SEARCH (DECLARATIVE + LONG_TERM)
        # =====================================================================
        if self.embedding_fn:
            # Get query embedding
            try:
                if asyncio.iscoroutinefunction(self.embedding_fn):
                    q_emb = await self.embedding_fn(query)
                else:
                    q_emb = await asyncio.to_thread(self.embedding_fn, query)
            except Exception as e:
                print(f"⚠️ Query embedding failed: {e}")
                q_emb = None

            if q_emb:
                # Search DECLARATIVE store
                if MemoryType.DECLARATIVE in memory_types:
                    vec_results = self.declarative.search(
                        query_embedding=q_emb,
                        top_k=top_k * 2,
                        min_score=min_score,
                        filter_fn=self._filter_fn(user_id, session_id, tags)
                    )
                    for item in vec_results:
                        mem = Memory.from_dict(item)
                        results.append(mem)

                # Search LONG_TERM store
                if MemoryType.LONG_TERM in memory_types:
                    vec_results = self.long_term.search(
                        query_embedding=q_emb,
                        top_k=top_k * 2,
                        min_score=min_score,
                        filter_fn=self._filter_fn(user_id, session_id, tags)
                    )
                    for item in vec_results:
                        mem = Memory.from_dict(item)
                        results.append(mem)

        # =====================================================================
        # REPAIR MEMORY (conflict-aware dedup)
        # =====================================================================
        if MemoryType.REPAIR in memory_types:
            repair_map = {}

            for mem in self.repair:
                if mem.user_id != user_id:
                    continue
                if tags and not any(t in (mem.tags or []) for t in tags):
                    continue

                key = (
                    mem.metadata.get("tool"),
                    json.dumps(mem.metadata.get("broken", {}), sort_keys=True)
                )

                # Keep best performer
                if key not in repair_map:
                    repair_map[key] = mem
                else:
                    if mem.calculate_priority() > repair_map[key].calculate_priority():
                        repair_map[key] = mem

            results.extend(repair_map.values())

        # =====================================================================
        # EPISODIC MEMORY (indexed + fallback)
        # =====================================================================
        if MemoryType.EPISODIC in memory_types:
            indexed = self._search_episodic_index(query, user_id)
            results.extend(indexed)

            # Fallback: include recent episodic if index misses
            for mem in self.episodic[-10:]:  # Last 10
                if mem.user_id == user_id and mem not in results:
                    results.append(mem)

        # =====================================================================
        # WORKING MEMORY (session-local)
        # =====================================================================
        if MemoryType.WORKING in memory_types:
            for mem in self.working:
                if mem.user_id == user_id and mem not in results:
                    results.append(mem)

        # Sort by priority and limit
        results.sort(key=lambda m: m.calculate_priority(), reverse=True)
        final = results[:top_k]

        # Update access stats
        for mem in final:
            mem.access_count += 1
            mem.last_accessed = datetime.now().isoformat()

            # Promote repair memory if frequently accessed
            if (mem.memory_type == MemoryType.REPAIR and
                mem.access_count >= 3 and
                mem.session_id != "GLOBAL"):
                self._promote_repair_to_global(mem)

        self._save()
        return final

    # =====================================================================
    # HELPERS
    # =====================================================================
    def _filter_fn(self, user_id, session_id, tags):
        """Create filter function for vector search."""
        def fn(item):
            if item.get("user_id") != user_id:
                return False
            if session_id and item.get("session_id") != session_id:
                return False
            if tags:
                itags = item.get("tags", [])
                if not any(t in itags for t in tags):
                    return False
            return True
        return fn

    def _decay_score(self, mem: Memory) -> float:
        """Calculate decay score for memory (higher = more stale)."""
        now = datetime.now()
        last = datetime.fromisoformat(mem.last_accessed)
        hours_old = (now - last).total_seconds() / 3600

        age_factor = min(1.0, hours_old / 24)
        importance_factor = 1 - mem.importance
        access_factor = 1 - min(mem.access_count / 10, 1)

        return (age_factor * 0.6) + (importance_factor * 0.3) + (access_factor * 0.1)

    def _compress_episodic(self):
        """Compress old episodic memories into long-term summary."""
        if self.embedding_fn is None or len(self.episodic) < 10:
            return

        # Take all but the most recent 5
        old_mems = self.episodic[:-5]
        if not old_mems:
            return

        # Build summary
        summary = " | ".join([m.content for m in old_mems])[:500]

        # Create compressed long-term memory
        try:
            if asyncio.iscoroutinefunction(self.embedding_fn):
                embedding = asyncio.run(self.embedding_fn(summary))
            else:
                embedding = self.embedding_fn(summary)
        except Exception as e:
            print(f"⚠️ Compression embedding failed: {e}")
            embedding = None

        compressed = Memory(
            id=f"cmp_{int(time.time() * 1000000)}",
            content=f"Compressed: {summary}",
            memory_type=MemoryType.LONG_TERM,
            user_id=old_mems[0].user_id,
            session_id=old_mems[0].session_id,
            timestamp=datetime.now().isoformat(),
            embedding=embedding,
            importance=0.6,
            tags=["compressed", "episodic"]
        )

        self.long_term.add(compressed.to_dict())
        self.episodic = self.episodic[-5:]  # Keep only last 5

    # =====================================================================
    # PRUNING
    # =====================================================================
    def _prune(self):
        """Prune old/low-priority memories to stay under limits."""
        
        # Prune DECLARATIVE store
        if self.declarative.size() > self.max_memories:
            items = [Memory.from_dict(x) for x in self.declarative.get_all()]
            scored = sorted(items, key=lambda m: m.calculate_priority(), reverse=True)
            keep = [m.to_dict() for m in scored[:int(self.max_memories * 0.8)]]

            self.declarative.clear()
            for item in keep:
                self.declarative.add(item)

        # Prune EPISODIC store
        self.episodic.sort(key=self._decay_score, reverse=True)
        
        # Remove memories older than 24 hours
        self.episodic = [
            m for m in self.episodic
            if (datetime.now() - datetime.fromisoformat(m.timestamp)).total_seconds() < 24 * 3600
        ]

        # Keep only last N episodic
        if len(self.episodic) > self.max_episodic:
            self.episodic = self.episodic[-self.max_episodic:]

        # Auto-compress episodic
        self._compress_episodic()

    # =====================================================================
    # SAVE / LOAD
    # =====================================================================
    def _save(self):
        """Persist memories to disk."""
        try:
            data = {
                "declarative": self.declarative.get_all(),
                "long_term": self.long_term.get_all(),
                "episodic": [m.to_dict() for m in self.episodic[-self.max_episodic:]],
                "repair": [m.to_dict() for m in self.repair],
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Save failed: {e}")

    def _load(self):
        """Load memories from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Load DECLARATIVE
            for item in data.get("declarative", []):
                self.declarative.add(item)

            # Load LONG_TERM
            for item in data.get("long_term", []):
                self.long_term.add(item)

            # Load EPISODIC
            for item in data.get("episodic", []):
                mem = Memory.from_dict(item)
                self.episodic.append(mem)
                self._index_episodic(mem)

            # Load REPAIR
            for item in data.get("repair", []):
                self.repair.append(Memory.from_dict(item))

            print(f"✅ Loaded {self.declarative.size()} declarative + {self.long_term.size()} long-term memories")

        except Exception as e:
            print(f"⚠️ Load failed: {e}")

__INTERNAL__ = True
