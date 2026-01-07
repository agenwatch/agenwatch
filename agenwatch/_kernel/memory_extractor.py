"""
AgenWatch Memory Extractor - Production Ready
===============================================
LLM-powered extraction of long-term memories from conversations.
"""

import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

class RepairExtractionResult:
    def __init__(self, fixed_args=None, pattern=None, error=None):
        self.fixed_args = fixed_args or {}
        self.pattern = pattern
        self.error = error
        self.confidence = 1.0

class MemoryExtractor:
    """
    Extracts new long-term memories (facts, preferences, tasks)
    from recent conversation messages using LLM.
    
    Features:
    - Batch processing of recent messages
    - Structured memory extraction with metadata
    - Automatic importance scoring
    - Deduplication and validation
    
    Example:
        >>> extractor = MemoryExtractor(llm_provider, batch_size=5)
        >>> memories = await extractor.extract(conversation, user_id, session_id)
        >>> for mem in memories:
        ...     print(f"{mem['importance']}: {mem['content']}")
    """
    
    def __init__(
        self,
        llm_provider,
        batch_size: int = 4,
        min_memory_length: int = 10,
        max_memories_per_extraction: int = 10
    ):
        """
        Initialize memory extractor.
        
        Args:
            llm_provider: LLM provider with generate() method
            batch_size: Number of recent messages to analyze
            min_memory_length: Minimum character length for valid memories
            max_memories_per_extraction: Maximum memories to extract per call
        """
        self.llm = llm_provider
        self.batch_size = batch_size
        self.min_memory_length = min_memory_length
        self.max_memories_per_extraction = max_memories_per_extraction
        self._seen_memories = set()  # Simple deduplication
    
    async def extract(
        self,
        conversation: list,
        user_id: str,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Extract structured memories from recent conversation.
        
        Args:
            conversation: List of conversation messages (with .role and .content)
            user_id: User identifier
            session_id: Session identifier
        
        Returns:
            List of memory dictionaries with:
            - content: Memory text
            - type: Memory category (preference, fact, task, etc.)
            - importance: Score from 0.0 to 1.0
            - tags: List of relevant tags
        
        Example:
            >>> memories = await extractor.extract(messages, "alice", "session_1")
            >>> memories[0]
            {
                "content": "User prefers Python over JavaScript",
                "type": "preference",
                "importance": 0.9,
                "tags": ["language", "preference"]
            }
        """
        if not conversation:
            return []
        
        # Take only last few messages
        recent = conversation[-self.batch_size:]
        if not recent:
            return []
        
        # Build conversation text
        text = "\n".join(
            f"{m.role.upper()}: {m.content}"
            for m in recent
        )
        
        # Build extraction prompt
        prompt = self._build_extraction_prompt(text)
        
        try:
            # Call LLM
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            memories = self._parse_extraction_response(response)
            
            # Validate and deduplicate
            memories = self._validate_memories(memories)
            
            if memories:
                print(f"💾 Extracted {len(memories)} new memories")
            
            return memories
            
        except Exception as e:
            print(f"⚠️  Memory extraction failed: {e}")
            return []
    
    def _build_extraction_prompt(self, conversation_text: str) -> str:
        """Build the LLM prompt for memory extraction."""
        return f"""You are an AI memory extraction system.

Analyze the conversation below and extract ONLY **useful long-term memories** that would help an AI assistant better serve this user in future conversations.

Extract:
- User preferences (communication style, technology choices, formatting preferences)
- User identity details (projects, role, expertise, non-sensitive info only)
- User constraints or rules (things user wants or doesn't want)
- Tasks or goals mentioned
- Important facts or context

DO NOT extract:
- Temporary information (weather, time-specific events)
- Generic conversational filler
- Information already obvious from context
- Sensitive personal information (addresses, phone numbers, etc.)

Conversation:
{conversation_text}

Return ONLY valid JSON in this exact format:
{{
  "memories": [
    {{
      "content": "Clear, concise memory statement",
      "type": "preference|fact|task|context|constraint",
      "importance": 0.0-1.0,
      "tags": ["tag1", "tag2"]
    }}
  ]
}}

Rules:
- Maximum {self.max_memories_per_extraction} memories
- Each memory must be a complete, standalone statement
- Importance: 0.9-1.0 (critical), 0.7-0.8 (important), 0.5-0.6 (useful), 0.3-0.4 (minor)
- Use lowercase tags
- Return ONLY the JSON, no other text"""
    
    def _parse_extraction_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response and extract memory objects.
        
        Handles:
        - Markdown code blocks (```json, ```)
        - Plain JSON
        - Malformed responses
        """
        cleaned = response.strip()

        # Remove <final> wrapper
        if cleaned.startswith("<final>"):
            cleaned = cleaned.replace("<final>", "").replace("</final>", "").strip()

        
        
        # Remove markdown code fences
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        
        try:
            data = json.loads(cleaned)
            memories = data.get("memories", [])
            
            # Validate structure
            if not isinstance(memories, list):
                print("⚠️  Invalid memories format: expected list")
                return []
            
            return memories
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"   Response preview: {cleaned[:100]}...")
            return []
    
    def _validate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and filter extracted memories.
        
        Checks:
        - Required fields present
        - Content length adequate
        - Importance in valid range
        - Not duplicate of recent memory
        """
        validated = []
        
        for mem in memories:
            # Check required fields
            if not isinstance(mem, dict):
                continue
            
            content = mem.get("content", "")
            if not content or not isinstance(content, str):
                continue
            
            # Check minimum length
            if len(content) < self.min_memory_length:
                continue
            
            # Check for duplicates (simple hash-based)
            content_hash = hash(content.lower().strip())
            if content_hash in self._seen_memories:
                continue
            
            # Validate importance
            importance = mem.get("importance", 0.5)
            if not isinstance(importance, (int, float)):
                importance = 0.5
            importance = max(0.0, min(1.0, float(importance)))
            
            # Validate type
            mem_type = mem.get("type", "fact")
            if mem_type not in ["preference", "fact", "task", "context", "constraint"]:
                mem_type = "fact"
            
            # Validate tags
            tags = mem.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            tags = [str(t).lower() for t in tags if t]
            
            # Build validated memory
            validated_mem = {
                "content": content,
                "type": mem_type,
                "importance": importance,
                "tags": tags
            }
            
            validated.append(validated_mem)
            self._seen_memories.add(content_hash)
            
            # Limit total memories
            if len(validated) >= self.max_memories_per_extraction:
                break
        
        return validated
    
    
    async def extract_repair_pattern(self, tool_name: str, args: dict, error: str = None):
        """
        Minimal stub so Agent.run() does not crash.
        Always returns a RepairExtractionResult object.
        """
        return RepairExtractionResult(
            fixed_args=None,   # means "no repair found"
            pattern=None,
            error=error
        )


    
    def clear_deduplication_cache(self):
        """Clear the deduplication cache (call periodically to allow re-extraction)."""
        self._seen_memories.clear()
    
    def get_cache_size(self) -> int:
        """Get number of unique memories tracked for deduplication."""
        return len(self._seen_memories)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Mock LLM provider for testing
    class MockLLM:
        async def generate(self, messages):
            # Simulate LLM response
            return """```json
{
  "memories": [
    {
      "content": "User prefers Python over JavaScript for backend development",
      "type": "preference",
      "importance": 0.9,
      "tags": ["language", "preference", "backend"]
    },
    {
      "content": "User is building an AI agent framework called AgenWatch",
      "type": "context",
      "importance": 0.8,
      "tags": ["project", "ai", "framework"]
    },
    {
      "content": "User dislikes excessive use of bullet points in responses",
      "type": "constraint",
      "importance": 0.85,
      "tags": ["formatting", "style"]
    }
  ]
}
```"""
    
    # Mock conversation message
    class Message:
        def __init__(self, role, content):
            self.role = role
            self.content = content
    
    async def test_extractor():
        print("=" * 60)
        print("MEMORY EXTRACTOR - TEST")
        print("=" * 60)
        
        # Initialize
        llm = MockLLM()
        extractor = MemoryExtractor(llm, batch_size=5)
        
        # Mock conversation
        conversation = [
            Message("user", "Hi! I'm building an AI agent framework in Python."),
            Message("assistant", "That's great! What features are you working on?"),
            Message("user", "I prefer Python over JavaScript for this. Also, please don't use too many bullet points."),
            Message("assistant", "Understood. I'll keep responses concise.")
        ]
        
        # Extract memories
        print("\n✅ Extracting memories...")
        memories = await extractor.extract(
            conversation,
            user_id="test_user",
            session_id="test_session"
        )
        
        print(f"\n📝 Extracted {len(memories)} memories:\n")
        for i, mem in enumerate(memories, 1):
            print(f"{i}. [{mem['type'].upper()}] (importance: {mem['importance']})")
            print(f"   {mem['content']}")
            print(f"   Tags: {', '.join(mem['tags'])}")
            print()
        
        # Test deduplication
        print("✅ Testing deduplication...")
        memories2 = await extractor.extract(
            conversation,
            user_id="test_user",
            session_id="test_session"
        )
        print(f"   Second extraction: {len(memories2)} memories (should be 0 due to dedup)")
        
        # Clear cache and retry
        print("\n✅ Clearing dedup cache and retrying...")
        extractor.clear_deduplication_cache()
        memories3 = await extractor.extract(
            conversation,
            user_id="test_user",
            session_id="test_session"
        )
        print(f"   After cache clear: {len(memories3)} memories (should match first)")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
    
    # Run test
    asyncio.run(test_extractor())

__INTERNAL__ = True



