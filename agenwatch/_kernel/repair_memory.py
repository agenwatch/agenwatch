"""
AgenWatch Tool Repair Memory System
=====================================
Learn from past tool repair patterns and auto-apply fixes.
"""

import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from agenwatch._kernel.memory_types import MemoryType   # ✅ IMPORTANT FIX


class RepairMemoryManager:
    """
    Manages tool repair patterns and auto-applies learned fixes.
    """

    def __init__(self, memory_system, user_id: str, session_id: str):
        self.memory = memory_system
        self.user_id = user_id
        self.session_id = session_id

    async def store_repair(
        self,
        tool_name: str,
        broken_args: Dict[str, Any],
        fixed_args: Dict[str, Any],
        error_message: str,
        validation_schema: Optional[Dict[str, Any]] = None
    ):
        """
        Store a repair pattern for future use.
        """
        repair_data = {
            "tool": tool_name,
            "broken": broken_args,
            "fixed": fixed_args,
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "pattern": self._extract_pattern(broken_args, fixed_args)
        }

        if validation_schema:
            repair_data["schema"] = validation_schema

        content = f"Tool repair: {tool_name}. Error: {error_message}. Pattern: {repair_data['pattern']}"

        try:
            repair_data["mem_type"] = "repair"   # STORE AS STRING FOR SEARCH SAFELY

            await self.memory.add_memory(
                content=content,
                memory_type=MemoryType.REPAIR,
                user_id=self.user_id,
                session_id=self.session_id,
                importance=0.8,
                tags=["repair", tool_name, "auto-fix"],
                metadata=repair_data   # SAFE NOW
            )

        except Exception as e:
            print(f"[RepairMemory] ERROR storing repair: {e}")

    async def match_and_fix(
        self,
        tool_name: str,
        args: Dict[str, Any],
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:  # ✅ Return dict
        """
        Try to match args to a known broken pattern.
        
        Returns:
            {
                "fixed_args": Dict,
                "was_repaired": bool,
                "pattern": str or None,
                "confidence": float
            }
        """

        try:
            memories = await self.memory.recall(
                query=f"{tool_name} repair pattern auto-fix",
                user_id=self.user_id,
                top_k=10,
                tags=["repair", tool_name]
            )
        except Exception as e:
            print(f"[RepairMemory] Recall failed: {e}")
            return {
                "fixed_args": args,
                "was_repaired": False,
                "pattern": None,
                "confidence": 0.0
            }

        if not memories:
            return {
                "fixed_args": args,
                "was_repaired": False,
                "pattern": None,
                "confidence": 0.0
            }

        best_match = None
        best_confidence = 0.0

        for mem in memories:
            repair_data = mem.metadata
            if not isinstance(repair_data, dict):
                continue

            if repair_data.get("tool") != tool_name:
                continue

            broken = repair_data.get("broken", {})
            fixed = repair_data.get("fixed", {})

            confidence = self._calculate_match_confidence(args, broken)

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = (broken, fixed, repair_data.get("pattern", ""))

        if best_match and best_confidence >= min_confidence:
            broken, fixed, pattern = best_match
            fixed_args = self._apply_fix(args, broken, fixed)

            print(f"[RepairMemory] Auto-applied repair ({pattern}) confidence={best_confidence:.2f}")

            return {
                "fixed_args": fixed_args,
                "was_repaired": True,
                "pattern": pattern,
                "confidence": best_confidence
            }

        return {
            "fixed_args": args,
            "was_repaired": False,
            "pattern": None,
            "confidence": 0.0
        }

    def _extract_pattern(self, broken: Dict[str, Any], fixed: Dict[str, Any]) -> str:
        patterns = []
        broken_keys = set(broken.keys())
        fixed_keys = set(fixed.keys())

        removed = broken_keys - fixed_keys
        added = fixed_keys - broken_keys

        if removed and added:
            for old in removed:
                for new in added:
                    if broken.get(old) == fixed.get(new):
                        patterns.append(f"rename: {old} -> {new}")

        for key in added:
            patterns.append(f"add: {key}")

        for key in removed:
            patterns.append(f"remove: {key}")

        for key in broken_keys & fixed_keys:
            if broken[key] != fixed[key]:
                patterns.append(f"change: {key}")

        return ", ".join(patterns) if patterns else "general_fix"

    def _calculate_match_confidence(self, args: Dict[str, Any], broken: Dict[str, Any]) -> float:
        if not broken:
            return 0.0

        matches = 0
        total = len(broken)

        for key, value in broken.items():
            if key in args and args[key] == value:
                matches += 1

        return matches / total if total else 0.0

    def _apply_fix(self, args: Dict[str, Any], broken: Dict[str, Any], fixed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply fixes to args based on broken->fixed pattern.
        Handles renames properly (removes old key, adds new key).
        """
        # Start with a copy of the current args
        new_args = args.copy()
        
        # Check if this is a simple rename (one key to one key with same value)
        if len(broken) == 1 and len(fixed) == 1:
            old_key = list(broken.keys())[0]
            new_key = list(fixed.keys())[0]
            old_value = broken[old_key]
            new_value = fixed[new_key]
            
            # If values are the same, it's a rename
            if old_value == new_value:
                # If old key exists in args, rename it
                if old_key in new_args:
                    # Move value from old key to new key
                    new_args[new_key] = new_args.pop(old_key)
                    return new_args
        
        # General case for other patterns
        # First remove keys that are in broken but not in fixed
        for key in broken:
            if key not in fixed and key in new_args:
                new_args.pop(key, None)
        
        # Then add/update with fixed values
        for key in fixed:
            new_args[key] = fixed[key]
        
        return new_args
    
# Backward compatibility for older imports
RepairMemory = RepairMemoryManager

__INTERNAL__ = True



