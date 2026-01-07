"""
SafetyGuard - Tool Call Validation Layer
=========================================
Catches LLM hallucinations and invalid tool calls BEFORE execution.

Validates:
- Tool name exists in registry
- Required arguments present
- Argument types match schema
- No infinite loops detected
- No duplicate/redundant calls

Usage:
    guard = SafetyGuard(tool_registry)
    is_valid, errors = guard.validate_tool_call(tool_call)
    if not is_valid:
        # Handle errors
"""

from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json


class ValidationError(str, Enum):
    """Types of validation errors."""
    TOOL_NOT_FOUND = "tool_not_found"
    MISSING_REQUIRED_ARG = "missing_required_arg"
    INVALID_ARG_TYPE = "invalid_arg_type"
    LOOP_DETECTED = "loop_detected"
    REDUNDANT_CALL = "redundant_call"
    INVALID_SCHEMA = "invalid_schema"


@dataclass
class ToolCallValidation:
    """Result of tool call validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    tool_name: str
    args: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "tool_name": self.tool_name,
            "args": self.args
        }


class SafetyGuard:
    """
    Validates tool calls before execution.
    
    Prevents:
    - Hallucinated tool names
    - Missing required arguments
    - Type mismatches
    - Infinite loops
    - Redundant calls
    
    Example:
        >>> guard = SafetyGuard(tool_registry)
        >>> result = guard.validate_tool_call({
        ...     "tool_name": "search",
        ...     "args": {"query": "python"}
        ... })
        >>> if result.is_valid:
        ...     # Execute tool
        ... else:
        ...     # Handle errors: result.errors
    """
    
    def __init__(
        self,
        tool_registry,
        max_loop_history: int = 10,
        enable_redundancy_check: bool = True
    ):
        """
        Initialize SafetyGuard.
        
        Args:
            tool_registry: ToolRegistry instance
            max_loop_history: How many recent calls to track for loop detection
            enable_redundancy_check: Check for duplicate calls
        """
        self.tool_registry = tool_registry
        self.max_loop_history = max_loop_history
        self.enable_redundancy_check = enable_redundancy_check
        
        # Track recent calls for loop detection
        self.call_history: List[Tuple[str, str]] = []  # (tool_name, args_hash)
        
    # =====================================================================
    # MAIN VALIDATION
    # =====================================================================
    
    def validate_tool_call(
        self,
        tool_call: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolCallValidation:
        """
        Validate a single tool call.
        
        Args:
            tool_call: {"tool_name": str, "args": dict}
            context: Optional context for validation (e.g., recent history)
            
        Returns:
            ToolCallValidation with validation results
        """
        errors = []
        warnings = []
        
        tool_name = tool_call.get("tool_name", "")
        args = tool_call.get("args", {})
        
        # 1. Validate tool exists
        if not self._tool_exists(tool_name):
            errors.append(f"Tool '{tool_name}' not found in registry")
            return ToolCallValidation(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                tool_name=tool_name,
                args=args
            )
        
        # Get tool schema
        tool_schema = self._get_tool_schema(tool_name)
        
        # 2. Validate required arguments
        missing_args = self._check_required_args(args, tool_schema)
        if missing_args:
            errors.append(f"Missing required arguments: {', '.join(missing_args)}")
        
        # 3. Validate argument types
        type_errors = self._check_arg_types(args, tool_schema)
        if type_errors:
            errors.extend(type_errors)
        
        # 4. Check for infinite loops
        if self._is_loop_detected(tool_name, args):
            warnings.append(f"Potential loop detected: '{tool_name}' called repeatedly with same args")
        
        # 5. Check for redundant calls
        if self.enable_redundancy_check and self._is_redundant(tool_name, args):
            warnings.append(f"Redundant call: '{tool_name}' already called with these args")
        
        # Record this call
        self._record_call(tool_name, args)
        
        return ToolCallValidation(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            tool_name=tool_name,
            args=args
        )
    
    def validate_batch(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolCallValidation]:
        """
        Validate multiple tool calls at once.
        
        Returns:
            List of ToolCallValidation results
        """
        return [self.validate_tool_call(tc) for tc in tool_calls]
    
    # =====================================================================
    # VALIDATION CHECKS
    # =====================================================================
    
    def _tool_exists(self, tool_name: str) -> bool:
        """Check if tool exists in registry."""
        try:
            # Try multiple registry API patterns
            if hasattr(self.tool_registry, 'get_tool'):
                return self.tool_registry.get_tool(tool_name) is not None
            elif hasattr(self.tool_registry, 'list_tool_names'):
                return tool_name in self.tool_registry.list_tool_names()
            elif hasattr(self.tool_registry, 'tools'):
                return tool_name in self.tool_registry.tools
            else:
                return False
        except:
            return False
    
    def _get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get tool schema from registry."""
        try:
            if hasattr(self.tool_registry, 'get_tool'):
                tool = self.tool_registry.get_tool(tool_name)
                return tool if isinstance(tool, dict) else {}
            elif hasattr(self.tool_registry, 'get_tool_schema'):
                return self.tool_registry.get_tool_schema(tool_name)
            else:
                return {}
        except:
            return {}
    
    def _check_required_args(
        self,
        args: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[str]:
        """Check for missing required arguments."""
        missing = []
        
        # Handle different schema formats
        parameters = schema.get("parameters", {})
        required = parameters.get("required", [])
        
        # Also check "input_schema" format (Anthropic style)
        if not required:
            input_schema = schema.get("input_schema", {})
            required = input_schema.get("required", [])
        
        for req_arg in required:
            if req_arg not in args:
                missing.append(req_arg)
        
        return missing
    
    def _check_arg_types(
        self,
        args: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[str]:
        """Check argument type validity."""
        errors = []
        
        # Get properties from schema
        parameters = schema.get("parameters", {})
        properties = parameters.get("properties", {})
        
        # Also check "input_schema" format
        if not properties:
            input_schema = schema.get("input_schema", {})
            properties = input_schema.get("properties", {})
        
        for arg_name, arg_value in args.items():
            if arg_name not in properties:
                # Unknown arg (might be extra, but not necessarily error)
                continue
            
            expected_type = properties[arg_name].get("type")
            if not expected_type:
                continue
            
            # Validate type
            if not self._is_valid_type(arg_value, expected_type):
                errors.append(
                    f"Argument '{arg_name}' has invalid type. "
                    f"Expected: {expected_type}, Got: {type(arg_value).__name__}"
                )
        
        return errors
    
    def _is_valid_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_map.get(expected_type)
        if not expected_python_type:
            return True  # Unknown type, skip validation
        
        return isinstance(value, expected_python_type)
    
    def _is_loop_detected(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Detect if same tool+args called repeatedly."""
        args_hash = self._hash_args(args)
        call_signature = (tool_name, args_hash)
        
        # Count recent occurrences
        recent_calls = self.call_history[-self.max_loop_history:]
        count = sum(1 for call in recent_calls if call == call_signature)
        
        # If called 3+ times in recent history, it's likely a loop
        return count >= 3
    
    def _is_redundant(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if this exact call was made recently."""
        args_hash = self._hash_args(args)
        call_signature = (tool_name, args_hash)
        
        # Check last 3 calls
        recent_calls = self.call_history[-3:]
        return call_signature in recent_calls
    
    def _hash_args(self, args: Dict[str, Any]) -> str:
        """Create stable hash of arguments."""
        try:
            return json.dumps(args, sort_keys=True)
        except:
            return str(args)
    
    def _record_call(self, tool_name: str, args: Dict[str, Any]):
        """Record tool call in history."""
        args_hash = self._hash_args(args)
        self.call_history.append((tool_name, args_hash))
        
        # Keep only recent history
        if len(self.call_history) > self.max_loop_history:
            self.call_history.pop(0)
    
    # =====================================================================
    # UTILITIES
    # =====================================================================
    
    def reset_history(self):
        """Clear call history (e.g., at start of new session)."""
        self.call_history.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "total_calls_tracked": len(self.call_history),
            "unique_tools": len(set(tc[0] for tc in self.call_history)),
            "recent_calls": [
                {"tool": tc[0], "args_hash": tc[1][:20]}
                for tc in self.call_history[-5:]
            ]
        }
    
    def suggest_fix(self, validation: ToolCallValidation) -> Optional[str]:
        """Suggest how to fix validation errors."""
        if validation.is_valid:
            return None
        
        suggestions = []
        
        for error in validation.errors:
            if "not found" in error:
                # Suggest similar tool names
                similar = self._find_similar_tools(validation.tool_name)
                if similar:
                    suggestions.append(f"Did you mean: {', '.join(similar)}?")
            
            elif "Missing required" in error:
                suggestions.append("Add the missing arguments to your tool call")
            
            elif "invalid type" in error:
                suggestions.append("Check argument types match the schema")
        
        return " ".join(suggestions) if suggestions else "Check tool call format"
    
    def _find_similar_tools(self, tool_name: str, max_suggestions: int = 3) -> List[str]:
        """Find similar tool names (simple string distance)."""
        try:
            if hasattr(self.tool_registry, 'list_tool_names'):
                all_tools = self.tool_registry.list_tool_names()
            elif hasattr(self.tool_registry, 'tools'):
                all_tools = list(self.tool_registry.tools.keys())
            else:
                return []
            
            # Simple similarity: check if tool_name substring matches
            similar = [
                t for t in all_tools
                if tool_name.lower() in t.lower() or t.lower() in tool_name.lower()
            ]
            
            return similar[:max_suggestions]
        except:
            return []

__INTERNAL__ = True



