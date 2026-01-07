import json
import re
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

class SafetyAction(Enum):
    ACCEPT = "accept"
    AUTO_FIX = "auto_fix"
    REJECT = "reject"

@dataclass
class ValidationResult:
    valid: bool
    action: SafetyAction
    fixed_args: Optional[Dict[str, Any]] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class ToolSafetyGuard:
    """
    Hybrid safety layer with auto-fix for minor issues, 
    rejection for critical errors.
    """
    
    def __init__(self, available_tools: Dict[str, dict], strict_mode: bool = False):
        self.available_tools = available_tools
        self.strict_mode = strict_mode
        
    def validate_and_fix(self, tool_name: str, args: Dict[str, Any]) -> ValidationResult:
        """Main entry point: validate and auto-fix if possible."""
        
        # Step 1: Validate tool exists
        if tool_name not in self.available_tools:
            # Try fuzzy match for typo correction
            corrected_name = self._find_similar_tool(tool_name)
            if corrected_name and not self.strict_mode:
                return ValidationResult(
                    valid=True,
                    action=SafetyAction.AUTO_FIX,
                    fixed_args=args,
                    warnings=[f"Tool '{tool_name}' corrected to '{corrected_name}'"],
                    errors=[]
                )
            return ValidationResult(
                valid=False,
                action=SafetyAction.REJECT,
                errors=[f"Unknown tool: '{tool_name}'"]
            )
        
        schema = self.available_tools[tool_name]
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Step 2: Prepare validation result
        result = ValidationResult(
            valid=False,
            action=SafetyAction.REJECT,
            fixed_args=args.copy(),
            errors=[],
            warnings=[]
        )
        
        # Step 3: Auto-fix minor issues
        if not self.strict_mode:
            self._auto_fix_typos(result, properties, args)
            self._auto_cast_types(result, properties)
        
        # Step 4: Check required fields (CRITICAL - can't auto-fix)
        missing_required = []
        for field in required:
            if field not in result.fixed_args:
                missing_required.append(field)
        
        if missing_required:
            result.errors.append(f"Missing required fields: {missing_required}")
            result.action = SafetyAction.REJECT
            return result
        
        # Step 5: Validate types for known fields
        for key, value in result.fixed_args.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    if self.strict_mode:
                        result.errors.append(
                            f"Field '{key}' must be {expected_type}, got {type(value).__name__}"
                        )
                    else:
                        # Try to auto-cast in non-strict mode
                        cast_value = self._try_cast(value, expected_type)
                        if cast_value is not None:
                            result.fixed_args[key] = cast_value
                            result.warnings.append(
                                f"Auto-cast '{key}' from {type(value).__name__} to {expected_type}"
                            )
                        else:
                            result.errors.append(
                                f"Field '{key}' must be {expected_type}, got {type(value).__name__}"
                            )
        
        # Step 6: Remove unknown fields if strict, warn if not
        unknown_fields = [k for k in result.fixed_args.keys() if k not in properties]
        if unknown_fields:
            if self.strict_mode:
                for field in unknown_fields:
                    del result.fixed_args[field]
                result.warnings.append(f"Removed unknown fields: {unknown_fields}")
            else:
                result.warnings.append(f"Unknown fields present: {unknown_fields}")
        
        # Step 7: Determine final action
        if result.errors:
            result.valid = False
            result.action = SafetyAction.REJECT
        else:
            result.valid = True
            result.action = SafetyAction.AUTO_FIX if result.warnings else SafetyAction.ACCEPT
        
        return result
    
    def _find_similar_tool(self, tool_name: str) -> Optional[str]:
        """Fuzzy match for tool name typos."""
        tool_name_lower = tool_name.lower()
        available = list(self.available_tools.keys())
        
        # Direct case-insensitive match
        for avail in available:
            if avail.lower() == tool_name_lower:
                return avail
        
        # Levenshtein-like simple matching
        for avail in available:
            if self._similarity(avail.lower(), tool_name_lower) > 0.7:
                return avail
        
        return None
    
    def _similarity(self, a: str, b: str) -> float:
        """Simple string similarity score."""
        if a == b:
            return 1.0
        if len(a) < 2 or len(b) < 2:
            return 0.0
        
        # Check for common prefixes or contained strings
        if a in b or b in a:
            return 0.8
        if a.startswith(b[:3]) or b.startswith(a[:3]):
            return 0.6
        
        return 0.0
    
    def _auto_fix_typos(self, result: ValidationResult, properties: Dict, original_args: Dict):
        """Fix common field name typos."""
        property_names = list(properties.keys())
        fixed_args = result.fixed_args.copy()
        
        for key in list(fixed_args.keys()):
            if key not in properties:
                # Try to find similar property name
                for prop in property_names:
                    if self._similarity(key, prop) > 0.8:
                        result.warnings.append(f"Renamed field '{key}' to '{prop}'")
                        fixed_args[prop] = fixed_args.pop(key)
                        break
        
        result.fixed_args = fixed_args
    
    def _auto_cast_types(self, result: ValidationResult, properties: Dict):
        """Auto-cast string numbers to integers/floats."""
        for key, value in result.fixed_args.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type in ["integer", "number"] and isinstance(value, str):
                    cast_value = self._try_cast(value, expected_type)
                    if cast_value is not None:
                        result.fixed_args[key] = cast_value
                        result.warnings.append(
                            f"Auto-cast '{key}' from string to {expected_type}"
                        )
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        if expected_type not in type_map:
            return True  # Unknown type, skip validation
        
        expected = type_map[expected_type]
        if isinstance(expected, tuple):
            return any(isinstance(value, t) for t in expected)
        return isinstance(value, expected)
    
    def _try_cast(self, value: Any, target_type: str) -> Any:
        """Try to cast value to target type."""
        try:
            if target_type == "integer":
                return int(value)
            elif target_type == "number":
                return float(value)
            elif target_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ["true", "1", "yes", "y"]
                return bool(value)
            elif target_type == "string":
                return str(value)
        except (ValueError, TypeError):
            return None
        return None

# Usage example:
if __name__ == "__main__":
    # Define your tool schemas
    TOOL_SCHEMAS = {
        "calculator": {
            "properties": {
                "expression": {"type": "string"},
                "precision": {"type": "integer"}
            },
            "required": ["expression"]
        },
        "search": {
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["query"]
        }
    }
    
    # Create safety guard (non-strict/hybrid mode)
    guard = ToolSafetyGuard(TOOL_SCHEMAS, strict_mode=False)
    
    # Test cases
    test_cases = [
        ("calculator", {"expresion": "10+20", "precision": "5"}),  # Typo + string number
        ("calculater", {"expression": "10+20"}),  # Tool name typo
        ("calculator", {"expression": 123}),  # Wrong type
        ("calculator", {}),  # Missing required
        ("unknown_tool", {"param": "value"}),  # Unknown tool
    ]
    
    for tool_name, args in test_cases:
        print(f"\nðŸ” Validating: {tool_name}({args})")
        result = guard.validate_and_fix(tool_name, args)
        print(f"   Valid: {result.valid}")
        print(f"   Action: {result.action.value}")
        if result.warnings:
            print(f"   Warnings: {result.warnings}")
        if result.errors:
            print(f"   Errors: {result.errors}")
        if result.fixed_args:
            print(f"   Fixed args: {result.fixed_args}")

__INTERNAL__ = True
