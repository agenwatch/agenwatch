"""
AgenWatch Tool Validation
===========================
Validates tool arguments against JSON schemas.
"""

import json
import jsonschema
from jsonschema import validate
from typing import Tuple, Dict, Any, Optional


class ToolValidator:
    """
    Validates tool arguments against the tool's JSON schema.
    
    Example:
        >>> schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        >>> args = {"query": "hello"}
        >>> valid, error = ToolValidator.validate_args(schema, args)
        >>> valid
        True
    """
    
    @staticmethod
    def validate_args(
        schema: Dict[str, Any], 
        args: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate tool arguments against JSON schema.
        
        Args:
            schema: JSON schema defining expected argument structure
            args: Actual arguments to validate
        
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if validation passed
            - error_message: None if valid, error string if invalid
        
        Example:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "query": {"type": "string"},
            ...         "limit": {"type": "integer"}
            ...     },
            ...     "required": ["query"]
            ... }
            >>> ToolValidator.validate_args(schema, {"query": "test"})
            (True, None)
            >>> ToolValidator.validate_args(schema, {"limit": 10})
            (False, "'query' is a required property")
        """
        try:
            validate(instance=args, schema=schema)
            return True, None
        except jsonschema.exceptions.ValidationError as e:
            return False, str(e)
        except Exception as e:
            # Catch any other unexpected errors
            return False, f"Validation error: {str(e)}"

__INTERNAL__ = True



