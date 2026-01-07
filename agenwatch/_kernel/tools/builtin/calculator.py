import logging
from agenwatch._kernel.tools.base import BaseTool
from typing import Any, Dict


logger = logging.getLogger("agenwatch.tools")


class CalculatorTool(BaseTool):
    """
    Simple calculator tool for mathematical expressions.
    """

    name = "calculator"
    description = "Evaluate safe mathematical expressions"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate (e.g., '2 + 2 * 3')"
            },
            "precision": {
                "type": "integer",
                "description": "Number of decimal places",
                "default": 2
            }
        },
        "required": ["expression"]
    }

    async def run(self, expression: str, precision: int = None, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression safely.
        `precision` is optional and ignored by the calculator,
        but required for schema-repair compatibility.
        """
        try:
            # Safe evaluation
            result = eval(
                expression,
                {"__builtins__": None},
                {}
            )

            # If precision is supplied → round result
            if precision is not None and isinstance(result, (int, float)):
                result = round(result, precision)

            return {"result": result}

        except Exception as e:
            return {"error": str(e)}

__INTERNAL__ = True



