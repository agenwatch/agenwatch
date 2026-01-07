from typing import Any, Dict


class BaseTool:
    """
    Base class for all AgenWatch tools.
    Each tool defines:
      - name
      - description
      - parameters (JSON schema)
      - run()
    
    Usage:
        class MyTool(BaseTool):
            name = "my_tool"
            description = "Does something"
            parameters = {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"}
                },
                "required": ["arg1"]
            }
            
            async def run(self, arg1: str):
                return f"Result: {arg1}"
    """
    name: str = "tool"
    description: str = "Tool description"
    parameters: Dict[str, Any] = None
    
    def __init__(self):
        """Initialize tool with safe defaults."""
        if self.parameters is None:
            self.parameters = {}
    
    async def run(self, **kwargs) -> Any:
        """
        Execute the tool with given arguments.
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Tool-specific arguments
        
        Returns:
            Tool result (any type)
        
        Raises:
            Exception: On tool execution failure
        """
        raise NotImplementedError("Tool must implement run()")
    
    def spec(self) -> Dict[str, Any]:
        """
        Returns the OpenAI/Groq function schema.
        
        Returns:
            Dict with name, description, parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters or {}
        }

__INTERNAL__ = True



