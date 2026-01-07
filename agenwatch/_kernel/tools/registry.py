import logging
import threading
from typing import Dict, Optional, List, Any

from agenwatch._kernel.tools.base import BaseTool
from agenwatch._kernel.tools.schemas import (
    ToolSchema,
    DEFAULT_SCHEMAS,
    SchemaRepairEngine,
)

# ------------------------------------------------------------
# Custom Errors (SDK-grade)
# ------------------------------------------------------------

class ToolRegistryError(Exception):
    """Base class for registry errors."""


class ToolAlreadyExistsError(ToolRegistryError):
    """Raised when a tool is registered twice."""


class SchemaNotFoundError(ToolRegistryError):
    """Raised when no schema exists for a requested tool."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when tool is not found in registry."""


# ------------------------------------------------------------
# Registry Core
# ------------------------------------------------------------

logger = logging.getLogger("agenwatch.tools")


class ToolRegistry:
    """
    AgenWatch OS — Production-Grade Tool Registry.

    Responsibilities:
    ---------------------------------------
    ✔ Register tools
    ✔ Register schemas
    ✔ Enforce uniqueness
    ✔ Provide LLM tool specs
    ✔ Provide schema repair engine
    ✔ Sync schemas + tools
    ✔ Thread-safe operations
    ✔ SDK-ready architecture
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        # Singleton pattern — one registry for the entire runtime
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 🔒 Instance-level lock (THIS is what I broke earlier)
        self._lock = threading.RLock()

        # Tool + schema storage
        self.tools: Dict[str, BaseTool] = {}
        self.schemas: Dict[str, ToolSchema] = {}

        # Auto-load built-in schemas
        for schema in DEFAULT_SCHEMAS.values():
            self.schemas[schema.tool_name] = schema

        # Schema repair engine
        self.schema_engine = SchemaRepairEngine(self.schemas)

        logger.info(
            "[ToolRegistry] Initialized — %d built-in schemas loaded",
            len(self.schemas),
        )

    # --------------------------------------------------------
    # TOOL REGISTRATION
    # --------------------------------------------------------

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool instance (SDK-safe)."""
        
        if hasattr(tool, "schema") and tool.schema:
            self.schemas[tool.name] = tool.schema
        
        with self._lock:
            if tool.name in self.tools:
                return

            self.tools[tool.name] = tool
            logger.info("[ToolRegistry] Tool registered: %s", tool.name)

            # Warn if tool has no schema
            if tool.name not in self.schemas:
                logger.warning(
                    "[ToolRegistry] Tool '%s' has no schema — schema repair disabled",
                    tool.name,
                )

    def get_tool(self, name: str) -> BaseTool:
        with self._lock:
            tool = self.tools.get(name)
            if not tool:
                raise ToolNotFoundError(f"Tool not found: {name}")
            return tool

    # --------------------------------------------------------
    # SCHEMA REGISTRATION
    # --------------------------------------------------------

    def register_schema(self, schema: ToolSchema) -> None:
        with self._lock:
            self.schemas[schema.tool_name] = schema
            logger.info("[ToolRegistry] Schema registered: %s", schema.tool_name)

    def get_schema(self, tool_name: str) -> ToolSchema:
        schema = self.schemas.get(tool_name)
        if not schema:
            raise SchemaNotFoundError(f"No schema found for tool: {tool_name}")
        return schema

    # --------------------------------------------------------
    # REPAIR ENGINE ACCESS
    # --------------------------------------------------------

    def get_repair_engine(self) -> SchemaRepairEngine:
        """Return shared schema repair engine."""
        return self.schema_engine

    # --------------------------------------------------------
    # LLM SPECS (CONVERT TOOL → OPENAI FORMAT)
    # --------------------------------------------------------

    def list_llm_specs(self) -> List[Dict[str, Any]]:
        """
        Return LLM-compatible tool specifications.
        Auto-generates parameter schemas using ToolSchema.
        """
        specs = []

        for tool_name, tool in self.tools.items():
            schema = self.schemas.get(tool_name)

            if not schema:
                # Fallback: use tool.spec() if no schema exists
                specs.append(tool.spec())
                continue

            specs.append({
                "name": tool_name,
                "description": schema.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        field.name: {
                            "type": field.field_type.value,
                            "description": field.description,
                        }
                        for field in schema.fields.values()
                    },
                    "required": [
                        field.name
                        for field in schema.fields.values()
                        if field.required
                    ],
                },
            })

        return specs

    # --------------------------------------------------------
    # INTROSPECTION
    # --------------------------------------------------------

    def list_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def list_schema_names(self) -> List[str]:
        return list(self.schemas.keys())

    def tool_count(self) -> int:
        return len(self.tools)

    def schema_count(self) -> int:
        return len(self.schemas)
    
    def get(self, name: str):
        """Backward compatibility: return tool by name."""
        return self.tools.get(name)
    
    def list_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Return schemas in a uniform dict format for the LLM.
        Agent uses this when constructing tool selection prompts.
        """
        schemas = []
        for name, schema in self.schemas.items():
            schemas.append(schema.to_dict() if hasattr(schema, "to_dict") else schema)
        return schemas

__INTERNAL__ = True



