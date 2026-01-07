"""
AgenWatch Schema-Aware Repair System
=====================================
Tool schemas for intelligent validation and repair.
"""

import json
from typing import Dict, Any, Optional, List, Tuple, Union, Type
from dataclasses import dataclass, field
from difflib import get_close_matches
from enum import Enum


# ============================================================
# Field / Schema definitions
# ============================================================

class FieldType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    NUMBER = "number"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


@dataclass
class FieldSchema:
    name: str
    field_type: FieldType
    required: bool = False
    default: Optional[Any] = None
    description: str = ""
    aliases: List[str] = field(default_factory=list)

    # validation constraints
    enum: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None

    strict: bool = False  # reject instead of cast

    def __post_init__(self):
        pass


@dataclass
class ToolSchema:
    tool_name: str
    fields: Dict[str, FieldSchema]
    description: str = ""

    def get_field(self, field_name: str) -> Optional[FieldSchema]:
        if field_name in self.fields:
            return self.fields[field_name]

        for fs in self.fields.values():
            if field_name in fs.aliases:
                return fs

        return None

    def fuzzy_match_field(self, input_name: str, cutoff: float = 0.7) -> Optional[FieldSchema]:
        # direct match
        direct = self.get_field(input_name)
        if direct:
            return direct

        # all possible field names
        names = list(self.fields.keys())
        for fs in self.fields.values():
            names.extend(fs.aliases)

        names = list(set(names))

        match = get_close_matches(input_name, names, n=1, cutoff=cutoff)
        if not match:
            return None

        found = match[0]

        for k, fs in self.fields.items():
            if found == k or found in fs.aliases:
                return fs

        return None

    # NEW: needed for UI & debugging
    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "fields": {k: vars(v) for k, v in self.fields.items()}
        }


# ============================================================
# Schema Repair Engine
# ============================================================

class SchemaRepairEngine:
    def __init__(self, schemas: Dict[str, ToolSchema]):
        self.schemas = schemas

    # NEW: combine memory repair + schema repair
    def repair_with_memory(self, tool_name: str, raw_args: Dict[str, Any], memory_patterns):
        # simple memory-based text replacement for each key
        mem_fixed = {}
        for k, v in raw_args.items():
            fixed_k = memory_patterns.get(k, k)
            mem_fixed[fixed_k] = v
        return self.repair_arguments(tool_name, mem_fixed)

    def repair_arguments(
        self,
        tool_name: str,
        raw_args: Dict[str, Any],
        strict_mode: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        if tool_name not in self.schemas:
            return raw_args, {
                "changes": {},
                "confidence": 0.0,
                "errors": [f"No schema for tool: {tool_name}"],
                "repaired": False,
                "final_args": raw_args
            }

        schema = self.schemas[tool_name]
        repaired = {}
        changes = {
            "mapped": {},
            "cast": {},
            "removed": [],
            "filled": {},
            "validated": {},
            "errors": []
        }

        matched_fields = set()

        # Phase 1 — map, cast, validate
        for input_key, value in raw_args.items():
            fs = schema.fuzzy_match_field(input_key)

            if fs:
                cname = fs.name
                matched_fields.add(cname)

                if input_key != cname:
                    changes["mapped"][input_key] = cname

                try:
                    cast_value = self._cast_value(fs, value)
                    if cast_value != value:
                        changes["cast"][cname] = {"old": value, "new": cast_value, "type": fs.field_type.value}

                    repaired[cname] = cast_value

                    # validate
                    err = self._validate_value(fs, cast_value)
                    if err:
                        changes["validated"][cname] = {"valid": False, "error": err}
                        changes["errors"].append(f"{cname}: {err}")
                    else:
                        changes["validated"][cname] = {"valid": True}

                except ValueError as e:
                    changes["errors"].append(f"{cname}: {str(e)}")
                    if fs.strict:
                        raise
                    repaired[cname] = value

            else:
                changes["removed"].append(input_key)

        # Phase 2 — fill defaults
        for fname, fs in schema.fields.items():
            if fname not in matched_fields:
                if fs.default is not None:
                    repaired[fname] = fs.default
                    changes["filled"][fname] = fs.default
                elif fs.required:
                    msg = f"Missing required field: {fname}"
                    changes["errors"].append(msg)
                    if strict_mode:
                        raise ValueError(msg)

        confidence = self._calculate_confidence(raw_args, changes)

        metadata = {
            "changes": changes,
            "confidence": confidence,
            "errors": changes["errors"],
            "repaired": (
                len(changes["mapped"]) > 0
                or len(changes["cast"]) > 0
                or len(changes["filled"]) > 0
            ),
            "schema": tool_name,

            # NEW: unified final args for execution layer
            "final_args": repaired
        }

        return repaired, metadata

    # --------------------------------------------------------
    # casting, validation, scoring
    # --------------------------------------------------------

    def _cast_value(self, fs: FieldSchema, value: Any) -> Any:
        t = fs.field_type

        if self._check_type(value, t):
            return value

        if t == FieldType.STRING:
            return str(value)
        if t == FieldType.INTEGER:
            return int(value)
        if t == FieldType.FLOAT:
            return float(value)
        if t == FieldType.NUMBER:
            try:
                return int(value)
            except:
                return float(value)
        if t == FieldType.BOOLEAN:
            if isinstance(value, str):
                v = value.lower()
                if v in ("true", "yes", "1", "on"):
                    return True
                if v in ("false", "no", "0", "off"):
                    return False
            return bool(value)
        if t == FieldType.ARRAY:
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    pass
                return [x.strip() for x in value.split(",") if x.strip()]
            if not isinstance(value, list):
                return [value]
            return value
        if t == FieldType.OBJECT:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except:
                    return {"value": value}
            if not isinstance(value, dict):
                return {"value": value}
            return value

        return value

    def _check_type(self, value: Any, t: FieldType) -> bool:
        if t == FieldType.STRING:
            return isinstance(value, str)
        if t == FieldType.INTEGER:
            return isinstance(value, int)
        if t == FieldType.FLOAT:
            return isinstance(value, float)
        if t == FieldType.NUMBER:
            return isinstance(value, (int, float))
        if t == FieldType.BOOLEAN:
            return isinstance(value, bool)
        if t == FieldType.ARRAY:
            return isinstance(value, list)
        if t == FieldType.OBJECT:
            return isinstance(value, dict)
        return True

    def _validate_value(self, fs: FieldSchema, value: Any) -> Optional[str]:
        if fs.enum is not None and value not in fs.enum:
            return f"{value!r} not in allowed {fs.enum}"

        if isinstance(value, (int, float)):
            if fs.min_value is not None and value < fs.min_value:
                return f"value {value} < minimum {fs.min_value}"
            if fs.max_value is not None and value > fs.max_value:
                return f"value {value} > maximum {fs.max_value}"

        if isinstance(value, str):
            if fs.min_length is not None and len(value) < fs.min_length:
                return f"string too short (min {fs.min_length})"
            if fs.max_length is not None and len(value) > fs.max_length:
                return f"string too long (max {fs.max_length})"

        if isinstance(value, list):
            if fs.min_length is not None and len(value) < fs.min_length:
                return f"array too short (min {fs.min_length})"
            if fs.max_length is not None and len(value) > fs.max_length:
                return f"array too long (max {fs.max_length})"

        return None

    def _calculate_confidence(self, raw_args: Dict[str, Any], changes: Dict[str, Any]) -> float:
        if not raw_args:
            return 1.0

        c = 1.0
        c -= len(changes["removed"]) * 0.1
        c -= len(changes["mapped"]) * 0.05
        c -= len(changes["cast"]) * 0.03
        c -= len(changes["errors"]) * 0.15

        if changes["filled"]:
            c += 0.05

        return max(0.0, min(1.0, c))


# ============================================================
# Default schemas
# ============================================================

DEFAULT_SCHEMAS = {
    "calculator": ToolSchema(
        tool_name="calculator",
        description="Evaluate mathematical expressions",
        fields={
            "expression": FieldSchema(
                name="expression",
                field_type=FieldType.STRING,
                required=True,
                aliases=["expr", "expresion", "exp", "formula"]
            ),
            "precision": FieldSchema(
                name="precision",
                field_type=FieldType.INTEGER,
                default=10,
                aliases=["prec", "precison", "decimal_places"],
                min_value=1,
                max_value=20
            )
        }
    ),

    "search": ToolSchema(
        tool_name="search",
        description="Search the web or sources",
        fields={
            "query": FieldSchema(
                name="query",
                field_type=FieldType.STRING,
                required=True,
                aliases=["q", "search_term", "question"],
                min_length=2
            ),
            "limit": FieldSchema(
                name="limit",
                field_type=FieldType.INTEGER,
                default=5,
                aliases=["max_results", "count", "top_n"],
                min_value=1,
                max_value=50
            ),
            "sources": FieldSchema(
                name="sources",
                field_type=FieldType.ARRAY,
                default=["web"],
                aliases=["source", "domains"]
            )
        }
    )
}


def get_default_schema_engine() -> SchemaRepairEngine:
    return SchemaRepairEngine(DEFAULT_SCHEMAS)

__INTERNAL__ = True



