"""
AgenWatch DAG Planner
======================
Converts LLM output into structured execution plans.

How it works:
1. LLM produces a JSON plan with tools and dependencies
2. DAGPlanner parses and validates the plan
3. Returns structured (calls_dict, dependencies_dict)
4. Agent passes to ParallelExecutionEngine with ExecutionMode.DAG
5. Engine executes tools in dependency order

This enables:
- Multi-step reasoning
- Automatic dependency detection
- Optimal execution ordering
- Plan reuse and learning
"""

import json
import logging
import re
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("agenwatch.dag_planner")


class DAGPlannerConfig:
    """Configuration for DAG planner."""
    def __init__(
        self,
        max_tools: int = 20,
        require_valid_json: bool = True,
        allow_circular_deps: bool = False,
    ):
        self.max_tools = max_tools
        self.require_valid_json = require_valid_json
        self.allow_circular_deps = allow_circular_deps


class DAGPlan:
    """Structured execution plan."""
    def __init__(
        self,
        tools: Dict[str, Dict[str, Any]],
        dependencies: Dict[str, List[str]],
        reasoning: str = ""
    ):
        self.tools = tools
        self.dependencies = dependencies
        self.reasoning = reasoning
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tools": self.tools,
            "dependencies": self.dependencies,
            "reasoning": self.reasoning
        }


class DAGPlanner:
    """
    Converts LLM output into DAG execution plans.
    
    The LLM should output JSON in this format:
    {
        "reasoning": "Why this plan makes sense",
        "tools": {
            "search": {
                "query": "python async",
                "max_results": 5
            },
            "fetch": {
                "url": "..."
            },
            "summarize": {
                "text": "..."
            }
        },
        "dependencies": {
            "fetch": ["search"],
            "summarize": ["fetch"]
        }
    }
    """
    
    def __init__(self, config: Optional[DAGPlannerConfig] = None):
        self.config = config or DAGPlannerConfig()
    
    
    def get_system_prompt(self) -> str:
        """
        System prompt for LLM to generate DAG plans.
        
        Use this when calling LLM in planning mode.
        """
        return """You are an expert at planning multi-step workflows.

When asked to perform a task with multiple steps, you MUST respond with a structured JSON plan.

FORMAT:
{
    "reasoning": "Explain why this plan is optimal",
    "tools": {
        "tool_name_1": {"arg1": "value1", "arg2": "value2"},
        "tool_name_2": {"arg1": "value1"}
    },
    "dependencies": {
        "tool_name_2": ["tool_name_1"],
        "tool_name_3": ["tool_name_1", "tool_name_2"]
    }
}

RULES:
1. "tools" = {tool_name: arguments_dict}
2. "dependencies" = {tool_name: [list_of_tools_it_depends_on]}
3. Tools with no dependencies are executed first (in parallel)
4. Only include tools that are NEEDED for the task
5. reasoning explains the execution order
6. NO circular dependencies allowed

EXAMPLE:
Task: "Search for Python async tutorials, fetch the first result, then summarize it"

{
    "reasoning": "First search for tutorials, then fetch the top result, then summarize the content",
    "tools": {
        "search": {"query": "Python async tutorials", "max_results": 5},
        "fetch": {"url_from_search": true},
        "summarize": {"text_from_fetch": true}
    },
    "dependencies": {
        "fetch": ["search"],
        "summarize": ["fetch"]
    }
}

Now, create a plan for the task."""
    
    
    def plan_from_llm_output(
        self,
        llm_output: str,
        available_tools: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[DAGPlan], Optional[str]]:
        """
        Parse LLM output and extract DAG plan.
        
        Args:
            llm_output: Raw LLM response (should contain JSON)
            available_tools: List of tool names that exist (for validation)
        
        Returns:
            (success, plan_or_none, error_message_or_none)
        """
        
        # Step 1: Extract JSON from LLM output
        json_str = self._extract_json(llm_output)
        if not json_str:
            error = "No JSON found in LLM output"
            logger.error("[DAGPlanner] %s", error)
            return False, None, error
        
        # Step 2: Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            error = f"Invalid JSON: {str(e)}"
            logger.error("[DAGPlanner] %s", error)
            return False, None, error
        
        # Step 3: Validate structure
        is_valid, error = self._validate_plan_structure(data)
        if not is_valid:
            logger.error("[DAGPlanner] %s", error)
            return False, None, error
        
        # Step 4: Validate tools exist
        if available_tools:
            is_valid, error = self._validate_available_tools(data, available_tools)
            if not is_valid:
                logger.warning("[DAGPlanner] %s (continuing anyway)", error)
        
        # Step 5: Check for circular dependencies
        is_valid, error = self._check_circular_dependencies(data)
        if not is_valid:
            if self.config.allow_circular_deps:
                logger.warning("[DAGPlanner] %s (allowing anyway)", error)
            else:
                logger.error("[DAGPlanner] %s", error)
                return False, None, error
        
        # Step 6: Create plan
        tools = data.get("tools", {})
        dependencies = data.get("dependencies", {})
        reasoning = data.get("reasoning", "")
        
        plan = DAGPlan(
            tools=tools,
            dependencies=dependencies,
            reasoning=reasoning
        )
        
        logger.info(
            "[DAGPlanner] Plan created: %d tools, %d dependencies",
            len(tools),
            len(dependencies)
        )
        
        return True, plan, None
    
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON block from text."""
        # Try to find ```json ... ``` block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to find raw JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        
        return None
    
    
    def _validate_plan_structure(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate plan has required fields and all dependencies are valid."""
        if not isinstance(data, dict):
            return False, "Plan must be a JSON object"
        
        if "tools" not in data:
            return False, "Plan must have 'tools' field"
        
        if not isinstance(data["tools"], dict):
            return False, "'tools' must be a dict"
        
        if not data["tools"]:
            return False, "'tools' cannot be empty"
        
        if len(data["tools"]) > self.config.max_tools:
            return False, f"Too many tools (max: {self.config.max_tools})"
        
        # ✅ VALIDATION: Each tool must have arguments (dict, not None)
        for tool_name, args in data["tools"].items():
            if not isinstance(args, dict):
                return False, f"Tool '{tool_name}' arguments must be a dict, got {type(args).__name__}"
            # Note: empty dict {} is allowed (tool may have no required args)
        
        # dependencies is optional (defaults to empty)
        if "dependencies" not in data:
            data["dependencies"] = {}
        elif data["dependencies"] is None:
            # ✅ OPTIONAL 1: Handle null dependencies
            data["dependencies"] = {}
        
        if not isinstance(data["dependencies"], dict):
            return False, "'dependencies' must be a dict"
        
        # ✅ OPTIONAL 2: Check for duplicate dependencies
        for tool_name, deps in data["dependencies"].items():
            if not isinstance(deps, list):
                return False, f"Dependencies for '{tool_name}' must be a list"
            
            # Remove duplicates and warn if found
            unique_deps = list(set(deps))
            if len(unique_deps) < len(deps):
                logger.warning(
                    "[DAGPlanner] Tool '%s' has duplicate dependencies: %s → %s",
                    tool_name,
                    deps,
                    unique_deps
                )
                data["dependencies"][tool_name] = unique_deps
            
            # Validate all dependencies reference existing tools
            for dep in unique_deps:
                if dep not in data["tools"]:
                    return False, (
                        f"Dependency '{dep}' (required by '{tool_name}') "
                        f"not found in tools. Available: {list(data['tools'].keys())}"
                    )
        
        # ✅ VALIDATION: reasoning should be a string if present
        if "reasoning" in data:
            if not isinstance(data["reasoning"], str):
                return False, "'reasoning' must be a string"
        else:
            data["reasoning"] = ""
        
        return True, None
    
    
    def _validate_available_tools(
        self,
        data: Dict[str, Any],
        available_tools: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Check that all tools exist."""
        requested_tools = set(data["tools"].keys())
        available = set(available_tools)
        missing = requested_tools - available
        
        if missing:
            return False, f"Unknown tools: {missing}"
        
        return True, None
    
    
    def _check_circular_dependencies(
        self,
        data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Detect circular dependencies using DFS with detailed error reporting."""
        tools = data.get("tools", {})
        deps = data.get("dependencies", {})
        
        # Build adjacency list
        graph = {tool: deps.get(tool, []) for tool in tools}
        
        # DFS for cycles with path tracking
        visited = set()
        rec_stack = set()
        path = []
        
        def has_cycle(node: str) -> Tuple[bool, Optional[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    cycle_found, cycle_path = has_cycle(neighbor)
                    if cycle_found:
                        return True, cycle_path
                elif neighbor in rec_stack:
                    # Found cycle: node → neighbor (and neighbor is in current path)
                    cycle_start = path.index(neighbor)
                    cycle = " → ".join(path[cycle_start:] + [neighbor])
                    return True, cycle
            
            path.pop()
            rec_stack.remove(node)
            return False, None
        
        for tool in tools:
            if tool not in visited:
                cycle_found, cycle_path = has_cycle(tool)
                if cycle_found:
                    return False, f"Circular dependency detected: {cycle_path}"
        
        return True, None
    
    
    def get_topological_order(self, plan: DAGPlan) -> List[List[str]]:
        """
        Get execution order of tools (level by level).
        
        Returns list of levels, where each level can execute in parallel.
        Example: [["search"], ["fetch"], ["summarize"]]
        """
        tools = plan.tools
        deps = plan.dependencies
        
        levels = []
        executed = set()
        
        while len(executed) < len(tools):
            # Find tools ready to execute (all deps satisfied)
            ready = [
                t for t in tools
                if t not in executed
                and all(dep in executed for dep in deps.get(t, []))
            ]
            
            if not ready:
                break  # Should not happen if no cycles
            
            levels.append(ready)
            executed.update(ready)
        
        return levels
    
    
    def get_plan_summary(self, plan: DAGPlan) -> str:
        """Get human-readable summary of plan."""
        tools = plan.tools
        deps = plan.dependencies
        
        summary = f"DAG Plan ({len(tools)} tools):\n"
        summary += f"Reasoning: {plan.reasoning}\n\n"
        
        levels = self.get_topological_order(plan)
        for i, level in enumerate(levels):
            summary += f"Level {i+1} (parallel): {', '.join(level)}\n"
            for tool in level:
                args_str = json.dumps(tools[tool], indent=2)
                summary += f"  - {tool}: {args_str}\n"
        
        return summary

__INTERNAL__ = True



