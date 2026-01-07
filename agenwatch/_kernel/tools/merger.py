"""
Tool Result Merger
==================
Merges parallel tool outputs with:
- Conflict detection
- Gap detection (missing dependencies)
- Clean summary for LLM
- Structured output for UI/dashboard
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import json
from datetime import datetime


class MergeStrategy(Enum):
    """Strategy for handling conflicts when merging."""
    KEEP_FIRST = "keep_first"      # Keep first tool's value
    KEEP_LAST = "keep_last"        # Keep last tool's value
    COMBINE_LIST = "combine_list"  # Combine into list
    REPORT_CONFLICT = "report_conflict"  # Report as conflict


@dataclass
class ConflictRecord:
    """Record of a conflict between tool results."""
    field_path: str
    values: Dict[str, Any]  # {tool_name: value}
    resolved_to: Optional[str] = None  # Which tool's value was kept
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GapRecord:
    """Record of a missing dependency."""
    tool_name: str
    expected_from: List[str]  # Tools that should have provided this
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolStatus:
    """Status of a single tool execution."""
    tool_name: str
    success: bool
    error: Optional[str] = None
    result_size: int = 0  # Bytes
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MergeResult:
    """Complete merge result with metadata."""
    merged: Dict[str, Any]
    individual_results: Dict[str, Dict[str, Any]]
    conflicts: List[ConflictRecord] = field(default_factory=list)
    gaps: List[GapRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tool_statuses: List[ToolStatus] = field(default_factory=list)
    final_message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "merged": self.merged,
            "individual_results": self.individual_results,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "gaps": [g.to_dict() for g in self.gaps],
            "warnings": self.warnings,
            "tool_statuses": [ts.to_dict() for ts in self.tool_statuses],
            "final_message": self.final_message,
            "timestamp": self.timestamp,
            "summary": {
                "total_tools": len(self.individual_results),
                "successful_tools": sum(1 for ts in self.tool_statuses if ts.success),
                "failed_tools": sum(1 for ts in self.tool_statuses if not ts.success),
                "conflict_count": len(self.conflicts),
                "gap_count": len(self.gaps),
                "warning_count": len(self.warnings)
            }
        }


class ToolResultMerger:
    """
    Merges results from parallel tool execution.
    
    Features:
    - Deep merge of nested dicts
    - Conflict detection and recording
    - Gap detection (missing dependencies)
    - Clean summary generation
    - Structured output for UI/dashboard
    """
    
    def __init__(self, strategy: MergeStrategy = MergeStrategy.REPORT_CONFLICT):
        """
        Initialize merger.
        
        Args:
            strategy: How to handle conflicts
        """
        self.strategy = strategy
    
    def merge(
        self,
        individual_results: Dict[str, Dict[str, Any]],
        dependencies: Optional[Dict[str, List[str]]] = None
    ) -> MergeResult:
        """
        Merge individual tool results into single context.
        
        Args:
            individual_results: {tool_name: result_dict}
            dependencies: {tool_name: [dependent_tool_names]}
        
        Returns:
            MergeResult with merged data, conflicts, gaps, warnings
        """
        merged = {}
        conflicts: List[ConflictRecord] = []
        gaps: List[GapRecord] = []
        warnings: List[str] = []
        tool_statuses: List[ToolStatus] = []
        
        # 1. Build tool status records
        for tool_name, result in individual_results.items():
            is_error = isinstance(result, dict) and "error" in result and result["error"]
            status = ToolStatus(
                tool_name=tool_name,
                success=not is_error,
                error=result.get("error") if is_error else None,
                result_size=len(json.dumps(result)) if isinstance(result, dict) else 0
            )
            tool_statuses.append(status)
        
        # 2. Detect gaps (missing dependencies)
        if dependencies:
            gaps = self._detect_gaps(individual_results, dependencies)
            for gap in gaps:
                warnings.append(f"Gap: {gap.tool_name} expected output from {gap.expected_from}")
        
        # 3. Deep merge results
        for tool_name, result in individual_results.items():
            if isinstance(result, dict) and "error" not in result:
                merged_new, new_conflicts = self._deep_merge(
                    merged,
                    result,
                    source=tool_name
                )
                merged = merged_new
                conflicts.extend(new_conflicts)
        
        # 4. Build final message
        successful = sum(1 for ts in tool_statuses if ts.success)
        failed = sum(1 for ts in tool_statuses if not ts.success)
        final_message = self._build_final_message(successful, failed, conflicts, gaps)
        
        # 5. Add context warnings
        if conflicts:
            warnings.append(f"{len(conflicts)} field conflicts detected during merge")
        
        return MergeResult(
            merged=merged,
            individual_results=individual_results,
            conflicts=conflicts,
            gaps=gaps,
            warnings=warnings,
            tool_statuses=tool_statuses,
            final_message=final_message
        )
    
    def _deep_merge(
        self,
        base: Dict[str, Any],
        update: Dict[str, Any],
        source: str = "update",
        path: str = ""
    ) -> tuple[Dict[str, Any], List[ConflictRecord]]:
        """
        Recursively merge update into base.
        
        Returns:
            (merged_dict, conflict_records)
        """
        conflicts: List[ConflictRecord] = []
        result = base.copy()
        
        for key, value in update.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in result:
                # New key, just add it
                result[key] = value
            
            elif isinstance(result[key], dict) and isinstance(value, dict):
                # Both are dicts, recurse
                result[key], new_conflicts = self._deep_merge(
                    result[key],
                    value,
                    source=source,
                    path=current_path
                )
                conflicts.extend(new_conflicts)
            
            elif isinstance(result[key], list) and isinstance(value, list):
                # Both are lists, combine
                result[key] = result[key] + value
            
            else:
                # Conflict: different types or values
                conflict = ConflictRecord(
                    field_path=current_path,
                    values={
                        "existing": result[key],
                        source: value
                    }
                )
                conflicts.append(conflict)
                
                # Resolve based on strategy
                if self.strategy == MergeStrategy.KEEP_FIRST:
                    conflict.resolved_to = "existing"
                elif self.strategy == MergeStrategy.KEEP_LAST:
                    result[key] = value
                    conflict.resolved_to = source
                elif self.strategy == MergeStrategy.COMBINE_LIST:
                    result[key] = [result[key], value]
                    conflict.resolved_to = "combined"
                # REPORT_CONFLICT: just record, don't change
        
        return result, conflicts
    
    def _detect_gaps(
        self,
        individual_results: Dict[str, Any],
        dependencies: Dict[str, List[str]]
    ) -> List[GapRecord]:
        """
        Detect missing dependencies.
        
        If tool A depends on B, but B failed or missing, record gap.
        """
        gaps: List[GapRecord] = []
        
        for tool_name, deps in dependencies.items():
            if tool_name not in individual_results:
                continue
            
            result = individual_results[tool_name]
            has_error = isinstance(result, dict) and result.get("error")
            
            for dep in deps:
                dep_result = individual_results.get(dep)
                dep_error = (
                    isinstance(dep_result, dict) and dep_result.get("error")
                ) if dep_result else True
                
                if dep_error:
                    gap = GapRecord(
                        tool_name=tool_name,
                        expected_from=[dep],
                        reason=f"Dependency {dep} failed or missing"
                    )
                    gaps.append(gap)
        
        return gaps
    
    def _build_final_message(
        self,
        successful: int,
        failed: int,
        conflicts: List[ConflictRecord],
        gaps: List[GapRecord]
    ) -> str:
        """Build clean summary message for LLM."""
        total = successful + failed
        
        if total == 0:
            return "No tools executed."
        
        msg_parts = [
            f"Tool execution summary: {successful}/{total} successful"
        ]
        
        if failed > 0:
            msg_parts.append(f", {failed} failed")
        
        if conflicts:
            msg_parts.append(f", {len(conflicts)} conflicts detected")
        
        if gaps:
            msg_parts.append(f", {len(gaps)} dependency gaps")
        
        msg_parts.append(".")
        
        return "".join(msg_parts)
    
    def merge_and_format_for_llm(
        self,
        individual_results: Dict[str, Dict[str, Any]],
        dependencies: Optional[Dict[str, List[str]]] = None,
        max_length: int = 4000
    ) -> str:
        """
        Merge results and format as concise text for LLM consumption.
        
        Args:
            individual_results: Tool results
            dependencies: Tool dependencies
            max_length: Max characters before truncating
        
        Returns:
            Clean text summary
        """
        result = self.merge(individual_results, dependencies)
        
        # Build human-readable format
        lines = [
            result.final_message,
            "",
            "=== Merged Results ===",
        ]
        
        if result.merged:
            try:
                merged_str = json.dumps(result.merged, indent=2)
                lines.append(merged_str)
            except:
                lines.append(str(result.merged))
        else:
            lines.append("(empty)")
        
        if result.conflicts:
            lines.append("")
            lines.append("=== Detected Conflicts ===")
            for conflict in result.conflicts:
                lines.append(f"- {conflict.field_path}: {conflict.values}")
        
        if result.warnings:
            lines.append("")
            lines.append("=== Warnings ===")
            for warning in result.warnings:
                lines.append(f"- {warning}")
        
        text = "\n".join(lines)
        
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "\n... (truncated)"
        
        return text


# =====================================================================
# Utility functions
# =====================================================================

def simple_merge(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Quick merge without conflict detection."""
    merger = ToolResultMerger()
    merge_result = merger.merge(results)
    return merge_result.merged


def merge_with_conflicts(
    results: Dict[str, Dict[str, Any]]
) -> tuple[Dict[str, Any], List[ConflictRecord]]:
    """Merge and return conflicts separately."""
    merger = ToolResultMerger()
    merge_result = merger.merge(results)
    return merge_result.merged, merge_result.conflicts

__INTERNAL__ = True


