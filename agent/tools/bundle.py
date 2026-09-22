from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from api_schema import ToolSpec


@dataclass(frozen=True)
class AgentToolBundle:
    """Provider-neutral function definitions and their local implementations."""

    specs: list[ToolSpec]
    runtime: dict[str, Any]


def merge_tool_bundles(*bundles: Any) -> AgentToolBundle | None:
    """Merge independently-built request-bound tool bundles.

    Tool names are the public ABI exposed to models, so collisions are rejected
    instead of allowing the last subsystem to silently replace an implementation.
    """

    specs: list[ToolSpec] = []
    runtime: dict[str, Any] = {}
    for bundle in bundles:
        if bundle is None:
            continue
        for spec in getattr(bundle, "specs", ()) or ():
            if spec.name in runtime or any(existing.name == spec.name for existing in specs):
                raise ValueError(f"duplicate agent tool name: {spec.name}")
            specs.append(spec)
        for name, implementation in (getattr(bundle, "runtime", {}) or {}).items():
            if name in runtime:
                raise ValueError(f"duplicate agent tool runtime: {name}")
            runtime[name] = implementation
    return AgentToolBundle(specs=specs, runtime=runtime) if specs else None
