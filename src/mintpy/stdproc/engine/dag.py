#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Task DAG for the MintPy slc2ifg engine.

A :class:`TaskGraph` holds one node per tool instance (e.g. one
``generate_ifgram`` node per date pair) and directed edges expressing
file-level dependencies.  Edges are used for (a) topological ordering,
(b) static validation and (c) Dask scheduling (a node only runs after its
dependencies).

Validation enforces the engine invariants (see ``docs/engine_design.md``):
- ``*.cpx.coh`` products may only exist for the ``fullres`` variant;
- cpx coherence must never appear as an input of any downstream tool;
- variant chains follow ``naming.next_variant``;
- processor (isce2/isce3) consistency along every edge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set

from mintpy.stdproc.engine.tool import Tool, ToolContext

logger = logging.getLogger(__name__)


@dataclass
class TaskNode:
    """One tool instance in the graph.

    Parameters
    ----------
    key : str
        Unique instance key, e.g. ``'generate_ifgram#20240101_20240119'``.
    tool : Tool
        The tool instance (stateless across runs).
    label : str
        Human-readable label for logging/dot export.
    ctx : ToolContext
        Fully resolved context (inputs/outputs/params) — inputs are static
        file paths thanks to the fixed naming scheme.
    """

    key: str
    tool: Tool
    label: str
    ctx: ToolContext
    deps: List[str] = field(default_factory=list)


class TaskGraph:
    """Directed acyclic graph of tool instances."""

    def __init__(self) -> None:
        self.nodes: Dict[str, TaskNode] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_node(self, node: TaskNode) -> None:
        if node.key in self.nodes:
            raise ValueError(f"Duplicate node key: {node.key}")
        self.nodes[node.key] = node

    def add_edge(self, src_key: str, dst_key: str) -> None:
        """Add a dependency edge ``src_key -> dst_key`` (src runs first).

        ``dst_key`` is recorded as depending on ``src_key``.
        """
        if src_key not in self.nodes or dst_key not in self.nodes:
            raise KeyError(f"Edge {src_key}->{dst_key} references unknown nodes")
        if src_key not in self.nodes[dst_key].deps:
            self.nodes[dst_key].deps.append(src_key)

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    def topo_order(self) -> List[str]:
        """Return node keys in topological order (dependencies first)."""
        visited: Set[str] = set()
        result: List[str] = []

        def dfs(key: str, stack: Set[str]) -> None:
            if key in visited:
                return
            if key in stack:
                raise ValueError(f"Cycle detected at node '{key}'")
            stack.add(key)
            for dep in self.nodes[key].deps:
                dfs(dep, stack)
            stack.remove(key)
            visited.add(key)
            result.append(key)

        for key in self.nodes:
            dfs(key, set())
        return result

    def subgraph(self, keys: List[str]) -> 'TaskGraph':
        """Return a new graph containing only the given node keys (and the
        edges between them)."""
        sub = TaskGraph()
        keyset = set(keys)
        for k in keys:
            if k not in self.nodes:
                raise KeyError(f"Unknown node '{k}'")
            node = self.nodes[k]
            sub.add_node(TaskNode(key=node.key, tool=node.tool,
                                  label=node.label, ctx=node.ctx,
                                  deps=[d for d in node.deps if d in keyset]))
        return sub

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Static invariants (see module docstring)."""
        # 1. Topological order check (also catches cycles)
        self.topo_order()

        # 2. cpx coherence invariants
        for key, node in self.nodes.items():
            for port, path in node.ctx.outputs.items():
                if self._is_cpx_coh(path):
                    variant = self._variant_of_path(path, node)
                    if variant != 'fullres':
                        raise ValueError(
                            f"[{key}] cpx coherence product must be 'fullres' "
                            f"variant only, got '{path.name}'")
            # 'stitch' merges/archives per-burst products (incl. cpx coh) into
            # the stitched tree — it does not *consume* them for processing.
            # 'unwrap' may consume the fullres complex coherence as its weight
            # input (slc2ifg.unwrap.coh_type = complex); the engine guarantees
            # it only feeds a fullres unwrap (see engine._add_uniform_nodes).
            if node.tool.name in ('stitch', 'unwrap'):
                continue
            for port, value in node.ctx.inputs.items():
                paths = value if isinstance(value, (list, tuple)) else [value]
                for p in paths:
                    if isinstance(p, (str, Path)) and self._is_cpx_coh(p):
                        raise ValueError(
                            f"[{key}] cpx coherence file '{p}' must never be "
                            f"consumed by a downstream tool")

        # 3. Processor consistency along edges
        for key, node in self.nodes.items():
            proc = node.ctx.params.get('processor')
            for dep in node.deps:
                dep_proc = self.nodes[dep].ctx.params.get('processor')
                if proc and dep_proc and proc != dep_proc:
                    raise ValueError(
                        f"[{key}] processor '{proc}' conflicts with dependency "
                        f"[{dep}] processor '{dep_proc}'")

        logger.info("DAG validation passed (%d nodes)", len(self.nodes))

    @staticmethod
    def _is_cpx_coh(path: Any) -> bool:
        name = str(path)
        return '.cpx.coh' in name

    @staticmethod
    def _variant_of_path(path: Any, node: TaskNode) -> str:
        """Variant of a product path.

        Uses ``naming.variant_of`` when importable; falls back to a local
        extension-stripping parser (e.g. in environments without osgeo,
        which ``mintpy.stdproc`` pulls in at import time).
        """
        name = Path(str(path)).name
        processor = node.ctx.params.get('processor', 'isce3')
        try:
            from mintpy.stdproc.utils.naming import variant_of
            return variant_of(name, processor)
        except Exception:
            stem = name
            if processor == 'isce3' and stem.endswith('.tif'):
                stem = stem[:-4]
            for prod in ('.unw.conncomp', '.conncomp', '.int', '.unw', '.coh'):
                if stem.endswith(prod):
                    stem = stem[: -len(prod)]
                    break
            for kind in ('.cpx', '.phsig'):
                if stem.endswith(kind):
                    stem = stem[: -len(kind)]
                    break
            return stem if stem in ('fullres', 'mli', 'filt', 'filt_mli') else 'fullres'

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self, verbose: bool = True) -> str:
        """Multi-line summary of the graph.

        Parameters
        ----------
        verbose : bool
            ``True`` (default) prints every node with its dependencies and
            outputs (used by ``--dry-run`` and ``--verbose``).  ``False``
            prints a one-line-per-tool count breakdown so normal runs keep
            the log compact even for large DAGs.
        """
        if not verbose:
            counts: Dict[str, int] = {}
            for key in self.topo_order():
                tool = key.split('#')[0]
                counts[tool] = counts.get(tool, 0) + 1
            lines = [f"DAG: {len(self.nodes)} node(s)"]
            lines += [f"  {tool}: {n} task(s)"
                      for tool, n in sorted(counts.items())]
            return '\n'.join(lines)

        lines = [f"DAG: {len(self.nodes)} node(s)"]
        for key in self.topo_order():
            node = self.nodes[key]
            deps = ', '.join(node.deps) if node.deps else '-'
            outs = ', '.join(str(p) for p in node.ctx.outputs.values())
            lines.append(f"  {key}  (deps: {deps})")
            lines.append(f"      out: {outs}")
        return '\n'.join(lines)

    def to_dot(self) -> str:
        """Export as graphviz DOT text."""
        lines = ["digraph engine {"]
        for key in self.topo_order():
            node = self.nodes[key]
            lines.append(f'  "{key}" [label="{node.label}"];')
        for key in self.topo_order():
            for dep in self.nodes[key].deps:
                lines.append(f'  "{dep}" -> "{key}";')
        lines.append("}")
        return '\n'.join(lines)
