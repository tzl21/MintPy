#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Tool abstraction for the MintPy slc2ifg engine.

A *Tool* is an atomic processing unit of the engine (see
``docs/engine_design.md``).  Each tool declares its input/output ports,
device preference and resource estimates, and implements an idempotent
``run(ctx)`` that calls the underlying mintpy.stdproc.slc2ifg module functions directly
(no subprocess).

Tools can be executed standalone (``run_engine.py --tool <name>``) or
orchestrated by the engine through a DAG (``dag.py``).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# Ports
# ------------------------------------------------------------------------
@dataclass
class Port:
    """A declared input or output port of a tool.

    Parameters
    ----------
    name : str
        Unique port name within the tool, e.g. ``'slc1'``, ``'ifg'``.
    kind : str
        ``'file'`` single file path, ``'file_list'`` list of paths,
        ``'dir'`` directory, ``'value'`` scalar/string value,
        ``'pairs_file'`` ifgram pair-list file.
    processor : str, optional
        ``'isce2'`` / ``'isce3'`` constraint (None = any).
    variant : str, optional
        Interferogram variant constraint: one of
        ``fullres / mli / filt / filt_mli`` (None = any).
    ext : str, optional
        Expected product extension (informational, e.g. ``'.int.tif'``).
    """

    name: str
    kind: str = 'file'
    processor: Optional[str] = None
    variant: Optional[str] = None
    ext: Optional[str] = None


# ------------------------------------------------------------------------
# Resources
# ------------------------------------------------------------------------
@dataclass
class Resource:
    """Resource declaration/estimate of a tool instance.

    Parameters
    ----------
    device : str
        ``'cpu'`` or ``'gpu'`` (declared by the tool author; the engine
        degrades GPU tools to CPU at runtime when no backend is available).
    mem_estimate_gb : float
        Peak host-memory estimate in GB (used for concurrency budgeting).
    gpu_mem_estimate_gb : float
        Peak GPU-memory estimate in GB.
    tileable : bool
        Whether the tool supports intra-interferogram tiling (M4).
    tile_overlap : int
        Required edge overlap in pixels when tiled.
    """

    device: str = 'cpu'
    mem_estimate_gb: float = 0.0
    gpu_mem_estimate_gb: float = 0.0
    tileable: bool = False
    tile_overlap: int = 0


# ------------------------------------------------------------------------
# Param specs (data-driven config plumbing, replaces _tool_params branches)
# ------------------------------------------------------------------------
@dataclass
class ParamSpec:
    """Declarative description of one tool configuration parameter.

    Parameters
    ----------
    key : str
        Key under which the value lands in ``ctx.params``.
    cfg : str or None
        Config key to read, e.g. ``'slc2ifg.unwrap.snaphu.cost_mode'``.
        Backend-scoped keys use the ``slc2ifg.<stage>.<backend>.<param>``
        convention (one namespace per algorithm/backend implementation);
        ``None`` = engine-injected constant (``default`` is used as-is).
    legacy_cfg : str or None
        Deprecated config key that is consulted as a fallback (with a warning)
        when ``cfg`` is absent — used to migrate to a new naming scheme
        without breaking existing configs.
    kind : str
        ``'str' | 'int' | 'float' | 'bool' | 'tuple'`` (tuple = comma list
        of ints).
    default : any
        Value used when the config key is missing / ``auto``.  ``None``
        means the key is omitted from ``ctx.params`` (tool falls back).
    """

    key: str
    cfg: Optional[str] = None
    legacy_cfg: Optional[str] = None
    kind: str = 'str'
    default: Any = None


# ------------------------------------------------------------------------
# Stage specs (tool-declared processing-chain fragments, see engine.chain)
# ------------------------------------------------------------------------
@dataclass
class StageSpec:
    """A tool's self-declared place in the processing chain.

    Third-party tools set ``Tool.stage`` to declare where they belong;
    the engine merges these fragments with the built-in default chain
    (``engine/chain.py``, see ``docs/engine_stages_design.md``).
    """

    name: str
    tool: str = ''
    scope: str = 'global'
    optional: bool = True
    requires: Tuple[str, ...] = ()
    out_dir: str = 'ifgram'
    desc: str = ''

    def to_stage(self):
        from mintpy.stdproc.slc2ifg.engine.chain import Stage
        return Stage(
            name=self.name,
            tool=self.tool or self.name,
            scope=self.scope,
            optional=self.optional,
            requires=self.requires,
            out_dir=self.out_dir,
            desc=self.desc,
        )


# ------------------------------------------------------------------------
# Tool context
# ------------------------------------------------------------------------
class ToolContext:
    """Everything a tool needs to run one instance.

    ``inputs``  : resolved input values keyed by port name
                  (paths for files, lists for file_list, scalars for value).
    ``outputs`` : output paths keyed by port name (computed via naming).
    ``params``  : configuration parameters relevant for this tool.
    ``work_dir``: engine working directory (for scratch/temp files).
    """

    def __init__(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Path],
        params: Dict[str, Any],
        work_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        self.tool_name = tool_name
        self.inputs = inputs
        self.outputs = outputs
        self.params = params
        self.work_dir = Path(work_dir)
        self.logger = logger or logging.getLogger(f'mintpy.stdproc.slc2ifg.engine.{tool_name}')
        #: Set by the scheduler right before ``Tool.run`` so tools can append
        #: their actual processing time to the completion log line.
        self.started: Optional[float] = None
        #: Set by tools (via ``skip_if_exists`` or their own early-return) when
        #: the task was skipped because outputs already exist.
        self.skipped: bool = False

    def input(self, name: str) -> Any:
        """Get a resolved input by port name."""
        return self.inputs[name]

    def output(self, name: str) -> Path:
        """Get the output path by port name."""
        return self.outputs[name]

    def param(self, key: str, default: Any = None) -> Any:
        """Get a configuration parameter (dot-separated keys supported)."""
        return self.params.get(key, default)

    def elapsed_str(self) -> str:
        """``(took 1m02s)`` for this task's actual processing time, or ``''``
        when the context was not timed (e.g. standalone invocation)."""
        if self.started is None:
            return ''
        return f"(took {_fmt_dur(time.monotonic() - self.started)})"


def _fmt_dur(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f}s"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ------------------------------------------------------------------------
# Tool base class & registry
# ------------------------------------------------------------------------
class Tool(ABC):
    """Base class of all engine tools.

    Subclasses must set ``name``, ``inputs``, ``outputs`` and implement
    ``run(ctx)``.  ``run()`` MUST be idempotent: if the declared outputs
    already exist and are complete, it returns them without recomputing.
    """

    name: str = ''
    inputs: List[Port] = field(default_factory=list)  # type: ignore[assignment]
    outputs: List[Port] = field(default_factory=list)  # type: ignore[assignment]
    resource: Resource = Resource()
    #: Data-driven config plumbing (see ParamSpec) — replaces engine-side
    #: hardcoded per-tool parameter branches.
    params_spec: List[ParamSpec] = field(default_factory=list)
    #: Self-declared place in the processing chain (see StageSpec).
    stage: Optional[StageSpec] = None

    def __init__(self) -> None:
        # Per-instance copy of the declared Resource: runtime adjustments
        # (e.g. GPU -> CPU degrade in the engine) must never leak into the
        # shared class declaration, which `gpu_tools()` reads.
        self.resource = replace(self.resource)

    @abstractmethod
    def run(self, ctx: ToolContext) -> Dict[str, Path]:
        """Execute the tool and return ``{output_port_name: path}``."""

    def estimate(self, ctx: ToolContext) -> Resource:
        """Refine the resource estimate for this specific instance."""
        return self.resource

    # -- convenience helpers for subclasses --------------------------------
    @staticmethod
    def output_ready(paths) -> bool:
        """All outputs exist and are usable.

        Files must be non-empty — a zero-byte/partial product left by an
        interrupted run is NOT treated as a valid output (the tool re-runs).
        Directories only need to exist.
        """
        for p in paths:
            p = Path(p)
            if not p.exists():
                return False
            if p.is_file() and p.stat().st_size <= 0:
                return False
        return True

    def skip_if_exists(self, ctx: ToolContext) -> Optional[Dict[str, Path]]:
        """Return output mapping if every output already exists, else None."""
        if self.output_ready(ctx.outputs.values()):
            ctx.logger.info("All outputs exist, skipping: %s",
                            ', '.join(Path(p).name for p in ctx.outputs.values()))
            ctx.skipped = True
            return dict(ctx.outputs)
        return None


TOOL_REGISTRY: Dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator registering a Tool subclass by its ``name``."""
    if not cls.name:
        raise ValueError(f"Tool class {cls.__name__} must define 'name'")
    TOOL_REGISTRY[cls.name] = cls
    return cls


def get_tool(name: str) -> Tool:
    """Instantiate a registered tool by name."""
    if name not in TOOL_REGISTRY:
        raise KeyError(
            f"Unknown tool '{name}'. Available: {sorted(TOOL_REGISTRY)}")
    return TOOL_REGISTRY[name]()


def available_tools() -> List[str]:
    """Names of all registered tools, sorted."""
    return sorted(TOOL_REGISTRY)


def gpu_tools() -> List[str]:
    """Names of tools whose declared :class:`Resource` targets the GPU.

    Replaces the hardcoded ``gpu_tools`` set in the engine: a tool opts in
    by declaring ``Resource(device='gpu')``.
    """
    return sorted(
        n for n, cls in TOOL_REGISTRY.items()
        if getattr(cls.resource, 'device', 'cpu') == 'gpu'
    )


# import tool implementations so the registry is populated
def _load_tools() -> None:
    import importlib
    for mod in ('ifgram_crop', 'generate_ifgram', 'complex_coh',
                'uniform', 'unwrap_stitch', 'atmosphere'):
        try:
            importlib.import_module(f'mintpy.stdproc.slc2ifg.engine.tools.{mod}')
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to load engine tool '%s': %s", mod, e)


_load_tools()
