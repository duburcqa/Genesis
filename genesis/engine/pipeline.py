"""
Composition of the simulation from Systems, which own Data and add stages to the phases of the Systems they depend on.

A System declares what it depends on as the typed parameters of its constructor, and the roles it plays as the role
classes it derives from. The simulator assembles the Systems a scene calls for (see 'assemble_systems'): it creates each
one once its dependencies exist, builds it once they are built, then routes the stage of each of its roles into the slot
the role fills. A System that depends on a System or a struct the scene never has is absent, and so are its stages. See
https://genesis-world.readthedocs.io/en/latest/user_guide/developers/composable_pipeline.html for the design.

A Phase is an ordered sequence of stages, split into slots. A stage is a kernel, a func whose loops run at the top
level, a kernel method of a data-oriented object, a Python function, an extension point of a kernel (see Inline) or
another Phase. Its arguments are bound by name to the Data of the System it belongs to and of the Systems it depends on,
and an argument that none of them provides is a call argument, given by name when the Phase runs.

A Phase fuses the kernels it runs between two Python functions into one kernel, generated from source, which calls the
body of each of them as a top-level func. While the launches are recorded for a backward pass, it launches every kernel
on its own instead, since the launch recorder of the differentiable simulation reverses one launch per entry (see
genesis/grad/bridge.py), whose tags (`no_grad`, `register_custom_backward`, `reversed_with`) apply as usual.
"""

import ast
import contextlib
import hashlib
import importlib.util
import inspect
import os
import sys
import textwrap
import types
import typing
from collections.abc import Callable, Iterator, Sequence
from contextlib import nullcontext
from typing import Any, ClassVar, NamedTuple

import numpy as np
import torch

import pydantic

import quadrants as qd
from quadrants.lang._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable

import genesis as gs
from genesis.engine.states.solvers import SystemCheckpoint
from genesis.grad.bridge import custom_backward, is_recording_launches
from genesis.grad.kernel_writes import get_written_paths
from genesis.utils.cache import get_cache_dir
from genesis.utils.data import (
    CHECKPOINT_KINDS,
    ArrayItem,
    ConfigItem,
    Data,
    DataItem,
    DataKind,
    ScalarItem,
    StaticConfig,
    fill_data,
    iter_data,
)
from genesis.utils.misc import data_to_array, has_zerocopy_view, qd_to_torch

# The type of the annotation of a compile-time parameter
_Template = type(qd.template())

# The System classes the simulator creates when a scene calls for them, in registration order (see register_system)
_SYSTEM_CLASSES: list[type["System"]] = []
# Every System class by name, which resolves the dependencies a module names as strings to break an import cycle
_SYSTEM_CLASSES_BY_NAME: dict[str, type["System"]] = {}

# The System whose stages are being added, which they bind the Data of, and the System offering the slot they fill
# (see '_get_scope')
_scope_system: "System | None" = None
_scope_slot_owner: "System | None" = None
# The funcs and kernels of the API of every System class, by their Python function (see 'System.api')
_API_FUNCS: dict[Callable, type["System"]] = {}


class BoundaryItems(NamedTuple):
    """The items a boundary checkpoint holds, listed once by the System (see 'System.save_checkpoint')."""

    # Every item the checkpoint holds
    items: list[DataItem]
    # The names of the arrays copied at once, in the order of their views
    names: list[str]
    # The zero-copy views of the arrays copied at once
    views: list[torch.Tensor]
    # The items copied one by one: the scalars, and the arrays without a view
    items_copied: list[DataItem]
    # The zero-copy views of the gradients of the arrays, which a backward pass checks at every step, the environment
    # first with a trailing axis that gives every view one to reduce over
    grads: list[torch.Tensor]
    # The items whose gradient has no view, copied at every check
    items_grad_copied: list[ArrayItem]


class System:
    """
    A unit of the simulation that owns Data, depends on other Systems and Data, and plays roles in their phases.

    Its Data are its public attributes. Its dependencies are the parameters of its constructor, typed with a System
    class to use the API of that System, or with a Data struct to read it alone, `X | None` for an optional one (see
    'assemble_systems'). It allocates the Data whose sizes the scene decides in `build`, once its dependencies are
    built. The roles it plays are the role classes it derives from, each routing one of its stages (see '_get_roles').
    """

    # Whether the System supports a scene with `requires_grad`, which every System of such a scene must
    is_differentiable: bool = True
    # Whether the System holds its state in frames, one per substep of a window of them, frame 0 holding the state
    # between two steps. The simulator copies the last frame into frame 0 at the end of each window, and a boundary
    # checkpoint holds frame 0 alone (see 'Solver.n_frames').
    has_frames: bool = False
    # The items and views of the boundary checkpoints, built by the first one, which comes after the build, and valid
    # from then on since the arrays are never reallocated, which spares walking the structs at every step
    _boundary_items: BoundaryItems | None = None
    # The slots the roles of other Systems, or the stages their owner adds to a slot, move to while this System exists,
    # None dropping them (see RoleOf)
    relocations: ClassVar[dict["RoleOf | Phase", "_SlotRef | Phase | None"]] = {}
    # The Systems and the structs its constructor depends on, whose Data its stages bind, the structs read-only
    _dependencies: "tuple[System, ...]" = ()
    _data_dependencies: ClassVar[dict[str, Data]] = {}
    # The instances of the template a System class is, by their arguments (see '__class_getitem__')
    _template_instances: ClassVar[dict[tuple[type, ...], type["System"]]] = {}
    # The arguments of the template instance a System class is, by the type parameters of its template
    template_args: ClassVar[dict[typing.TypeVar, type]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._template_instances = {}
        _SYSTEM_CLASSES_BY_NAME.setdefault(cls.__name__, cls)
        # A condition names a field of an options model or a property of a System, checked once the class exists
        for hint in inspect.get_annotations(vars(cls)["__init__"]).values() if "__init__" in vars(cls) else ():
            for condition in typing.get_args(hint)[1:] if typing.get_origin(hint) is typing.Annotated else ():
                if isinstance(condition, Enabled):
                    condition.check(typing.get_args(hint)[0])

    def __class_getitem__(cls, args: "type | tuple[type, ...]") -> "type[System]":
        """
        Return the System class instantiating the template for the given arguments, the same class for the same ones.

        A template is a System class generic over type parameters, `class GJK[LayoutT: Layout, CapabilityT:
        Capability](System)`, whose instances read their arguments by parameter, `self.template_args[LayoutT]`. Its
        instances are distinct System classes, which a scene holds side by side, each owning the state its arguments
        call for, and which a dependency names apart. They are created on demand when the template is.
        """
        args = args if isinstance(args, tuple) else (args,)
        # An argument left to a type parameter of an enclosing template is substituted where it is resolved
        if any([isinstance(arg, typing.TypeVar) for arg in args]):
            return super().__class_getitem__(args)
        params = cls.__type_params__
        if len(args) != len(params):
            raise TypeError(f"{cls.__name__}[...] takes the arguments {[param.__name__ for param in params]}.")
        if args not in cls._template_instances:
            name = f"{cls.__name__}[{', '.join([arg.__name__ for arg in args])}]"
            namespace = {"template_args": dict(zip(params, args)), "__module__": cls.__module__}
            cls._template_instances[args] = type(name, (cls,), namespace)
        return cls._template_instances[args]

    @classmethod
    def api(cls, *funcs: QuadrantsCallable) -> QuadrantsCallable:
        """
        Declare funcs or kernels the API of the System class, `@GJK.api` above `@qd.func`, and return the first one.

        A stage writes the Data of the System adding it and of the System offering the slot it fills. It changes the
        Data of any other System through the API of that System alone, which writes it unchecked: the funcs of the API
        it calls, or a kernel of the API it runs.
        """
        for func in funcs:
            _API_FUNCS[func.fn] = cls
        return funcs[0]

    def build(self) -> None:
        """Allocate the Data whose sizes the scene decides, once the Systems it depends on are built."""

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        """
        Yield every array and static config the System holds, named by the attributes reaching it (see iter_data).

        They are the structs and static configs its class annotates, in declaration order, those not built left out.
        """
        for name in _get_owned_attrs(type(self)):
            if vars(self).get(name) is not None:
                yield from iter_data(self, name)

    def save_checkpoint(self, kinds: frozenset[DataKind], *, boundary_only: bool = False) -> SystemCheckpoint:
        """
        Return a checkpoint of the arrays of the given kinds the System holds, for 'load_checkpoint' to restore.

        The arrays are what 'data' yields of those kinds, copied with their gradients where they live: on the device as
        torch tensors when zero-copy views exist, which keeps a save and a restore within one process off the host, and
        as numpy arrays otherwise. The static configs come along as provenance. A boundary checkpoint holds the state at
        the boundary of two steps alone, from which the backward pass of a recorded step replays it: no gradient, frame
        0 of the arrays holding frames, on the device whatever the backend.
        """
        if not boundary_only:
            arrays = {
                item.name: np.array(getattr(item.owner, item.attr))
                if isinstance(item, ScalarItem)
                else data_to_array(item.array, to_torch=gs.use_zerocopy.is_supported, copy=True)
                for item in self.data
                if isinstance(item, DataItem) and item.kind in kinds
            }
            configs = {item.name: item.config for item in self.data if isinstance(item, ConfigItem)}
            return SystemCheckpoint(arrays, configs, kinds)
        arrays = {}
        if self._boundary_items is None:
            # The gradients ('iter_data' names each one after its array) and the placeholders of the features
            # switched off (see 'ArrayItem') hold nothing a replay reads
            self._boundary_items = BoundaryItems(
                items=[], names=[], views=[], items_copied=[], grads=[], items_grad_copied=[]
            )
            grad_names = set()
            for item in self.data:
                if isinstance(item, ConfigItem) or item.kind not in kinds:
                    continue
                if isinstance(item, ArrayItem):
                    is_placeholder = not isinstance(item.array, torch.Tensor) and not item.array.shape
                    if item.name in grad_names or is_placeholder:
                        continue
                    if not isinstance(item.array, torch.Tensor) and item.array.has_grad():
                        grad_names.add(f"{item.name}.grad")
                        if has_zerocopy_view(item.array.grad):
                            grad = qd_to_torch(item.array.grad, transpose=True, copy=False)
                            self._boundary_items.grads.append(grad[..., None])
                        else:
                            self._boundary_items.items_grad_copied.append(item)
                self._boundary_items.items.append(item)
                if isinstance(item, ScalarItem) or not gs.use_zerocopy.is_supported:
                    self._boundary_items.items_copied.append(item)
                    continue
                if isinstance(item.array, torch.Tensor):
                    view = item.array
                elif not has_zerocopy_view(item.array):
                    self._boundary_items.items_copied.append(item)
                    continue
                elif self.has_frames:
                    view = qd_to_torch(item.array, row_mask=0, keepdim=False, copy=False)
                else:
                    view = qd_to_torch(item.array, copy=False)
                self._boundary_items.names.append(item.name)
                self._boundary_items.views.append(view)
        if self._boundary_items.views:
            # One fused copy of every array with a view, rather than one launch each
            copies = [torch.empty_like(view) for view in self._boundary_items.views]
            # The views read the arrays as the last kernel left them once it is done (see qd_to_torch)
            if gs.backend == gs.metal:
                qd.sync()
            torch._foreach_copy_(copies, self._boundary_items.views)
            # The copy reads the arrays on the torch stream, before the next kernel writes them (see qd_zero_grad)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            arrays.update(zip(self._boundary_items.names, copies))
        items = self._boundary_items.items_copied
        for item in items:
            if isinstance(item, ScalarItem):
                arrays[item.name] = np.array(getattr(item.owner, item.attr))
            elif self.has_frames:
                arrays[item.name] = qd_to_torch(item.array, row_mask=0, keepdim=False, copy=True)
            else:
                arrays[item.name] = data_to_array(item.array, to_torch=True, copy=True)
        # The static configs are provenance, which a boundary checkpoint never leaving the process does without
        return SystemCheckpoint(arrays=arrays, configs={}, kinds=kinds)

    def __getstate__(self) -> SystemCheckpoint:
        """Return a SystemCheckpoint of the System, for '__setstate__' to restore (see 'CHECKPOINT_KINDS')."""
        return self.save_checkpoint(CHECKPOINT_KINDS)

    def __setstate__(self, state: SystemCheckpoint) -> None:
        """Write back the arrays of the kinds a record holds, rejecting a record read from another build."""
        self.load_checkpoint(state)

    def load_checkpoint(self, checkpoint: SystemCheckpoint, *, boundary_only: bool = False) -> None:
        """
        Write back the arrays of a checkpoint of 'save_checkpoint', rejecting a record read from another build.

        `boundary_only` states that 'save_checkpoint' saved a boundary checkpoint. Nothing is notified to the
        subscribers of a solver (see 'mutates'). A System whose step reads a DERIVED array before recomputing it
        overrides this to recompute or invalidate that array when the record lacks DERIVED (see 'DataKind'). On Metal,
        the caller synchronizes the torch stream before the next kernel (see 'fill_data').
        """
        if not boundary_only:
            fill_data(
                [item for item in self.data if isinstance(item, DataItem) and item.kind in checkpoint.kinds],
                checkpoint.arrays,
            )
            return
        # The arrays with a view are written back in one fused copy, from the copies 'save_checkpoint' made of them in
        # this process, which spares checking them against the scene
        if self._boundary_items.views:
            # The writes wait for the last kernel to be done with the arrays (see qd_to_torch)
            if gs.backend == gs.metal:
                qd.sync()
            values = [checkpoint.arrays[name] for name in self._boundary_items.names]
            torch._foreach_copy_(self._boundary_items.views, values)
        items = self._boundary_items.items_copied
        fill_data(items, {item.name: checkpoint.arrays[item.name] for item in items}, frame_only=self.has_frames)


def register_system(cls: type[System]) -> type[System]:
    """Declare a System the simulator creates whenever a scene provides every dependency of its constructor."""
    _SYSTEM_CLASSES.append(cls)
    return cls


class _SlotRef(NamedTuple):
    """A slot of a phase, or an extension point, that a System class offers, by attribute and slot name."""

    owner: type
    attr: str
    slot: str | None = None


class RoleOf(NamedTuple):
    """A role a System class plays, which another System relocates (see 'System.relocations')."""

    system: type
    role: type


# The entry point of a role: one stage or several, run in order (see '_get_roles')
type Stages = Callable | tuple[Callable, ...]


def _get_roles(cls: type) -> list[type]:
    """
    Return the roles a System class plays: the classes it derives from that declare a slot.

    A role is a System class declaring the slot it fills, `slot = Collider.detection.narrowphase`, and annotating the
    entry point a System playing it provides, `detect: Stages`: a method, a property, a kernel, a func, or a tuple of
    them run in order. The stages of every System playing the role are routed into the slot (see 'assemble_systems').
    """
    return [base for base in cls.__mro__ if "slot" in vars(base)]


def _get_stages(system: System, role: type) -> tuple[Any, ...]:
    """
    Return the stages of the entry point of a role a System plays, its methods bound, its properties evaluated and its
    phases those of the System.
    """
    (entry,) = inspect.get_annotations(role)
    value = inspect.getattr_static(system, entry)
    if isinstance(value, property):
        value = value.fget(system)
    elif isinstance(value, types.FunctionType):
        value = value.__get__(system)
    elif isinstance(value, (Phase, Inline)):
        value = _get_declared(system, entry)
    return value if isinstance(value, tuple) else (value,)


class Enabled:
    """
    Make a System exist only while flags of one of its dependencies hold, the metadata of its annotation.

    `options: Annotated[RigidOptions, Enabled("enable_collision")]` reads a field of an options model, and
    `ipc: Annotated[IPCCoupler, Enabled("has_any_rigid_coupling")]` a property of a System, read once that System is
    built. A name given a value compares with it, `Enabled(contact_depth_query="sdf")`.
    """

    def __init__(self, *flags: str, **values: Any) -> None:
        self.conditions = {**dict.fromkeys(flags, True), **values}

    def check(self, owner: type) -> None:
        """Refuse a name that is neither a field of the options model nor a property of the System class."""
        for name in self.conditions:
            is_field = issubclass(owner, pydantic.BaseModel) and name in owner.model_fields
            if not (is_field or isinstance(inspect.getattr_static(owner, name, None), property)):
                raise TypeError(f"Enabled('{name}') names no field or property of {owner.__name__}.")

    def holds(self, dependency: object) -> bool:
        """Return whether every condition holds for the given dependency."""
        for name, value in self.conditions.items():
            attr = inspect.getattr_static(dependency, name)
            actual = attr.fget(dependency) if isinstance(attr, property) else attr
            if (bool(actual) if value is True else actual) != value:
                return False
        return True


class _CallArg(NamedTuple):
    """A parameter that no Data provides, given by name when the Phase runs instead."""

    name: str


class _Launch(NamedTuple):
    """
    A kernel fusing a run of kernel stages, or a Python stage, with its arguments.

    An argument is its bound value, or the _CallArg naming it, whose positions `call_args` lists.
    """

    fn: Callable
    args: tuple[Any, ...]
    call_args: tuple[tuple[int, str], ...]


class _FusedStage(NamedTuple):
    """
    A kernel stage of a fused kernel: the kernel, and the names its parameters take in the fused kernel, in order.

    The first parameter of a kernel method (`is_method`) is its object.
    """

    kernel: QuadrantsCallable
    names: tuple[str, ...]
    is_method: bool


def _make_fused_kernel(name: str, stages: Sequence[_FusedStage], is_graph: bool) -> tuple[Callable, tuple[str, ...]]:
    """
    Generate the kernel calling the bodies of the given kernels in order, and return it with its parameter names.

    Its parameters are those of the kernels, each name once, with the annotation it has in every kernel taking it, the
    object of a kernel method being a compile-time parameter. The source is written to the cache directory of Genesis
    and imported from there: the fastcache of Quadrants reads the source of a kernel from its file, and a file named
    after the hash of its content stays valid across processes. The kernel is defined in that module, so that every
    scene fusing the same kernels shares it and its compiled variants.
    """
    # The annotation of every parameter, as an object to compare and as the source evaluating to it
    annotations: dict[str, tuple[Any, str]] = {}
    lines, calls = ["import inspect", "", "import quadrants as qd", ""], []
    for i, (kernel, names, is_method) in enumerate(stages):
        # A kernel method is reached through its class, at module level
        owner_name, *path = kernel.fn.__qualname__.split(".")
        assert "<locals>" not in path, "A kernel stage must be defined at the top level of its module."
        lines.append(f"from {kernel.fn.__module__} import {owner_name} as kernel_{i}")
        lines.append(f"fn_{i} = kernel_{i}{''.join([f'.{attr}' for attr in path])}.fn")
        lines.append(f"func_{i} = qd.func(requires_top_level=True)(fn_{i})")
        for i_param, (name_, parameter) in enumerate(zip(names, inspect.signature(kernel.fn).parameters.values())):
            if is_method and i_param == 0:
                annotation = (_Template, "qd.template()")
            else:
                source = f"inspect.signature(fn_{i}).parameters['{parameter.name}'].annotation"
                annotation = (parameter.annotation, source)
            if name_ in annotations:
                # FIXME: quadrants#1020 - a func ignores the class 'qd.Template' as annotation, so kernels keep the
                # 'qd.template()' instances, which never compare equal and are compared by type instead.
                annotation_other = annotations[name_][0]
                assert (
                    type(annotation_other) is type(annotation[0])
                    if isinstance(annotation_other, _Template)
                    else annotation_other == annotation[0]
                ), f"Parameter '{name_}' of '{kernel.fn.__qualname__}' is annotated otherwise in another kernel."
            annotations.setdefault(name_, annotation)
        calls.append(f"    func_{i}({', '.join(names)})")
    lines += [f"{name_}_annotation = {source}" for name_, (_, source) in annotations.items()]
    signature = ", ".join([f"{name_}: {name_}_annotation" for name_ in annotations])
    # Fastcache takes no object as argument, and a kernel left out of it stays out once fused, funcs having no say
    is_fastcached = all(
        [not stage_.is_method and (stage_.kernel.is_pure or not stage_.kernel._is_wrapped_kernel) for stage_ in stages]
    )
    decorator = f"@qd.kernel(graph={is_graph}, fastcache={is_fastcached})"
    lines += ["", "", decorator, f"def {name}({signature}):", *calls, ""]
    source = "\n".join(lines)

    return _load_generated(source, name), tuple(annotations)


def _load_generated(source: str, name: str) -> Any:
    """
    Return the object of the given name a generated source defines, written to the cache directory and imported from
    there.

    The fastcache of Quadrants reads the source of a kernel from its file, and a file named after the hash of its
    content stays valid across processes. The module is imported once per process.
    """
    path = os.path.join(get_cache_dir(), "pipeline", f"{hashlib.sha1(source.encode()).hexdigest()[:16]}.py")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Written aside then moved, so that a concurrent process never imports a partial file
        path_tmp = f"{path}.{os.getpid()}"
        with open(path_tmp, "w") as file:
            file.write(source)
        os.replace(path_tmp, path)
    module_name = f"genesis_pipeline_{os.path.basename(path)[:-3]}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return vars(module)[name]


class Phase:
    """
    An ordered sequence of stages, which its System runs, split into the `slots` that roles fill.

    Declared on a System class, `detection = Phase("broadphase", "narrowphase", "pruning")`, it names the slots,
    `Collider.detection.narrowphase`, and every System of the class runs a Phase of its own, `is_graph` being then a
    flag or a function of the System.

    The kernels of a run are fused into one kernel between two Python stages, and captured as a graph with `is_graph`,
    which a kernel holding a device loop (`qd.graph.do_while`) requires. A nested Phase runs inline, fused with the
    stages around it, unless it `is_launched_apart`, its own launches then bounding the runs around it. While the
    launches are recorded for a backward pass, every stage runs on its own instead, all of them reversed as one by
    `backward` when given (see custom_backward).
    """

    def __init__(
        self,
        *slots: str,
        is_graph: "bool | Callable[[System], bool]" = False,
        is_launched_apart: bool = False,
        backward: Callable[[], None] | None = None,
    ) -> None:
        self.is_graph = is_graph
        self.is_launched_apart = is_launched_apart
        self.backward = backward
        # The phases of the slots, which run in declaration order
        self.slots = {name: Phase() for name in slots}
        # The stages, in order, with their arguments
        self.items: list[tuple[Callable, tuple[Any, ...]]] = [(phase, ()) for phase in self.slots.values()]
        self._launches: tuple[_Launch, ...] | None = None
        # The System class declaring the Phase and its attribute, for a declaration (see '__set_name__')
        self.owner: type | None = None
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        # A role naming the phase as its slot assigns it again, which leaves its owner
        if self.owner is None:
            self.owner, self.name = owner, name

    def __getattr__(self, slot: str) -> _SlotRef:
        if slot not in vars(self).get("slots", {}):
            raise AttributeError(f"The phase has no slot '{slot}'.")
        return _SlotRef(self.owner, self.name, slot)

    def __get__(self, system: "System | None", owner: type) -> "Phase":
        if system is None:
            return self
        is_graph = self.is_graph(system) if callable(self.is_graph) else self.is_graph
        phase = Phase(*self.slots, is_graph=is_graph, is_launched_apart=self.is_launched_apart, backward=self.backward)
        vars(system)[self.name] = phase
        return phase

    def add(self, *stages: Callable) -> None:
        """Append stages, binding their arguments to the Data of the System they belong to and of its dependencies."""
        scope, writable_ids = _get_scope()
        for stage in stages:
            if isinstance(stage, (Phase, Inline)):
                self.items.append((stage, ()))
                continue
            if isinstance(stage, BoundQuadrantsCallable):
                names = list(inspect.signature(stage.fn).parameters)[1:]
            elif isinstance(stage, QuadrantsCallable):
                names = list(inspect.signature(stage.fn).parameters)
            else:
                names = list(inspect.signature(stage).parameters)
            args = []
            for name in names:
                args.append(_bind(scope, name))
            if isinstance(stage, (QuadrantsCallable, BoundQuadrantsCallable)):
                owner = stage.instance if isinstance(stage, BoundQuadrantsCallable) else None
                _check_writes(stage.fn, owner, names, args, writable_ids)
            # A func, whose loops run in parallel at the top level of a kernel, runs through the kernel generated around
            # it
            if isinstance(stage, QuadrantsCallable) and not stage._is_wrapped_kernel:
                kernel_name = f"kernel_{stage.fn.__name__.removeprefix('func_')}"
                stage, _ = _make_fused_kernel(kernel_name, [_FusedStage(stage, tuple(names), False)], is_graph=False)
            self.items.append((stage, tuple(args)))
        self._launches = None

    def get_items(self) -> list[tuple[Callable, tuple[Any, ...]]]:
        """Return the stages the Phase runs, its nested Phases and slots unrolled, but those launched apart."""
        return [
            item
            for stage, args in self.items
            for item in (
                stage.get_items() if isinstance(stage, Phase) and not stage.is_launched_apart else [(stage, args)]
            )
        ]

    def get_launches(self) -> tuple[_Launch, ...]:
        """Return the launches running the Phase, the kernel stages between two Python stages fused."""
        if self._launches is not None:
            return self._launches
        # The kernels and Python stages to launch, with their arguments
        calls: list[tuple[Callable, tuple[Any, ...]]] = []
        run: list[tuple[_FusedStage, tuple[Any, ...]]] = []
        # The name of each object of the run in the fused kernel, by parameter name and identity: objects of one name,
        # such as the states of two instances of a template, are numbered apart
        fused_names: dict[tuple[str, int], str] = {}
        for item, args in [*self.get_items(), (None, ())]:
            if isinstance(item, (QuadrantsCallable, BoundQuadrantsCallable)):
                names = list(inspect.signature(item.fn).parameters)
                is_method = isinstance(item, BoundQuadrantsCallable)
                if is_method:
                    names[0] = "owner"
                    item, args = item.quadrants_callable, (item.instance, *args)
                for i, (name, arg) in enumerate(zip(names, args)):
                    if not isinstance(arg, _CallArg):
                        n_named = len([key for key in fused_names if key[0] == name])
                        names[i] = fused_names.setdefault((name, id(arg)), f"{name}_{n_named}" if n_named else name)
                run.append((_FusedStage(item, tuple(names), is_method), args))
                continue
            if run:
                stages = [stage_ for stage_, _ in run]
                kernel, names = _make_fused_kernel(f"fused_{stages[0].kernel.fn.__name__}", stages, self.is_graph)
                args_by_name = {n: a for stage_, args_ in run for n, a in zip(stage_.names, args_)}
                calls.append((kernel, tuple([args_by_name[name] for name in names])))
                run.clear()
                fused_names.clear()
            if item is not None:
                calls.append((item, args))
        self._launches = tuple(
            [
                _Launch(fn, args, tuple([(i, arg.name) for i, arg in enumerate(args) if isinstance(arg, _CallArg)]))
                for fn, args in calls
            ]
        )
        return self._launches

    def __call__(self, **values: Any) -> None:
        """Run the stages, in order, with the given values of their call arguments."""
        if not self.get_launches():
            return
        if not is_recording_launches():
            for launch in self.get_launches():
                if isinstance(launch.fn, (Inline, Phase)):
                    launch.fn(**values)
                    continue
                args = launch.args
                if launch.call_args:
                    args = list(args)
                    for i, name in launch.call_args:
                        args[i] = values[name]
                launch.fn(*args)
            return
        with custom_backward(self.backward) if self.backward is not None else nullcontext():
            for stage, args in self.items:
                if isinstance(stage, (Phase, Inline)):
                    stage(**values)
                else:
                    stage(*[values[arg.name] if isinstance(arg, _CallArg) else arg for arg in args])


class Inline:
    """
    An extension point of a kernel: the funcs the Systems add run in turn where the kernel calls a placeholder func.

    The kernel calls the placeholder `hook` with the values the extension point hands over, `x = hook(a, b, x)`, and the
    placeholder returns the last of them unchanged. Every func added takes the same leading parameters, then the Data
    it reads, bound by name to the Data of the System adding it and of its dependencies like the stages of a Phase. The
    kernel launched is the kernel itself while no func is added, and otherwise a kernel generated from its source,
    whose call to the placeholder is replaced by the calls of the funcs in turn, taking their Data as parameters of
    their own. The values handed over stay in registers, which the autodiff of the kernel reverses as a whole, unlike a
    sequence of kernels updating the same array in place. The other parameters of the kernel are bound to the Data of
    the System declaring the extension point once it is built, and those that none provides are call arguments, given
    by name when it runs. A parameter of a func that the scope of its System leaves unbound takes the value the kernel
    binds to that name.

    Declared on a System class, `grid_couplings = Inline(kernel_mpm_grid_op, func_mpm_grid_couplings)`, it is the slot
    that roles name, `MPMSolver.grid_couplings`, and every System of the class runs an extension point of its own.

    `run_without_funcs` false skips the kernel while no func is added, for a kernel doing nothing but hand the values
    over: its writes of the values it read back would race on their adjoints in a parallel reverse pass.
    """

    def __init__(self, kernel: QuadrantsCallable, hook: Callable, *, run_without_funcs: bool = True) -> None:
        self.kernel = kernel
        self.hook = hook
        self._is_run_without_funcs = run_without_funcs
        # The funcs added, in order, with the arguments of their Data parameters
        self.items: list[tuple[Callable, tuple[Any, ...]]] = []
        self._args: tuple[Any, ...] | None = None
        self._launch: tuple[Callable, tuple[Any, ...]] | None = None
        # The System class declaring the extension point and its attribute, for a declaration (see '__set_name__')
        self.owner: type | None = None
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        # A role naming the extension point as its slot assigns it again, which leaves its owner
        if self.owner is None:
            self.owner, self.name = owner, name

    def __get__(self, system: "System | None", owner: type) -> "Inline":
        if system is None:
            return self
        inline = Inline(self.kernel, self.hook, run_without_funcs=self._is_run_without_funcs)
        vars(system)[self.name] = inline
        return inline

    def bind(self) -> None:
        """Bind the parameters of the kernel to the Data of the System declaring the extension point, once built."""
        scope, writable_ids = _get_scope()
        names = list(inspect.signature(self.kernel.fn).parameters)
        self._args = tuple([_bind(scope, name) for name in names])
        _check_writes(self.kernel.fn, None, names, self._args, writable_ids)

    def add(self, *funcs: Callable) -> None:
        """Append funcs, binding their Data parameters to the Data of the System composing and of its dependencies."""
        scope, writable_ids = _get_scope()
        n_handed = len(inspect.signature(self.hook).parameters)
        for func in funcs:
            names = list(inspect.signature(func.fn).parameters)[n_handed:]
            args = tuple([_bind(scope, name) for name in names])
            _check_writes(func.fn, None, names, args, writable_ids)
            self.items.append((func, args))
        self._launch = None

    def get_launch(self) -> tuple[Callable, tuple[Any, ...]]:
        """Return the kernel running the extension point with its arguments, generated once the funcs are added."""
        assert self._args is not None, "An extension point is bound before it runs."
        if self._launch is None:
            if not self.items:
                self._launch = (self.kernel, self._args)
            else:
                self._launch = self._make_kernel()
        return self._launch

    def _make_kernel(self) -> tuple[Callable, tuple[Any, ...]]:
        fn = self.kernel.fn
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        funcdef = tree.body[0]
        funcdef.decorator_list = []
        funcdef.name = f"{fn.__name__}_inline"
        lines = ["import inspect", "import sys", "", "import quadrants as qd", ""]
        # The names of the module of the kernel, which its body refers to
        lines.append(f"globals().update(vars(sys.modules[{fn.__module__!r}]))")
        args_by_name = dict(zip(inspect.signature(fn).parameters, self._args))
        # The name of each object in the kernel generated, by parameter name and identity: objects of one name, such as
        # the info structs of two couplings, are numbered apart (see 'Phase.get_launches')
        bound_names = {(name, id(arg)): name for name, arg in args_by_name.items() if not isinstance(arg, _CallArg)}
        params_new: list[ast.arg] = []
        calls_data: list[list[str]] = []
        for i, (func, args) in enumerate(self.items):
            owner_name, *path = func.fn.__qualname__.split(".")
            assert "<locals>" not in path, "An extension point func must be defined at the top level of its module."
            lines.append(f"from {func.fn.__module__} import {owner_name} as func_module_{i}")
            lines.append(f"func_{i} = func_module_{i}{''.join([f'.{attr}' for attr in path])}")
            names = list(inspect.signature(func.fn).parameters)[-len(args) :] if args else []
            names_bound = []
            for name, arg in zip(names, args):
                # A parameter that the scope of the func leaves unbound takes the value the kernel binds to its name
                if isinstance(arg, _CallArg):
                    name_bound = name
                else:
                    n_named = len([key for key in bound_names if key[0] == name])
                    name_bound = bound_names.setdefault((name, id(arg)), f"{name}_{n_named}" if n_named else name)
                names_bound.append(name_bound)
                if name_bound not in args_by_name:
                    args_by_name[name_bound] = arg
                    annotation = f"inspect.signature(func_{i}.fn).parameters['{name}'].annotation"
                    lines.append(f"{name_bound}_annotation = {annotation}")
                    annotation_node = ast.Name(id=f"{name_bound}_annotation", ctx=ast.Load())
                    params_new.append(ast.arg(arg=name_bound, annotation=annotation_node))
            calls_data.append(names_bound)
        funcdef.args.args += params_new

        hook_name = self.hook.__name__
        items = self.items

        class ReplaceHook(ast.NodeTransformer):
            def visit_Assign(self, node: ast.Assign) -> Any:
                call = node.value
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == hook_name:
                    return [
                        ast.Assign(
                            targets=node.targets,
                            value=ast.Call(
                                func=ast.Name(id=f"func_{i}", ctx=ast.Load()),
                                args=[*call.args, *[ast.Name(id=name, ctx=ast.Load()) for name in calls_data[i]]],
                                keywords=[],
                            ),
                        )
                        for i in range(len(items))
                    ]
                return node

        tree = ast.fix_missing_locations(ReplaceHook().visit(tree))
        lines += ["", "", f"@qd.kernel(fastcache={self.kernel.is_pure})", ast.unparse(funcdef), ""]
        kernel = _load_generated("\n".join(lines), funcdef.name)
        return kernel, tuple(args_by_name.values())

    def __call__(self, **values: Any) -> None:
        """Run the kernel with the given values of its call arguments."""
        if not self.items and not self._is_run_without_funcs:
            return
        kernel, args = self.get_launch()
        kernel(*[values[arg.name] if isinstance(arg, _CallArg) else arg for arg in args])


def _check_writes(
    fn: Callable, owner: object | None, names: Sequence[str], args: Sequence[Any], writable_ids: set[int]
) -> None:
    """Refuse a kernel or a func writing Data of another System than through its API (see 'System.api')."""
    if fn in _API_FUNCS:
        return
    written_names = {path[0] for path in get_written_paths(fn, owner, frozenset(_API_FUNCS))}
    for name, arg in zip(names, args):
        if name in written_names and not isinstance(arg, _CallArg) and id(arg) not in writable_ids:
            raise TypeError(
                f"'{fn.__qualname__}' writes '{name}', which {type(_scope_system).__name__} changes through the API of "
                "its owner alone."
            )


def _bind(scope: dict[str, list[Any]], name: str) -> Any:
    """Return the Data of the given name in a scope, the _CallArg naming it if none (see '_get_scope')."""
    values = scope.get(name, ())
    assert len(values) < 2, f"Data '{name}' is ambiguous among the dependencies of the System."
    return values[0] if values else _CallArg(name)


class _Dependency(NamedTuple):
    """A parameter of the constructor of a System: the classes it takes, whether it is optional, its conditions."""

    name: str
    classes: tuple[type, ...]
    is_optional: bool
    conditions: tuple[Enabled, ...]


def _get_dependencies(cls: type[System]) -> list[_Dependency]:
    """Return the dependencies a System class declares."""
    hints = typing.get_type_hints(cls.__init__, localns=_SYSTEM_CLASSES_BY_NAME, include_extras=True)
    dependencies = []
    for name in inspect.signature(cls).parameters:
        hint, conditions = hints[name], ()
        if typing.get_origin(hint) is typing.Annotated:
            metadata = hint.__metadata__
            hint = hint.__origin__
            # A condition compared with a type parameter of the owner compares with its argument
            conditions = tuple(
                [
                    Enabled(**{name: cls.template_args.get(value, value) for name, value in meta.conditions.items()})
                    for meta in metadata
                    if isinstance(meta, Enabled)
                ]
            )
        hint_args = typing.get_args(hint)
        is_optional = typing.get_origin(hint) in (typing.Union, types.UnionType) and type(None) in hint_args
        classes = tuple([arg for arg in hint_args if arg is not type(None)]) if is_optional else (hint,)
        # A template instance named with the type parameters of the owner takes the arguments the owner was given
        classes = tuple(
            [
                typing.get_origin(c)[tuple([cls.template_args.get(arg, arg) for arg in typing.get_args(c)])]
                if typing.get_origin(c) is not None
                else c
                for c in classes
            ]
        )
        dependencies.append(_Dependency(name, classes, is_optional, conditions))
    return dependencies


def _is_component(cls: type) -> bool:
    """
    Return whether a System class is a component: playing no role and unregistered, created on demand.

    A component exists while some System depends on it, and is built once every System that could use it is decided.
    """
    if not (isinstance(cls, type) and issubclass(cls, System)):
        return False
    return not _get_roles(cls) and cls not in _SYSTEM_CLASSES


def _get_owned_attrs(cls: type) -> dict[str, type]:
    """
    Return the attributes a System class declares it owns, by name: its Data structs and static configs.

    A System declares them as the annotations of its class attributes, `dyn_state: DynState`, in declaration order,
    those of its bases first.
    """
    owned = {}
    for base in reversed(cls.__mro__):
        for name, hint in inspect.get_annotations(base).items():
            for arg in typing.get_args(hint) or (hint,):
                if isinstance(arg, type) and issubclass(arg, (Data, StaticConfig)):
                    owned[name] = arg
    return owned


@contextlib.contextmanager
def _scoped(system: System, slot_owner: System | None = None) -> Iterator[None]:
    """Bind the stages added within the block to the Data of the System and of what it depends on (see '_get_scope')."""
    global _scope_system, _scope_slot_owner

    _scope_system, _scope_slot_owner = system, slot_owner
    try:
        yield
    finally:
        _scope_system, _scope_slot_owner = None, None


def _get_scope() -> tuple[dict[str, list[Any]], set[int]]:
    """
    Return the Data the stages being added bind by name, and the identities of those they may write.

    The scope holds the public attributes of the System and of the Systems it depends on, as they are when a stage is
    added, and the structs it depends on as Data under the name of their parameter. The stages write the Data of the
    System and of the System offering their slot, but never the Data another System declares it owns (see
    'System.api').
    """
    system = _scope_system
    assert system is not None, "A stage is added within the scope of its System (see 'assemble_systems')."
    scope: dict[str, list[Any]] = {}
    for dependency in (system, *system._dependencies):
        for name, value in vars(dependency).items():
            if not name.startswith("_") and all([value is not other for other in scope.get(name, ())]):
                scope.setdefault(name, []).append(value)
    for name, struct in system._data_dependencies.items():
        if all([struct is not other for other in scope.get(name, ())]):
            scope.setdefault(name, []).append(struct)
    units = [unit for unit in (system, _scope_slot_owner) if unit is not None]
    # The Data another System of the scope declares it owns stays read-only, even held by a System of the unit
    owned_ids = {
        id(vars(other).get(name))
        for other in (system, *system._dependencies)
        if all([other is not unit for unit in units])
        for name in _get_owned_attrs(type(other))
    }
    writable_ids = {
        id(value)
        for unit in units
        for name, value in vars(unit).items()
        if not name.startswith("_") and id(value) not in owned_ids
    }
    return scope, writable_ids


def assemble_systems(
    systems: Sequence[System],
    systems_built: Sequence[System],
    context: Sequence[object] = (),
    requested: Sequence[type[System]] = (),
) -> list[System]:
    """
    Create and build the Systems a scene calls for, then route their roles and return them.

    `systems` are the Systems the scene holds already, `systems_built` those among them built elsewhere, `context` the
    other objects a dependency may name, such as the options of the scene, and `requested` the classes requested from
    outside the pipeline, decided as if a System depended on them.

    A registered System is created once every System it depends on exists, every struct it depends on is built, and
    every condition on them holds, and is absent once one of them never will, the reason being logged. A component
    exists once an existing System depends on it (see '_is_component'). A System is built once the Systems it depends
    on are built and every System that could still use it is decided, so that every request they make of it in their
    constructor is in, those needing its build aside (its Data, a condition on it). Its extension points
    then bind their kernel, and the stage of each role is routed into its slot, in the order of the Systems: those
    given, then those created in registration order, then the components.
    """
    systems = list(systems)
    # A System the scene gives depends on the Systems its public attributes hold, which its stages bind
    for system in systems:
        system._dependencies = tuple(
            [value for name, value in vars(system).items() if not name.startswith("_") and isinstance(value, System)]
        )
    built_ids = {id(system) for system in systems_built}
    # A dependency outside the scene, such as a solver holding no entity, is built by the scene itself
    built_ids |= {
        id(dependency)
        for system in systems
        for dependency in system._dependencies
        if all([dependency is not other for other in systems])
    }
    classes_pending = [cls for cls in _SYSTEM_CLASSES if all([type(system) is not cls for system in systems])]
    classes_pending += [cls for cls in requested if cls not in classes_pending]
    dependencies_by_cls: dict[type, list[_Dependency]] = {}

    def decide(cls: type[System]) -> "dict[str, Any] | str | None":
        """Return the arguments of a pending class to create, why it is absent, or None while it waits."""
        # The values a dependency may take, and the classes that may still provide one: the pending classes, the
        # Systems not built yet, and what they own
        values = [
            *systems,
            *context,
            *[
                value
                for system in systems
                if id(system) in built_ids
                for name, value in vars(system).items()
                if name[0] != "_"
            ],
        ]
        classes_coming = [
            owned
            for owner in (*classes_pending, *[type(system) for system in systems if id(system) not in built_ids])
            for owned in (owner, *_get_owned_attrs(owner).values())
        ]
        args = {}
        if cls not in dependencies_by_cls:
            dependencies_by_cls[cls] = _get_dependencies(cls)
        for dependency in dependencies_by_cls[cls]:
            candidates = list({id(value): value for value in values if isinstance(value, dependency.classes)}.values())
            if len(candidates) > 1:
                raise TypeError(f"Dependency '{dependency.name}' of {cls.__name__} is ambiguous: {candidates}.")
            if not candidates:
                components = [c for c in dependency.classes if not dependency.is_optional and _is_component(c)]
                classes_pending.extend([c for c in components if c not in classes_pending])
                # A class waits at the first dependency still undecided, before requiring a component it may not use
                if components or any([issubclass(c, dependency.classes) for c in classes_coming]):
                    return None
                if not dependency.is_optional:
                    return f"no {' or '.join([c.__name__ for c in dependency.classes])} for '{dependency.name}'"
                args[dependency.name] = None
                continue
            (value,) = candidates
            # A condition on a System is read once that System is built
            if dependency.conditions and isinstance(value, System) and id(value) not in built_ids:
                return None
            if not all([condition.holds(value) for condition in dependency.conditions]):
                if not dependency.is_optional:
                    return f"a condition on '{dependency.name}' does not hold"
                value = None
            args[dependency.name] = value
        return args

    while True:
        classes_decided = list(classes_pending)
        is_progressing = False
        for cls in classes_decided:
            decision = decide(cls)
            if decision is None:
                continue
            classes_pending.remove(cls)
            is_progressing = True
            if isinstance(decision, str):
                gs.logger.debug(f"{cls.__name__} is absent: {decision}.")
                continue
            system = cls(**decision)
            system._dependencies = tuple([arg for arg in decision.values() if isinstance(arg, System)])
            system._data_dependencies = {name: arg for name, arg in decision.items() if isinstance(arg, Data)}
            systems.append(system)
        for system in [system for system in systems if id(system) not in built_ids]:
            # A pending class needing what a build provides (the Data of this System, a condition on a System
            # not built yet) is decided after that build, so it makes no request of this one in its constructor
            owned = tuple(_get_owned_attrs(type(system)).values())
            types_built = [type(other) for other in systems if id(other) in built_ids]
            is_used_pending = False
            for cls in classes_pending:
                if cls not in dependencies_by_cls:
                    dependencies_by_cls[cls] = _get_dependencies(cls)
                dependencies = dependencies_by_cls[cls]
                if any(
                    [
                        issubclass(c, owned)
                        or (d.conditions and issubclass(c, System) and not any([issubclass(t, c) for t in types_built]))
                        for d in dependencies
                        for c in d.classes
                    ]
                ):
                    continue
                is_used_pending |= any(
                    [issubclass(type(system), c) for d in dependencies for c in d.classes if issubclass(c, System)]
                )
            if is_used_pending or not all([id(dependency) in built_ids for dependency in system._dependencies]):
                continue
            with _scoped(system):
                system.build()
            built_ids.add(id(system))
            is_progressing = True
        # A component joins the pending classes while they are decided
        if not is_progressing and all([cls in classes_decided for cls in classes_pending]):
            break
    unbuilt = [type(system).__name__ for system in systems if id(system) not in built_ids]
    if classes_pending or unbuilt:
        raise TypeError(
            f"The declarations of {[cls.__name__ for cls in classes_pending]} and {unbuilt} contradict one another: "
            "each waits for another one."
        )

    # The Systems in an order that does not depend on when they were created (the sort is stable)
    registration = {cls: i for i, cls in enumerate(_SYSTEM_CLASSES)}
    systems.sort(
        key=lambda system: (
            (1, registration[type(system)]) if type(system) in registration else (2 * _is_component(type(system)), 0)
        )
    )
    # The extension points a System declares bind their kernel to its Data, whether or not a role fills them. A role
    # holds the extension point it fills as its slot, which its owner alone declares.
    for system in systems:
        for base in type(system).__mro__:
            for name, value in vars(base).items():
                if isinstance(value, Inline) and value.owner is base and value.name == name:
                    with _scoped(system):
                        _get_declared(system, name).bind()
    _route_roles(systems)
    return systems


def _route_roles(systems: Sequence[System]) -> None:
    """Add the stage of every role of every System to its slot, or to the slot another System relocates it to."""
    # The stages the owner of a slot adds to it are moved or dropped by a System relocating the slot, before the roles
    # fill it
    for system in systems:
        for slot, target in type(system).relocations.items():
            if not isinstance(slot, Phase):
                continue
            owner = _get_owner(systems, system, slot.owner)
            if owner is None:
                continue
            source = _get_declared(owner, slot.name)
            if target is not None:
                _get_declared(_get_owner(systems, system, target.owner), target.name).items += source.items
            source.items.clear()
    relocations: dict[RoleOf, tuple[_SlotRef | Phase | None, System]] = {}
    for system in systems:
        for role_of, slot in type(system).relocations.items():
            if not isinstance(role_of, RoleOf):
                continue
            if role_of in relocations:
                raise TypeError(
                    f"{type(system).__name__} and {type(relocations[role_of][1]).__name__} both relocate {role_of}."
                )
            relocations[role_of] = (slot, system)
    for system in systems:
        for role in _get_roles(type(system)):
            role_ofs = [
                role_of for role_of in relocations if role_of.role is role and isinstance(system, role_of.system)
            ]
            slot = relocations[role_ofs[0]][0] if role_ofs else role.slot
            if slot is None:
                continue
            slot = _SlotRef(slot.owner, slot.name) if isinstance(slot, (Phase, Inline)) else slot
            owner = _get_owner(systems, system, slot.owner)
            if owner is None:
                continue
            target = _get_declared(owner, slot.attr)
            if slot.slot is not None:
                target = target.slots[slot.slot]
            with _scoped(system, owner):
                target.add(*_get_stages(system, role))


def _get_owner(systems: Sequence[System], system: System, owner_cls: type) -> System | None:
    """
    Return the System of a class offering a slot to a System: the one it depends on, else the one of the scene.

    A slot of a System the scene does not hold is None, the roles filling it and the stages moving it doing nothing.
    """
    owners = [dep for dep in (system, *system._dependencies) if isinstance(dep, owner_cls)] or [
        other for other in systems if isinstance(other, owner_cls)
    ]
    if len(owners) > 1:
        raise TypeError(f"{type(system).__name__} names a slot that {len(owners)} {owner_cls.__name__} offer.")
    return owners[0] if owners else None


def _get_declared(system: System, name: str) -> "Phase | Inline":
    """Return the Phase or the extension point a System class declares under a name, that of the System itself."""
    return vars(system).get(name) or inspect.getattr_static(system, name).__get__(system, type(system))
