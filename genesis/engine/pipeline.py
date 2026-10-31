"""
Composition of the simulation from Systems, which own Data and add stages to the phases of the Systems they depend on.

A System declares what it depends on as the typed parameters of its constructor, and the roles it plays as the role
classes it derives from. The simulator assembles the Systems a scene calls for (see 'assemble_systems'): it creates each
one once its dependencies exist, builds it once they are built, then routes the stage of each of its roles into the slot
the role fills. A System that depends on a System or a struct the scene never has is absent, and so are its stages. See
.github/contributing/PIPELINE.md for the design.

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
    ConfigItem,
    Data,
    DataItem,
    DataKind,
    StaticConfig,
    fill_data,
    iter_data,
)
from genesis.utils.misc import data_to_array

# The type of the annotation of a compile-time parameter
Template = type(qd.template())

# The System classes the simulator creates when a scene calls for them, in registration order (see register_system)
_SYSTEM_CLASSES: list[type["System"]] = []
# Every System class by name, which resolves the dependencies a module names as strings to break an import cycle
_SYSTEM_CLASSES_BY_NAME: dict[str, type["System"]] = {}

# The System whose stages are being added, which they bind the Data of (see '_get_scope')
_scope_system: "System | None" = None


class System:
    """
    A unit of the simulation that owns Data, depends on other Systems and Data, and plays roles in their phases.

    Its Data are its public attributes. Its dependencies are the parameters of its constructor, typed with a System
    class to use the API of that System, or with a Data struct to read it alone, `X | None` for an optional one (see
    'assemble_systems'). It allocates the Data whose sizes the scene decides in `build`, once its dependencies are
    built. The roles it plays are the role classes it derives from, each routing one of its stages (see 'get_roles').
    """

    # Whether the System supports a scene with `requires_grad`, which every System of such a scene must
    is_differentiable: bool = True
    # The slots the roles of other Systems move to while this System exists, by role (see 'RoleOf')
    relocations: ClassVar[dict["RoleOf", "SlotRef"]] = {}
    # The Systems, or roles of Systems, absent while this System exists
    supersedes: ClassVar["tuple[type[System] | RoleOf, ...]"] = ()
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

    def build(self) -> None:
        """Allocate the Data whose sizes the scene decides, once the Systems it depends on are built."""

    def get_parts(self) -> "tuple[System, ...]":
        """Return the Systems this one builds and holds itself, its annotated parts, which join the scene once built."""
        parts = [vars(self).get(name) for name, cls in _get_owned_attrs(type(self)).items() if issubclass(cls, System)]
        return tuple([part for part in parts if part is not None])

    @property
    def requires(self) -> "tuple[type[System], ...]":
        """The components a System the scene holds already uses beyond its constructor, created on demand for it."""
        return ()

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        """
        Yield every array and static config the System holds, named by the attributes reaching it (see iter_data).

        They are the structs and static configs its class annotates, in declaration order, those not built left out.
        """
        for name, cls in _get_owned_attrs(type(self)).items():
            if not issubclass(cls, System) and vars(self).get(name) is not None:
                yield from iter_data(self, name)

    def save_checkpoint(self, kinds: frozenset[DataKind]) -> SystemCheckpoint:
        """Return a checkpoint of the arrays of the given kinds the System holds, for 'load_checkpoint' to restore."""
        items = [item for item in self.data if isinstance(item, DataItem) and item.kind in kinds]
        arrays = {
            item.name: data_to_array(item.array, to_torch=gs.use_zerocopy.is_supported, copy=True) for item in items
        }
        return SystemCheckpoint(arrays, {}, kinds)

    def __getstate__(self) -> SystemCheckpoint:
        """Return a SystemCheckpoint of the System, for '__setstate__' to restore (see 'CHECKPOINT_KINDS')."""
        return self.save_checkpoint(CHECKPOINT_KINDS)

    def __setstate__(self, state: SystemCheckpoint) -> None:
        """Write back the arrays of the kinds a record holds, rejecting a record read from another build."""
        self.load_checkpoint(state)

    def load_checkpoint(self, checkpoint: SystemCheckpoint) -> None:
        """Write back the arrays of a checkpoint of 'save_checkpoint', rejecting a record read from another build."""
        fill_data(
            [item for item in self.data if isinstance(item, DataItem) and item.kind in checkpoint.kinds],
            checkpoint.arrays,
        )


class Coupler(System):
    """
    A System solving the interactions of several solvers at once, which the solvers it couples arrange their substep
    around.
    """

    @property
    def is_solving_rigid_constraints(self) -> bool:
        """Whether the coupler solves the constraints of the rigid entities, between their dynamics and integration."""
        return False

    @property
    def is_using_rigid_coup_type(self) -> bool:
        """Whether the coupler couples every rigid entity the way the `coup_type` of its material sets."""
        return False

    @property
    def rigid_links_idx_simulated(self) -> set[int]:
        """Return the indices of the rigid links the coupler simulates alone, which take part in no rigid contact."""
        return set()

    @property
    def rigid_links_idx_contacting(self) -> set[int]:
        """Return the indices of the rigid links whose contacts with each other the coupler resolves."""
        return set()


def register_system(cls: type[System]) -> type[System]:
    """Declare a System the simulator creates whenever a scene provides every dependency of its constructor."""
    _SYSTEM_CLASSES.append(cls)
    return cls


class SlotRef(NamedTuple):
    """A slot of a phase, or an extension point, that a System class offers, by attribute and slot name."""

    owner: type
    attr: str
    slot: str | None = None


class RoleOf(NamedTuple):
    """A role a System class plays, which another System relocates or supersedes (see 'System.relocations')."""

    system: type
    role: type


class PhaseSpec:
    """
    The declaration of a phase a System offers, `detection = PhaseSpec("broadphase", "narrowphase", "pruning")`.

    On the class it names the slots that roles fill, `Collider.detection.narrowphase`, and on a System it is the Phase
    of that System, its slots running in declaration order. `is_graph` is a flag, or a function of the System, and
    `is_launched_apart` is passed on to the Phase.
    """

    def __init__(
        self, *slots: str, is_graph: "bool | Callable[[System], bool]" = False, is_launched_apart: bool = False
    ) -> None:
        self.slots, self.is_graph, self.is_launched_apart = slots, is_graph, is_launched_apart
        self.owner: type | None = None
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        # A role naming the phase as its slot assigns it again, which leaves its owner
        if self.owner is None:
            self.owner, self.name = owner, name

    def __getattr__(self, slot: str) -> SlotRef:
        if slot not in vars(self).get("slots", ()):
            raise AttributeError(f"The phase has no slot '{slot}'.")
        return SlotRef(self.owner, self.name, slot)

    def __get__(self, system: "System | None", owner: type) -> "PhaseSpec | Phase":
        if system is None:
            return self
        is_graph = self.is_graph(system) if callable(self.is_graph) else self.is_graph
        vars(system)[self.name] = phase = Phase(
            slots=self.slots, is_graph=is_graph, is_launched_apart=self.is_launched_apart
        )
        return phase


class InlineSpec:
    """
    The declaration of an extension point a System offers in one of its kernels (see Inline), which roles fill.

    On the class it is the slot a role names, `MPMSolver.grid_couplings`. The System creates the Inline under the same
    name in its constructor, which shadows the declaration.
    """

    def __init__(self) -> None:
        self.owner: type | None = None
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        # A role naming the extension point as its slot assigns it again, which leaves its owner
        if self.owner is None:
            self.owner, self.name = owner, name

    def __get__(self, system: "System | None", owner: type) -> "InlineSpec":
        assert system is None, f"{owner.__name__} creates its extension point '{self.name}' in its constructor."
        return self


def get_roles(cls: type) -> list[type]:
    """
    Return the roles a System class plays: the classes it derives from that declare a slot.

    A role is a System class declaring the slot it fills, `slot = Collider.detection.narrowphase`, and the name of the
    entry point of a System playing it, `entry = "detect"`: a method, a property, a kernel, or a tuple of them run in
    order. The stage of every System playing the role is routed into the slot (see 'assemble_systems').
    """
    return [base for base in cls.__mro__ if "slot" in vars(base)]


def _get_stages(system: System, entry: str) -> tuple[Any, ...]:
    """Return the stages an entry point of a System names, its methods bound and its properties evaluated."""
    value = inspect.getattr_static(system, entry)
    if isinstance(value, property):
        value = value.fget(system)
    elif isinstance(value, types.FunctionType):
        value = value.__get__(system)
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


class CallArg(NamedTuple):
    """A parameter that no Data provides, given by name when the Phase runs instead."""

    name: str


class Launch(NamedTuple):
    """
    A kernel fusing a run of kernel stages, or a Python stage, with its arguments.

    An argument is its bound value, or the CallArg naming it, whose positions `call_args` lists.
    """

    fn: Callable
    args: tuple[Any, ...]
    call_args: tuple[tuple[int, str], ...]


class FusedStage(NamedTuple):
    """
    A kernel stage of a fused kernel: the kernel, and the names its parameters take in the fused kernel, in order.

    The first parameter of a kernel method (`is_method`) is its object.
    """

    kernel: QuadrantsCallable
    names: tuple[str, ...]
    is_method: bool


def make_fused_kernel(name: str, stages: Sequence[FusedStage], is_graph: bool) -> tuple[Callable, tuple[str, ...]]:
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
                annotation = (Template, "qd.template()")
            else:
                source = f"inspect.signature(fn_{i}).parameters['{parameter.name}'].annotation"
                annotation = (parameter.annotation, source)
            if name_ in annotations:
                # FIXME: quadrants#1020 - a func ignores the class 'qd.Template' as annotation, so kernels keep the
                # 'qd.template()' instances, which never compare equal and are compared by type instead.
                annotation_other = annotations[name_][0]
                assert (
                    type(annotation_other) is type(annotation[0])
                    if isinstance(annotation_other, Template)
                    else annotation_other == annotation[0]
                ), f"Parameter '{name_}' of '{kernel.fn.__qualname__}' is annotated otherwise in another kernel."
            annotations.setdefault(name_, annotation)
        calls.append(f"    func_{i}({', '.join(names)})")
    lines += [f"{name_}_annotation = {source}" for name_, (_, source) in annotations.items()]
    signature = ", ".join([f"{name_}: {name_}_annotation" for name_ in annotations])
    # Fastcache takes no object as argument
    is_fastcached = not any([stage_.is_method for stage_ in stages])
    decorator = f"@qd.kernel(graph={is_graph}, fastcache={is_fastcached})"
    lines += ["", "", decorator, f"def {name}({signature}):", *calls, ""]
    source = "\n".join(lines)

    return load_generated(source, name), tuple(annotations)


def load_generated(source: str, name: str) -> Any:
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
    An ordered sequence of stages, which its System runs, split into the `slots` that roles fill (see PhaseSpec).

    The kernels of a run are fused into one kernel between two Python stages, and captured as a graph with `is_graph`,
    which a kernel holding a device loop (`qd.graph.do_while`) requires. A nested Phase runs inline, fused with the
    stages around it, unless it `is_launched_apart`, its own launches then bounding the runs around it. While the
    launches are recorded for a backward pass, every stage runs on its own instead, all of them reversed as one by
    `backward` when given (see custom_backward).
    """

    def __init__(
        self,
        *,
        slots: Sequence[str] = (),
        is_graph: bool = False,
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
        self._launches: tuple[Launch, ...] | None = None

    def add(self, *stages: Callable) -> None:
        """Append stages, binding their arguments to the Data of the System they belong to and of its dependencies."""
        scope, read_only_ids = _get_scope()
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
                _check_read_only(stage.fn, owner, names, args, read_only_ids)
            # A func, whose loops run in parallel at the top level of a kernel, runs through the kernel generated around
            # it
            if isinstance(stage, QuadrantsCallable) and not stage._is_wrapped_kernel:
                kernel_name = f"kernel_{stage.fn.__name__.removeprefix('func_')}"
                stage, _ = make_fused_kernel(kernel_name, [FusedStage(stage, tuple(names), False)], is_graph=False)
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

    def get_launches(self) -> tuple[Launch, ...]:
        """Return the launches running the Phase, the kernel stages between two Python stages fused."""
        if self._launches is not None:
            return self._launches
        # The kernels and Python stages to launch, with their arguments
        calls: list[tuple[Callable, tuple[Any, ...]]] = []
        run: list[tuple[FusedStage, tuple[Any, ...]]] = []
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
                    if not isinstance(arg, CallArg):
                        n_named = len([key for key in fused_names if key[0] == name])
                        names[i] = fused_names.setdefault((name, id(arg)), f"{name}_{n_named}" if n_named else name)
                run.append((FusedStage(item, tuple(names), is_method), args))
                continue
            if run:
                stages = [stage_ for stage_, _ in run]
                kernel, names = make_fused_kernel(f"fused_{stages[0].kernel.fn.__name__}", stages, self.is_graph)
                args_by_name = {n: a for stage_, args_ in run for n, a in zip(stage_.names, args_)}
                calls.append((kernel, tuple([args_by_name[name] for name in names])))
                run.clear()
                fused_names.clear()
            if item is not None:
                calls.append((item, args))
        self._launches = tuple(
            [
                Launch(fn, args, tuple([(i, arg.name) for i, arg in enumerate(args) if isinstance(arg, CallArg)]))
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
                    stage(*[values[arg.name] if isinstance(arg, CallArg) else arg for arg in args])


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
    the System declaring the extension point when it binds it (see 'bind'), and those that none provides are call
    arguments, given by name when it runs. A parameter of a func that the scope of its System leaves unbound takes the
    value the kernel binds to that name.

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

    def bind(self, **values: Any) -> None:
        """
        Bind the parameters of the kernel to the Data of the System declaring the extension point, as it builds.

        `values` binds parameters by name to values of its own, such as objects the System reaches through a property.
        """
        scope, _ = _get_scope()
        parameters = inspect.signature(self.kernel.fn).parameters
        self._args = tuple([values[name] if name in values else _bind(scope, name) for name in parameters])

    def add(self, *funcs: Callable) -> None:
        """Append funcs, binding their Data parameters to the Data of the System composing and of its dependencies."""
        scope, read_only_ids = _get_scope()
        n_handed = len(inspect.signature(self.hook).parameters)
        for func in funcs:
            names = list(inspect.signature(func.fn).parameters)[n_handed:]
            args = tuple([_bind(scope, name) for name in names])
            _check_read_only(func.fn, None, names, args, read_only_ids)
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
        params_new: list[ast.arg] = []
        calls_data: list[list[str]] = []
        for i, (func, args) in enumerate(self.items):
            owner_name, *path = func.fn.__qualname__.split(".")
            assert "<locals>" not in path, "An extension point func must be defined at the top level of its module."
            lines.append(f"from {func.fn.__module__} import {owner_name} as func_module_{i}")
            lines.append(f"func_{i} = func_module_{i}{''.join([f'.{attr}' for attr in path])}")
            names = list(inspect.signature(func.fn).parameters)[-len(args) :] if args else []
            for name, arg in zip(names, args):
                # A parameter that the scope of the func leaves unbound takes the value the kernel binds to its name
                if name in args_by_name:
                    assert args_by_name[name] is arg or isinstance(arg, CallArg), (
                        f"Parameter '{name}' of '{func.fn.__qualname__}' is bound to another value by another func."
                    )
                    continue
                args_by_name[name] = arg
                annotation = f"inspect.signature(func_{i}.fn).parameters['{name}'].annotation"
                lines.append(f"{name}_annotation = {annotation}")
                params_new.append(ast.arg(arg=name, annotation=ast.Name(id=f"{name}_annotation", ctx=ast.Load())))
            calls_data.append(names)
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
        lines += ["", "", "@qd.kernel", ast.unparse(funcdef), ""]
        kernel = load_generated("\n".join(lines), funcdef.name)
        return kernel, tuple(args_by_name.values())

    def __call__(self, **values: Any) -> None:
        """Run the kernel with the given values of its call arguments."""
        if not self.items and not self._is_run_without_funcs:
            return
        kernel, args = self.get_launch()
        kernel(*[values[arg.name] if isinstance(arg, CallArg) else arg for arg in args])


def _check_read_only(
    fn: Callable, owner: object | None, names: Sequence[str], args: Sequence[Any], read_only_ids: set[int]
) -> None:
    """Refuse a kernel or a func writing a struct that its System depends on as Data alone (see 'get_dependencies')."""
    written_names = {path[0] for path in get_written_paths(fn, owner)}
    for name, arg in zip(names, args):
        if id(arg) in read_only_ids and name in written_names:
            raise TypeError(f"'{fn.__qualname__}' writes '{name}', which its System depends on as Data, read-only.")


def _bind(scope: dict[str, list[Any]], name: str) -> Any:
    """Return the Data of the given name in a scope, the CallArg naming it if none (see '_get_scope')."""
    values = scope.get(name, ())
    assert len(values) < 2, f"Data '{name}' is ambiguous among the dependencies of the System."
    return values[0] if values else CallArg(name)


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
        # A type alias stands for its value, its type parameters taking the arguments it is given
        while isinstance(typing.get_origin(hint) or hint, typing.TypeAliasType):
            alias = typing.get_origin(hint) or hint
            hint = alias.__value__[typing.get_args(hint)] if typing.get_args(hint) else alias.__value__
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
    Return whether a System class is a component: playing no role, registered and owned by none, it is created on demand.

    A component exists while some System depends on it, and is built once every System that could use it is decided.
    """
    owned = {owned for owner in _SYSTEM_CLASSES_BY_NAME.values() for owned in _get_owned(owner)}
    return not get_roles(cls) and cls not in _SYSTEM_CLASSES and cls not in owned


def _get_owned_attrs(cls: type) -> dict[str, type]:
    """
    Return the attributes a System class declares it owns, by name: its Data structs, static configs and parts.

    A System declares them as the annotations of its class attributes, `dyn_state: DynState`, `collider: Collider |
    None`, in declaration order, those of its bases first.
    """
    owned = {}
    for base in reversed(cls.__mro__):
        for name, hint in inspect.get_annotations(base).items():
            for arg in typing.get_args(hint) or (hint,):
                if isinstance(arg, type) and issubclass(arg, (Data, StaticConfig, System)) and arg is not cls:
                    owned[name] = arg
    return owned


def _get_owned(cls: type) -> list[type]:
    """Return the classes of the Data structs and of the parts a System class owns, those of its parts included."""
    return [
        owned
        for owned_cls in _get_owned_attrs(cls).values()
        for owned in (owned_cls, *(_get_owned(owned_cls) if issubclass(owned_cls, System) else ()))
    ]


@contextlib.contextmanager
def _scoped(system: System) -> Iterator[None]:
    """Bind the stages added within the block to the Data of the System and of what it depends on (see '_get_scope')."""
    global _scope_system

    _scope_system = system
    try:
        yield
    finally:
        _scope_system = None


def _get_scope() -> tuple[dict[str, list[Any]], set[int]]:
    """
    Return the Data the stages being added bind by name, and the identities of those they may read alone.

    The scope holds the public attributes of the System, of the Systems it depends on, and of their parts, as they are
    when a stage is added, and the structs it depends on as Data under the name of their parameter, read-only unless it
    reaches them otherwise.
    """
    system = _scope_system
    assert system is not None, "A stage is added within the scope of its System (see 'assemble_systems')."
    scope: dict[str, list[Any]] = {}
    for dependency in (system, *system._dependencies):
        for owner in (dependency, *dependency.get_parts()):
            for name, value in vars(owner).items():
                if not name.startswith("_") and all([value is not other for other in scope.get(name, ())]):
                    scope.setdefault(name, []).append(value)
    writable_ids = {id(value) for values in scope.values() for value in values}
    read_only_ids = set()
    for name, struct in system._data_dependencies.items():
        if id(struct) not in writable_ids:
            scope.setdefault(name, []).append(struct)
            read_only_ids.add(id(struct))
    return scope, read_only_ids


def assemble_systems(
    systems: Sequence[System], systems_built: Sequence[System], context: Sequence[object] = ()
) -> list[System]:
    """
    Create and build the Systems a scene calls for, then route their roles and return them.

    `systems` are the Systems the scene holds already, `systems_built` those among them built elsewhere, and `context`
    the other objects a dependency may name, such as the options of the scene. A registered System is created as soon as
    every System it depends on exists, every struct it depends on is built, and every condition on them holds. A System
    created on demand, such as an instance of a component template, exists once an existing System depends on it. A
    System is built once the Systems it depends on are built and every System that could still use its API is decided,
    so that every request they make of it in their constructor is in, except those conditioned on it, which are decided
    once it is built. A dependency on a struct never delays the build of its owner. The stage of each role is then
    routed into its slot, in the order of the Systems: those given, then those created in registration order. A System
    that a mandatory dependency or a condition rules out is absent, and so is a System another one supersedes, the
    reason being logged.
    """
    systems = list(systems)
    built_ids = {id(system) for system in systems_built}
    classes_pending = [cls for cls in _SYSTEM_CLASSES if all([type(system) is not cls for system in systems])]
    classes_pending += list(dict.fromkeys([cls for system in systems for cls in system.requires]))

    def decide(cls: type[System]) -> tuple[str, Any]:
        """Return whether a pending class is created (with its arguments), absent (with why), or waits."""
        for system in systems:
            for superseded in type(system).supersedes:
                if isinstance(superseded, type) and issubclass(cls, superseded):
                    return "absent", f"superseded by {type(system).__name__}"
        # A class waits at the first dependency still undecided, before requiring any component it may never use
        args = {}
        for dependency in _get_dependencies(cls):
            classes = dependency.classes
            owners_unbuilt = [type(system) for system in systems if id(system) not in built_ids]
            if all([isinstance(c, type) and issubclass(c, Data) for c in classes]):
                candidates = [
                    value
                    for system in systems
                    if id(system) in built_ids
                    for name, value in vars(system).items()
                    if not name.startswith("_") and isinstance(value, classes)
                ]
                may_come = any(
                    [
                        issubclass(owned, classes)
                        for owner in (*owners_unbuilt, *classes_pending)
                        for owned in _get_owned(owner)
                    ]
                )
            elif all([isinstance(c, type) and issubclass(c, System) for c in classes]):
                candidates = [system for system in systems if isinstance(system, classes)]
                # A part joins once its owner is built
                may_come = any(
                    [
                        issubclass(c, classes)
                        for owner in (*classes_pending, *owners_unbuilt)
                        for c in (owner, *_get_owned(owner))
                    ]
                )
                if not candidates and not dependency.is_optional:
                    for needed in classes:
                        if _is_component(needed) and needed not in classes_pending:
                            classes_pending.append(needed)
                            may_come = True
            else:
                candidates = [value for value in context if isinstance(value, classes)]
                may_come = False
            unique = []
            for candidate in candidates:
                if all([candidate is not other for other in unique]):
                    unique.append(candidate)
            if len(unique) > 1:
                raise TypeError(f"Dependency '{dependency.name}' of {cls.__name__} is ambiguous: {unique}.")
            if not unique:
                if may_come:
                    return "wait", None
                if dependency.is_optional:
                    args[dependency.name] = None
                    continue
                return "absent", f"no {' or '.join([c.__name__ for c in classes])} for '{dependency.name}'"
            value = unique[0]
            if dependency.conditions:
                # A condition on a System is read once that System is built
                if isinstance(value, System) and id(value) not in built_ids:
                    return "wait", None
                if not all([condition.holds(value) for condition in dependency.conditions]):
                    if dependency.is_optional:
                        args[dependency.name] = None
                        continue
                    return "absent", f"a condition on '{dependency.name}' does not hold"
            args[dependency.name] = value
        return "create", args

    while True:
        classes_decided = list(classes_pending)
        is_progressing = False
        for cls in classes_decided:
            status, value = decide(cls)
            if status == "wait":
                continue
            classes_pending.remove(cls)
            is_progressing = True
            if status == "absent":
                gs.logger.debug(f"{cls.__name__} is absent: {value}.")
                continue
            system = cls(**value)
            system._dependencies = tuple([arg for arg in value.values() if isinstance(arg, System)])
            system._data_dependencies = {name: arg for name, arg in value.items() if isinstance(arg, Data)}
            systems.append(system)
        for system in list(systems):
            if id(system) in built_ids:
                continue
            # A pending class that needs what the build of the System provides, its parts or its Data, is decided after
            # it
            owned = tuple(_get_owned(type(system)))
            users_pending = [
                cls
                for cls in classes_pending
                if not any([issubclass(c, owned) for dependency in _get_dependencies(cls) for c in dependency.classes])
            ]
            is_ready = all([id(dependency) in built_ids for dependency in system._dependencies]) and not any(
                [
                    issubclass(type(system), c)
                    for cls in users_pending
                    for dependency in _get_dependencies(cls)
                    if not dependency.conditions
                    for c in dependency.classes
                    if issubclass(c, System)
                ]
            )
            if not is_ready:
                continue
            with _scoped(system):
                system.build()
            built_ids.add(id(system))
            is_progressing = True
            # The parts of a System join the Systems of the scene, built with it, and depend on it
            for part in system.get_parts():
                if all([part is not other for other in systems]):
                    part._dependencies = (system, *part._dependencies)
                    systems.append(part)
                    built_ids.add(id(part))
        # A class created on demand joins the pending ones while they are decided
        if not is_progressing and all([cls in classes_decided for cls in classes_pending]):
            break
    unbuilt = [type(system).__name__ for system in systems if id(system) not in built_ids]
    if classes_pending or unbuilt:
        raise TypeError(
            f"The declarations of {[cls.__name__ for cls in classes_pending]} and {unbuilt} contradict one another: "
            "each waits for another one."
        )

    # The Systems in an order that does not depend on when they were created: those the scene held and their parts, then
    # the registered ones in registration order, then those created on demand in creation order (the sort is stable)
    registration = {cls: i for i, cls in enumerate(_SYSTEM_CLASSES)}

    def get_rank(system: System) -> tuple[int, int]:
        if type(system) in registration:
            return 1, registration[type(system)]
        return (2 if _is_component(type(system)) else 0), 0

    systems = sorted(systems, key=get_rank)
    _route_roles(systems)
    return systems


def _route_roles(systems: Sequence[System]) -> None:
    """Add the stage of every role of every System to its slot, or to the slot another System relocates it to."""
    relocations: dict[RoleOf, tuple[SlotRef, System]] = {}
    superseded_roles: list[RoleOf] = []
    for system in systems:
        for role_of, slot in type(system).relocations.items():
            if role_of in relocations:
                raise TypeError(
                    f"{type(system).__name__} and {type(relocations[role_of][1]).__name__} both relocate {role_of}."
                )
            relocations[role_of] = (slot, system)
        superseded_roles += [item for item in type(system).supersedes if isinstance(item, RoleOf)]
    for system in systems:
        for role in get_roles(type(system)):
            role_ofs = [
                role_of
                for role_of in (*relocations, *superseded_roles)
                if role_of.role is role and isinstance(system, role_of.system)
            ]
            if any([role_of in superseded_roles for role_of in role_ofs]):
                continue
            slot = relocations[role_ofs[0]][0] if role_ofs else role.slot
            slot = SlotRef(slot.owner, slot.name) if isinstance(slot, (PhaseSpec, InlineSpec)) else slot
            owners = [dep for dep in system._dependencies if isinstance(dep, slot.owner)] or [
                other for other in systems if isinstance(other, slot.owner)
            ]
            if len(owners) != 1:
                raise TypeError(
                    f"{type(system).__name__} plays {role.__name__}, which {len(owners)} {slot.owner.__name__} offer."
                )
            target = inspect.getattr_static(owners[0], slot.attr)
            if isinstance(target, PhaseSpec):
                target = target.__get__(owners[0], type(owners[0]))
            if slot.slot is not None:
                target = target.slots[slot.slot]
            with _scoped(system):
                target.add(*_get_stages(system, role.entry))
