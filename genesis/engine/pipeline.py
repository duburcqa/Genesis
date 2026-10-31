"""
Composition of the simulation from Systems, which own Data and add stages to the phases of the Systems they depend on.

A System declares the Systems it depends on as the typed parameters of its constructor and of its `compose` method, and
nothing else of the others: the Engine builds the Systems a scene calls for, then lets each one compose with its
dependencies. A System that depends on a System the scene does not have is absent, and so are the stages it would add.

A Phase is an ordered sequence of stages that its System runs, and that the Systems depending on it add to. A stage is a
kernel, a kernel method of a data-oriented object, a Python function or another Phase. Its arguments are bound by name to
the Data of the System adding it and of the Systems it depends on, and an argument that none of them provides is a call
argument, given by name when the Phase runs.

A Phase fuses the kernels it runs between two Python functions into one kernel, generated from source, which calls the
body of each of them as a top-level func. While the launches are recorded for a backward pass, it launches every kernel on
its own instead, since the launch recorder of the differentiable simulation reverses one launch per entry (see
genesis/grad/bridge.py), whose tags (`no_grad`, `register_custom_backward`, `reversed_with`) apply as usual.
"""

import hashlib
import importlib.util
import inspect
import os
import sys
import types
import typing
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any, NamedTuple

import quadrants as qd
from quadrants.lang._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable

from genesis.grad.bridge import custom_backward, is_recording_launches
from genesis.utils.cache import get_cache_dir

# The type of the annotation of a compile-time parameter
Template = type(qd.template())

# The System classes the Engine builds when a scene calls for them, in registration order (see register_system)
_SYSTEM_CLASSES: list[type["System"]] = []

# The Data a stage may bind, by name, while a System composes (see Engine)
_scope: dict[str, list[Any]] | None = None


class System:
    """
    A unit of the simulation that owns Data and composes with the Systems it depends on.

    Its Data are its public attributes. The Systems it depends on are the parameters of `compose`, typed with their
    class, `X | None` for an optional one. A System the Engine builds also takes them as the parameters of its
    constructor, which decide whether it exists (see register_system).
    """

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> "System | None":
        """Build the System from its dependencies, or return None where the scene has no use of it."""
        return cls(*args, **kwargs)

    def compose(self) -> None:
        """Add the stages of the System to the phases of the Systems it depends on, its parameters."""

    def get_parts(self) -> "tuple[System, ...]":
        """Return the Systems this one builds and holds itself, which compose like any other once it exists."""
        return ()


def register_system(cls: type[System]) -> type[System]:
    """Declare a System the Engine builds whenever a scene provides every dependency of its constructor."""
    _SYSTEM_CLASSES.append(cls)
    return cls


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
    Engine fusing the same kernels shares it and its compiled variants.
    """
    # The annotation of every parameter, as an object to compare and as the source evaluating to it
    annotations: dict[str, tuple[Any, str]] = {}
    lines, calls = ["import inspect", "", "import quadrants as qd", ""], []
    for i, (kernel, names, is_method) in enumerate(stages):
        # A kernel method is reached through its class, at module level
        owner_name, *path = kernel.fn.__qualname__.split(".")
        assert "<locals>" not in path, "A kernel stage is defined at module level."
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
    return vars(module)[name], tuple(annotations)


class Phase:
    """
    An ordered sequence of stages, which its System runs and the Systems depending on it add to while they compose.

    The kernels of a run are fused into one kernel between two Python stages, a nested Phase running inline, and captured
    as a graph with `is_graph`, which a kernel holding a device loop (`qd.graph.do_while`) requires. While the launches
    are recorded for a backward pass, every stage runs on its own instead, all of them reversed as one by `backward` when
    given (see custom_backward).
    """

    def __init__(self, *, is_graph: bool = False, backward: Callable[[], None] | None = None) -> None:
        self.is_graph = is_graph
        self.backward = backward
        # The stages, in order, with their arguments
        self.items: list[tuple[Callable, tuple[Any, ...]]] = []
        self._launches: tuple[Launch, ...] | None = None

    def add(self, *stages: Callable) -> None:
        """Append stages, binding their arguments to the Data of the System composing and of its dependencies."""
        assert _scope is not None, "A stage is added while its System composes."
        for stage in stages:
            if isinstance(stage, Phase):
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
                values = _scope.get(name, ())
                assert len(values) < 2, f"Data '{name}' is ambiguous among the dependencies of the System composing."
                args.append(values[0] if values else CallArg(name))
            self.items.append((stage, tuple(args)))
        self._launches = None

    def get_items(self) -> list[tuple[Callable, tuple[Any, ...]]]:
        """Return the stages the Phase runs, its nested Phases unrolled."""
        return [
            item
            for stage, args in self.items
            for item in (stage.get_items() if isinstance(stage, Phase) else [(stage, args)])
        ]

    def get_launches(self) -> tuple[Launch, ...]:
        """Return the launches running the Phase, the kernel stages between two Python stages fused."""
        if self._launches is not None:
            return self._launches
        # The kernels and Python stages to launch, with their arguments
        calls: list[tuple[Callable, tuple[Any, ...]]] = []
        run: list[tuple[FusedStage, tuple[Any, ...]]] = []
        # The objects of the kernel methods of the run, whose rank names them in the fused kernel
        owners: list[object] = []
        for item, args in [*self.get_items(), (None, ())]:
            if isinstance(item, (QuadrantsCallable, BoundQuadrantsCallable)):
                names = list(inspect.signature(item.fn).parameters)
                is_method = isinstance(item, BoundQuadrantsCallable)
                if is_method:
                    if all([item.instance is not owner for owner in owners]):
                        owners.append(item.instance)
                    names[0] = f"owner_{[owner is item.instance for owner in owners].index(True)}"
                    item, args = item.quadrants_callable, (item.instance, *args)
                run.append((FusedStage(item, tuple(names), is_method), args))
                continue
            if run:
                stages = [stage_ for stage_, _ in run]
                kernel, names = make_fused_kernel(f"fused_{stages[0].kernel.fn.__name__}", stages, self.is_graph)
                args_by_name = {n: a for stage_, args_ in run for n, a in zip(stage_.names, args_)}
                calls.append((kernel, tuple([args_by_name[name] for name in names])))
                run.clear()
                owners.clear()
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
        if not is_recording_launches():
            for launch in self.get_launches():
                args = launch.args
                if launch.call_args:
                    args = list(args)
                    for i, name in launch.call_args:
                        args[i] = values[name]
                launch.fn(*args)
            return
        with custom_backward(self.backward) if self.backward is not None else nullcontext():
            for stage, args in self.items:
                if isinstance(stage, Phase):
                    stage(**values)
                else:
                    stage(*[values[arg.name] if isinstance(arg, CallArg) else arg for arg in args])


def get_dependencies(fn: Callable, systems: Sequence[object]) -> dict[str, Any] | None:
    """
    Return the arguments of a System class or of a bound `compose` method, None when a mandatory one is missing.

    A parameter typed with a class takes the one object of `systems` of that class, None for an optional one missing.
    """
    hints = typing.get_type_hints(fn.__init__ if isinstance(fn, type) else fn)
    args = {}
    for name in inspect.signature(fn).parameters:
        hint = hints[name]
        hint_args = typing.get_args(hint)
        is_optional = isinstance(hint, types.UnionType) and type(None) in hint_args
        classes = tuple([arg for arg in hint_args if arg is not type(None)]) if is_optional else (hint,)
        matches = [system for system in systems if isinstance(system, classes)]
        assert len(matches) < 2, f"Dependency '{name}' of '{fn.__qualname__}' is ambiguous."
        if not matches and not is_optional:
            return None
        args[name] = matches[0] if matches else None
    return args


class Engine:
    """
    Build the Systems a scene calls for, then let each one compose with its dependencies.

    `systems` are the Systems that exist already, their parts included, and `context` the other objects a dependency may
    name, such as the options of the scene. A registered System class is built once every System class its constructor depends on is
    settled, in registration order otherwise, and only when every mandatory dependency exists (see register_system).
    """

    def __init__(self, systems: Sequence[System], context: Sequence[object] = ()) -> None:
        global _scope

        self.systems: list[System] = []
        systems_pending = list(systems)
        while systems_pending:
            system = systems_pending.pop(0)
            if all([system is not other for other in self.systems]):
                self.systems.append(system)
                systems_pending[:0] = system.get_parts()
        # The dependencies of every System, by System
        dependencies: dict[int, list[System]] = {id(system): [] for system in self.systems}
        classes_pending = [cls for cls in _SYSTEM_CLASSES if all([type(s) is not cls for s in self.systems])]
        while classes_pending:
            for cls in classes_pending:
                hints = typing.get_type_hints(cls.__init__)
                classes_needed = [
                    arg
                    for hint in hints.values()
                    for arg in (typing.get_args(hint) or (hint,))
                    if isinstance(arg, type)
                ]
                if any([issubclass(other, needed) for other in classes_pending for needed in classes_needed]):
                    continue
                classes_pending.remove(cls)
                args = get_dependencies(cls, [*self.systems, *context])
                system = None if args is None else cls.create(**args)
                if system is not None:
                    self.systems.append(system)
                    dependencies[id(system)] = [arg for arg in args.values() if isinstance(arg, System)]
                break
            else:
                assert False, f"The dependencies of {classes_pending} form a cycle."

        for system in self.systems:
            args = get_dependencies(system.compose, [*self.systems, *context])
            assert args is not None, f"A dependency of '{type(system).__qualname__}.compose' is missing."
            dependencies[id(system)] += [arg for arg in args.values() if isinstance(arg, System)]
            _scope = {}
            for owner in (system, *dependencies[id(system)]):
                for name, value in vars(owner).items():
                    if not name.startswith("_") and all([value is not other for other in _scope.get(name, ())]):
                        _scope.setdefault(name, []).append(value)
            try:
                system.compose(**args)
            finally:
                _scope = None

    def find(self, cls: type[System]) -> System | None:
        """Return the System of the given class, None when the scene has none."""
        matches = [system for system in self.systems if isinstance(system, cls)]
        return matches[0] if matches else None
