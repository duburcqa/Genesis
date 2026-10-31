"""
Composition of the simulation from Systems, which own Data and add stages to the phases of the Systems they depend on.

A System declares the Systems it depends on as the typed parameters of its constructor, of its `setup` method and of its
`compose` method, and nothing else of the others. The simulator creates the Systems a scene calls for, sets every System
up, builds them, then lets each one compose with its dependencies (see 'create_systems'). A System that depends on a
System the scene does not have is absent, and so are the stages it would add.

A Phase is an ordered sequence of stages that its System runs, and that the Systems depending on it add to. A stage is a
kernel, a kernel method of a data-oriented object, a Python function or another Phase. Its arguments are bound by name to
the Data of the System adding it and of the Systems it depends on, and an argument that none of them provides is a call
argument, given by name when the Phase runs.

A Phase fuses the kernels it runs between two Python functions into one kernel, generated from source, which calls the
body of each of them as a top-level func. While the launches are recorded for a backward pass, it launches every kernel on
its own instead, since the launch recorder of the differentiable simulation reverses one launch per entry (see
genesis/grad/bridge.py), whose tags (`no_grad`, `register_custom_backward`, `reversed_with`) apply as usual.
"""

import ast
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

import quadrants as qd
from quadrants.lang._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable

import genesis as gs
from genesis.engine.states.solvers import SystemCheckpoint
from genesis.grad.bridge import custom_backward, is_recording_launches
from genesis.grad.kernel_writes import get_written_paths
from genesis.utils.cache import get_cache_dir
from genesis.utils.data import CHECKPOINT_KINDS, ConfigItem, Data, DataItem, DataKind, fill_data
from genesis.utils.misc import data_to_array

# The type of the annotation of a compile-time parameter
Template = type(qd.template())

# The System classes the simulator creates when a scene calls for them, in registration order (see register_system)
_SYSTEM_CLASSES: list[type["System"]] = []

# The Data a stage may bind, by name, while a System composes (see compose_systems)
_scope: dict[str, list[Any]] | None = None
# The structs among them the System composing depends on as Data alone, which its kernels may read and never write
_read_only_ids: set[int] = set()


class System:
    """
    A unit of the simulation that owns Data and composes with the Systems it depends on.

    Its Data are its public attributes. The Systems it depends on are the parameters of `setup` and `compose`, typed
    with their class, `X | None` for an optional one. A System the simulator creates also takes them as the parameters
    of its constructor, which decide whether it exists (see register_system).
    """

    # Whether the System supports a scene with `requires_grad`, which every System of such a scene must
    is_differentiable: bool = True
    # Whether the System is created when another one depends on it, rather than registered (see 'create_systems')
    is_created_on_demand: bool = False
    # The Systems its constructor depends on, which its stages bind the Data of as those of its `compose` dependencies
    _dependencies: "tuple[System, ...]" = ()
    # The instances of the template a System class is, by their arguments (see '__class_getitem__')
    _template_instances: ClassVar[dict[tuple[type, ...], type["System"]]] = {}
    # The arguments of the template instance a System class is, by the type parameters of its template
    template_args: ClassVar[dict[typing.TypeVar, type]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._template_instances = {}

    def __class_getitem__(cls, args: "type | tuple[type, ...]") -> "type[System]":
        """
        Return the System class instantiating the template for the given arguments, the same class for the same ones.

        A template is a System class generic over type parameters, `class GJK[LayoutT: Layout, CapabilityT:
        Capability](System)`, whose instances read their arguments by parameter, `self.template_args[LayoutT]`. Its
        instances are distinct System classes, which a scene holds side by side, each owning the state its arguments
        call for, and which a dependency names apart. They are created on demand when the template is (see
        'create_systems').
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
    def create(cls, *args: Any, **kwargs: Any) -> "System | None":
        """Build the System from its dependencies, or return None where the scene has no use of it."""
        return cls(*args, **kwargs)

    def setup(self) -> None:
        """Opt in what the System uses of the Systems it depends on, its parameters, before any of them builds."""

    def build(self) -> None:
        """
        Allocate the Data of a System the simulator creates, once the solvers and the Systems it depends on built.

        Its parameters are the Systems and Data structs it depends on (see 'get_dependencies').
        """

    def compose(self) -> None:
        """Add the stages of the System to the phases of the Systems it depends on, its parameters."""

    def get_parts(self) -> "tuple[System, ...]":
        """Return the Systems this one builds and holds itself, which compose like any other once it exists."""
        return ()

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        """Yield every array and static config the System holds, named by the attributes reaching it (see 'iter_data')."""
        return iter(())

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
        """Write back the arrays of a checkpoint 'save_checkpoint' returned, rejecting a record read from another build."""
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
    def is_simulating_rigid_first(self) -> bool:
        """Whether the coupler simulates the rigid entities it couples before the rigid solver runs its substep."""
        return False

    @property
    def is_simulating_fem(self) -> bool:
        """Whether the coupler computes the dynamics of the FEM entities in place of the FEM solver."""
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

    The fastcache of Quadrants reads the source of a kernel from its file, and a file named after the hash of its content
    stays valid across processes. The module is imported once per process.
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
                values = _scope.get(name, ())
                assert len(values) < 2, f"Data '{name}' is ambiguous among the dependencies of the System composing."
                args.append(values[0] if values else CallArg(name))
            if isinstance(stage, (QuadrantsCallable, BoundQuadrantsCallable)):
                _check_read_only(
                    stage.fn, stage.instance if isinstance(stage, BoundQuadrantsCallable) else None, names, args
                )
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
        if not self.items:
            return
        if not is_recording_launches():
            for launch in self.get_launches():
                if isinstance(launch.fn, Inline):
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
        Bind the parameters of the kernel to the Data of the System declaring the extension point, as it composes.

        `values` binds parameters by name to values of its own, such as objects the System reaches through a property.
        """
        assert _scope is not None, "An extension point is bound while its System composes."
        parameters = inspect.signature(self.kernel.fn).parameters
        self._args = tuple([values[name] if name in values else _bind(name) for name in parameters])

    def add(self, *funcs: Callable) -> None:
        """Append funcs, binding their Data parameters to the Data of the System composing and of its dependencies."""
        assert _scope is not None, "A func is added while its System composes."
        n_handed = len(inspect.signature(self.hook).parameters)
        for func in funcs:
            names = list(inspect.signature(func.fn).parameters)[n_handed:]
            args = tuple([_bind(name) for name in names])
            _check_read_only(func.fn, None, names, args)
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


def _check_read_only(fn: Callable, owner: object | None, names: Sequence[str], args: Sequence[Any]) -> None:
    """Refuse a kernel or a func writing a struct that its System depends on as Data alone (see 'get_dependencies')."""
    written_names = {path[0] for path in get_written_paths(fn, owner)}
    for name, arg in zip(names, args):
        if id(arg) in _read_only_ids and name in written_names:
            raise TypeError(f"'{fn.__qualname__}' writes '{name}', which its System depends on as Data, read-only.")


def _bind(name: str) -> Any:
    """Return the Data of the given name in the scope of the System composing, the CallArg naming it if none."""
    values = _scope.get(name, ())
    assert len(values) < 2, f"Data '{name}' is ambiguous among the dependencies of the System composing."
    return values[0] if values else CallArg(name)


class OnlyIf(NamedTuple):
    """
    Make a dependency of a template exist in its instances taking the given argument alone (see 'System').

    It is the metadata of a generic type alias declared once with its argument, `type IfQueued[T] = Annotated[T | None,
    OnlyIf(Queued)]`, which annotates the parameter it conditions, `gjk: IfQueued[GJK[Queued, WithoutEPA]]`: a mandatory
    dependency of the instances taking `Queued`, absent from the others, and an optional one for type checkers.
    """

    arg: type


def _get_dependency_hints(fn: Callable) -> dict[str, tuple[tuple[type, ...], bool]]:
    """Return the classes each parameter of a System class or bound method takes, and whether it is optional."""
    owner = fn if isinstance(fn, type) else type(fn.__self__)
    hints = typing.get_type_hints(fn.__init__ if isinstance(fn, type) else fn, include_extras=True)
    dependency_hints = {}
    for name in inspect.signature(fn).parameters:
        hint, is_conditional = hints[name], False
        # A type alias stands for its value, its type parameters taking the arguments it is given
        while isinstance(typing.get_origin(hint) or hint, typing.TypeAliasType):
            alias = typing.get_origin(hint) or hint
            hint = alias.__value__[typing.get_args(hint)] if typing.get_args(hint) else alias.__value__
        if typing.get_origin(hint) is typing.Annotated:
            conditions = [meta for meta in hint.__metadata__ if isinstance(meta, OnlyIf)]
            hint = hint.__origin__
            if conditions:
                if not all([condition.arg in owner.template_args.values() for condition in conditions]):
                    dependency_hints[name] = ((), True)
                    continue
                is_conditional = True
        hint_args = typing.get_args(hint)
        is_optional = typing.get_origin(hint) in (typing.Union, types.UnionType) and type(None) in hint_args
        classes = tuple([arg for arg in hint_args if arg is not type(None)]) if is_optional else (hint,)
        # A template instance named with the type parameters of the owner takes the arguments the owner was given
        classes = tuple(
            [
                typing.get_origin(cls)[tuple([owner.template_args.get(arg, arg) for arg in typing.get_args(cls)])]
                if typing.get_origin(cls) is not None
                else cls
                for cls in classes
            ]
        )
        dependency_hints[name] = (classes, is_optional and not is_conditional)
    return dependency_hints


def get_dependencies(fn: Callable, systems: Sequence[object]) -> dict[str, Any] | None:
    """
    Return the arguments of a System class or of a bound method of a System, None when a mandatory one is missing.

    A parameter typed with a class takes the one object of `systems` of that class, None for an optional one missing. A
    parameter typed with a Data struct takes the one struct of that type a System holds as a public attribute: a System
    depends on another System to use its API, which may change its state, and on a struct to read it alone. The Data of
    a System exists once it is built, so a struct is a dependency of `build` and `compose` alone.
    """
    structs = [
        value
        for system in systems
        if isinstance(system, System)
        for name, value in vars(system).items()
        if not name.startswith("_") and isinstance(value, Data)
    ]
    args = {}
    for name, (classes, is_optional) in _get_dependency_hints(fn).items():
        if not classes:
            args[name] = None
            continue
        candidates = structs if all([issubclass(cls, Data) for cls in classes]) else systems
        matches = []
        for candidate in candidates:
            if isinstance(candidate, classes) and all([candidate is not match for match in matches]):
                matches.append(candidate)
        assert len(matches) < 2, f"Dependency '{name}' of '{fn.__qualname__}' is ambiguous."
        if not matches and not is_optional:
            return None
        args[name] = matches[0] if matches else None
    return args


def create_systems(systems: Sequence[System], context: Sequence[object] = ()) -> list[System]:
    """
    Create the registered Systems that the given Systems and context call for, and return them in creation order.

    `context` holds the other objects a dependency may name, such as the options of the scene. A registered System class
    is created once every System class its constructor depends on is settled, in registration order otherwise, and only
    when every mandatory dependency exists (see register_system). A System class created on demand, such as the instances
    of a component template, is created when a System being created depends on it. They are created before the solvers
    allocate their Data, so that a System opts in what it uses of another one.
    """
    systems_created: list[System] = []
    classes_pending = [cls for cls in _SYSTEM_CLASSES if all([type(system) is not cls for system in systems])]
    while classes_pending:
        for cls in classes_pending:
            dependency_hints = _get_dependency_hints(cls)
            classes_needed = [needed for classes, _ in dependency_hints.values() for needed in classes]
            # A mandatory dependency that nothing registers is created on demand
            for classes, is_optional in dependency_hints.values():
                for needed in () if is_optional else classes:
                    is_demanded = issubclass(needed, System) and needed.is_created_on_demand
                    is_present = any([isinstance(other, needed) for other in (*systems, *systems_created)])
                    if is_demanded and needed not in classes_pending and not is_present:
                        classes_pending.append(needed)
            if any([issubclass(other, needed) for other in classes_pending for needed in classes_needed]):
                continue
            classes_pending.remove(cls)
            args = get_dependencies(cls, [*systems, *systems_created, *context])
            system = None if args is None else cls.create(**args)
            if system is not None:
                system._dependencies = tuple([arg for arg in args.values() if isinstance(arg, System)])
                systems_created.append(system)
            break
        else:
            assert False, f"The dependencies of {classes_pending} form a cycle."
    return systems_created


def call_with_dependencies(
    methods: Sequence[Callable], systems: Sequence[System], context: Sequence[object] = ()
) -> None:
    """Call bound lifecycle methods of Systems in turn, each with the Systems and Data it depends on."""
    for method in methods:
        args = get_dependencies(method, [*systems, *context])
        assert args is not None, f"A dependency of '{method.__qualname__}' is missing."
        method(**args)


def compose_systems(systems: Sequence[System], context: Sequence[object] = ()) -> list[System]:
    """
    Let every System add its stages to the phases of its dependencies, and return the Systems with their parts.

    The parts of a System, which exist once it is built (see 'System.get_parts'), join the Systems first. The stages of a
    System bind the Data of the System and of the Systems its constructor and `compose` depend on.
    """
    global _scope, _read_only_ids

    systems_composed: list[System] = []
    systems_pending = list(systems)
    while systems_pending:
        system = systems_pending.pop(0)
        if all([system is not other for other in systems_composed]):
            systems_composed.append(system)
        systems_pending[:0] = [part for part in system.get_parts() if all([part is not s for s in systems_composed])]

    for system in systems_composed:
        args = get_dependencies(system.compose, [*systems_composed, *context])
        assert args is not None, f"A dependency of '{type(system).__qualname__}.compose' is missing."
        dependencies = [*system._dependencies, *[arg for arg in args.values() if isinstance(arg, System)]]
        _scope = {}
        for owner in (system, *dependencies):
            for name, value in vars(owner).items():
                if not name.startswith("_") and all([value is not other for other in _scope.get(name, ())]):
                    _scope.setdefault(name, []).append(value)
        # A struct depended on as Data joins the scope under its parameter, read-only unless the System reaches it as
        # its own or through a System it depends on
        writable_ids = {id(value) for values in _scope.values() for value in values}
        for name, arg in args.items():
            if isinstance(arg, Data) and id(arg) not in writable_ids:
                _scope.setdefault(name, []).append(arg)
                _read_only_ids.add(id(arg))
        try:
            system.compose(**args)
        finally:
            _scope = None
            _read_only_ids = set()
    return systems_composed
