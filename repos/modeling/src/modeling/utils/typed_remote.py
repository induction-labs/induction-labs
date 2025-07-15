# https://chatgpt.com/share/68769b6d-b38c-8006-8221-e7f845ec2439
from __future__ import annotations

import functools
import types
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Concatenate,
    Coroutine,
    Generic,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    Self,
    TypeVar,
    Union,
    cast,
    final,
    overload,
    runtime_checkable,
)

import ray
from pydantic import BaseModel

# from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

P = ParamSpec("P")  # parameters after  ‘self’
R = TypeVar("R", covariant=True)  # return type
ActorArgs = TypeVar("ActorArgs", bound=BaseModel)
ConfiguredState = TypeVar("ConfiguredState")
SelfT = TypeVar("SelfT")  # the class on which the decorator is used

# ray.remote


@runtime_checkable
class _WithCustomCall(Protocol[P, R]):
    """What the decorated method looks like to the type-checker."""

    # Disable the default `__call__` method to avoid confusion with the remote call.
    # Similar to CUDA `__device__` vs `__host__` qualifiers.
    # def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def remote_call(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


@overload
def remote_method(
    func: Callable[Concatenate[SelfT, P], Coroutine[Any, Any, R]],
) -> _WithCustomCall[P, Coroutine[Any, Any, R]]: ...


@overload
def remote_method(
    func: Callable[Concatenate[SelfT, P], R],
) -> _WithCustomCall[P, Coroutine[Any, Any, R]]: ...


def remote_method(  # type: ignore
    func,
):
    """
    Decorator to turn a normal instance method into one that also exposes
    `.custom_call()`, while preserving the original type signature.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # -- normal call path, you can add extra behaviour here
        return func(self, *args, **kwargs)

    def _custom(self, *args, **kwargs):
        # -- “alternative” call path
        # put whatever custom logic you need before/after the real call
        # Assert that func has the `remote` attribute
        assert hasattr(func, "remote"), (
            f"Function {func.__name__} does not have a 'remote' attribute."
        )
        return func.remote(self, *args, **kwargs)  # type: ignore

    # ---- Descriptor trick so attribute access works on the *bound* method ----
    def _get_bound(
        unbound: Callable[..., Any],  # the function object itself
        instance: SelfT | None,
        owner: type[SelfT] | None = None,
    ):
        if instance is None:  # accessed on the class
            return unbound
        bound = types.MethodType(unbound, instance)  # normal bound method
        # attach *another* bound method that already has the same `self`
        object.__setattr__(bound, "custom_call", types.MethodType(_custom, instance))
        return bound

    wrapper.__get__ = _get_bound  # type: ignore[attr-defined]
    # keep an unbound `.custom_call` so static analysers see it on the class
    wrapper.remote_call = _custom  # type: ignore[attr-defined]

    return wrapper  # type: ignore[return-value]


# ────────────────────────  how you use it  ─────────────────────────


class BaseActor(Generic[ActorArgs, ConfiguredState], ABC):
    state: ConfiguredState

    @final
    def __init__(self, args: ActorArgs):
        self.args = args
        # We can't do configure_state here because it is async so we rely on the caller to call it
        # right after instantiation.

    @abstractmethod
    async def _configure_state(self) -> ConfiguredState:
        """
        Configure the state of the actor based on the provided arguments.
        This method should be implemented by subclasses to define how the
        actor's state is initialized.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @final
    @remote_method
    async def configure_state(self):
        """
        Configure the state of the actor.
        This method is called to initialize the actor's state.
        """
        assert not hasattr(self, "state"), (
            "State has already been configured. "
            "This method should only be called once per actor instance."
        )
        self.state = await self._configure_state()

    @remote_method
    async def health_check(self) -> None:
        """
        Run a health check on the actor.
        This method can be overridden by subclasses to implement custom health checks.
        """
        return None

    @remote_method
    async def shutdown(self):
        """
        Shutdown the actor.
        """
        pass

    @classmethod
    async def create(
        cls, *, args: ActorArgs, remote_args: Optional[RemoteArgs] = None
    ) -> Self:
        """
        A no-op method that allows the class to be used with `ray.remote`.
        It does not change the type of the class.
        """
        kwargs = remote_args.model_dump(exclude_none=True) if remote_args else {}
        # Just a lil cursed
        instance = cast(Self, ray.remote(**kwargs)(cls).remote(args))
        await instance.configure_state.remote_call()
        return instance


# Passing options
# @overload
# def remote(
#     *,
#     num_returns: Union[int, Literal["streaming"]] = Undefined,
#     num_cpus: Union[int, float] = Undefined,
#     num_gpus: Union[int, float] = Undefined,
#     resources: Dict[str, float] = Undefined,
#     accelerator_type: str = Undefined,
#     memory: Union[int, float] = Undefined,
#     max_calls: int = Undefined,
#     max_restarts: int = Undefined,
#     max_task_retries: int = Undefined,
#     max_retries: int = Undefined,
#     runtime_env: Dict[str, Any] = Undefined,
#     retry_exceptions: bool = Undefined,
#     scheduling_strategy: Union[
#         None, Literal["DEFAULT"], Literal["SPREAD"], PlacementGroupSchedulingStrategy
#     ] = Undefined,
#     label_selector: Dict[str, str] = Undefined,
# ) -> RemoteDecorator:
#     ...


class RemoteArgs(BaseModel):
    num_cpus: Optional[float] = None
    num_gpus: Optional[float] = None
    resources: Optional[dict[str, float]] = None
    runtime_env: Optional[dict[str, Any]] = None
    label_selector: Optional[dict[str, str]] = None
    scheduling_strategy: Optional[
        Union[
            Literal["DEFAULT"], Literal["SPREAD"]
        ]  # , PlacementGroupSchedulingStrategy]
    ] = None


C = TypeVar("C", bound=type[BaseActor])  # “any class object”


def actor_class(
    remote_args: Optional[RemoteArgs] = None,
):  # → returns *the same* class
    """A no-op decorator that preserves the original type."""
    # Get kwargs dict, remove all None values
    kwargs = remote_args.model_dump(exclude_none=True) if remote_args else {}

    def decorator(cls: C) -> C:
        return ray.remote(**kwargs)(cls)  # type: ignore[return-value]

    return decorator


if __name__ == "__main__":

    @actor_class()
    class Greeter(BaseActor[BaseModel, None]):
        @remote_method
        def greet(self, name: str) -> None:
            print(f"Hello, {name}!")

        @remote_method
        async def greet_async(self, name: str) -> int:
            print(f"⚡ custom path\nHello, {name}!")
            return 4

    async def test():
        # g = Greeter(BaseModel())
        g = await Greeter.create(args=BaseModel())
        # g.greet("Alice")  # → Hello, Alice!
        await g.greet.remote_call("Bob")  # → ⚡ custom path
        await g.greet_async.remote_call("Bob")
        #    Hello, Bob!
