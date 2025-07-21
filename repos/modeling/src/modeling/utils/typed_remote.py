# https://chatgpt.com/share/68769b6d-b38c-8006-8221-e7f845ec2439
from __future__ import annotations

import types
from abc import ABC
from collections.abc import Callable, Coroutine, Mapping
from typing import (
    Any,
    Concatenate,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
    Self,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

import ray
from pydantic import BaseModel, ConfigDict
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

P = ParamSpec("P")  # parameters after  'self'
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
    def remote(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


#! For now, in practice we should not be using async remote methods because Ray
#! does not allow for threading if the method is async for some reason. :/
@overload
def remote_method[SelfT, **P, R](
    func: Callable[Concatenate[SelfT, P], Coroutine[Any, Any, R]],
) -> _WithCustomCall[P, Coroutine[Any, Any, R]]: ...


@overload
def remote_method[SelfT, **P, R](
    func: Callable[Concatenate[SelfT, P], R],
) -> _WithCustomCall[P, Coroutine[Any, Any, R]]: ...


def remote_method(  # type: ignore[override]
    func,
):  # -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], Any]:
    """
    Decorator to turn a normal instance method into one that also exposes
    `.custom_call()`, while preserving the original type signature.
    """

    #! Note: Don't wrap the original function, because ray checks whether the original
    #! function is sync or async and does different things based on that.

    # @functools.wraps(func)
    # def wrapper(self, *args, **kwargs):
    #     # -- normal call path, you can add extra behaviour here
    #     return func(self, *args, **kwargs)

    def _custom(self, *args, **kwargs):
        #! Note that this is never actually called, this is only used for typing
        #! because ray overrides with `setattr(obj,method, ActorMethod(...))` which
        #! ignores the additional properties on the method we add.
        raise NotImplementedError("This method should be called on a remote actor.")
        assert hasattr(func, "remote"), (
            f"Function {func.__name__} does not have a 'remote' attribute."
        )
        return func.remote(self, *args, **kwargs)  # type: ignore [return-value]

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

    func.__get__ = _get_bound  # type: ignore[attr-defined]
    # keep an unbound `.custom_call` so static analysers see it on the class
    func.remote_call = _custom  # type: ignore[attr-defined]

    return func  # type: ignore[return-value]


# ────────────────────────  how you use it  ─────────────────────────


class BaseActor(Generic[ActorArgs], ABC):
    def __init__(self, args: ActorArgs):
        self.args = args

    @remote_method
    def shutdown(self):
        """
        Shutdown the actor.
        """

    @remote_method
    def set_environ(self, env: Mapping[str, str | None]) -> None:
        """
        Set the environment variables for the actor.
        """
        import os

        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @classmethod
    async def create(
        cls, *, args: ActorArgs, remote_args: RemoteArgs | None = None
    ) -> Self:
        """
        A no-op method that allows the class to be used with `ray.remote`.
        It does not change the type of the class.
        """
        kwargs = remote_args.model_dump(exclude_none=True) if remote_args else {}
        # Just a lil cursed
        instance = cast(Self, ray.remote(**kwargs)(cls).remote(args))
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
    model_config = ConfigDict(arbitrary_types_allowed=True)
    num_cpus: float | None = None
    num_gpus: float | None = None
    resources: dict[str, float] | None = None
    runtime_env: dict[str, Any] | None = None
    label_selector: dict[str, str] | None = None
    scheduling_strategy: (
        Literal["DEFAULT", "SPREAD"] | PlacementGroupSchedulingStrategy | None
    ) = None


C = TypeVar("C", bound=type[BaseActor])  # “any class object”


# def actor_class(
#     remote_args: Optional[RemoteArgs] = None,
# ):  # → returns *the same* class
#     """A no-op decorator that preserves the original type."""
#     # Get kwargs dict, remove all None values
#     kwargs = remote_args.model_dump(exclude_none=True) if remote_args else {}

#     def decorator(cls: C) -> C:
#         return ray.remote(**kwargs)(cls)  # type: ignore[return-value]

#     return decorator


if __name__ == "__main__":
    # @actor_class()
    class Greeter(BaseActor[BaseModel]):
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
        await g.greet.remote("Bob")  # → ⚡ custom path
        await g.greet_async.remote("Bob")
        #    Hello, Bob!
