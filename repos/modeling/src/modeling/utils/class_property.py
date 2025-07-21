from abc import abstractmethod


# TODO: Make this not show up on pydantic
class class_property:
    def __init__(self, fget):
        self.fget = fget
        self.__doc__ = fget.__doc__
        # pick up the abstract flag from the function
        self.__isabstractmethod__ = getattr(fget, "__isabstractmethod__", False)

    def __get__(self, obj, owner=None):
        owner = owner or type(obj)
        return self.fget(owner)

    # allow stacking .getter() if you need it
    def getter(self, fget):
        return type(self)(fget)


def abstract_classproperty(func):
    # first mark the function as abstract, then wrap it
    return class_property(abstractmethod(func))


# —— example —— #


# class MyBase(metaclass=ABCMeta):
#     @abstract_classproperty
#     def identifier(cls):
#         """Must be overridden as a class-level property."""


# class Good(MyBase):
#     @class_property
#     def identifier(cls):
#         return "I work"


# class Bad(MyBase):
#     pass


# print(Good.identifier)   # → "I work"
# _ = Bad()                # → TypeError: Can't instantiate abstract class Bad with abstract methods identifier
