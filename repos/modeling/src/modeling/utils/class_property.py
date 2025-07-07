class class_property(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super().__init__(fget, fset, fdel, doc)
        # Propagate the abstract flag from the getter (or setter)
        self.__isabstractmethod__ = bool(
            getattr(fget, "__isabstractmethod__", False)
            or getattr(fset, "__isabstractmethod__", False)
        )

    def getter(self, fget):
        # Return a new classproperty, copying over any abstract flag
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)
