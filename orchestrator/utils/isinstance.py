from typing import Any, Union


def isinstance_no_import(object: Any,
                         classinfo: Union[str, tuple[str, ...]]) -> bool:
    """
    Replacement for isinstance to avoid importing modules for type checking

    :param object: object to query
    :type object: Any
    :param classinfo: string of the class name to check against. Simple
        replacement of the class object with a string of the same name.
    :type classinfo: str
    :returns: True if object is of type classinfo (including inheritance),
        False otherwise
    :rtype: bool
    """
    mro_class_list = [t.__name__ for t in type(object).__mro__]
    if isinstance(classinfo, str):
        return True if classinfo in mro_class_list else False
    elif isinstance(classinfo, tuple):
        for c in classinfo:
            if c in mro_class_list:
                return True
        return False
    else:
        raise ValueError('classinfo is not a string or tuple!')
