
class NotFoundObject():
    pass

NOT_FOUND = NotFoundObject()


def deep_access_nested_dict(nested_dict, keys, strict=True):
    for key in keys:
        nested_dict = nested_dict.get(key, NOT_FOUND)
        
        if isinstance(nested_dict, NotFoundObject):
            if strict:
                raise KeyError(f"Key {key} not found")
            else:
                return None

    return nested_dict
