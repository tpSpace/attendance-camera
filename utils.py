from typing import Union, Tuple, TypeVar

T = TypeVar('T')
Result = Union[Tuple[T, None], Tuple[None, Exception]]

def Ok(val: T) -> Result[T]:
    return val, None

def Err(err: Union[str,Exception]) -> Result:
    if isinstance(err, str):
        err = Exception(err)
    return None, err
