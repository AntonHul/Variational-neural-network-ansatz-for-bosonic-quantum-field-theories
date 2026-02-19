from typing import Callable, Optional, Sequence, Union
from functools import partial
import jax

def _argnums_partial(fun: Callable, args: Sequence, dyn_argnums: Sequence[int]):
    """
    Partially apply `fun` to arguments not in `dyn_argnums`.  
    Returns a function expecting only dynamic args.
    """
    sentinel = object()
    args_template = [sentinel] * len(args)
    dyn_args = []

    for i, arg in enumerate(args):
        if i in dyn_argnums:
            dyn_args.append(arg)
        else:
            args_template[i] = arg

    def fun_partial(*new_dyn_args):
        arg_iter = iter(new_dyn_args)
        interpolated_args = tuple(
            next(arg_iter) if arg == sentinel else arg for arg in args_template
        )
        return fun(*interpolated_args)

    return fun_partial, dyn_args


def vmap(
    fun: Callable,
    in_axes: Union[int, Sequence[Optional[int]]] = 0,
    out_axes: Union[int, Sequence[Optional[int]]] = 0,
    chunk_size: Optional[int] = None,
    *args,
    **kwargs,
) -> Callable:
    """
    Chunked vmap: maps `fun` over leading dimension of inputs in chunks.  
    Falls back to standard `jax.vmap` if `chunk_size` is None.

    Only supports in_axes = 0 or None currently.
    """
    if chunk_size is None:
        return jax.vmap(fun, in_axes, out_axes, *args, **kwargs)

    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    if not set(in_axes).issubset((0, None)):
        raise NotImplementedError("Only in_axes 0/None are currently supported")

    # Identify which arguments are mapped
    argnums = tuple(i for i, ix in enumerate(in_axes) if ix is not None)

    def f_chunked(*args, **kwargs):
        f_partial, dyn_args = _argnums_partial(partial(fun, **kwargs), args, argnums)
        # Map the function over chunks
        return jax.lax.map(lambda args: f_partial(*args), dyn_args, batch_size=chunk_size)

    return f_chunked
