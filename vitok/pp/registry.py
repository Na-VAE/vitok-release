"""Preprocessing DSL parser.

Build transform pipelines from string specifications like:
    "random_resized_crop(512)|flip|to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)"
"""

from __future__ import annotations

import ast
import re
from functools import reduce
from typing import Any, Callable, Dict, Tuple

from vitok.pp.ops import OPS


def parse_op(op_str: str) -> Tuple[str, Tuple[Any, ...], Dict[str, Any]]:
    """Parse an op string into (name, args, kwargs).

    Examples:
        'resize(256)' -> ('resize', (256,), {})
        'resize(256, 256)' -> ('resize', (256, 256), {})
        'random_resized_crop(256, scale=(0.8, 1.0))' ->
            ('random_resized_crop', (256,), {'scale': (0.8, 1.0)})
        'flip' -> ('flip', (), {})
        'normalize(minus_one_to_one)' -> ('normalize', ('minus_one_to_one',), {})
    """
    op_str = op_str.strip()
    if not op_str:
        raise ValueError("Empty op string")

    # Match: name(args) or just name
    match = re.match(r'^(\w+)(?:\((.*)\))?$', op_str, re.DOTALL)
    if not match:
        raise ValueError(f"Invalid op syntax: '{op_str}'")

    name = match.group(1)
    args_str = match.group(2)

    if args_str is None or args_str.strip() == '':
        return name, (), {}

    # Parse arguments using ast
    # Wrap in a fake function call for parsing
    try:
        fake_call = f"_({args_str})"
        tree = ast.parse(fake_call, mode='eval')
        call_node = tree.body

        args = tuple(_eval_arg(arg) for arg in call_node.args)
        kwargs = {kw.arg: _eval_arg(kw.value) for kw in call_node.keywords}

        return name, args, kwargs
    except SyntaxError as e:
        raise ValueError(f"Invalid arguments in '{op_str}': {e}")


def _eval_arg(node: ast.AST) -> Any:
    """Safely evaluate an AST node to a Python value."""
    # Handle string literals that aren't quoted (like normalize(minus_one_to_one))
    if isinstance(node, ast.Name):
        return node.id
    # Use literal_eval for safe evaluation of literals
    return ast.literal_eval(ast.unparse(node))


def build_transform(pp_string: str) -> Callable:
    """Build a composed transform from a DSL string.

    Args:
        pp_string: Pipe-separated ops, e.g.:
            "random_resized_crop(512)|flip|to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)"

    Returns:
        Callable that applies all ops in sequence.

    Example:
        transform = build_transform("center_crop(256)|flip|to_tensor|normalize(minus_one_to_one)|patchify(256, 16, 256)")
        patch_dict = transform(pil_image)
    """
    if not pp_string or not pp_string.strip():
        return lambda x: x

    ops = []
    for op_str in pp_string.split('|'):
        op_str = op_str.strip()
        if not op_str:
            continue
        name, args, kwargs = parse_op(op_str)
        if name not in OPS:
            available = ", ".join(sorted(OPS.keys()))
            raise KeyError(f"Unknown op: '{name}'. Available: {available}")
        factory = OPS[name]
        ops.append(factory(*args, **kwargs))

    if not ops:
        return lambda x: x

    def composed(x):
        return reduce(lambda v, f: f(v), ops, x)

    return composed


__all__ = ["build_transform", "parse_op", "OPS"]
