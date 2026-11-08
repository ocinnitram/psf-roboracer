"""Parse vehicle/params.h and write vehicle/params.yaml. stdlib only."""

from __future__ import annotations

import ast
import math
import operator
import re
import sys
from pathlib import Path

# Capture the type, name, and expression of one numeric C++ constant.
CONST_RE = re.compile(
    r"^\s*(?:static\s+)?constexpr\s+(double|int)\s+(k[A-Za-z0-9]+)\s*=\s*(.*?)\s*;"
)

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def yaml_key(kname: str) -> str:
    body = kname[1:]
    out = []
    for i, ch in enumerate(body):
        if ch.isupper():
            if i > 0:
                prev = body[i - 1]
                nxt = body[i + 1] if i + 1 < len(body) else ""
                if prev.islower() or prev.isdigit() or (nxt.islower() and prev.isupper()):
                    out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _to_python_expr(expr: str) -> str:
    return (
        expr.replace("std::asin", "asin")
        .replace("std::tan", "tan")
        .replace("M_PI", "pi")
    )


class _SafeEval(ast.NodeVisitor):
    def __init__(self, names: dict) -> None:
        self.names = names

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"disallowed operator {type(node.op).__name__}")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"disallowed unary {type(node.op).__name__}")
        return op(self.visit(node.operand))

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"disallowed constant {node.value!r}")

    def visit_Name(self, node: ast.Name):
        if node.id not in self.names:
            raise ValueError(f"unknown name {node.id}")
        return self.names[node.id]

    def visit_Call(self, node: ast.Call):
        if node.keywords:
            raise ValueError("keyword calls are not allowed")
        func = self.visit(node.func)
        if not callable(func):
            raise ValueError("call target is not a function")
        args = [self.visit(a) for a in node.args]
        return func(*args)

    def generic_visit(self, node: ast.AST):
        raise ValueError(f"disallowed expression {type(node).__name__}")


def eval_expr(expr: str, names: dict):
    tree = ast.parse(_to_python_expr(expr), mode="eval")
    return _SafeEval(names).visit(tree)


def format_yaml_value(value, type_name: str) -> str:
    if type_name == "int":
        return str(int(value))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value {number}")
    text = format(number, ".15g")
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text


def parse_header(text: str) -> list[tuple[str, str, object]]:
    names = {
        "pi": math.pi,
        "asin": math.asin,
        "tan": math.tan,
    }
    rows: list[tuple[str, str, object]] = []
    for line in text.splitlines():
        match = CONST_RE.match(line)
        if match is None:
            continue
        type_name, kname, expr = match.group(1), match.group(2), match.group(3)
        value = eval_expr(expr, names)
        names[kname] = value
        rows.append((yaml_key(kname), type_name, value))
    return rows


def render_yaml(rows: list[tuple[str, str, object]]) -> str:
    lines = ["# generated from vehicle/params.h; do not edit"]
    for key, type_name, value in rows:
        lines.append(f"{key}: {format_yaml_value(value, type_name)}")
    lines.append("")
    return "\n".join(lines)


def generate(header_path: Path, output_path: Path) -> None:
    rows = parse_header(header_path.read_text())
    if not rows:
        raise SystemExit(f"no constexpr kName lines in {header_path}")
    output_path.write_text(render_yaml(rows))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    vehicle_dir = Path(__file__).resolve().parent
    header = Path(argv[0]) if len(argv) >= 1 else vehicle_dir / "params.h"
    output = Path(argv[1]) if len(argv) >= 2 else vehicle_dir / "params.yaml"
    generate(header, output)


if __name__ == "__main__":
    main()
