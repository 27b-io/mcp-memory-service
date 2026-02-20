"""Verify default min_similarity is 0.5 after v11.10.0 cosine score fix."""

import ast
from pathlib import Path


def test_default_min_similarity_is_0_5():
    """Default min_similarity should be 0.5, not 0.6.

    Parses the AST to check the default value directly, avoiding
    FastMCP tool wrapper complexity.
    """
    server_path = Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "mcp_server.py"
    tree = ast.parse(server_path.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "search":
            for arg, default in zip(
                reversed(node.args.args),
                reversed(node.args.defaults),
            ):
                if arg.arg == "min_similarity":
                    assert isinstance(default, ast.Constant)
                    assert default.value == 0.5, f"Expected 0.5, got {default.value}"
                    return

    raise AssertionError("Could not find min_similarity parameter in search function")
