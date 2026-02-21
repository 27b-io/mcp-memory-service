"""Verify default min_similarity is 0.3 after #115 threshold reduction."""

import ast
from pathlib import Path


def test_default_min_similarity_is_0_3():
    """Default min_similarity should be 0.3 (lowered from 0.5 in #115).

    0.5 was too strict — silently dropped relevant results for short/fuzzy queries.
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
                    assert default.value == 0.3, f"Expected 0.3, got {default.value}"
                    return

    raise AssertionError("Could not find min_similarity parameter in search function")
