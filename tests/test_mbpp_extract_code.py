# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx/eval/mbpp.py's _extract_code helper — the regex/
heuristic parser that pulls Python source out of an LLM response
before MBPP runs it through the subprocess executor. The
benchmark-runner logic itself is exercised by test_eval.py; here we
just pin the code-extraction branches.
"""

from __future__ import annotations

import pytest

from omlx.eval.mbpp import _extract_code


class TestFencedBlocks:
    def test_python_fenced_block_extracted(self):
        response = """Here's the solution:
```python
def add(a, b):
    return a + b
```
Done."""
        assert _extract_code(response) == "def add(a, b):\n    return a + b"

    def test_unspecified_fenced_block_extracted(self):
        """Some models emit ``` without a language tag — must still
        work, otherwise we'd silently fail half the runs."""
        response = """Solution:
```
def mul(a, b):
    return a * b
```"""
        assert _extract_code(response) == "def mul(a, b):\n    return a * b"

    def test_python_fence_wins_over_unspecified(self):
        """When both fence styles appear, the ```python regex runs
        first — the language-tagged block is canonical."""
        response = """Wrong:
```
print("plain")
```
Right:
```python
def f():
    return 42
```"""
        assert _extract_code(response) == "def f():\n    return 42"

    def test_multiline_fenced_block(self):
        response = """```python
import math

def area(r):
    return math.pi * r ** 2

# ready
```"""
        assert _extract_code(response) == (
            "import math\n\ndef area(r):\n    return math.pi * r ** 2\n\n# ready"
        )

    def test_fence_strips_surrounding_whitespace(self):
        """The .strip() inside the regex branch must remove leading/
        trailing whitespace inside the fence — the subprocess executor
        wants clean source."""
        response = "```python\n\n   def f():\n       return 1\n\n```"
        assert _extract_code(response) == "def f():\n       return 1"


class TestHeuristicLineScan:
    def test_def_starts_code_region(self):
        """No fences → scan for first ``def `` line and take everything
        from there. The chatty preamble must be stripped."""
        response = """Sure, here's a solution.
This function adds two numbers.

def add(a, b):
    return a + b"""
        result = _extract_code(response)
        assert result == "def add(a, b):\n    return a + b"

    def test_class_starts_code_region(self):
        response = """Here it is:

class Counter:
    def __init__(self):
        self.n = 0"""
        result = _extract_code(response)
        assert result.startswith("class Counter:")
        assert "self.n = 0" in result

    def test_import_starts_code_region(self):
        response = """The solution uses math:
import math
def area(r):
    return math.pi * r ** 2"""
        result = _extract_code(response)
        assert result.startswith("import math")

    def test_from_import_starts_code_region(self):
        response = """We need typing:
from typing import List
def first(xs: List[int]) -> int:
    return xs[0]"""
        result = _extract_code(response)
        assert result.startswith("from typing import List")

    def test_comment_starts_code_region(self):
        """A leading ``# `` comment is treated as part of the code —
        models sometimes start with ``# Solution`` before the actual
        def."""
        response = """Here's my approach:
# Add two numbers
def add(a, b):
    return a + b"""
        result = _extract_code(response)
        assert result.startswith("# Add two numbers")
        assert "def add" in result

    def test_includes_lines_after_code_start(self):
        """Once code starts, everything (including blank lines and
        trailing text) is included — we don't try to find the END of
        the code, just the start."""
        response = """def f():
    return 1

# This is part of the code now
print(f())"""
        result = _extract_code(response)
        assert "def f():" in result
        assert "print(f())" in result


class TestFallback:
    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            pytest.param(
                "  I don't know how to solve this.  ",
                "I don't know how to solve this.",
                id="no-code-markers",
            ),
            pytest.param("", "", id="empty"),
            pytest.param("   \n\n  \t  ", "", id="whitespace-only"),
            pytest.param("print('hello')", "print('hello')", id="bare-print"),
        ],
    )
    def test_returns_stripped_response(self, response, expected):
        """A bare print is not a code-start marker, so it uses the fallback."""
        assert _extract_code(response) == expected
