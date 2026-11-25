# SPDX-License-Identifier: Apache-2.0
"""Safety patch for the mlx-video LTX-2 tokenizer loader."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import transformers


class _SafeAutoTokenizer(transformers.AutoTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs["trust_remote_code"] = False
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


@contextmanager
def safe_ltx_tokenizer() -> Iterator[None]:
    """Prevent the LTX-2 loader from executing checkpoint tokenizer code."""
    with patch.object(transformers, "AutoTokenizer", _SafeAutoTokenizer):
        yield
