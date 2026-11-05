# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.tokenizer module."""

import json

import pytest

from omlx.utils.tokenizer import (
    apply_qwen3_fix,
    create_streaming_detokenizer,
    get_tokenizer_config,
    is_gemma4_model,
    is_harmony_model,
    is_qwen3_model,
    repair_misconverted_unlimited_ocr_tokenizer,
)


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _spm_decoder(strip_space=True):
    decoders = [
        {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
    ]
    if strip_space:
        decoders.append({"type": "Strip", "content": " ", "start": 1, "stop": 0})
    return {"type": "Sequence", "decoders": decoders}


class _ByteFallbackTokenizer:

    def __len__(self):
        return max(self.vocab.values()) + 1

    def convert_ids_to_tokens(self, ids):
        reverse = {v: k for k, v in self.vocab.items()}
        return [reverse[i] for i in ids]

    clean_up_tokenization_spaces = False
    vocab = {
        "<pad>": 0,
        "<0xEC>": 1,
        "<0x9E>": 2,
        "<0xA0>": 3,
    }

    def decode(self, token_ids, skip_special_tokens: bool = True):
        table = {
            0: b"",
            1: bytes([0xEC]),
            2: bytes([0x9E]),
            3: bytes([0xA0]),
        }
        raw = b"".join(table[token_id] for token_id in token_ids)
        if not raw:
            return ""
        if raw == bytes([0xEC, 0x9E, 0xA0]):
            return "\uc7a0"
        return "\ufffd" * sum(1 for token_id in token_ids if token_id != 0)


class _BpeTokenizer:

    def __len__(self):
        return max(self.vocab.values()) + 1

    def convert_ids_to_tokens(self, ids):
        reverse = {v: k for k, v in self.vocab.items()}
        return [reverse[i] for i in ids]

    clean_up_tokenization_spaces = False
    vocab = {"A": 0, "B": 1}

    def decode(self, token_ids, skip_special_tokens: bool = True):
        reverse = {token_id: token for token, token_id in self.vocab.items()}
        return "".join(reverse[token_id] for token_id in token_ids)


class BPEStreamingDetokenizer:
    __module__ = "mlx_vlm.tokenizer_utils"

    def reset(self):
        pass


class _MlxVlmBpeTokenizer:

    def __len__(self):
        return max(self.vocab.values()) + 1

    def convert_ids_to_tokens(self, ids):
        reverse = {v: k for k, v in self.vocab.items()}
        return [reverse[i] for i in ids]

    clean_up_tokenization_spaces = False

    def __init__(self, vocab):
        self.vocab = vocab
        self.detokenizer = BPEStreamingDetokenizer()

    def decode(self, token_ids, skip_special_tokens: bool = True):
        reverse = {token_id: token for token, token_id in self.vocab.items()}
        return "".join(reverse[token_id] for token_id in token_ids)


class _ExplicitNoDetokenizer:
    detokenizer = None

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return ""


def _bpe_byte_chars(*byte_values):
    from mlx_lm.tokenizer_utils import _byte_decoder

    byte_encoder = {byte_value: char for char, byte_value in _byte_decoder().items()}
    return [byte_encoder[byte_value] for byte_value in byte_values]


def _make_misconverted_unlimited_ocr_tokenizer(
    tmp_path,
    *,
    model_type="unlimited-ocr",
):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    ni_bytes = _bpe_byte_chars(0xE4, 0xBD, 0xA0)
    vocab = {
        "Ġ": 0,
        "Ċ": 1,
        "A": 2,
        "B": 3,
        ni_bytes[0]: 4,
        ni_bytes[1]: 5,
        ni_bytes[2]: 6,
    }
    backend = Tokenizer(
        models.BPE(
            vocab=vocab,
            merges=[],
            fuse_unk=True,
            byte_fallback=True,
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        prepend_scheme="always",
        split=False,
    )
    backend.decoder = decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(" ", 1, 0),
        ]
    )
    backend.save(str(tmp_path / "tokenizer.json"))
    _write_json(tmp_path / "config.json", {"model_type": model_type})
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tmp_path / "tokenizer.json"))
    return tokenizer, ni_bytes


class TestCreateStreamingDetokenizer:
    def test_uses_spm_decoder_from_tokenizer_json(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": _spm_decoder()})

        detokenizer = create_streaming_detokenizer(
            _ByteFallbackTokenizer(),
            model_path=tmp_path,
        )
        assert detokenizer is not None

        parts = []
        for token_id in [1, 2, 3]:
            detokenizer.add_token(token_id)
            parts.append(detokenizer.last_segment)

        assert "".join(parts) == "\uc7a0"

    def test_uses_bpe_decoder_from_tokenizer_json(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": {"type": "ByteLevel"}})

        detokenizer = create_streaming_detokenizer(
            _BpeTokenizer(),
            model_path=tmp_path,
        )

        assert type(detokenizer).__name__ == "BPEStreamingDetokenizer"

    def test_replaces_mlx_vlm_bpe_detokenizer_from_tokenizer_json(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": {"type": "ByteLevel"}})
        chars = _bpe_byte_chars(0xEC, 0x9E, 0xA0, 0x20)
        tokenizer = _MlxVlmBpeTokenizer(
            {
                chars[0]: 0,
                chars[1]: 1,
                chars[2]: 2,
                chars[3] + "A": 3,
            }
        )

        detokenizer = create_streaming_detokenizer(tokenizer, model_path=tmp_path)

        assert type(detokenizer).__module__ == "mlx_lm.tokenizer_utils"
        parts = []
        for token_id in [0, 1, 2, 3]:
            detokenizer.add_token(token_id)
            parts.append(detokenizer.last_segment)
        detokenizer.finalize()
        parts.append(detokenizer.last_segment)

        assert "".join(parts) == "\uc7a0 A"

    def test_mlx_vlm_bpe_replacement_buffers_incomplete_utf8(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": {"type": "ByteLevel"}})
        lead_byte, space = _bpe_byte_chars(0xEC, 0x20)
        tokenizer = _MlxVlmBpeTokenizer({lead_byte: 0, space: 1})

        detokenizer = create_streaming_detokenizer(tokenizer, model_path=tmp_path)

        detokenizer.add_token(0)
        detokenizer.add_token(1)

        assert detokenizer.last_segment == ""

    def test_decoder_aware_detokenizer_scans_the_vocabulary_once(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": {"type": "ByteLevel"}})
        tokenizer = _MlxVlmBpeTokenizer({"A": 0, "B": 1})
        scans = []
        convert = tokenizer.convert_ids_to_tokens
        tokenizer.convert_ids_to_tokens = lambda ids: (scans.append(len(ids)), convert(ids))[1]

        first = create_streaming_detokenizer(tokenizer, model_path=tmp_path)
        second = create_streaming_detokenizer(tokenizer, model_path=tmp_path)

        # One vocabulary conversion; each request still gets its own instance
        # with fresh streaming state, sharing only the immutable token table.
        assert scans == [2]
        assert first is not second
        assert first.tokenmap is second.tokenmap
        first.add_token(0)
        first.add_token(1)
        first.finalize()
        assert first.text == "AB"
        assert second.tokens == [] and second.text == ""
        second.add_token(1)
        second.finalize()
        assert second.text == "B"
        assert first.text == "AB"

    def test_unhashable_tokenizer_still_gets_fresh_detokenizers(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": {"type": "ByteLevel"}})

        class _Unhashable(_MlxVlmBpeTokenizer):
            __hash__ = None

        tokenizer = _Unhashable({"A": 0, "B": 1})
        first = create_streaming_detokenizer(tokenizer, model_path=tmp_path)
        second = create_streaming_detokenizer(tokenizer, model_path=tmp_path)

        assert first is not second
        assert first.tokenmap is not second.tokenmap
        first.add_token(0)
        first.finalize()
        assert first.text == "A"
        assert second.text == ""

    def test_explicit_none_detokenizer_without_model_path_stays_none(self):
        assert create_streaming_detokenizer(_ExplicitNoDetokenizer()) is None

    def test_missing_tokenizer_json_uses_naive_fallback(self, tmp_path):
        detokenizer = create_streaming_detokenizer(
            _ByteFallbackTokenizer(),
            model_path=tmp_path,
        )

        assert type(detokenizer).__name__ in {
            "NaiveStreamingDetokenizer",
            "_CompatNaiveStreamingDetokenizer",
        }
        for token_id in [1, 2, 3]:
            detokenizer.add_token(token_id)
        detokenizer.finalize()

        assert detokenizer.text == "\uc7a0"


class TestRepairMisconvertedUnlimitedOCRTokenizer:
    def test_repairs_prompt_encoding_and_utf8_decode(self, tmp_path):
        tokenizer, _ = _make_misconverted_unlimited_ocr_tokenizer(tmp_path)

        assert tokenizer.encode(" A\nB", add_special_tokens=False) == [2, 3]
        assert tokenizer.decode([4, 5, 6]) != "你"

        repaired = repair_misconverted_unlimited_ocr_tokenizer(
            tokenizer,
            model_path=tmp_path,
        )

        assert repaired is True
        assert tokenizer.encode(" A\nB", add_special_tokens=False) == [0, 2, 1, 3]
        assert tokenizer.decode([0, 2, 1, 3]) == " A\nB"
        assert tokenizer.decode([4, 5, 6]) == "你"
        assert tokenizer.backend_tokenizer.model.fuse_unk is False
        assert tokenizer.backend_tokenizer.model.byte_fallback is False

    def test_uses_fresh_bpe_detokenizer_for_misconverted_export(self, tmp_path):
        tokenizer, _ = _make_misconverted_unlimited_ocr_tokenizer(tmp_path)

        first = create_streaming_detokenizer(tokenizer, model_path=tmp_path)
        second = create_streaming_detokenizer(tokenizer, model_path=tmp_path)

        assert type(first).__module__ == "mlx_lm.tokenizer_utils"
        assert type(first).__name__ == "BPEStreamingDetokenizer"
        assert first is not second

        parts = []
        for token_id in [0, 2, 4, 5, 6, 1]:
            first.add_token(token_id)
            parts.append(first.last_segment)
        first.finalize()
        parts.append(first.last_segment)

        assert "".join(parts) == "A你\n"

    def test_leaves_non_unlimited_model_untouched(self, tmp_path):
        tokenizer, _ = _make_misconverted_unlimited_ocr_tokenizer(
            tmp_path,
            model_type="llama",
        )

        repaired = repair_misconverted_unlimited_ocr_tokenizer(
            tokenizer,
            model_path=tmp_path,
        )

        assert repaired is False
        assert tokenizer.encode(" A\nB", add_special_tokens=False) == [2, 3]

    def test_leaves_canonical_unlimited_tokenizer_untouched(self, tmp_path):
        tokenizer, _ = _make_misconverted_unlimited_ocr_tokenizer(tmp_path)
        tokenizer_content = json.loads((tmp_path / "tokenizer.json").read_text())
        tokenizer_content["pre_tokenizer"] = {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        }
        tokenizer_content["decoder"] = {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        }
        _write_json(tmp_path / "tokenizer.json", tokenizer_content)

        repaired = repair_misconverted_unlimited_ocr_tokenizer(
            tokenizer,
            model_path=tmp_path,
        )

        assert repaired is False


class TestIsHarmonyModel:
    """Test cases for is_harmony_model function."""

    @pytest.mark.parametrize(
        "name, config, expected",
        [
            ("some-model", {"model_type": "gpt_oss"}, True),
            ("gpt-oss-1.0", None, True),
            ("GPT-OSS-v2", None, True),
            ("my-gpt-oss-model", None, True),
            ("gptoss", None, True),
            ("GPTOSS-large", None, True),
            ("my-gptoss", None, True),
            ("llama-3.1-8b", None, False),
            ("qwen2.5-32b", None, False),
            ("mistral-7b", None, False),
            ("some-model", {"model_type": "llama"}, False),
            pytest.param("", None, False, id="empty-name"),
            ("gpt-oss", None, True),
            ("llama", None, False),
            ("gpt-oss", {}, True),
            ("llama", {}, False),
        ],
    )
    def test_detects_harmony_model(self, name, config, expected):
        assert is_harmony_model(name, config) is expected


class TestIsGemma4Model:
    """Test cases for is_gemma4_model function."""

    @pytest.mark.parametrize(
        "name, config, expected",
        [
            ("some-model", {"model_type": "gemma4"}, True),
            ("some-model", {"model_type": "gemma4_unified"}, True),
            ("google/gemma-4b", None, True),
            ("GEMMA-4-27B", None, True),
            ("my-gemma4-model", None, True),
            ("gemma-3-27b", None, False),
            ("llama-3.1-8b", None, False),
            ("some-model", {"model_type": "gemma"}, False),
        ],
    )
    def test_detects_gemma4_model(self, name, config, expected):
        assert is_gemma4_model(name, config) is expected


class TestIsQwen3Model:
    """Test cases for is_qwen3_model function."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("qwen3-8b", True),
            ("my-qwen3-model", True),
            ("qwen3", True),
            ("Qwen3-8B", True),
            ("My-Qwen3-Model", True),
            ("Qwen3", True),
            ("qwen2.5-32b", False),
            ("Qwen2-7B", False),
            ("llama-3.1", False),
            ("qwen-7b", False),
            pytest.param("", False, id="empty-name"),
            # Preserve the current substring-matching behavior.
            ("qwen30-model", True),
        ],
    )
    def test_detects_qwen3_model(self, name, expected):
        assert is_qwen3_model(name) is expected


class TestGetTokenizerConfig:
    """Test cases for get_tokenizer_config function."""

    def test_basic_config(self):
        """Test basic config generation."""
        config = get_tokenizer_config("llama-3.1-8b")
        assert "trust_remote_code" in config
        assert config["trust_remote_code"] is False

    def test_config_with_trust_remote_code(self):
        """Test config with trust_remote_code enabled."""
        config = get_tokenizer_config("some-model", trust_remote_code=True)
        assert config["trust_remote_code"] is True

    def test_laguna_config_enables_mistral_regex_fix(self, tmp_path):
        """Laguna's Mistral-derived tokenizer needs the corrected regex."""
        _write_json(
            tmp_path / "config.json",
            {
                "model_type": "laguna",
                "architectures": ["LagunaForCausalLM"],
            },
        )

        config = get_tokenizer_config(str(tmp_path))

        assert config["fix_mistral_regex"] is True

    def test_laguna_config_pins_laguna_tool_parser(self, tmp_path):
        """Laguna templates contain <arg_key>, which mlx-lm's template
        sniffing misreads as glm47; the vendored parser must be pinned."""
        _write_json(
            tmp_path / "config.json",
            {
                "model_type": "laguna",
                "architectures": ["LagunaForCausalLM"],
            },
        )

        config = get_tokenizer_config(str(tmp_path))

        assert config["tool_parser_type"] == "laguna"

    def test_qwen3_model_config(self):
        """Test Qwen3 model gets eos_token fix."""
        config = get_tokenizer_config("qwen3-8b")
        assert config["eos_token"] == "<|im_end|>"

    def test_non_qwen3_model_no_eos_fix(self):
        """Test non-Qwen3 models don't get eos_token."""
        config = get_tokenizer_config("llama-3.1-8b")
        assert "eos_token" not in config

    def test_qwen3_with_trust_remote_code(self):
        """Test Qwen3 model with trust_remote_code."""
        config = get_tokenizer_config("Qwen3-72B", trust_remote_code=True)
        assert config["trust_remote_code"] is True
        assert config["eos_token"] == "<|im_end|>"


class TestApplyQwen3Fix:
    """Test cases for apply_qwen3_fix function."""

    def test_apply_fix_to_qwen3(self):
        """Test applying Qwen3 fix."""
        config = {"trust_remote_code": True}
        result = apply_qwen3_fix(config, "qwen3-8b")
        assert result["eos_token"] == "<|im_end|>"
        assert result["trust_remote_code"] is True

    def test_no_fix_for_non_qwen3(self):
        """Test no fix applied for non-Qwen3 models."""
        config = {"trust_remote_code": True}
        result = apply_qwen3_fix(config, "llama-3.1-8b")
        assert "eos_token" not in result
        assert result["trust_remote_code"] is True

    def test_apply_fix_modifies_original(self):
        """Test that apply_qwen3_fix modifies the original config."""
        config = {"trust_remote_code": True}
        result = apply_qwen3_fix(config, "qwen3-8b")
        # The function modifies in place and returns the same dict
        assert config is result
        assert config["eos_token"] == "<|im_end|>"

    def test_apply_fix_overwrites_existing_eos(self):
        """Test that apply_qwen3_fix overwrites existing eos_token."""
        config = {"eos_token": "<|endoftext|>"}
        result = apply_qwen3_fix(config, "qwen3-8b")
        assert result["eos_token"] == "<|im_end|>"

    def test_apply_fix_empty_config(self):
        """Test applying fix to empty config."""
        config = {}
        result = apply_qwen3_fix(config, "qwen3-8b")
        assert result["eos_token"] == "<|im_end|>"

    def test_apply_fix_preserves_other_keys(self):
        """Test that apply_qwen3_fix preserves other config keys."""
        config = {
            "trust_remote_code": True,
            "use_fast": True,
            "padding_side": "left",
        }
        result = apply_qwen3_fix(config, "qwen3-8b")
        assert result["trust_remote_code"] is True
        assert result["use_fast"] is True
        assert result["padding_side"] == "left"
        assert result["eos_token"] == "<|im_end|>"


class TestMistralCommonTokenizerConfig:
    """get_tokenizer_config forces the HF-native backend for mistral-common repos.

    transformers routes repos that ship tekken.json (Devstral 2 / Mistral
    Small Tekken exports) to MistralCommonBackend, whose rendered chat
    template cannot be re-encoded faithfully — control tokens become literal
    text and every prompt built via render-then-encode is corrupted. Passing
    fix_mistral_regex selects TokenizersBackend instead (gate shipped in
    transformers 5.12.1, the pyproject floor).
    """

    def _make_mistral_repo(self, tmp_path):
        _write_json(tmp_path / "tekken.json", {"version": "v13"})
        _write_json(tmp_path / "tokenizer.json", {"version": "1.0"})
        return tmp_path

    def test_mistral_common_repo_gets_fix_mistral_regex(self, tmp_path):
        repo = self._make_mistral_repo(tmp_path)
        config = get_tokenizer_config(str(repo))
        assert config["fix_mistral_regex"] is True

    def test_symlinked_repo_detected(self, tmp_path):
        """Served model dirs are frequently symlinks; is_file() must follow."""
        real = tmp_path / "real"
        real.mkdir()
        self._make_mistral_repo(real)
        link = tmp_path / "link"
        link.symlink_to(real)
        config = get_tokenizer_config(str(link))
        assert config["fix_mistral_regex"] is True

    def test_plain_repo_without_tekken_json_untouched(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"version": "1.0"})
        config = get_tokenizer_config(str(tmp_path))
        assert "fix_mistral_regex" not in config

    def test_audio_only_export_without_tokenizer_json_untouched(self, tmp_path):
        """Voxtral-shaped exports ship tekken.json with no HF-native
        tokenizer.json; there is no TokenizersBackend to select — do not
        inject the kwarg."""
        _write_json(tmp_path / "tekken.json", {"version": "v13"})
        config = get_tokenizer_config(str(tmp_path))
        assert "fix_mistral_regex" not in config

    def test_hf_repo_id_untouched(self):
        """Remote repo ids are not local paths; detection stays conservative."""
        config = get_tokenizer_config("mlx-community/Devstral-Small-2-24B-4bit")
        assert "fix_mistral_regex" not in config

    def test_other_family_fixes_unaffected(self, tmp_path):
        repo = self._make_mistral_repo(tmp_path)
        config = get_tokenizer_config(str(repo))
        assert "eos_token" not in config
