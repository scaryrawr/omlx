import json
import re
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

mx.set_default_device(mx.cpu)

import numpy as np
import pytest
from PIL import Image

from mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7 import (
    MiniCPMV4_7ImageProcessor,
    MiniCPMV4_7Processor,
    MiniCPMV4_7VideoProcessor,
)

pinned_hf = pytest.mark.skipif(
    version("transformers") != "5.18.0",
    reason="Processor oracle requires pinned Transformers 5.18.0",
)


class Tokenizer:
    tokens = {
        "<|image_pad|>": 10,
        "<|video_pad|>": 11,
        "<image>": 12,
        "</image>": 13,
        "<slice>": 14,
        "</slice>": 15,
        "<image_id>": 16,
        "</image_id>": 17,
        "<|im_end|>": 18,
        "<|endoftext|>": 19,
        "\n": 20,
    }
    pad_token_id = 0
    unk_token_id = -1
    eos_token_id = 18
    eos_token = "<|im_end|>"
    pad_token = "<pad>"
    chat_template = "local-template"
    init_kwargs = {}

    def convert_tokens_to_ids(self, token):
        return self.tokens.get(token, self.unk_token_id)

    def encode(self, text, **kwargs):
        pattern = "|".join(
            re.escape(token) for token in sorted(self.tokens, key=len, reverse=True)
        )
        return [
            self.tokens[chunk] if chunk in self.tokens else 100 + ord(chunk)
            for chunk in re.findall(pattern + r"|[\s\S]", text)
        ]

    def decode(self, ids, **kwargs):
        inverse = {value: key for key, value in self.tokens.items()}
        return "".join(
            inverse.get(value, chr(value - 100) if value >= 100 else "")
            for value in ids
        )

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]

    def apply_chat_template(self, messages, **kwargs):
        tools = kwargs.get("tools", [])
        output = json.dumps(tools) + "".join(message["content"] for message in messages)
        if kwargs.get("add_generation_prompt"):
            output += (
                "<think>"
                if kwargs.get("enable_thinking", True)
                else "<think>\n\n</think>\n\n"
            )
        return self.encode(output) if kwargs.get("tokenize") else output


def processor(**kwargs):
    return MiniCPMV4_7Processor(
        image_processor=MiniCPMV4_7ImageProcessor(
            scale_resolution=16, patch_size=2, **kwargs
        ),
        tokenizer=Tokenizer(),
    )


def image(width=16, height=16, color=(255, 0, 0)):
    return Image.new("RGB", (width, height), color)


def test_defaults_and_native_input_names():
    defaults = MiniCPMV4_7ImageProcessor()
    assert (
        defaults.max_slice_nums,
        defaults.scale_resolution,
        defaults.patch_size,
    ) == (9, 448, 14)
    assert defaults.image_mean == defaults.image_std == (0.5, 0.5, 0.5)
    assert (
        defaults.use_image_id
        and defaults.slice_mode
        and defaults.downsample_mode == "16x"
    )
    assert "target_sizes" in processor().model_input_names


@pytest.mark.parametrize("mode,token_count", [("16x", 4), ("4x", 16)])
def test_packed_pixels_rgb_normalization_tokens_and_controls(mode, token_count):
    p = processor()
    output = p(text="see <|image_pad|>", images=image(), downsample_mode=mode)
    assert output["pixel_values"].shape == (1, 3, 2, 128)
    np.testing.assert_array_equal(output["target_sizes"], [[8, 8]])
    np.testing.assert_array_equal(output["pixel_values"][0, 0], 1)
    np.testing.assert_array_equal(output["pixel_values"][0, 1:], -1)
    assert np.count_nonzero(output["mm_token_type_ids"] == 1) == token_count
    assert output["downsample_mode"] == mode
    assert (
        p.image_processor.downsample_mode == p.video_processor.downsample_mode == "16x"
    )


def test_patch_packing_matches_independent_patch_order():
    array = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    output = processor()(text="<|image_pad|>", images=array)
    expected = []
    for top in range(0, 16, 2):
        for left in range(0, 16, 2):
            expected.append(
                array[top : top + 2, left : left + 2].transpose(2, 0, 1)
                / np.float32(255)
                * 2
                - 1
            )
    expected = np.stack(expected, axis=2).reshape(3, 2, -1)
    np.testing.assert_allclose(output["pixel_values"][0], expected, atol=1e-7)


def test_thumbnail_first_row_major_slices_and_official_grid_tie():
    p = processor(max_slice_nums=2)
    # Equal aspect error for 1x2 and 2x1: the pinned release prefers more rows.
    output = p.image_processor(image(24, 24))
    np.testing.assert_array_equal(output["grids"], [[2, 1]])
    assert output["num_patches_per_image"] == [3]
    np.testing.assert_array_equal(output["target_sizes"], [[8, 8], [4, 12], [4, 12]])
    array = np.zeros((16, 32, 3), dtype=np.uint8)
    array[:, :16, 0] = 255
    array[:, 16:, 2] = 255
    output = p(text="<image>", images=array)
    assert output["target_sizes"].shape == (3, 2)
    # Thumbnail uses 64 packed columns; left and right crops follow it.
    sizes = output["target_sizes"].prod(-1) * 2
    slices = np.split(output["pixel_values"][0], np.cumsum(sizes)[:-1], axis=-1)
    assert np.all(slices[1][0] == 1) and np.all(slices[1][2] == -1)
    assert np.all(slices[2][0] == -1) and np.all(slices[2][2] == 1)
    decoded = p.decode(output["input_ids"][0])
    assert decoded.startswith("<image_id>0</image_id><image>")
    assert decoded.count("<slice>") == 2


def test_image_local_ids_reset_per_sample_and_padding_modalities_zero():
    p = processor()
    output = p(
        text=["<|image_pad|><|image_pad|>", "x<|image_pad|>"],
        images=[[image(), image()], [image()]],
    )
    first, second = p.batch_decode(output["input_ids"])
    assert "<image_id>0</image_id>" in first and "<image_id>1</image_id>" in first
    assert "<image_id>0</image_id>" in second and "<image_id>1</image_id>" not in second
    assert np.all(output["mm_token_type_ids"][output["attention_mask"] == 0] == 0)
    assert output["target_sizes"].shape == (3, 2)


def test_video_frames_timestamps_and_mixed_prompt_modality_order():
    p = processor()
    frames = {"frames": [image(color=(0, 0, 255)), image()], "timestamps": [0.5, 1.25]}
    output = p(text="<|video_pad|> then <|image_pad|>", images=image(), videos=frames)
    assert output["pixel_values_videos"].shape == (1, 3, 2, 256)
    np.testing.assert_array_equal(output["target_sizes_videos"], [[8, 8], [8, 8]])
    ids = output["mm_token_type_ids"][0]
    assert np.flatnonzero(ids == 2)[-1] < np.flatnonzero(ids == 1)[0]
    assert p.decode(output["input_ids"][0]).count("<image_id>") == 1
    video = p.video_processor(frames)
    assert video["frame_timestamps"] == [[0.5, 1.25]]
    assert video["num_frames_per_video"] == [2]


def test_existing_video_loader_processor_and_native_embedding_transport(monkeypatch):
    from mlx_vlm import utils
    from mlx_vlm.models.minicpmv4_7 import Model, ModelConfig
    from mlx_vlm.tests.test_minicpmv4_7 import tiny_config

    class CompactTokenizer(Tokenizer):
        tokens = {
            **Tokenizer.tokens,
            "<|image_pad|>": 8,
            "<|video_pad|>": 9,
            "<image>": 10,
            "</image>": 11,
            "<slice>": 12,
            "</slice>": 13,
            "\n": 2,
        }

        def encode(self, text, **kwargs):
            return [
                token if token < 32 else 3 for token in super().encode(text, **kwargs)
            ]

    p = MiniCPMV4_7Processor(
        image_processor=MiniCPMV4_7ImageProcessor(scale_resolution=16, patch_size=2),
        tokenizer=CompactTokenizer(),
    )
    metadata = utils.VideoMetadata(
        total_num_frames=3,
        fps=2.0,
        frames_indices=[0, 2],
    )
    loaded = []

    def load_video(path, sampling, **kwargs):
        loaded.append(path)
        return np.zeros((2, 3, 16, 16), np.uint8), metadata

    monkeypatch.setattr(utils, "load_video", load_video)
    inputs = utils.prepare_inputs(
        p,
        images=[image()],
        videos=["synthetic.mp4"],
        prompts=["<|video_pad|> then <|image_pad|>"],
        downsample_mode="4x",
        max_slice_nums=2,
    )
    assert loaded == ["synthetic.mp4"]
    assert inputs["downsample_mode"] == "4x"
    assert inputs["target_sizes_videos"].shape == (2, 2)
    model = Model(ModelConfig.from_dict(tiny_config()))
    features = model.get_input_embeddings(**inputs)
    assert features.inputs_embeds.shape[:2] == inputs["input_ids"].shape
    assert features.position_ids.shape == (3, *inputs["input_ids"].shape)
    assert features.rope_deltas.shape == (1, 1)
    mx.eval(features.inputs_embeds, features.position_ids, features.rope_deltas)


def test_video_loader_metadata_and_frame_limit():
    p = processor()
    output = p.video_processor(
        [np.zeros((5, 3, 16, 16), dtype=np.uint8)],
        max_num_frames=2,
        video_metadata=[{"frames_indices": [0, 3, 6, 9, 12], "fps": 6}],
    )
    assert output["num_frames_per_video"] == [2]
    assert output["frame_timestamps"] == [[0.0, 2.0]]
    assert p.video_processor.video_sampling_defaults() == {"max_frames": 128}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"downsample_mode": "8x"},
        {"max_slice_nums": 0},
        {"max_slice_nums": -1},
        {"max_slice_nums": True},
        {"max_slice_nums": 2.0},
        {"max_slice_nums": "2"},
        {"stack_frames": 1},
        {"videos_kwargs": {"stack_frames": 2}},
        {"images_kwargs": {"max_slice_nums": False}},
        {"max_num_frames": 0},
        {"pack_image_by_patch": False},
        {"slice_mode": "false"},
    ],
)
def test_invalid_controls_are_rejected_even_without_media(kwargs):
    with pytest.raises((ValueError, TypeError)):
        processor()(text="hello", **kwargs)


@pytest.mark.parametrize(
    "text,images,videos",
    [
        ("hello", image(), None),
        ("<|image_pad|>", None, None),
        ("<|image_pad|><|image_pad|>", image(), None),
        ("hello", None, [image()]),
        ("<|video_pad|>", None, None),
    ],
)
def test_input_placeholder_mismatches(text, images, videos):
    with pytest.raises(ValueError, match="placeholders"):
        processor()(text=text, images=images, videos=videos)


def test_per_sample_mismatch_is_not_hidden_by_matching_total():
    with pytest.raises(ValueError, match="Per-sample"):
        processor()(
            text=["<|image_pad|>", "<|image_pad|>"], images=[[], [image(), image()]]
        )


def test_truncation_cannot_drop_or_split_visual_spans():
    p = processor()
    for maximum in (1, 5, 8):
        with pytest.raises(ValueError, match="Truncation"):
            p(text="<|image_pad|>", images=image(), truncation=True, max_length=maximum)
    output = p(text="text after", truncation=True, max_length=4)
    assert output["input_ids"].shape == (1, 4)


def test_tools_thinking_template_and_eos_are_delegated_unchanged():
    p = processor()
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    rendered = p.apply_chat_template(
        [{"role": "user", "content": "<|image_pad|>"}],
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    assert "lookup" in rendered and "<think>\n\n</think>" in rendered
    output = p(text=rendered, images=image())
    assert "</think>" in p.decode(output["input_ids"][0])
    assert p.post_process_image_text_to_text([[18, 19]]) == ["<|im_end|><|endoftext|>"]
    assert p.tokenizer.eos_token_id == 18


def test_pinned_template_thinking_tools_and_native_media_format():
    from transformers.utils.chat_template_utils import render_jinja_template
    from mlx_vlm.prompt_utils import apply_chat_template

    template = (
        Path(__file__).parent / "fixtures" / "minicpmv4_7_chat_template.jinja"
    ).read_text()

    class OfficialTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            kwargs.pop("tokenize", None)
            kwargs.pop("chat_template", None)
            return render_jinja_template(
                conversations=[messages], chat_template=template, **kwargs
            )[0][0]

    p = MiniCPMV4_7Processor(
        image_processor=MiniCPMV4_7ImageProcessor(scale_resolution=16, patch_size=2),
        tokenizer=OfficialTokenizer(),
    )
    config = {"model_type": "minicpmv4_7"}
    for thinking in (True, False):
        rendered = apply_chat_template(
            p,
            config,
            "describe",
            num_images=2,
            enable_thinking=thinking,
        )
        assert rendered.count("<|image_pad|>") == 2
        assert "<image>" not in rendered
        assert rendered.endswith("<think>\n" if thinking else "<think>\n\n</think>\n\n")
        packed = p(text=rendered, images=[image(), image()])
        assert packed["target_sizes"].shape == (2, 2)
    video = apply_chat_template(p, config, "describe", video="clip.mp4")
    assert "<|video_pad|>" in video
    messages = [
        {"role": "user", "content": "lookup"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"query": "value"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "found", "tool_call_id": "call_1"},
    ]
    rendered = apply_chat_template(
        p,
        config,
        messages,
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )
    assert "<function=lookup>" in rendered
    assert "<parameter=query>\nvalue\n</parameter>" in rendered
    assert "<tool_response>" in rendered and "found" in rendered


def test_concurrent_calls_do_not_leak_slicing_or_compression_controls():
    p = processor()

    def call(index):
        mode, maximum = ("4x", 2) if index % 2 else ("16x", 1)
        return p(
            text="<|image_pad|>",
            images=image(32, 16),
            downsample_mode=mode,
            max_slice_nums=maximum,
            use_image_id=False,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(call, range(24)))
    for index, result in enumerate(results):
        assert result["downsample_mode"] == ("4x" if index % 2 else "16x")
        assert result["target_sizes"].shape[0] == (3 if index % 2 else 1)
        assert "<image_id>" not in p.decode(result["input_ids"][0])
    assert p.image_processor.max_slice_nums == 9
    assert p.image_processor.use_image_id
    assert p.image_processor.downsample_mode == "16x"


def test_empty_metadata_loads_native_local_tokenizer_without_remote_code():
    # No directory fixture or tokenizer checkpoint is needed for this loader test.
    def read_json(directory, filename):
        return (
            {"model_type": "minicpmv4_7", "downsample_mode": "4x"}
            if filename == "config.json"
            else {}
        )

    module = "mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7"
    with (
        patch(module + "._read_json", side_effect=read_json),
        patch(
            module + ".Qwen2TokenizerFast.from_pretrained", return_value=Tokenizer()
        ) as load,
        patch(module + ".load_chat_template"),
    ):
        p = MiniCPMV4_7Processor.from_pretrained(".", trust_remote_code=True)
    assert p.image_processor.downsample_mode == "4x"
    assert p.video_processor.downsample_mode == "4x"
    assert load.call_args.kwargs["trust_remote_code"] is False
    assert load.call_args.kwargs["local_files_only"] is True


@pytest.mark.parametrize("mismatch", [None, "vocab", "marker"])
def test_local_tokenizer_binding_and_extra_token_mapping(mismatch):
    class BoundTokenizer(Tokenizer):
        def __len__(self):
            return 63 if mismatch == "vocab" else 64

    config = {
        "model_type": "minicpmv4_7",
        "text_config": {"vocab_size": 64},
        "image_token_id": 99 if mismatch == "marker" else 10,
        "video_token_id": 11,
        "image_start_id": 12,
        "image_end_id": 13,
        "slice_start_id": 14,
        "slice_end_id": 15,
    }
    extras = {"image_token": "<|image_pad|>"}

    def read_json(directory, name):
        return (
            config
            if name == "config.json"
            else (
                {"extra_special_tokens": extras}
                if name == "tokenizer_config.json"
                else {}
            )
        )

    module = "mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7"
    with (
        patch(module + "._read_json", side_effect=read_json),
        patch(
            module + ".Qwen2TokenizerFast.from_pretrained",
            return_value=BoundTokenizer(),
        ) as load,
        patch(module + ".load_chat_template"),
    ):
        if mismatch:
            with pytest.raises(ValueError, match="tokenizer"):
                MiniCPMV4_7Processor.from_pretrained(".")
        else:
            MiniCPMV4_7Processor.from_pretrained(".")
        assert load.call_args.kwargs["extra_special_tokens"] == extras


def test_native_loader_dispatch_is_strict_and_keeps_both_eos_ids(tmp_path):
    from mlx_vlm.utils import load_processor

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minicpmv4_7",
                "text_config": {"eos_token_id": [18, 21]},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "processor_config.json").write_text("{}", encoding="utf-8")
    module = "mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7"
    with (
        patch(module + ".Qwen2TokenizerFast.from_pretrained", return_value=Tokenizer()),
        patch(module + ".load_chat_template"),
        patch("mlx_vlm.utils.AutoProcessor.from_pretrained") as fallback,
    ):
        loaded = load_processor(tmp_path, add_detokenizer=False)
    assert isinstance(loaded, MiniCPMV4_7Processor)
    assert loaded.additional_eos_token_ids == (18, 21)
    fallback.assert_not_called()
    with (
        patch(
            module + ".Qwen2TokenizerFast.from_pretrained",
            side_effect=ValueError("bad local tokenizer"),
        ),
        patch("mlx_vlm.utils.AutoProcessor.from_pretrained") as fallback,
        pytest.raises(ValueError, match="bad local tokenizer"),
    ):
        load_processor(tmp_path, add_detokenizer=False)
    fallback.assert_not_called()


@pinned_hf
def test_native_media_matches_pinned_hf_pil_backend():
    from transformers.models.minicpmv4_6.image_processing_pil_minicpmv4_6 import (
        MiniCPMV4_6ImageProcessorPil,
    )

    array = np.arange(24 * 40 * 3, dtype=np.uint8).reshape(24, 40, 3)
    native = processor(max_slice_nums=3).image_processor(array)
    official = MiniCPMV4_6ImageProcessorPil(
        scale_resolution=16,
        patch_size=2,
        max_slice_nums=3,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )(Image.fromarray(array), return_tensors="np")
    np.testing.assert_array_equal(native["grids"], official["grids"])
    np.testing.assert_array_equal(native["target_sizes"], official["target_sizes"])
    np.testing.assert_allclose(
        native["pixel_values"], official["pixel_values"], atol=1e-7
    )


@pinned_hf
def test_replacements_match_pinned_hf_native_processor():
    from transformers.models.minicpmv4_7.processing_minicpmv4_7 import (
        MiniCPMV4_7Processor as OfficialProcessor,
    )

    native = processor(max_slice_nums=3)
    # Placeholder generation is pure; no Torch-backed video processor is needed.
    official = OfficialProcessor.__new__(OfficialProcessor)
    for name in (
        "image_token",
        "video_token",
        "image_start_token",
        "image_end_token",
        "slice_start_token",
        "slice_end_token",
        "image_id_start_token",
        "image_id_end_token",
    ):
        setattr(official, name, getattr(native, name))
    official.slice_mode = official.video_slice_mode = True
    image_inputs = native.image_processor([image(40, 24), image()])
    video_inputs = native.video_processor([[image(40, 24), image()]])
    for mode, divisor in (("4x", 4), ("16x", 16)):
        for index in range(2):
            assert native.replace_image_token(
                image_inputs, index, downsample_mode=mode
            ) == official.replace_image_token(
                image_inputs, index, image_token_divisor=divisor
            )
        assert native.replace_video_token(
            video_inputs, 0, downsample_mode=mode
        ) == official.replace_video_token(video_inputs, 0, video_token_divisor=divisor)


def test_real_local_tokenizer_special_tokens_and_concurrent_calls():
    from tokenizers import Tokenizer as Backend
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen2TokenizerFast

    backend = Backend(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=list(Tokenizer.tokens),
    )
    p = MiniCPMV4_7Processor(
        MiniCPMV4_7ImageProcessor(scale_resolution=16, patch_size=2),
        tokenizer=tokenizer,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda mode: p(
                    text="<|image_pad|>", images=image(), downsample_mode=mode
                ),
                ["4x", "16x"] * 4,
            )
        )
    assert [int((row["mm_token_type_ids"] == 1).sum()) for row in results] == [
        16,
        4,
    ] * 4


def test_nested_options_are_consumed_without_mutation():
    p = processor()
    options = {"downsample_mode": "4x", "max_slice_nums": 1}
    nested = {"downsample_mode": "4x", "use_image_id": False}
    output = p(text="<|image_pad|>", images=image(), options=options)
    assert output["downsample_mode"] == "4x"
    output = p(text="<|image_pad|>", images=image(), images_kwargs=nested)
    assert output["downsample_mode"] == "4x"
    assert "<image_id>" not in p.decode(output["input_ids"][0])
    assert options == {"downsample_mode": "4x", "max_slice_nums": 1}
    assert nested == {"downsample_mode": "4x", "use_image_id": False}
    assert p.image_processor.downsample_mode == "16x"
    with pytest.raises(ValueError, match="same downsample"):
        p(
            text="<|image_pad|><|video_pad|>",
            images=image(),
            videos=[image()],
            images_kwargs={"downsample_mode": "4x"},
        )


def test_unmatched_visual_markers_cannot_reach_the_model():
    with pytest.raises(ValueError, match="complete"):
        processor()(text="bad </image>")
    with pytest.raises(ValueError, match="closing"):
        processor()(text="<slice>")


def test_existing_prepare_inputs_api_preserves_native_layout_and_controls():
    from mlx_vlm.utils import prepare_inputs

    output = prepare_inputs(
        processor(),
        images=[image()],
        prompts="<|image_pad|>",
        downsample_mode="4x",
        max_slice_nums=1,
    )
    assert output["downsample_mode"] == "4x"
    assert output["pixel_values"].shape == (1, 3, 2, 128)
    assert output["target_sizes"].shape == (1, 2)
    assert output["mm_token_type_ids"].shape == output["input_ids"].shape
    mx.eval(output["input_ids"], output["pixel_values"])
