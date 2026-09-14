import re

from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


class BailingMoeV3VLProcessor(Qwen3VLProcessor):
    def __init__(self, image_processor, tokenizer, video_processor=None, **kwargs):
        super().__init__(image_processor, tokenizer, video_processor, **kwargs)
        self.video_start_token = (
            getattr(tokenizer, "video_bos_token", None) or "<|video_start|>"
        )
        self.video_end_token = (
            getattr(tokenizer, "video_eos_token", None) or "<|video_end|>"
        )

    def _normalize_media_wrappers(self, text):
        if not isinstance(text, str):
            return text

        image_wrapper = (
            self.vision_start_token + self.image_token + self.vision_end_token
        )
        text = re.sub(
            re.escape(image_wrapper) + r"\n?",
            image_wrapper + "\n",
            text,
        )

        video_wrapper = self.video_start_token + self.video_token + self.video_end_token
        qwen_video_wrapper = (
            self.video_start_token
            + self.vision_start_token
            + self.video_token
            + self.vision_end_token
            + self.video_end_token
        )
        return text.replace(video_wrapper, qwen_video_wrapper)

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        if isinstance(text, list):
            text = [self._normalize_media_wrappers(sample) for sample in text]
        else:
            text = self._normalize_media_wrappers(text)
        return super().__call__(images=images, text=text, videos=videos, **kwargs)


__all__ = ["BailingMoeV3VLProcessor"]
