def format_message(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        role: str = "user",
        skip_image_token: bool = False,
        skip_audio_token: bool = False,
        num_images: int = 1,
        num_audios: int = 1,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """Format a message based on the model type."""

        if isinstance(prompt, list):
            explicit_images = _content_media_count(
                prompt, ("image", "image_url", "input_image")
            )
            if not explicit_images:
                prompt = extract_text_from_content(prompt)
            elif role == "user" and not skip_image_token:
                num_images = max(num_images, explicit_images)

        # Check multi-image support
        if num_images > 1 and self.model_name in SINGLE_IMAGE_ONLY_MODELS:
            raise ValueError(
                f"Model {self.model_name} does not support multi-image chat. "
                f"Please only use 1 image."
            )

        # Handle video format for specific models
        if self.model_name in [
            "qwen2_vl",
            "qwen2_5_vl",
            "mimo_v2",
            "qwen3_vl",
            "mage_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "prism_hadamard_qwen35",
            "qwen3_5_moe",
            "qwen4_exp",
            "qwen3_omni_moe",
            "gemma4",
            "gemma4_unified",
            "diffusion_gemma",
            "minicpmv4_6",
            "minicpmv4_7",
            "minimax_m3_vl",
            "llava_onevision",
            "glm5_next",
        ] and kwargs.get("video"):
            return self._format_video_message(
                prompt,
                role,
                skip_image_token,
                skip_audio_token,
                num_images,
                num_audios,
                **kwargs,
            )

        # Route to appropriate formatter
        formatter_map = {
            MessageFormat.LIST_WITH_IMAGE: self._format_list_with_image,
            MessageFormat.LIST_WITH_IMAGE_FIRST: partial(
                self._format_list_with_image, image_first=True
            ),
            MessageFormat.LIST_WITH_IMAGE_URL_FIRST: partial(
                self._format_list_with_image, image_first=True, use_image_url=True
            ),
            MessageFormat.LIST_WITH_IMAGE_TYPE: self._format_list_with_image_type,
            MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT: partial(
                self._format_list_with_image_type, message_type="text"
            ),
            MessageFormat.LIST_WITH_IMAGE_TYPE_TEXT_IMAGE_LAST: partial(
                self._format_list_with_image_type,
                message_type="text",
                image_first=False,
            ),
            MessageFormat.IMAGE_TOKEN: partial(
                self._format_with_token, token="<image>"
            ),
            MessageFormat.IMAGE_TOKEN_PIPE: partial(
                self._format_with_token, token="<|image|>"
            ),
            MessageFormat.IMAGE_PATCH_TOKEN: partial(
                self._format_with_token, token="<im_patch>"
            ),
            MessageFormat.START_IMAGE_TOKEN: partial(
                self._format_with_token, token="<start_of_image>", image_first=False
            ),
            MessageFormat.IMAGE_TOKEN_NEWLINE: partial(
                self._format_with_token, token="<image>\n"
            ),
            MessageFormat.SINGLE_IMAGE_TOKEN: partial(
                self._format_with_token, token="<image>", repeat_image_token=False
            ),
            MessageFormat.IMAGE_TOKEN_WRAPPED: partial(
                self._format_with_token, token="(<image>./</image>)\n"
            ),
            MessageFormat.NUMBERED_IMAGE_TOKENS: self._format_numbered_tokens,
            MessageFormat.PROMPT_ONLY: lambda *args, **kw: extract_text_from_content(
                prompt
            ),
            MessageFormat.TEXT_ONLY: self._format_text_only,
            MessageFormat.PROMPT_WITH_IMAGE_TOKEN: lambda *args, **kw: "<image>"
            * num_images
            + extract_text_from_content(prompt),
            MessageFormat.PROMPT_WITH_START_IMAGE_TOKEN: lambda *args, **kw: extract_text_from_content(
                prompt
            )
            + "<start_of_image>" * num_images,
            MessageFormat.VIDEO_WITH_TEXT: self._format_video_message,
        }

        formatter = formatter_map.get(self.format_type)
        return formatter(
            prompt,
            role,
            skip_image_token,
            skip_audio_token,
            num_images,
            num_audios,
            **kwargs,
        )
