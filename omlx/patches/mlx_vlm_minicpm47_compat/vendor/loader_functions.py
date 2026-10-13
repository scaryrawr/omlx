def update_module_configs(model_config, model_class, config, modules):
    """Updates configuration for model modules like text and vision modules.

    Args:
        model_config: The model configuration object that will be updated
        model_class: The model class containing component config classes
        config: Dictionary containing configuration parameters
        modules: List of module names to update configs for (e.g. ["text", "vision"])

    Returns:
        The updated model_config object
    """
    for config_name in modules:
        config_attr = f"{config_name}_config"
        if hasattr(model_config, config_attr) and config.get(config_attr) is not None:
            config_class = getattr(model_class, f"{config_name.title()}Config")
            if isinstance(getattr(model_config, config_attr), config_class):
                continue
            setattr(
                model_config, config_attr, config_class.from_dict(config[config_attr])
            )
    return model_config

def load_processor(
    model_path, add_detokenizer=True, eos_token_ids=None, **kwargs
) -> ProcessorMixin:
    config_path = Path(model_path) / "config.json"
    config = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    if config.get("model_type") == "minicpmv4_7":
        from .models.minicpmv4_7 import Processor

        processor = Processor.from_pretrained(model_path, **kwargs)
    else:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    if add_detokenizer:
        detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)

        # Get the tokenizer object
        tokenizer_obj = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

        # Non-text models (depth, detection) have no decode(); skip detokenizer
        try:
            processor.detokenizer = detokenizer_class(tokenizer_obj)
        except AttributeError:
            return processor

        # Create and assign the StoppingCriteria
        criteria = StoppingCriteria(
            eos_token_ids,
            tokenizer_obj,
            additional_eos_token_ids=getattr(processor, "additional_eos_token_ids", ()),
        )
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.stopping_criteria = criteria
        else:
            processor.stopping_criteria = criteria

    return processor
