from ..base import install_auto_processor_patch
from .bailing_moe_v3_vl import (
    LanguageModel as LanguageModel,
)
from .bailing_moe_v3_vl import (
    Model as Model,
)
from .bailing_moe_v3_vl import (
    VisionModel as VisionModel,
)
from .config import (
    ModelConfig as ModelConfig,
)
from .config import (
    TextConfig as TextConfig,
)
from .config import (
    VisionConfig as VisionConfig,
)
from .processing_bailing_moe_v3_vl import (
    BailingMoeV3VLProcessor as BailingMoeV3VLProcessor,
)

install_auto_processor_patch("bailing_moe_v3_vl", BailingMoeV3VLProcessor)
