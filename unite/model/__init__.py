import os
from unite.utils import print_red

AVAILABLE_MODELS = {
    "modeling_unite": "UniteQwen2VL, UniteQwen2VLConfig",
}

for model_name, model_classes in AVAILABLE_MODELS.items():

    exec(f"from .{model_name} import {model_classes}")
    try:
        exec(f"from .{model_name} import {model_classes}")
    except ImportError:
        print_red(f"Failed to import {model_name} from unite.model.{model_name}")
        pass
