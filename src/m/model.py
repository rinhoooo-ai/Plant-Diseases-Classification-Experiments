import torch
import timm
from pathlib import Path

def build_model(num_classes: int, pretrained_path: str = None, device: str = 'cpu'):
    """
    Build ConvNeXt model and optionally load pretrained weights.
    - num_classes: number of classes
    - pretrained_path: path to pretrained weights
    - device: 'cpu' or 'cuda'
    Returns: model on CPU (caller moves to device)
    """

    # If you use another model, change its name here
    model = timm.create_model('convnext_large', pretrained=False, num_classes=num_classes)

    if pretrained_path and Path(pretrained_path).exists():
        try:
            # safe load weights only (avoids FutureWarning)
            state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            loaded = False

            if isinstance(state, dict):
                for key in ('model', 'state_dict', 'model_state', 'state_dict_ema', 'model_state_dict'):
                    if key in state:
                        try:
                            model.load_state_dict(state[key], strict=False)
                            print(f"Loaded weights from checkpoint key: {key}")
                            loaded = True
                            break
                        except Exception as e:
                            print(f"Failed loading from key {key}: {e}")
                if not loaded:
                    # fallback: try matching keys
                    sd = state
                    if any(k.startswith('module.') for k in sd.keys()):
                        sd = {k.replace('module.',''):v for k,v in sd.items()}
                    model.load_state_dict(sd, strict=False)
                    print("Loaded state_dict by best-effort matching (strict=False)")
            else:
                print("Checkpoint not dict, skipping load")
        except Exception as e:
            print("Failed to load pretrained weights:", e)
    else:
        print("No pretrained path found, using random init")

    return model
