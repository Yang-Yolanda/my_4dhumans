import torch
import torch.nn as nn
import timm
import os

class TimmBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_name = cfg.MODEL.BACKBONE.get('MODEL_NAME', 'fastvit_sa24')
        pretrained = cfg.MODEL.BACKBONE.get('PRETRAINED', True)
        
        # Create model without classifier
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained and (model_name != 'fastvit_sa24'), # Disable auto-download for fastvit
            num_classes=0,
            global_pool='',
        )
        
        # Local weight fallback for fastvit_sa24
        if model_name == 'fastvit_sa24' and pretrained:
            local_weight = '/home/yangz/weights/fastvit_sa24.safetensors'
            if os.path.exists(local_weight):
                print(f"   [TimmBackbone] Loading local weights from {local_weight}")
                from safetensors.torch import load_file
                try:
                    state_dict = load_file(local_weight)
                    self.model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    # Fallback to standard torch load if safetensors fails
                    print(f"   [TimmBackbone] safetensors load failed, trying torch.load: {e}")
                    self.model.load_state_dict(torch.load(local_weight), strict=False)
            else:
                print(f"   [TimmBackbone] ⚠️ Local weight {local_weight} not found. Attempting online download.")
                self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')

        # Determine output dimension
        with torch.no_grad():
            dummy_in = torch.zeros(1, 3, 256, 192)
            dummy_out = self.model(dummy_in)
            if isinstance(dummy_out, (list, tuple)):
                dummy_out = dummy_out[-1]
            self.num_features = dummy_out.shape[1]
            print(f"   [TimmBackbone] Created {model_name}, Output Channels: {self.num_features}")

    def forward(self, x):
        # x is (B, 3, 256, 192) or similar
        # Most timm models return (B, C, H, W) when global_pool=''
        feat = self.model(x)
        
        # Handle multi-scale outputs if needed
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
            
        return feat

def timm_backbone(cfg):
    return TimmBackbone(cfg)
