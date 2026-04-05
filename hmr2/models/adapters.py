# -*- coding: utf-8 -*-
from typing import Optional, Dict
import torch

try:
    from hmr2.models import load_hmr2
except Exception:
    load_hmr2 = None  # type: ignore

class ModelAdapter:
    def __init__(self, device: torch.device):
        self.device = device
    def infer(self, batch: Dict) -> Dict:
        raise NotImplementedError

class HMR2Adapter(ModelAdapter):
    def __init__(self, model: torch.nn.Module, device: torch.device):
        super().__init__(device)
        self.model = model.to(device)
        self.model.eval()
    @torch.no_grad()
    def infer(self, batch: Dict) -> Dict:
        return self.model(batch)

class ROMPAdapter(ModelAdapter):
    def __init__(self, checkpoint: Optional[str], device: torch.device):
        super().__init__(device)
    def infer(self, batch: Dict) -> Dict:
        raise NotImplementedError("ROMPAdapter 待实现")

class TRACEAdapter(ModelAdapter):
    def __init__(self, checkpoint: Optional[str], device: torch.device):
        super().__init__(device)
    def infer(self, batch: Dict) -> Dict:
        raise NotImplementedError("TRACEAdapter 待实现")

class BEVAdapter(ModelAdapter):
    def __init__(self, checkpoint: Optional[str], device: torch.device):
        super().__init__(device)
    def infer(self, batch: Dict) -> Dict:
        raise NotImplementedError("BEVAdapter 待实现")

def load_model_adapter(method: str, checkpoint: Optional[str], device: torch.device, preloaded_model: Optional[torch.nn.Module] = None) -> ModelAdapter:
    m = method.lower()
    if m == 'hmr2':
        model = preloaded_model
        if model is None:
            if load_hmr2 is None:
                raise RuntimeError('hmr2.models.load_hmr2 不可用')
            model, _ = load_hmr2(checkpoint)
        return HMR2Adapter(model, device)
    if m == 'romp':
        return ROMPAdapter(checkpoint, device)
    if m == 'trace':
        return TRACEAdapter(checkpoint, device)
    if m == 'bev':
        return BEVAdapter(checkpoint, device)
    raise ValueError(f'未知的方法: {method}')