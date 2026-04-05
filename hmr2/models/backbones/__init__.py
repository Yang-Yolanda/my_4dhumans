from .vit import vit
from .timm_backbone import timm_backbone

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'timm':
        return timm_backbone(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
