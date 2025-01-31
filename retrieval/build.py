from fvcore.common.registry import Registry

EVALUATION_REGISTRY = Registry("Inference")

def build_evaluation_module(cfg):
    return EVALUATION_REGISTRY.get(cfg.inference_module)(cfg)