from fvcore.common.registry import Registry

ENCODER1D_REGISTRY = Registry("1D")
ENCODER2D_REGISTRY = Registry("2D")
ENCODER3D_REGISTRY = Registry("3D")


def build_module(modality_type, model_name, **kwargs):
    if modality_type == '1D':
        return ENCODER1D_REGISTRY.get(model_name)()
    elif modality_type == '2D':
        return ENCODER2D_REGISTRY.get(model_name)(**kwargs)
    elif modality_type == '3D':
        return ENCODER3D_REGISTRY.get(model_name)()
    else:
        raise NotImplementedError