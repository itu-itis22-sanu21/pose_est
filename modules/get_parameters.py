def get_parameters_conv(model, key):
    for m in model.modules():
        if hasattr(m, key) and m.__class__.__name__.startswith('Conv'):
            yield getattr(m, key)

def get_parameters_bn(model, key):
    for m in model.modules():
        if hasattr(m, key) and m.__class__.__name__.startswith('BatchNorm'):
            yield getattr(m, key)

def get_parameters_conv_depthwise(model, key):
    for m in model.modules():
        if hasattr(m, key) and m.__class__.__name__.startswith('Conv') and m.groups == m.in_channels:
            yield getattr(m, key)
