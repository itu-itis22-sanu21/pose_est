import torch

def load_state(net, checkpoint):
    """
    Load a full checkpoint into the model.
    Arguments:
        net: model instance
        checkpoint: checkpoint dict loaded by torch.load()
    """
    net.load_state_dict(checkpoint['state_dict'])
