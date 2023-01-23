def freeze_parameters(model):
    """
    freezes model paramters
    model: model
    returns: model with freezed paramters
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_parameters(model):
    """
    unfreezes model paramters
    model: model
    returns: model with unfreezed paramters
    """
    for param in model.parameters():
        param.requires_grad = True
    return model


# loading and saving
def load_model(model, name="model.pt"):
    """
    Helper function for Load model
    :param model: current model
    :param name: model name
    :return: loaded model default model.pt
    """
    model.load_state_dict(torch.load(name))
    return model


def save_model(model, name='model.pt'):
    """
    Helper function for save model
    :param model: current model
    :param name: model name, default model.pt
    :return: None
    """
    torch.save(model.state_dict(), name)


# model
import torch.nn as nn

class M(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.list = nn.ModuleList()
        self.list.append(nn.Linear(in_size, 16*16))
        self.list.append(nn.Linear(16*16, 8*8))
        self.list.append(nn.Linear(8*8, out_size))
        self.relu = nn.functional.relu

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.list[:-1]:
            x = layer(x)
            x = self.relu(x)
        x = self.list[-1](x)
        return x

def get_model():
    return M
