import torch
import torch.utils.model_zoo as model_zoo


def rotate_channels(x):
    out = torch.transpose(x, 1, 3)  # 0, 3, 2, 1
    out = torch.transpose(out, 2, 3)  # 0, 3, 1, 2
    return out


def load_pretrained_weights_no_fc(model, settings):
    skip = ['last_linear.weight', 'last_linear.bias']
    pretrained_weights = model_zoo.load_url(settings['url'])
    state_dict = model.state_dict()
    for key in state_dict.keys():
        if key in skip:
            continue
        state_dict[key] = pretrained_weights[key]
    model.load_state_dict(state_dict)
    return model
