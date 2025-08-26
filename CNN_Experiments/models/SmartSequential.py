from torch import nn


def parseConv2d(x: list):
    # set Conv2d parameters to those specified in x otherwise use default values
    return nn.Conv2d(in_channels=x[0],
                     out_channels=x[1],
                     kernel_size=3 if len(x) < 3 else x[2],
                     stride=1 if len(x) < 4 else x[3],
                     padding=1 if len(x) < 5 else x[4])


class SmartSequential(nn.Module):
    module_dict = {}

    activation_functions = {
         "relu": nn.ReLU(),
         "leakyrelu": nn.LeakyReLU(negative_slope=0.01),
         "gelu": nn.GELU(),
         "tanh": nn.Tanh(),
    }

    # M: nn.MaxPool2d()
    # F: nn.Flatten()
    # C: nn.Conv2d()
    # L: nn.Linear()
    # A: activation
    # D: nn.Dropout

    config2layer = {
        "L": lambda x: nn.Linear(in_features=x[0], 
                                 out_features=x[1]),
        "C": lambda x: parseConv2d(x),
        "A": lambda x: SmartSequential.activation_functions[x],
        "F": lambda x: nn.Flatten(start_dim=x),
        "M": lambda x: nn.MaxPool2d(kernel_size=x),
        "D": lambda x: nn.Dropout(p=x),
    }

    def __init__(self):
        super().__init__()

        self.architecture = nn.Sequential(
            *self.get_architecture_layers()
        )
    
    @staticmethod
    def extend_layers(layers, layer_config):
        for l in layer_config:
            if isinstance(l[0], int):
                for _ in range(l[0]):
                    layers = SmartSequential.extend_layers(layers, l[1])
                continue

            layers.append(SmartSequential.config2layer[l[0]](l[1]))
        return layers

    def get_architecture_layers(self):        
        layers = SmartSequential.extend_layers(layers=[], layer_config=self.CONFIG)
        return layers
    
    def state_dict(self):
        sd = super().state_dict()
        sd["MODEL_CLASS"] = self.MODEL_CLASS
        sd["CONFIG"] = self.CONFIG
        return sd

    def load_state_dict(self, state_dict, strict=False):
        # override the default artument to strict=False, since SmartModules 
        # contain the extra key "CONFIG"
        return super().load_state_dict(state_dict, strict)