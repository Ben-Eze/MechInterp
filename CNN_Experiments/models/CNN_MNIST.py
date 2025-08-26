from models.SmartSequential import SmartSequential


class CNN_MNIST(SmartSequential):
    # TODO: make a hooked model
    MODEL_CLASS = "CNN_MNIST"

    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        super().__init__()
    
    
    def forward(self, X):
        return self.architecture(X)

SmartSequential.module_dict[CNN_MNIST.MODEL_CLASS] = CNN_MNIST