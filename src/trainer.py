
from .model import Model


class Trainer(Model):
    def __init__(self):
        super().__init__()
        self.model = Model()
    
    def train(self):
        pass

    def visulize(self):
        pass

    def save_model(self):
        pass