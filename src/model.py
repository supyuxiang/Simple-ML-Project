import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,config):
        super(self).__init__()
        self.config = config
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.config.model.input_size,self.config.model.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.model.hidden_size,self.config.model.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.model.hidden_size,self.config.model.output_size),
            nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.config.model.output_size,self.config.model.output_size),
            nn.ReLU(),
            nn.Linear(self.config.model.output_size,self.config.model.output_size)
            )
            
    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x