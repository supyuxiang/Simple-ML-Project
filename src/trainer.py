import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 直接导入，避免相对导入问题
from src.model import Model
from utils.utils import DataPreprocessing,Timer
from utils.utils import select_optimizer,select_loss,select_scheduler
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Trainer:
    def __init__(self,model,config):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.data_train_loader, self.data_test_loader = DataPreprocessing(config.data.data_path_train,config.data.data_path_test).build_dataloader()
        self.optimizer = select_optimizer(config.train.optimizer)
        self.loss = select_loss(config.train.loss)
        self.scheduler = select_scheduler(config.train.scheduler)
    
    @Timer
    def train(self):
        self.model.train()
        metrics = {
            'loss':[],
            'mse':[],
            'mae':[],
            'r2':[]
        }

        for epoch in tqdm(range(self.config.train.epochs)):
            self.scheduler.step()
            for batch_idx, (data, target) in tqdm(enumerate(self.data_train_loader),desc=f'Epoch:{epoch+1}/{self.config.train.epochs}'):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                mse = mean_squared_error(output, target)
                mae = mean_absolute_error(output, target)
                r2 = r2_score(output, target)
                metrics['loss'].append(loss.item())
                metrics['mse'].append(mse.item())
                metrics['mae'].append(mae.item())
                metrics['r2'].append(r2.item())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            if self.config.train.save_visualization:
                if epoch % self.config.train.save_visualization_interval == 0:
                    self.visulize(metrics)
            if self.config.save.save_model:
                if epoch % self.config.save.save_model_interval == 0:
                    self.save_model()
            if self.config.save.save_metrics:
                if epoch % self.config.save.save_metrics_interval == 0:
                    self.save_metrics(metrics)
    
    def save_metrics(self,metrics):
        np.save(self.config.train.save_path+'/metrics.npy',metrics)
    

    def visulize(self,metrics):
        plt.figure(figsize=(6,10))
        loss = metrics['loss']
        r2 = metrics['r2']
        mse = metrics['mse']
        mae = metrics['mae']

        fig,axes = plt.subplots(2,2,figsize=(10,10))
        sns.lineplot(x=range(len(loss)),y=loss,ax=axes[0,0])
        sns.lineplot(x=range(len(r2)),y=r2,ax=axes[0,1])
        sns.lineplot(x=range(len(mse)),y=mse,ax=axes[1,0])
        sns.lineplot(x=range(len(mae)),y=mae,ax=axes[1,1])
        plt.savefig(self.config.train.save_path+'/metrics.png')
        plt.close()

    def save_model(self):
        torch.save(self.model.state_dict(), self.config.train.save_path+'/model.pth')
    
if __name__ == "__main__":
    # 测试代码
    trainer = Trainer()
    print('Trainer 初始化成功!')