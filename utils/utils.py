import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys
from tqdm import tqdm
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    LabelBinarizer,
    MultiLabelBinarizer,
    KBinsDiscretizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.nn import MSELoss, L1Loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
import time





class DataPreprocessing:
    def __init__(self,data_path_train,data_path_test):
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test
        self.load_data()
        #self.check_data()
        self.numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        self.categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        self.data_proprocessing()
        self.build_dataset()
        self.build_dataloader()
    
    def load_data(self):
        self.data_train = pd.read_csv(self.data_path_train)
        self.data_test = pd.read_csv(self.data_path_test)
        return self.data_train, self.data_test
    
    def check_data(self):
        print('train_data:')
        print('*'*100)
        print(f'data_head:{self.data_train.head()}')
        print('*'*100)
        print(f'data_describe:{self.data_train.describe()}')
        print('*'*100)
        print(f'data_isnull:{self.data_train.isnull()}')
        print('*'*100)
        print(f'data_shape:{self.data_train.shape}')
        print('*'*100)
        print(f'data_columns:{self.data_train.columns}')
        print('*'*100)
        print(f'data_info:{self.data_train.info()}')
        print('*'*100)
        print(f'data_len:{len(self.data_train)}')
        print('*'*100)
        print('test_data:')
        print('*'*100)
        print(f'data_head:{self.data_test.head()}')
        print('*'*100)
        print(f'data_describe:{self.data_test.describe()}')
        print('*'*100)
        print(f'data_isnull:{self.data_test.isnull()}')
        print('*'*100)
        print(f'data_shape:{self.data_test.shape}')
        print('*'*100)
        print(f'data_columns:{self.data_test.columns}')
        print('*'*100)
        print(f'data_info:{self.data_test.info()}')
        print('*'*100)
        print(f'data_len:{len(self.data_test)}')
        print('*'*100)
    
    def data_proprocessing(self):
        self.data_train.dropna(inplace=True)
        self.data_train.fillna(self.data_train.median(), inplace=True)
        self.data_train.drop_duplicates(inplace=True)
        self.data_train.reset_index(drop=True, inplace=True)
        for col in self.numerical_columns:
            if col in self.data_train.columns:
                self.data_train[col] = self.data_train[col].fillna(self.data_train[col].median())
        for col in self.categorical_columns:
            if col in self.data_train.columns:
                self.data_train[col] = self.data_train[col].fillna('Unknown')
        mask = self.data_train[self.categorical_columns].eq('Unknown').any(axis=1)
        self.data_train = self.data_train[~mask]
        self.data_train.reset_index(drop=True, inplace=True)


    def build_dataset(self):
        self.data_train_dataset = Dataset(self.data_train)
        self.data_test_dataset = Dataset(self.data_test)

    def build_dataloader(self):
        self.data_train_loader = DataLoader(self.data_train_dataset, batch_size=32, shuffle=True)
        self.data_test_loader = DataLoader(self.data_test_dataset, batch_size=32, shuffle=True)
        return self.data_train_loader, self.data_test_loader



def Timer(func):
    def wrapper(*args,**kwargs):
        start_time = time.time()
        result = func(*args,**kwargs)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        return result
    return wrapper

def select_optimizer(config):
    assert config.train.optimizer.name in ['Adam', 'SGD', 'AdamW'], "Optimizer name must be one of: Adam, SGD, AdamW"
    if config.train.optimizer.name == 'Adam':
        return Adam(config.train.optimizer.lr)
    elif config.train.optimizer.name == 'SGD':
        return SGD(config.train.optimizer.lr)
    elif config.train.optimizer.name == 'AdamW':
        return AdamW(config.train.optimizer.lr)

def select_loss(config):
    assert config.train.loss.name in ['MSELoss', 'L1Loss', 'L2Loss'], "Loss name must be one of: MSELoss, L1Loss, L2Loss"
    if config.train.loss.name == 'MSELoss':
        return MSELoss()
    elif config.train.loss.name == 'L1Loss':
        return L1Loss()
    elif config.train.loss.name == 'L2Loss':
        return L2Loss()

def select_scheduler(config):
    assert config.train.scheduler.name in ['StepLR', 'MultiStepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'], "Scheduler name must be one of: StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau"
    if config.train.scheduler.name == 'StepLR':
        return StepLR(config.train.scheduler.lr)
    elif config.train.scheduler.name == 'MultiStepLR':
        return MultiStepLR(config.train.scheduler.lr)
    elif config.train.scheduler.name == 'CosineAnnealingLR':
        return CosineAnnealingLR(config.train.scheduler.lr)
    elif config.train.scheduler.name == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(config.train.scheduler.lr)
    



    




