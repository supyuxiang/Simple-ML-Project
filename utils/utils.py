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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report





class DataPreprocessing:
    def __init__(self,data_path_train,data_path_test):
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test
        self.load_data()
        self.check_data()
        self.data_proprocessing()
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
        

    def build_dataloader(self):
        pass
        # return train_loader, test_loader



    




