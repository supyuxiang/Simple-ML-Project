import pandas as pd

data_path = '/home/yxfeng/project2/ML1/data/train_u6lujuX_CVtuZ9i.csv'

# 读取数据
df = pd.read_csv(data_path)

# 查看数据
print(df.head())
print('*'*100)

# 查看数据类型
print(df.dtypes)
print('*'*100)

# 查看数据缺失值
print(df.isnull().sum())
print('*'*100)
    # 查看数据描述
print(df.describe())
print('*'*100)
print(len(df))

print(df.columns)
