import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tqdm import tqdm

data_path = '/home/yxfeng/project2/ML1/data/train_u6lujuX_CVtuZ9i.csv'

# 读取数据
df = pd.read_csv(data_path)

print("=== 原始数据信息 ===")
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print("\n缺失值统计:")
print(df.isnull().sum())

print("\n=== 处理缺失值 ===")


# 方法1: 查看缺失值详情
print("各列缺失值详情:")
for col in df.columns:
    missing_count = df[col].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    print(f"{col}: {missing_count} ({missing_pct:.1f}%)")

# 方法2: 处理缺失值
df_processed = df.copy()

# 对于分类列，用 'Unknown' 填充
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for col in categorical_columns:
    if col in df_processed.columns:
        df_processed[col] = df_processed[col].fillna('Unknown')

# 对于数值列，用中位数填充
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in numerical_columns:
    if col in df_processed.columns:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())

print("\n处理后的缺失值统计:")
print(df_processed.isnull().sum())

print('*'*100)
print('删除包含 Unknown 的行')

# 方法1: 使用 pandas 的布尔索引（推荐）
print("删除前数据形状:", df_processed.shape)

# 检查哪些行包含 'Unknown'
mask = df_processed[categorical_columns].eq('Unknown').any(axis=1)
print(f"包含 'Unknown' 的行数: {mask.sum()}")

# 删除包含 'Unknown' 的行
df_clean = df_processed[~mask].copy()
print("删除后数据形状:", df_clean.shape)

# 重置索引
df_clean = df_clean.reset_index(drop=True)
print("重置索引后数据形状:", df_clean.shape)

print("\n删除后的数据预览:")
print(df_clean.head())
print(f"\n最终数据形状: {df_clean.shape}")

# 验证是否还有 'Unknown' 值
print("\n验证是否还有 'Unknown' 值:")
for col in categorical_columns:
    unknown_count = (df_clean[col] == 'Unknown').sum()
    print(f"{col}: {unknown_count} 个 'Unknown'")
            














