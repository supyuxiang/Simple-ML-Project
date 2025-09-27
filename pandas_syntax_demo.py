import pandas as pd
import numpy as np

# 创建示例数据
data = {
    'ID': [1, 2, 3, 4, 5],
    'Gender': ['Male', 'Female', 'Unknown', 'Male', 'Female'],
    'Married': ['Yes', 'No', 'Yes', 'Unknown', 'No'],
    'Education': ['Graduate', 'Graduate', 'Unknown', 'Graduate', 'Graduate'],
    'Income': [5000, 6000, 0, 7000, 5500],
    'Age': [25, 30, 0, 35, 28]
}

df = pd.DataFrame(data)
print("=== 原始数据 ===")
print(df)
print()

# 1. 列选择语法详解
print("=== 1. 列选择语法 ===")

# 基础选择
print("选择单列:")
print(df['Gender'])
print()

print("选择多列:")
categorical_columns = ['Gender', 'Married', 'Education']
selected = df[categorical_columns]
print(selected)
print()

# 按数据类型选择
print("按数据类型选择:")
print("字符串类型列:")
print(df.select_dtypes(include=['object']))
print()

print("数值类型列:")
print(df.select_dtypes(include=['int64', 'float64']))
print()

# 2. 比较操作语法详解
print("=== 2. 比较操作语法 ===")

# 基础比较
print("检查是否等于 'Unknown':")
print(df['Gender'] == 'Unknown')
print()

print("使用 .eq() 方法:")
print(df['Gender'].eq('Unknown'))
print()

# 多值比较
print("检查是否在多个值中:")
print(df['Gender'].isin(['Male', 'Female']))
print()

# 字符串操作
print("字符串包含检查:")
print(df['Education'].str.contains('Graduate'))
print()

# 3. 聚合函数语法详解
print("=== 3. 聚合函数语法 ===")

# 创建布尔 DataFrame
bool_df = df[categorical_columns].eq('Unknown')
print("布尔 DataFrame:")
print(bool_df)
print()

print("每行是否有任何 True (.any(axis=1)):")
print(bool_df.any(axis=1))
print()

print("每行是否全部为 True (.all(axis=1)):")
print(bool_df.all(axis=1))
print()

# 4. 组合使用
print("=== 4. 组合使用示例 ===")

# 原始代码的完整分解
print("步骤1: 选择分类列")
step1 = df[categorical_columns]
print(step1)
print()

print("步骤2: 检查是否等于 'Unknown'")
step2 = step1.eq('Unknown')
print(step2)
print()

print("步骤3: 检查每行是否有任何 True")
step3 = step2.any(axis=1)
print(step3)
print()

print("步骤4: 使用掩码过滤数据")
mask = step3
df_clean = df[~mask]  # ~ 表示取反
print("删除包含 'Unknown' 的行后:")
print(df_clean)
print()

# 5. 其他有用的语法
print("=== 5. 其他有用的语法 ===")

# 条件过滤
print("收入大于5000的行:")
print(df[df['Income'] > 5000])
print()

# 多条件过滤
print("男性且收入大于5000的行:")
print(df[(df['Gender'] == 'Male') & (df['Income'] > 5000)])
print()

# 使用 query 方法
print("使用 query 方法:")
print(df.query("Gender == 'Male' and Income > 5000"))
print()

# 6. 高级语法
print("=== 6. 高级语法 ===")

# 使用 loc 和 iloc
print("使用 loc 选择:")
print(df.loc[df['Gender'] == 'Male', ['Gender', 'Income']])
print()

print("使用 iloc 选择:")
print(df.iloc[0:3, 1:4])  # 前3行，第2-4列
print()

# 链式操作
print("链式操作:")
result = (df
          .query("Income > 0")  # 过滤收入大于0的行
          .groupby('Gender')    # 按性别分组
          .agg({'Income': 'mean', 'Age': 'mean'})  # 聚合
          .round(2))            # 保留2位小数
print(result)
