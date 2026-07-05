import pandas as pd

# 1. 读取 CSV 文件
file_path = 'LA_snowy.csv'  # 替换为你的实际文件名
df = pd.read_csv(file_path)

df['city_id'] = 'LA_snowy'

df.to_csv(file_path, index=False)
