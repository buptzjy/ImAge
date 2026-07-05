import os
from tqdm import tqdm

# 定义目标目录
target_dir = '/media/data/zhangjingyi/IClight/additional_cities/Images/3city'

# 1. 获取目录下所有文件
all_files = os.listdir(target_dir)

# 2. 筛选出以 "LosAngeles" 开头，但还没被改名为 "LosAngeles_mix" 的文件
# 这样做可以防止重复运行脚本时名字变得越来越长（例如：LosAngeles_mix_mix_...）
files_to_rename = [
    f for f in all_files 
    if f.startswith('LosAngeles') and not f.startswith('LA_snowy')
]

print(f"找到 {len(files_to_rename)} 个需要改名的文件。")

# 3. 开始改名并显示进度条
for filename in tqdm(files_to_rename, desc="正在改名"):
    # 构建旧路径
    old_path = os.path.join(target_dir, filename)
    
    # 构建新文件名：将开头的 "LosAngeles" 替换为 "LosAngeles_mix"
    # 使用 replace(old, new, 1) 确保只替换开头的第一个匹配项
    new_filename = filename.replace('LosAngeles', 'LA_snowy', 1)
    new_path = os.path.join(target_dir, new_filename)
    
    # 执行改名操作
    try:
        os.rename(old_path, new_path)
    except Exception as e:
        print(f"\n修改 {filename} 出错: {e}")

print("任务完成！")