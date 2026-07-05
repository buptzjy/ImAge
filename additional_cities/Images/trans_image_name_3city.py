import os
from tqdm import tqdm

# 定义目标目录
target_dir = '/media/data/zhangjingyi/IClight/additional_cities/Images/3city'

# 1. 获取目录下所有文件
try:
    all_files = os.listdir(target_dir)
except FileNotFoundError:
    print(f"错误：找不到目录 {target_dir}")
    exit()

# 2. 定义匹配逻辑
# 使用元组 (Tuple) 可以让 startswith 一次性检查多个开头
old_prefixes = ('London', 'Osaka', 'Phoenix')
new_prefix = '3city'

# 筛选出以这三个城市开头，且还没改名为 3city 的文件
files_to_rename = [
    f for f in all_files 
    if f.startswith(old_prefixes) and not f.startswith(new_prefix)
]

print(f"共找到 {len(files_to_rename)} 个需要改名的文件。")

# 3. 开始改名并显示进度条
for filename in tqdm(files_to_rename, desc="正在统一前缀"):
    # 构建旧路径
    old_path = os.path.join(target_dir, filename)
    
    # 动态确定当前文件匹配的是哪个前缀
    # 因为 startswith(old_prefixes) 已经帮我们过滤了，这里只需找到具体是哪个
    matched_city = None
    for city in old_prefixes:
        if filename.startswith(city):
            matched_city = city
            break
            
    if matched_city:
        # 构建新文件名：将匹配到的城市名替换为 3city
        # 使用 replace(..., 1) 确保只替换开头的第一个匹配项
        new_filename = filename.replace(matched_city, new_prefix, 1)
        new_path = os.path.join(target_dir, new_filename)
        
        # 执行改名操作
        try:
            # 增加一个判断，防止目标文件名已存在导致报错
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
            else:
                print(f"\n跳过 {filename}：目标文件 {new_filename} 已存在")
        except Exception as e:
            print(f"\n修改 {filename} 出错: {e}")

print("任务完成！所有前缀已统一为 3city。")