import os
import torch
import parser
import network
import util
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# ================= 1. 绝对路径配置区 =================
# 原数据集路径
ORIGINAL_IMAGES_DIR = "/media/data1/chenshunpeng1/datasets/gsv_cities/Images/"
ORIGINAL_CSV_DIR = "/media/data1/chenshunpeng1/datasets/gsv_cities/Dataframes/"

# # 生成数据集路径
FAKE_IMAGES_DIR = "/media/data/zhangjingyi/datasets/additional_cities/Images/genimages_pix2pix/"
# FAKE_IMAGES_DIR = "/media/data/zhangjingyi/datasets/additional_cities/Images/3city/"
# FAKE_CSV_DIR = "/media/data/zhangjingyi/datasets/additional_cities/Dataframes/3city/" # 本脚本直接按规则寻找图片，不需要读生成的CSV

OUTPUT_CSV = "similarity_results_pix2pix_3cities.csv"

# 仅处理这3个有生成图的城市
TARGET_CITIES = ['Osaka', 'London', 'Phoenix']
WEATHER_TYPES = ['night', 'occlusion']  # 同时匹配雨天和雪天

# ================= 2. 阶段 1：跨目录数据匹配测试 =================
print(">>> 阶段 1：开始跨目录进行图像路径匹配测试...")
valid_pairs = []
missing_count = 0

for city in TARGET_CITIES:
    csv_path = os.path.join(ORIGINAL_CSV_DIR, f"{city}.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到 CSV 文件: {csv_path}，跳过该城市。")
        continue
        
    print(f"正在匹配城市: {city} ...")
    df = pd.read_csv(csv_path)
    
    for idx, row in df.iterrows():
        # 获取相对路径 (GSV-Cities 通常是 "place_id/image.jpg")
        img_rel_path = row['image_path'] 
        
        # 如果 csv 里的路径没带城市名前缀，我们手动加上
        if not img_rel_path.startswith(city):
            img_rel_path = os.path.join(city, img_rel_path)
            
        # 拼接原图绝对路径
        real_full_path = os.path.join(ORIGINAL_IMAGES_DIR, img_rel_path)
        
        # 遍历雨天和雪天，拼接生成图绝对路径
        for weather in WEATHER_TYPES:
            # 替换后缀，例如 .jpg 变成 _rain.jpg
            if img_rel_path.endswith('.jpg'):
                fake_rel_path = img_rel_path.replace('.jpg', f'_{weather}.jpg')
            elif img_rel_path.endswith('.png'):
                fake_rel_path = img_rel_path.replace('.png', f'_{weather}.png')
            else:
                fake_rel_path = img_rel_path + f'_{weather}' # 备用防错
                
            fake_full_path = os.path.join(FAKE_IMAGES_DIR, fake_rel_path)
            
            # 核心测试：两边的物理文件是否都存在？
            if os.path.exists(real_full_path) and os.path.exists(fake_full_path):
                valid_pairs.append({
                    'city': city,
                    'weather': weather,
                    'real': real_full_path, 
                    'fake': fake_full_path
                })
            else:
                missing_count += 1

print("\n匹配测试完成！")
print(f"✅ 成功匹配到原图与生成图: {len(valid_pairs)} 对")
print(f"❌ 未找到对应生成图 (或原图丢失): {missing_count} 次")

if len(valid_pairs) == 0:
    print("致命错误：没有找到任何有效配对！请检查FAKE_IMAGES_DIR下的文件命名规则。")
    exit()

# 交互确认
response = input("\n匹配成功，是否加载 ImAge 模型开始漫长的相似度计算？(y/n): ")
if response.lower() != 'y':
    print("已终止计算。你可以根据上面的输出核对路径逻辑。")
    exit()

# ================= 3. 阶段 2：加载 ImAge 裁判模型 =================
print("\n>>> 阶段 2：加载 ImAge 模型计算相似度...")
args = parser.parse_arguments()
args.features_dim = args.num_learnable_aggregation_tokens * 768

model = network.VPRmodel(args)
model = model.to(args.device)

if args.resume is not None:
    print(f"正在加载裁判权重: {args.resume}")
    model = util.resume_model(args, model)

model = torch.nn.DataParallel(model)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

similarities = []

# ================= 4. 阶段 3：特征提取与相似度计算 =================
print("开始计算余弦相似度...")
with torch.no_grad():
    for pair in tqdm(valid_pairs):
        try:
            img_real = Image.open(pair['real']).convert('RGB')
            img_fake = Image.open(pair['fake']).convert('RGB')
            
            tensor_real = transform(img_real).unsqueeze(0).to(args.device)
            tensor_fake = transform(img_fake).unsqueeze(0).to(args.device)
            
            feat_real = model(tensor_real)
            feat_fake = model(tensor_fake)
            
            sim = F.cosine_similarity(feat_real, feat_fake).item()
            similarities.append(sim)
            
        except Exception as e:
            print(f"处理出错 {pair['real']}: {e}")
            similarities.append(0.0)

results_df = pd.DataFrame(valid_pairs)
results_df['similarity_score'] = similarities
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n打分完毕！结果已保存至 {OUTPUT_CSV}")

# ================= 5. 阶段 4：绘制分布图 =================
print("正在生成相似度分布图...")
plt.figure(figsize=(10, 6))

# 提取 rainy 和 snowy 的分数
rainy_scores = results_df[results_df['weather'] == 'rainy']['similarity_score']
snowy_scores = results_df[results_df['weather'] == 'snowy']['similarity_score']

if not rainy_scores.empty:
    plt.hist(rainy_scores, bins=50, alpha=0.6, color='blue', label='Rainy (Pix2Pix)')
if not snowy_scores.empty:
    plt.hist(snowy_scores, bins=50, alpha=0.6, color='cyan', label='Snowy (Pix2Pix)')

plt.title('Pix2Pix Generation Quality (ImAge Similarity)', fontsize=14)
plt.xlabel('Cosine Similarity Score', fontsize=12)
plt.ylabel('Number of Image Pairs', fontsize=12)
plt.axvline(x=0.65, color='green', linestyle='--', label='Good Quality (>0.65)')
plt.axvline(x=0.40, color='red', linestyle='--', label='Poor Quality (<0.40)')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('similarity_distribution_pix2pix.png')
print("分析完成！请查看 similarity_distribution_pix2pix.png。")