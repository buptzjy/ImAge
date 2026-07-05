import pandas as pd
import os
from tqdm import tqdm

def generate_unified_mix_csv(input_dir, output_csv_path, image_dir):
    # 1. 获取目录下所有的 CSV 文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"错误：在 {input_dir} 中没找到任何 CSV 文件。")
        return

    print(f"检测到 {len(csv_files)} 个城市文件，准备开始批量处理并合并...")

    all_dfs = []  # 用于存储每个城市处理后的 DataFrame

    for csv_name in csv_files:
        input_path = os.path.join(input_dir, csv_name)
        print(f"\n[正在处理]: {csv_name}")
        
        try:
            # 读取原始城市 CSV
            df = pd.read_csv(input_path, encoding='utf-8')
            new_rows = []
            
            # 遍历当前城市的所有行
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"匹配 {csv_name}"):
                # 格式化文件名（保持你的业务逻辑）
                pl_id = str(row['place_id'] % 10**5).zfill(7)
                city = str(row['city_id'])
                panoid = str(row['panoid'])
                year = str(row['year']).zfill(4)
                month = str(row['month']).zfill(2)
                northdeg = str(row['northdeg']).zfill(3)
                lat, lon = str(row['lat']), str(row['lon'])
                
                base_name = f"3city_{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}"
                
                # 在混合图库中检查是否存在雨/雪图
                # 雨天检查
                if os.path.exists(os.path.join(image_dir, f"{base_name}_rainy.jpg")):
                    new_row = row.to_dict()
                    new_row['panoid'] = f"{panoid}_rainy"
                    new_rows.append(new_row)
                    
                # 雪天检查
                if os.path.exists(os.path.join(image_dir, f"{base_name}_snowy.jpg")):
                    new_row = row.to_dict()
                    new_row['panoid'] = f"{panoid}_snowy"
                    new_rows.append(new_row)

            # 如果有新增行，合并到当前城市的 df 中
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                df = pd.concat([df, new_df], ignore_index=True)
                print(f"  - 该城市新增了 {len(new_rows)} 条雨雪数据")
            
            all_dfs.append(df)

        except Exception as e:
            print(f"  - 处理文件 {csv_name} 时出错: {e}")

    # 2. 合并所有城市的 DataFrame 并输出
    if all_dfs:
        print("\n正在合并所有城市数据并保存...")
        final_mix_df = pd.concat(all_dfs, ignore_index=True)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        final_mix_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"成功！最终文件已保存至: {output_csv_path}")
        print(f"最终总行数: {len(final_mix_df)}")
    else:
        print("没有处理任何数据，未生成文件。")

if __name__ == "__main__":
    # --- 最终路径配置 ---
    # 原始城市 CSV 目录
    ORIGINAL_CSV_DIR = "/media/data1/chenshunpeng1/datasets/gsv_cities/Dataframes"
    
    # 混合后的图片存放目录
    IMAGE_MIX_DIR = "/media/data/zhangjingyi/datasets/GSV_Mixed/Images/gsv_mix"
    
    # 输出的统一 CSV 路径
    FINAL_OUTPUT_CSV = "/media/data/zhangjingyi/datasets/GSV_Mixed/Dataframes/gsv_mix.csv"
    
    batch_process_cities = generate_unified_mix_csv(ORIGINAL_CSV_DIR, FINAL_OUTPUT_CSV, IMAGE_MIX_DIR)