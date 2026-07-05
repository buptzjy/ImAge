import pandas as pd
import os
def read_csv_select_columns(file_path):
    """读取CSV并只保留指定列"""
    try:
        # 方法1：读取时直接指定要保留的列（推荐）
        # 提前指定usecols参数，只读取需要的列，效率更高
        df = pd.read_csv(
            file_path,
            encoding='utf-8'
        )
        # df = df.set_index('place_id')
        x = 0
        new_rows = []
        for index, row in df.iterrows():
            # 通过列名访问每行的具体值
            pl_id = row['place_id'] % 10**5  #row.name是行索引，和图片名无关
            pl_id = str(pl_id).zfill(7)
            city = row['city_id']
            # 3. 提取并格式化各类信息：
            panoid = row['panoid']          # 全景图片的唯一标识（街景类图片常用）
            year = str(row['year']).zfill(4)    # 年份补零到4位（如23→0023，2023→2023）
            month = str(row['month']).zfill(2)  # 月份补零到2位（如5→05，12→12）
            northdeg = str(row['northdeg']).zfill(3)  # 朝向角度（北偏角度）补零到3位（如10→010）
            lat, lon = str(row['lat']), str(row['lon'])  # 纬度、经度
            name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
                northdeg+'_'+lat+'_'+lon+'_'+panoid
            if os.path.exists(f'/media/data/zhangjingyi/datasets/GSV_Mixed/Images/{name}.jpg'):
                x += 1
            if os.path.exists(f'/media/data/zhangjingyi/datasets/GSV_Mixed/Images//{name}_rainy.jpg'):
                x += 1
                new_row = row.to_dict()
                # 修改panoid字段，添加_rainy后缀
                new_row['panoid'] = panoid + '_rainy'
                # 将新增行加入列表
                new_rows.append(new_row)
            if os.path.exists(f'/media/data/zhangjingyi/datasets/GSV_Mixed/Images/{name}_snowy.jpg'):
                x += 1
                # 复制当前行数据
                new_row = row.to_dict()
                # 修改panoid字段，添加_snowy后缀
                new_row['panoid'] = panoid + '_snowy'
                # 将新增行加入列表
                new_rows.append(new_row)
        print(x)
        if new_rows:  # 只有当有新增行时才合并
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            csv_save_path = '/media/data1/zhangjingyi/IC-Light/LosAngeles_mix/Dataframes/LosAngeles_mix.csv'
            # 写入CSV文件（最常用的基础写法）
            df.to_csv(
                csv_save_path,          # 保存路径
                index=False,            # 不保存行索引（关键！避免多余的索引列）
                encoding='utf-8'        # 指定编码，避免中文乱码（如果有中文的话）
            )
        return df
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
    except Exception as e:
        print(f"处理失败：{e}")

# 调用示例
if __name__ == "__main__":
    csv_file_path = "/media/data1/zhangjingyi/IC-Light/LosAngeles_mix/Dataframes/LosAngeles_original.csv"  # 替换为你的CSV文件路径
    filtered_df = read_csv_select_columns(csv_file_path)