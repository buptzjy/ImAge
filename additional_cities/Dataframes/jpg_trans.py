import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

path = '/media/data/zhangjingyi/datasets/GSV_Mixed/Images/3city'

def convert(f):
    if f.lower().endswith('.png'):
        try:
            full_path = os.path.join(path, f)
            new_name = os.path.splitext(f)[0] + '.jpg'
            target_path = os.path.join(path, new_name)
            if os.path.exists(target_path):
                return 'skipped'
            img = Image.open(full_path).convert('RGB')
            img.save(target_path, 'JPEG', quality=95)
            return 'success'
        except Exception as e:
            return f'error: {e}'
    return 'not_png'

if __name__ == "__main__":
    if not os.path.exists(path):
        print(f"错误：路径 {path} 不存在")
    else:
        all_files = [f for f in os.listdir(path) if f.lower().endswith('.png')]
        total = len(all_files)
        print(f"准备处理 {total} 张图片...")
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("提示：未安装 tqdm 库，将使用简易进度显示。")

        success_count, skip_count, error_count = 0, 0, 0
        with ProcessPoolExecutor() as exe:
            futures = {exe.submit(convert, f): f for f in all_files}
            if use_tqdm:
                with tqdm(total=total, desc="转换进度", unit="img") as pbar:
                    for future in as_completed(futures):
                        res = future.result()
                        if res == 'success': success_count += 1
                        elif res == 'skipped': skip_count += 1
                        else: error_count += 1
                        pbar.update(1)
            else:
                processed = 0
                for future in as_completed(futures):
                    res = future.result()
                    processed += 1
                    if res == 'success': success_count += 1
                    elif res == 'skipped': skip_count += 1
                    else: error_count += 1
                    if processed % 500 == 0:
                        print(f"进度: {processed}/{total} ({(processed/total)*100:.1f}%)")
        print(f"\n任务完成！成功: {success_count}, 跳过: {skip_count}, 失败: {error_count}")
