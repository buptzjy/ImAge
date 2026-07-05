import pandas as pd
from pathlib import Path
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

#0407修改：返回值加入similarities

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the root constant
ROOT_PATH_LIST = [
    path for path in [
        os.getenv("IMAGE_DATA_ROOT"),
        '/data_nvme/zhangjingyi/Unified_cities',
        '/media/data1/chenshunpeng1/datasets/gsv_cities',
        '/media/data/zhangjingyi/datasets/additional_cities',
    ]
    if path
]
WARN_TIMES = 10

class GSVCitiesDataset(Dataset):
    def __init__(self,
                 cities=['LA_rainy'],
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform
                 ):
        super(GSVCitiesDataset, self).__init__()
        
        # Set base_path as ROOT_PATH + training_dataset
        # self.base_path = ROOT_PATH
        
        # # Ensure it ends with a slash for the string concatenation used later
        # if not self.base_path.endswith('/'):
        #     self.base_path += '/'
        
        self.cities = cities

        # if not os.path.exists(self.base_path):
        #     raise FileNotFoundError(f"Path not found: {self.base_path}")
        
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        self.dataframe = self.__getdataframes()
        self.places_ids = pd.unique(self.dataframe.index)
        self.place_groups = self.dataframe.groupby(level=0)
        self.place_data = {pid: group for pid, group in self.place_groups}
        self.total_nb_images = len(self.dataframe)
    
    def get_real_path(self, path):
        ret_path = None
        global WARN_TIMES
        for ROOT_PATH in ROOT_PATH_LIST:
            if os.path.exists(os.path.join(ROOT_PATH, path)):
                if ret_path is None:
                    ret_path = os.path.join(ROOT_PATH, path)
                else:
                    if WARN_TIMES:
                        print(f'警告：同时存在路径：{ret_path} 和 {os.path.join(ROOT_PATH, path)}')
                        WARN_TIMES -= 1
        if ret_path is None:
            raise RuntimeError(f'不存在的文件：{path}')
        return ret_path

    def __getdataframes(self):
        # Now uses self.base_path which points to /.../IC-Light/training_dataset/
        try:
            df = pd.read_csv(self.get_real_path(os.path.join('Dataframes/csv_with_gen', f'{self.cities[0]}.csv')))
        except:
            df = pd.read_csv(self.get_real_path(os.path.join('Dataframes', f'{self.cities[0]}.csv')))
        df = df.sample(frac=1)
        
        for i in range(1, len(self.cities)):
            try:
                tmp_df = pd.read_csv(self.get_real_path(os.path.join('Dataframes/csv_with_gen', f'{self.cities[i]}.csv')))
            except:
                tmp_df = pd.read_csv(self.get_real_path(os.path.join('Dataframes', f'{self.cities[i]}.csv')))
            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)
            df = pd.concat([df, tmp_df], ignore_index=True)

        res = df[df.groupby('place_id')['place_id'].transform('size') >= self.min_img_per_place]
        return res.set_index('place_id')  

    
    

    def __getitem__(self, index):
        place_id = self.places_ids[index]
        place = self.dataframe.loc[place_id]
        
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:
            place = place.sort_values(by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]
            
        imgs = []
        similarities = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row, place_id)
            # Path points to /.../IC-Light/training_dataset/Images/city_id/img_name
            img_path = self.get_real_path(os.path.join('Images', row['city_id'], img_name))
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
            try:
                sim_value = float(row['similarity'])
            except:
                sim_value = 1.0
            similarities.append(sim_value)

        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place), torch.tensor(similarities, dtype=torch.float32)
    

    def __len__(self):
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row, place_id):
        city = row['city_id']
        # Use the passed place_id to ensure naming consistency
        pl_id = place_id % 10**5
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        
        name = f"{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
        return name
