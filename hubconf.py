dependencies = ['torch']

import torch
import network

class SimpleArgs:
    def __init__(self, **kwargs):
        self.backbone = kwargs.pop('backbone', 'dinov2')
        self.aggregator = kwargs.pop('aggregator', None)
      
        self.num_learnable_aggregation_tokens = kwargs.pop('num_learnable_aggregation_tokens', 8)
        self.freeze_te = kwargs.pop('freeze_te', 8)

        self.resume = kwargs.pop("resume", True)
        self.foundation_model_path = kwargs.pop('foundation_model_path', None)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

# def ImAge(training_set="Merged", **kwargs):
#     args = SimpleArgs(**kwargs)
#     model = network.VPRmodel(args)
#     model = torch.nn.DataParallel(model)
#     if training_set == "Merged":
#       model.load_state_dict(
#           torch.hub.load_state_dict_from_url(f'https://github.com/Lu-Feng/ImAge/releases/download/v1.0.0/ImAge_Merged.pth', map_location=torch.device('cpu'))["model_state_dict"]
#       )
#     elif training_set == "GSV_Cities":
#       model.load_state_dict(
#           torch.hub.load_state_dict_from_url(f'https://github.com/Lu-Feng/ImAge/releases/download/v1.0.0/ImAge_GSV.pth', map_location=torch.device('cpu'))["model_state_dict"]
#       )
#     return model
def ImAge(training_set="Merged", checkpoint_path=None, **kwargs):
    args = SimpleArgs(**kwargs)
    model = network.VPRmodel(args)
    
    # 注意：这里的 DataParallel 会给 key 加上 "module." 前缀
    model = torch.nn.DataParallel(model)
    
    # 逻辑 1：如果指定了本地路径，优先加载本地权重
    if checkpoint_path is not None:
        print(f"Loading local checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        # 兼容两种保存格式：直接保存 dict 或 保存了 {'model_state_dict': ...}
        actual_dict = state_dict.get("model_state_dict", state_dict.get("state_dict", state_dict))
        model.load_state_dict(actual_dict)
        
    # 逻辑 2：否则根据 training_set 参数从 GitHub 下载
    elif training_set == "Merged":
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                'https://github.com/Lu-Feng/ImAge/releases/download/v1.0.0/ImAge_Merged.pth', 
                map_location=torch.device('cpu')
            )["model_state_dict"]
        )
    elif training_set == "GSV_Cities":
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                'https://github.com/Lu-Feng/ImAge/releases/download/v1.0.0/ImAge_GSV.pth', 
                map_location=torch.device('cpu')
            )["model_state_dict"]
        )
    
    return model
