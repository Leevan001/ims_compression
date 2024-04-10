import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM
import warnings
import torch
import os
import sys
import math
import argparse
import time
import json
from tqdm import tqdm
import warnings
import numpy as np
import pandas as pd
from pytorch_msssim import ms_ssim
from PIL import Image
import matplotlib.pyplot as plt
import base64
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())
def prepare_data(out_enc, min_val, max_val, img_name):
    # 将bytes类型的strings转换为Base64编码的字符串
    strings_base64 = [base64.b64encode(s[0]).decode('utf-8') for s in out_enc["strings"]]
    
    # 准备单个图像的数据
    data = {
        # "img_name": img_name,
        "strings": strings_base64,
        # "strings": out_enc["strings"],
        "shape": list(out_enc["shape"]),  # 假设shape是np.ndarray类型，转换为列表
        "min_val": float(min_val),
        "max_val": float(max_val)
    }
    # print(out_enc["strings"])
    return data



def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)



class HistogramStretchingTransform16Bit(object):
    def __call__(self, img):
        img_np = np.array(img, dtype=np.float32)
        min_val = np.min(img_np)
        max_val = np.max(img_np)

        if max_val - min_val == 0:
            img_stretched = np.zeros_like(img_np)
        else:
            img_stretched = (img_np - min_val) / (max_val - min_val)

        img_stretched_pil = Image.fromarray(img_stretched, mode='F')
        
        # 返回变换后的图像和变换参数
        return img_stretched_pil, min_val, max_val

def inverse_histogram_stretching_transform(img_transformed, min_val, max_val):
    """
    根据直方图拉伸变换的参数，逆转变换过程并恢复原始图像。
    
    参数:
        img_transformed (PIL Image): 经过直方图拉伸变换的图像。
        min_val (float): 变换中使用的最小值。
        max_val (float): 变换中使用的最大值。
        
    返回:
        PIL Image: 恢复后的原始图像。
    """
    # 将PIL图像转换为NumPy数组
    img_stretched_np = np.array(img_transformed, dtype=np.float32)
    
    # 应用逆变换公式
    original_img_np = img_stretched_np * (max_val - min_val) + min_val
    
    # 确保数据类型和范围是正确的
    original_img_np = np.clip(original_img_np, min_val, max_val).astype(np.uint16)
    
    # 将NumPy数组转换回PIL图像
    original_img = Image.fromarray(original_img_np, mode='I;16')
    
    return original_img


def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.add_argument("--sf",type=str)
    # parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    result = pd.DataFrame(columns=['imageNmae', 'Compression time', 'Decompression time', 'PSNR', 'Bitrate'])
    args = parse_args(argv)
    p = 128
    path = args.data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg","tif"]:
            img_list.append(file)
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    net = TCM(config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320)
    net = net.to(device)
    net.eval()
    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    dictory = {}
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
    if args.real:
        net.update()
        print("保存")
        all_data = {} 
        savepath=str(args.sf)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        for img_name in tqdm(img_list, desc='Processing images'):
            img_path = os.path.join(path, img_name)
            transform = HistogramStretchingTransform16Bit()
            # img = transforms.ToTensor()(Image.open(img_path).convert('L')).to(device)
            img_transformed, min_val, max_val = transform(Image.open(img_path))
            img = transforms.ToTensor()(img_transformed).to(device)                        
            
            x = img.unsqueeze(0)
            x_padded, padding = pad(x, p)
            count += 1
            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()
                # 压缩
                out_enc = net.compress(x_padded)
                
                # print(out_enc["strings"])
                
                all_data[img_name] = prepare_data(out_enc, min_val, max_val, img_name)
                
                # json_file_path = os.path.join(savepath,"compressed.json")
                # with open(json_file_path, 'w') as json_file:
                #     json.dump(all_data, json_file)
                # print("save successfully!",str(json_file_path))
                

        json_file_path = os.path.join(savepath,"compressed.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(all_data, json_file)
        print("save successfully!",str(json_file_path))


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
