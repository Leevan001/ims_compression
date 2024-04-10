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
import warnings
import numpy as np
import pandas as pd
from pytorch_msssim import ms_ssim
from PIL import Image
import matplotlib.pyplot as plt
import base64
import tqdm
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())




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
    parser.add_argument("--js", type=str, help="Path to json")
    parser.add_argument("--sf", type=str)
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    # parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    result = pd.DataFrame(columns=['imageNmae', 'Compression time', 'Decompression time', 'PSNR', 'Bitrate'])
    args = parse_args(argv)
    p = 128
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
        with open(str(args.js), 'r') as f:
            compressed_data = json.load(f)
        for img_name, img_data in tqdm.tqdm(compressed_data.items(),desc='decompressing images'):
            min_val, max_val=img_data["min_val"],img_data["max_val"]
                 
            padding=(0, 0, 54, 54)
            
            part1 =  [[base64.b64decode(s)] for s in img_data["strings"]]
            part2=  torch.tensor(img_data["shape"]).to(device)
            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()

                out_dec = net.decompress(part1, part2)
                if args.cuda:
                    torch.cuda.synchronize()

                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)

                tensor_permuted = out_dec["x_hat"]
                
                array = np.squeeze(tensor_permuted.detach().cpu().numpy())
                # print(array.shape)
                

                # 然后构造save_path，最后显式添加.tif扩展名
                
                save_path =os.path.join(savepath,img_name)
                               
                original_img=inverse_histogram_stretching_transform(array, min_val, max_val)

                original_img.save(save_path, format='TIFF')
                





if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])