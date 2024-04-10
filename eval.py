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
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())
def prepare_data(out_enc, min_val, max_val, img_name):
    # 将bytes类型的strings转换为Base64编码的字符串
    strings_base64 = [base64.b64encode(s[0]).decode('utf-8') for s in out_enc["strings"]]
    
    # 准备单个图像的数据
    data = {
        # "img_name": img_name,
        "strings": strings_base64,
        "shape": list(out_enc["shape"]),  # 假设shape是np.ndarray类型，转换为列表
        "min_val": float(min_val),
        "max_val": float(max_val)
    }
    
    return data

def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.).item())


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    a = 0
    for likelihoods in out_net['likelihoods'].values():
        print(likelihoods.shape)
        print(torch.log(likelihoods).sum() / (-math.log(2)))
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels) for likelihoods in
               out_net['likelihoods'].values()).item()


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
    parser.set_defaults(real=False)
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
        if not os.path.exists("/data/yifanli/mycompress/result"):
            os.makedirs("/data/yifanli/mycompress/result")
        for img_name in img_list:
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
                s = time.time()
                out_enc = net.compress(x_padded)
                e = time.time()
                c_t = round(e - s, 2)
                all_data[img_name] = prepare_data(out_enc, min_val, max_val, img_name)
                # out_enc是一个字典 keys有strings和shape
                # shape 16*32
                # strings是一个列表，长度为2，其中第两个元素也都是列表
                # strings[0]长度为1 strings[1]长度为1 二者的元素均是<class 'bytes'>

                # 解压
                s = time.time()
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                e = time.time()
                dc_t = round(e - s, 2)
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                num_pixels = x.size(0) * x.size(2) * x.size(3)
                #                 print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
                #                 print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
                #                 print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')

                Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                PSNR += compute_psnr(x, out_dec["x_hat"])
                # print("------x-x_hat---------")
                # print(x.shape)
                # print(out_dec["x_hat"].shape)
                # print("------x-x_hat---------")
                
                MS_SSIM += compute_msssim(x, out_dec["x_hat"])

                ps = round(compute_psnr(x, out_dec["x_hat"]), 2)
                br = round(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels, 2)
                #                 for s in out_enc["strings"]:
                #                     print("here")
                #                     # print(out_net['likelihoods']['y'].shape) # [8, 320, 16, 16]
                #                     # print(out_net['likelihoods']['z'].shape) # [8, 192, 4, 4])
                #                     # print(len(s[0])) # 92996、40260
                #                     print(max(s[0]))
                #                     print(min(s[0]))
                #                     for j, label in enumerate(s[0]):
                #                         print(label)
                #                         if j>=10:
                #                             exit(0)
                #                 exit(0)

                tensor_permuted = out_dec["x_hat"]
                # print(tensor_permuted.shape)# 1,1,788,2048
                # tensor_permuted = tensor_permuted.permute(0, 3, 2, 1)
                
                array = np.squeeze(tensor_permuted.detach().cpu().numpy())
                # print(array.shape)
                
                # 去除img_name中的.tif扩展名（如果存在）
                if img_name.lower().endswith(".tif"):
                    img_name = img_name[:-4]

                # 然后构造save_path，最后显式添加.tif扩展名
                save_path = "./result/{}_{}_{}.tif".format(img_name, ps, br)
                
                
                original_img=inverse_histogram_stretching_transform(array, min_val, max_val)

                original_img.save(save_path, format='TIFF')
                
                
                mycopy=inverse_histogram_stretching_transform(np.squeeze(img.cpu().numpy()),min_val,max_val)
                sp="./result/{}.tif".format(img_name)
                mycopy.save(sp,format="TIFF")
                
                #将原图和修改图上下拼接
                combined_img=Image.new('I;16',(max(mycopy.width,original_img.width),mycopy.height+original_img.height))
                combined_img.paste(original_img,(0,0))
                combined_img.paste(mycopy,(0,original_img.height))
                cp=sp="./result/{}_combined.tif".format(img_name)
                combined_img.save(cp,format="TIFF")
                
                
                
                # array = tensor_permuted.detach().cpu().numpy() * 255
                # array = np.squeeze(array).astype(np.uint8)
                # image = Image.fromarray(array)
                # # 将RGB图像转换为灰度图像
                # gray_image = image.convert("L")

                # gray_image.save("./result/{}_{}_{}.png".format(img_name, ps, br))
                result.loc[result.shape[0], :] = [img_name, c_t, dc_t, ps, br]
                result.to_csv('./result/result.csv', index=False)
        json_file_path = '/data/yifanli/mycompress/result/all_images_data.json'
        with open(json_file_path, 'w') as json_file:
            json.dump(all_data, json_file)

    else:
        print("不保存")
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            # img = Image.open(img_path)
            transform = HistogramStretchingTransform16Bit()
            # img = transforms.ToTensor()(Image.open(img_path).convert('L')).to(device)
            img_transformed, min_val, max_val = transform(Image.open(img_path))
            x = transforms.ToTensor()(img_transformed).to(device).unsqueeze(0)     

            
            # x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            x_padded, padding = pad(x, p)
            count += 1
            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_net = net.forward(x_padded)
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                out_net['x_hat'].clamp_(0, 1)
                out_net["x_hat"] = crop(out_net["x_hat"], padding)

                ps = round(compute_psnr(x, out_net["x_hat"]), 2)
                br = round(compute_bpp(out_net), 2)

                PSNR += compute_psnr(x, out_net["x_hat"])
                MS_SSIM += compute_msssim(x, out_net["x_hat"])
                Bit_rate += compute_bpp(out_net)

                # tensor_permuted = out_net["x_hat"]
                # tensor_permuted = tensor_permuted.permute(0, 2, 3, 1)
                # array = tensor_permuted.detach().cpu().numpy() * 255
                # array = np.squeeze(array).astype(np.uint8)
                # image = Image.fromarray(array)
                # # 将RGB图像转换为灰度图像
                # gray_image = image.convert("L")

                # 保存灰度图像
    #                 gray_image_path = "path_to_save_gray_image.jpg"
    #                 gray_image.save(gray_image_path)

    #                 gray_image.save("./result/{}_{}_{}.png".format(img_name, ps, br))

    PSNR = PSNR / count
    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    total_time = total_time / count
    print(f'average_PSNR: {PSNR:.2f}dB')
    print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
    print(f'average_time: {total_time:.3f} ms')


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
