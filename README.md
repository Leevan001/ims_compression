# ims_compression
# ims_压缩

## ims转tiff

使用脚本ims_compression/get_tif.py

## 训练

一个例子（支持单卡）

```bash
 CUDA_VISIBLE_DEVICES=1 python -u ./train.py -d /data/yifanli/data/tiff --cuda --N 128 --lambda 0.05 --epochs 100  --batch-size 16 --save_path ./cm_save/tiff --save  --checkpoint /data/yifanli/mycompress/cm_save/lambda_0.01/0.01checkpoint_best.pth.tar --learning-rate 1e-5 --lr_epoch 45 48
```

## 评估

```bash
CUDA_VISIBLE_DEVICES=3 python eval.py  --checkpoint /data/yifanli/mycompress/tif_save/1.0checkpoint_best.pth.tar  --data /data/yifanli/data/tiff/test --cuda  --real
```

一张图片将得到三张对应图片，分别为：

1. 解压后的图片
2. 原图片
3. 上面两张上下拼接在一起的图片

其中`CUDA_VISIBLE_DEVICES=3`和`--cuda`用于设置是否设置在显卡上评估

`checkpoint`用于加载模型权重

## 实际应用

### 压缩

将tiff压缩为json

```bash
CUDA_VISIBLE_DEVICES=7 python compress2json.py  --checkpoint /data/yifanli/mycompress/tif_save/0.1checkpoint_best.pth.tar  --data /data/yifanli/mycompress/ddebug  --sf /data/yifanli/mycompress/jsons --cuda

python compress2json.py  --checkpoint /data/yifanli/mycompress/tif_save/0.1checkpoint_best.pth.tar  --data /data/yifanli/mycompress/ddebug  --sf /data/yifanli/ims_compression/jsons
```

第一行是通过显卡压缩，第二行五显卡

- `sf`控制输出json文件所在的文件夹（即压缩产物）
- `--data`选择要压缩的文件所在的文件夹（里面的图像格式应为tiff，且长宽为2048x788）

### 解压

将json文件解压为tiff

```bash
CUDA_VISIBLE_DEVICES=7 python decompress2tif.py  --checkpoint /data/yifanli/mycompress/tif_save/0.1checkpoint_best.pth.tar  --js /data/yifanli/ims_compression/jsons/compressed.json   --sf /data/yifanli/ims_compression/decompress   --cuda


python decompress2tif.py  --checkpoint /data/yifanli/mycompress/tif_save/0.1checkpoint_best.pth.tar  --js /data/yifanli/ims_compression/jsons/compressed.json   --sf /data/yifanli/ims_compression/decompress
```

- `js`指定被压缩后的json文件
- `sf`指定解压后的文件所在的文件夹

## 模型权重

提供了三种权重文件

0.1,0.5.1.0

数字越小对应的压缩率越高，但质量下降

权重网盘链接为：

