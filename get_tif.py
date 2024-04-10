import h5py
import numpy as np
import cv2
file_path = "/data/yifanli/data/ims/DAPI-USTC_THY1-YFP_1779_1_026_1_column-2.ims"
save_path = "/data/yifanli/data/tiff/train"  # 指定保存处理后图像的文件夹
def string2value(h5str):
    vs = []
    for v in h5str.tolist():
        vs.append(v.decode('utf-8'))
    return float(''.join(vs))
with h5py.File(file_path, 'r') as file:
    res_level_0_path = 'DataSet/ResolutionLevel 0'
    
    # 获取总时间点数量
    total_time_points = len(file[res_level_0_path].keys())

    for time_point in range(total_time_points):
        time_point_path = f'{res_level_0_path}/TimePoint {time_point}'

        if time_point_path in file:
            for channel in file[time_point_path].keys():
                channel_path = f'{time_point_path}/{channel}'

                if 'Data' in file[channel_path]:
                    dataset_data = np.array(file[channel_path]['Data'])  
                    dims_data = []
                    for k in ["ImageSizeX", "ImageSizeY", "ImageSizeZ"]:
                        dims_data.append(int(string2value(file[channel_path].attrs.get(k))))
                    print('dims_data:', dims_data)
                    dataset_data = dataset_data[0:dims_data[2], 0:dims_data[1], 0:dims_data[0]]

                    file_name = f'DAPI-USTC_THY1-YFP_1779_1_026_1_column-2_{time_point}.tif'
                    cv2.imwrite(f'{save_path}/{file_name}', np.transpose(dataset_data, (1, 2, 0)))
                    print(f'已保存处理后的图像：{file_name}')
                    
