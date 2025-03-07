import matplotlib.pyplot as plt

from model.transformer import make_model as transformer
import numpy as np
import os
from glob import glob
from cugdt.Tiff import *
from tqdm import tqdm
from scipy.ndimage import binary_erosion, label
import scipy.ndimage as ndi
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.morphology import remove_small_objects, remove_small_holes

# 选择2020年和2021年两年的数据的起始位置
Cairo_2020 = [24, 36]
Cairo_2021 = [36, 48]
Melbourne_2020 = [12, 24]
Melbourne_2021 = [24, 36]
Mexicali_2020 = [0, 12]
Mexicali_2021 = [12, 24]
San_Francisco_2020 = [12, 24]
San_Francisco_2021 = [24, 36]
Sao_Paulo_2020 = [0, 12]
Sao_Paulo_2021 = [12, 24]
shenzhen_2020 = [24, 36]
shenzhen_2021 = [36, 48]
Wuhan_2020 = [13, 25]
Wuhan_2021 = [25, 37]
Xiongan_New_Area_2020 = [12, 24]
Xiongan_New_Area_2021 = [24, 36]


def built(config):
    model = transformer
    return model(config.src_vocab_size, config.tgt_vocab_size, config.num_layers, config.hidden_dim, config.feedforward_dim, config.num_heads,
                 config.dropout_rate)


def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    area = 'Mexicali'
    data = np.load(rf'data/{area}/{area}.npy')
    da = data[:, :, eval(f'{area}_2020')[0]:eval(f'{area}_2021')[1]]
    print(da.shape)
    np.save(rf'data/{area}.npy', da)
    # data, geo, pro = ReadGeoTIFF('data/S.tif')
    # print(data.shape, geo)
    # data = np.load(rf'data/Sao_Paulo/Sao_Paulo.npy')
    # print(data[:, 12, 0])
    # print(np.max(data[:, 11, 0]))
    # print(np.max(data[:, 12, 0]))
    # # 12 是经度，11是纬度
    # jingdu = geo[0] + data[:, 12, 0] * geo[1]
    # weidu = geo[3] + data[:, 11, 0] * geo[5]
    #
    # jingdu = np.tile(jingdu[:, np.newaxis, np.newaxis], (1, 1, 48))
    # weidu = np.tile(weidu[:, np.newaxis, np.newaxis], (1, 1, 48))
    #
    # data[:, 12:13, :] = weidu
    # data[:, 11:12, :] = jingdu
    # np.save(rf'data/Sao_Paulo/Sao_Paulo.npy', data)

