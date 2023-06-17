from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

from read_dataset import to_numpy


def PSNR(x, y):
    return peak_signal_noise_ratio(to_numpy(x)[0], to_numpy(y)[0])


def SSIM(x, y):
    return structural_similarity(to_numpy(x)[0], to_numpy(y)[0], channel_axis=2, data_range=1)

def MSE(x, y):
    return mean_squared_error(to_numpy(x)[0], to_numpy(y)[0])