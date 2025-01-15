import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
from skimage import io
from odl.contrib.torch import OperatorModule

import odl
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim1
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import math
import glob
from torch.utils.data import Dataset, DataLoader
import pydicom
from skimage.transform import resize

def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return x

def compare(recon0, recon1, verbose=True):
    mse_recon = mean_squared_error(recon0, recon1)
    # np.mean((recon0-recon1)**2)

    small_side = np.min(recon0.shape)
    if small_side < 7:
        if small_side % 2:  # if odd
            win_size = small_side
        else:
            win_size = small_side - 1
    else:
        win_size = None

    ssim_recon = ssim1(recon0, recon1,
                       data_range=recon0.max() - recon0.min(), win_size=win_size)

    psnr_recon = peak_signal_noise_ratio(recon0, recon1,
                                         data_range=recon0.max() - recon0.min())

    if verbose:
        err_string = 'MSE: {:.8f}, SSIM: {:.3f}, PSNR: {:.3f}'
        print(err_string.format(mse_recon, ssim_recon, psnr_recon))
    return (mse_recon, ssim_recon, psnr_recon)

class DicomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the DICOM images.
            transform (callable, optional): Optional transform to be applied on an image.
            size (int, optional): The size to which the images will be resized.

        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = glob.glob(os.path.join(self.root_dir, '**/*.IMA'), recursive=True)


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.file_list[idx]
        image = pydicom.read_file(img_path).pixel_array
        image = np.float32(image)
        image[image > 2500] =0
        image = image / 2500

        image = np.expand_dims(image, axis=0)  # add channel dimension
        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)


        return image
    
class ProjOperator:
    def __init__(self, N=512, M=512, pixel_size_x=0.15, pixel_size_y=0.15,
                 det_pixels=624, det_pixel_size=0.2, angles=720, src_origin=950,
                 det_origin=200):
        self.N = N
        self.M = M
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        self.det_pixels = det_pixels
        self.det_pixel_size = det_pixel_size
        self.angles = angles
        self.src_origin = src_origin
        self.det_origin = det_origin

    def forward(self, img):
        # create a uniform discretization reconstruction space
        reco_space = odl.uniform_discr(
            min_pt=[-self.N * self.pixel_size_x // 2, -self.N * self.pixel_size_x // 2],
            max_pt=[self.M * self.pixel_size_y // 2, self.M * self.pixel_size_y // 2],
            shape=[self.N, self.M], dtype='float32')

        # create angle partition
        grid = odl.uniform_grid(0, np.pi + 3.6 * np.pi / 18 - 21.6 * np.pi / 18 / self.angles + 0,
                                self.angles)
        angle_partition = odl.uniform_partition_fromgrid(grid)

        # create detector partition
        detector_partition = odl.uniform_partition(-self.det_pixels * self.det_pixel_size // 2,
                                                   self.det_pixels * self.det_pixel_size // 2,
                                                   self.det_pixels)

        # create geometry
        geometry = odl.tomo.FanBeamGeometry(
            angle_partition, detector_partition,
            src_radius=self.src_origin,
            det_radius=self.det_origin)

        # create ray transform
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

        # if using cuda
        if isinstance(img, torch.Tensor):
            fwd_op_mod = OperatorModule(ray_trafo)
            proj_data = fwd_op_mod(img)
        else:
            proj_data = ray_trafo(img).data

        return proj_data

    def __call__(self, img):
        return self.forward(img)
    

class FilteredProjOperator(ProjOperator):
    def __init__(self, N=512, M=512, pixel_size_x=0.15, pixel_size_y=0.15,
                 det_pixels=624, det_pixel_size=0.2, angles=720, src_origin=950,
                 det_origin=200, filter_type='Ram-Lak', frequency_scaling=0.7):
        # 继承父类的初始化
        super().__init__(N, M, pixel_size_x, pixel_size_y, det_pixels, 
                        det_pixel_size, angles, src_origin, det_origin)
        
        # 添加滤波相关的参数
        self.filter_type = filter_type
        self.frequency_scaling = frequency_scaling
        
        # 预先创建空间和几何信息（避免重复创建）
        self.reco_space = odl.uniform_discr(
            min_pt=[-self.N * self.pixel_size_x // 2, -self.N * self.pixel_size_x // 2],
            max_pt=[self.M * self.pixel_size_y // 2, self.M * self.pixel_size_y // 2],
            shape=[self.N, self.M], dtype='float32')

        grid = odl.uniform_grid(0, np.pi + 3.6 * np.pi / 18 - 21.6 * np.pi / 18 / self.angles + 0,
                                self.angles)
        self.angle_partition = odl.uniform_partition_fromgrid(grid)

        self.detector_partition = odl.uniform_partition(
            -self.det_pixels * self.det_pixel_size // 2,
            self.det_pixels * self.det_pixel_size // 2,
            self.det_pixels)

        self.geometry = odl.tomo.FanBeamGeometry(
            self.angle_partition, self.detector_partition,
            src_radius=self.src_origin,
            det_radius=self.det_origin)

        # 创建射线变换
        self.ray_trafo = odl.tomo.RayTransform(self.reco_space, self.geometry)
        
        # 创建滤波器
        self.fourier_filter = odl.tomo.fbp_filter_op(self.ray_trafo,
                                                    filter_type=self.filter_type,
                                                    frequency_scaling=self.frequency_scaling)

    def forward(self, img):
        # 首先进行投影
        if isinstance(img, torch.Tensor):
            fwd_op_mod = OperatorModule(self.ray_trafo)
            proj_data = fwd_op_mod(img)
            # 对投影数据进行滤波
            filter_op_mod = OperatorModule(self.fourier_filter)
            filtered_proj = filter_op_mod(proj_data)
            return filtered_proj
        else:
            proj_data = self.ray_trafo(img).data
            # 对投影数据进行滤波
            filtered_proj = self.fourier_filter(proj_data).data
            return filtered_proj

    def __call__(self, img):
        return self.forward(img)

class FBPOperator:
    def __init__(self, N=512, M=512, pixel_size_x=0.15, pixel_size_y=0.15,
                 det_pixels=624, det_pixel_size=0.2, angles=720, src_origin=950,
                 det_origin=200, filter_type='Ram-Lak', frequency_scaling=0.7):
        self.N = N
        self.M = M
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        self.det_pixels = det_pixels
        self.det_pixel_size = det_pixel_size
        self.angles = angles
        self.src_origin = src_origin
        self.det_origin = det_origin
        self.filter_type = filter_type
        self.frequency_scaling = frequency_scaling

        # create a uniform discretization reconstruction space
        self.reco_space = odl.uniform_discr(
            min_pt=[-self.N * self.pixel_size_x // 2, -self.N * self.pixel_size_x // 2],
            max_pt=[self.M * self.pixel_size_y // 2, self.M * self.pixel_size_y // 2],
            shape=[self.N, self.M], dtype='float32')

        # create angle partition
        grid = odl.uniform_grid(0, np.pi + 3.6 * np.pi / 18 - 21.6 * np.pi / 18 / self.angles + 0,
                                self.angles)
        self.angle_partition = odl.uniform_partition_fromgrid(grid)

        # create detector partition
        self.detector_partition = odl.uniform_partition(-self.det_pixels * self.det_pixel_size // 2,
                                                        self.det_pixels * self.det_pixel_size // 2,
                                                        self.det_pixels)

        # create geometry
        self.geometry = odl.tomo.FanBeamGeometry(
            self.angle_partition, self.detector_partition,
            src_radius=self.src_origin,
            det_radius=self.det_origin)

        # create ray transform
        self.ray_trafo = odl.tomo.RayTransform(self.reco_space, self.geometry)

        # create FBP operator
        self.fbp = odl.tomo.fbp_op(self.ray_trafo, filter_type=self.filter_type,
                                   frequency_scaling=self.frequency_scaling)

        # create parker weighting
        self.parker_weighting = odl.tomo.parker_weighting(self.ray_trafo, 1)

        # create parker weighted FBP operator
        self.parker_weighted_fbp = self.fbp * self.parker_weighting

    def forward(self, proj):
        # if using cuda
        if isinstance(proj, torch.Tensor):
            fwd_op_adj_mod = OperatorModule(self.ray_trafo.adjoint)
            parker_weighted_fbp_mod = OperatorModule(self.parker_weighted_fbp)
            reconstructed_img = parker_weighted_fbp_mod(proj)
        else:
            reconstructed_img = self.parker_weighted_fbp(proj).data

        return reconstructed_img

    def __call__(self, proj):
        return self.forward(proj)
    
class BackProjectionOperator:
    def __init__(self, N=512, M=512, pixel_size_x=0.15, pixel_size_y=0.15,
                det_pixels=624, det_pixel_size=0.2, angles=720, src_origin=950,
                det_origin=200):
        self.N = N
        self.M = M
        self.pixel_size_x = pixel_size_x 
        self.pixel_size_y = pixel_size_y
        self.det_pixels = det_pixels
        self.det_pixel_size = det_pixel_size
        self.angles = angles
        self.src_origin = src_origin
        self.det_origin = det_origin

        # 创建重建空间
        self.reco_space = odl.uniform_discr(
            min_pt=[-self.N * self.pixel_size_x // 2, -self.N * self.pixel_size_x // 2],
            max_pt=[self.M * self.pixel_size_y // 2, self.M * self.pixel_size_y // 2],
            shape=[self.N, self.M], dtype='float32')

        # 创建角度划分
        grid = odl.uniform_grid(0, np.pi + 3.6 * np.pi / 18 - 21.6 * np.pi / 18 / self.angles + 0,
                                self.angles)
        self.angle_partition = odl.uniform_partition_fromgrid(grid)

        # 创建探测器划分
        self.detector_partition = odl.uniform_partition(
            -self.det_pixels * self.det_pixel_size // 2,
            self.det_pixels * self.det_pixel_size // 2,
            self.det_pixels)

        # 创建扇束几何
        self.geometry = odl.tomo.FanBeamGeometry(
            self.angle_partition, self.detector_partition,
            src_radius=self.src_origin,
            det_radius=self.det_origin)

        # 创建射线变换
        self.ray_trafo = odl.tomo.RayTransform(self.reco_space, self.geometry)

    def forward(self, proj):
        # 如果输入是torch tensor
        if isinstance(proj, torch.Tensor):
            # 只使用伴随算子进行反投影
            fwd_op_adj_mod = OperatorModule(self.ray_trafo.adjoint)
            reconstructed_img = fwd_op_adj_mod(proj)
        else:
            # 对numpy数组进行操作
            reconstructed_img = self.ray_trafo.adjoint(proj).data

        return reconstructed_img

    def __call__(self, proj):
        return self.forward(proj)
    
def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def window_image(img, window_center, window_width, intercept, slope, rescale=True):
    img = (img * slope + intercept)  # for translation adjustments given in the dicom file.
    img_min = window_center - window_width // 2  # minimum HU level
    img_max = window_center + window_width // 2  # maximum HU level
    img[img < img_min] = img_min  # set img_min for all HU levels less than minimum HU level
    img[img > img_max] = img_max  # set img_max for all HU levels higher than maximum HU level
    if rescale:
        img = (img - img_min) / (img_max - img_min)
    return img
    
# x =np.float32(pydicom.read_file('/home/w/fpb_untrain/2755.IMA').pixel_array)
# x = x / np.max(x)
# x[x < 0] = 0
# plt.imshow(x, cmap='gray')
# plt.show()
# size=512
# angles=64
# radon_pro = ProjOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
#                  det_pixels=624, det_pixel_size=0.2, angles=angles, src_origin=950,
#                  det_origin=200)

# fbp_op_angel = FBPOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
#                  det_pixels=624, det_pixel_size=0.2, angles=angles, src_origin=950,
#                  det_origin=200, filter_type='Ram-Lak', frequency_scaling=0.7)

# p = radon_pro(x)
# plt.imshow(p, cmap='gray')
# plt.show()
# x_recon = fbp_op_angel(p)
# compare(x, x_recon)
# plt.imshow(x_recon, cmap='gray')
# plt.show()