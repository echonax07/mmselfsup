import numpy as np
import xarray as xr
from mmcv.image import imread
from mmcv.transforms import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from icecream import ic
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from functools import partial
import torch


@TRANSFORMS.register_module()
class LoadImageFromNetCDFFile(BaseTransform):
    """Load an image from an xarray dataset.

    Required Keys:
        - img_path

    Modified Keys:
        - img
        - img_shape
        - ori_shape

    Args:
        channels (list[str]): List of variable names to load as channels of the image.
        to_float32 (bool): Whether to convert the loaded image to a float32 numpy array.
            If set to False, the loaded image is a uint8 array. Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`. Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. Defaults to 'cv2'.
        ignore_empty (bool): Whether to allow loading empty image or file path not existent.
            Defaults to False.
    """

    def __init__(self,
                 channels,
                 mean=[-14.508254953309349, -24.701211250236728],
                 std=[5.659745919326586, 4.746759336539111],
                 to_float32=True,
                 color_type='color',
                 imdecode_backend='cv2',
                 nan=255,
                 ignore_empty=False):
        self.channels = channels
        self.mean = mean
        self.std = std
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.ignore_empty = ignore_empty

    def transform(self, results):
        """Functions to load image.

        Args:
            results (dict): Result dict from :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']
        # ic(filename)
        try:
            xarr = xr.open_dataset(filename, engine='h5netcdf')
            image_data = []
            for channel in self.channels:
                if channel in xarr:
                    channel_data = xarr[channel].values
                    # if channel_data.dtype == np.float32 or channel_data.dtype == np.float64:
                    #     channel_data = (channel_data * 255).astype(np.uint8)
                    # ic(image_data)
                    image_data.append(channel_data)
                else:
                    raise ValueError(
                        f"Variable '{channel}' not found in the NetCDF file.")
            img = np.stack(image_data, axis=-1)
            mean = np.array(self.mean)
            std = np.array(self.std)
            b = np.stack(image_data, axis=-1)
            img = (img-mean)/std
            # img = np.nan_to_num(img, nan=255)

            # # Plot the data
            # # ic(np.unique(img))
            # # ic(img.shape)
            # plt.imshow(img[:,:,0], cmap='gray')
            # # Add labels and title if needed
            # plt.xlabel('X-axis Label')
            # plt.ylabel('Y-axis Label')
            # plt.title('Plot Title')
            # # Save the plot to an image file (e.g., PNG format)
            # plt.savefig('/home/m32patel/projects/def-dclausi/AI4arctic/m32patel/mmselfsup/mmselfsup/transforms/plot_a.png')

            # plt.imshow(b[:,:,0], cmap='gray')
            # # Add labels and title if needed
            # plt.xlabel('X-axis Label')
            # plt.ylabel('Y-axis Label')
            # plt.title('Plot Title')
            # # Save the plot to an image file (e.g., PNG format)
            # plt.savefig('/home/m32patel/projects/def-dclausi/AI4arctic/m32patel/mmselfsup/mmselfsup/transforms/plot_b.png')

            # mask = (img==255)
            # img[mask]= np.nan

            # plt.imshow(img[:,:,0], cmap='gray')
            # # Add labels and title if needed
            # plt.xlabel('X-axis Label')
            # plt.ylabel('Y-axis Label')
            # plt.title('Plot Title')
            # # Save the plot to an image file (e.g., PNG format)
            # plt.savefig('/home/m32patel/projects/def-dclausi/AI4arctic/m32patel/mmselfsup/mmselfsup/transforms/plot_c.png')

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'channels={self.channels}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'ignore_empty={self.ignore_empty})')
        return repr_str


@TRANSFORMS.register_module()
class PreLoadImageFromNetCDFFile(BaseTransform):
    """Load an image from an xarray dataset.

    Required Keys:
        - img_path

    Modified Keys:
        - img
        - img_shape
        - ori_shape

    Args:
        channels (list[str]): List of variable names to load as channels of the image.
        to_float32 (bool): Whether to convert the loaded image to a float32 numpy array.
            If set to False, the loaded image is a uint8 array. Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`. Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. Defaults to 'cv2'.
        ignore_empty (bool): Whether to allow loading empty image or file path not existent.
            Defaults to False.
    """

    def __init__(self,
                 channels,
                 data_root,
                 ann_file = None,
                 mean=[-14.508254953309349, -24.701211250236728],
                 std=[5.659745919326586, 4.746759336539111],
                 to_float32=True,
                 color_type='color',
                 imdecode_backend='cv2',
                 nan=255,
                 downsample_factor=10,
                 ignore_empty=False):
        self.channels = channels
        self.mean = mean
        self.std = std
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.ignore_empty = ignore_empty
        self.data_root = data_root
        self.downsample_factor = downsample_factor
        self.ann_file = ann_file
        self.nc_files = self.list_nc_files(data_root, ann_file)
        # key represents full path of the image and value represents the np image loaded
        self.pre_loaded_image_dic = {}
        ic('Starting to load all the images into memory...')
        for filename in tqdm(self.nc_files):
            xarr = xr.open_dataset(filename, engine='h5netcdf')
            img = xarr[self.channels].to_array().data
            # reorder from (2, H, W) to (H, W, 2)
            img = np.transpose(img, (1, 2, 0))
            mean = np.array(self.mean)
            std = np.array(self.std)
            img = (img-mean)/std
            shape = img.shape
            if self.downsample_factor != 1:
                # downsample by taking max over a 10x10 block
                # img = torch.from_numpy(np.expand_dims(img, 0))
                img = torch.from_numpy(img)
                img = img.unsqueeze(0).permute(0, 3, 1, 2)
                img = torch.nn.functional.interpolate(img,
                                                      size=(shape[0]//self.downsample_factor,
                                                            shape[1]//self.downsample_factor),
                                                      mode='nearest')
                img = img.permute(0,2,3,1).squeeze(0)        
                # Take the average over each 10x10 block
                # img = img.mean(axis=(1, 3))
                img = img.numpy()
            if to_float32:
                img = img.astype(np.float32)
            self.pre_loaded_image_dic[filename] = img
        ic('Finished loading all the images into memory...')

    # # not speeding up
    # def load_image(self, filename, channels, mean, std, to_float32):
    #     xarr = xr.open_dataset(filename, engine='h5netcdf')
    #     img = xarr[channels].to_array().data
    #     img = np.transpose(img, (1, 2, 0))
    #     mean = np.array(mean)
    #     std = np.array(std)
    #     img = (img - mean) / std
    #     shape = img.shape
    #     ic(self.downsample_factor)
    #     if self.downsample_factor != 1:
    #         # downsample by taking max over a 10x10 block
    #         img = img.reshape(shape[0], self.downsample_factor,
    #                           shape[1], self.downsample_factor, shape[2])
    #         # Take the average over each 10x10 block
    #         img = img.mean(axis=(1, 3))
    #         ic(img.shape)
    #     if to_float32:
    #         img = img.astype(np.float32)
    #     return filename, img
    # # not speeding up

    # def load_images_parallel(self, nc_files, channels, mean, std, to_float32):
    #     pool = multiprocessing.Pool(processes=4)
    #     func = partial(self.load_image, channels=channels,
    #                    mean=mean, std=std, to_float32=to_float32)

    #     results = []
    #     with tqdm(total=len(nc_files)) as pbar:
    #         for result in pool.imap_unordered(func, nc_files):
    #             results.append(result)
    #             pbar.update()

    #     pool.close()
    #     pool.join()
    #     return dict(results)

    def list_nc_files(self, folder_path, ann_file):
        nc_files = []
        if ann_file != None:
            with open(ann_file, "r") as file:
                # Read the lines of the file into a list
                filenames = file.readlines()
            nc_files = [os.path.join(folder_path, filename.strip()) for filename in filenames]
        else:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".nc"):
                        nc_files.append(os.path.join(root, file))
        return nc_files

    def transform(self, results):
        """Functions to load image.

        Args:
            results (dict): Result dict from :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']
        img = self.pre_loaded_image_dic[filename]
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'channels={self.channels}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'ignore_empty={self.ignore_empty})')
        return repr_str
