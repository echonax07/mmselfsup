import numpy as np
import xarray as xr
from mmcv.image import imread
from mmcv.transforms import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from icecream import ic

import numpy as np
import matplotlib.pyplot as plt


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
