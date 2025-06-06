import glob
import torch.utils.data as data
from PIL import Image
import torch
import sys
import os, cv2
import pickle
import numpy as np
from tifffile import imread
sys.path.append('..')
from csbdeep.utils import normalize, axes_dict, axes_check_and_normalize, backend_channels_last, move_channel_for_backend
from csbdeep.io import load_training_data
from tifffile import imwrite as imsave
from tqdm import tqdm

CSB_path = '/home/user2/dataset/microscope/CSB'
VCD_path = '/home/user2/dataset/microscope/VCD'

datamin, datamax = 0, 100  #

def augment(img1:torch.Tensor, img2:torch.Tensor):
    if np.random.rand() > 0.5:
        img1 = torch.flip(img1, [1])
        img2 = torch.flip(img2, [1])
    if np.random.rand() > 0.5:
        img1 = torch.flip(img1, [2])
        img2 = torch.flip(img2, [2])
    return img1, img2

def random_crop(img1:torch.Tensor, img2:torch.Tensor, patch_size:int):
    _, h, w = img1.size()
    x = np.random.randint(0, w - patch_size)
    y = np.random.randint(0, h - patch_size)
    img1 = img1[:, y:y + patch_size, x:x + patch_size]
    img2 = img2[:, y:y + patch_size, x:x + patch_size]
    return img1, img2

def np2Tensor(*args):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        return tensor
    
    return [_np2Tensor(a) for a in args]


def loadData(traindatapath, axes='SCYX', validation_split=0.05):
    print('Load data npz')
    if validation_split > 0:
        (X, Y), (X_val, Y_val), axes = load_training_data(traindatapath, validation_split=validation_split, axes=axes, verbose=True)
    else:
        (X, Y), _, axes = load_training_data(traindatapath, validation_split=validation_split, axes=axes, verbose=True)
        X_val, Y_val = 0, 1
    print(X.shape, Y.shape)  # (18468, 128, 128, 1) (18468, 256, 256, 1)
    return X, Y, X_val, Y_val

def add_gaussian_blur(image, blur_level):
    if image.ndim == 3:
        for i in range(image.shape[0]):
            image[i] = add_gaussian_blur(image[i], blur_level)
        return image

    dst = cv2.GaussianBlur(image, (blur_level, blur_level), 0)
    return dst

def add_gaussian_noise(image, noise_level):
    if image.ndim == 3:
        for i in range(image.shape[0]):
            image[i] = add_gaussian_noise(image[i], noise_level)
        return image

    row, col = image.shape
    mean = 0
    std_dev = noise_level
    
    gauss = np.random.normal(mean, std_dev, (row, col))
    noisy = image + gauss
    
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

class DIV2K_Agents(data.Dataset):
    def __init__(self, args, train=True, repeat=1, shuffle=False):
        self.args = args
        self.train = train
        self.dir_data = ['dataset/%s/%s' % ('train' if train else 'test', s) for s in args.data_test.split('+')]
        self.images_hr = []
        for path in self.dir_data:
            files = glob.glob(path+'/HR*.tif')
            self.images_hr.extend(files)
        if shuffle:
            np.random.shuffle(self.images_hr)
        self.repeat = repeat
        self.args.degrade = self.args.degrade.split('+') 

    def __getitem__(self, idx):
        idx = self._get_index(idx)

        hr_file = self.images_hr[idx]

        # print('hr_file', hr_file)
        hr = imread(hr_file)
        # print('hr.shape', hr.shape, hr.max(), hr.min())

        lr = hr
        for degrade in self.args.degrade:
            if degrade == 'None':
                continue
            elif 'b' in degrade:
                lr = cv2.GaussianBlur(lr, (int(degrade[1:]), int(degrade[1:])), 0)
            else:
                lr = add_gaussian_noise(lr, int(degrade[1:]))
        
        hr = normalize(hr, datamin, datamax, clip=True) * self.args.rgb_range
        lr = normalize(lr, datamin, datamax, clip=True) * self.args.rgb_range
        pair = (lr[..., None], hr[..., None])   # (256, 256, 1) (256, 256, 1)
        pair_t = np2Tensor(*pair)
        
        if self.train:
            pair_t = augment(pair_t[0], pair_t[1])
            pair_t = random_crop(pair_t[0], pair_t[1], self.args.patch_size)
        
        return pair_t[0], pair_t[1], os.path.basename(hr_file)

    def __len__(self):
        # print('len(self.images_hr)', len(self.images_hr))
        return int(len(self.images_hr) * self.repeat)
        
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

class DIV2K(data.Dataset):
    def __init__(self, args, name='CCPs', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.benchmark = benchmark
        self.dir_data = f'{CSB_path}/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/train/%s/my_training_data.npz' % args.data_test
        self.dir_demo = f'{CSB_path}/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/test/%s/LR/' % args.data_test

        self.input_large = (self.dir_demo != '')
        self.scale = args.scale
        if train:
            X, Y, X_val, Y_val = self.loadData()  # (18468, 128, 128, 1) (18468, 256, 256, 1)
            print('np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
            list_hr, list_lr = Y, X
        else:
            self.filenames = glob.glob(self.dir_demo + '*.tif')
            list_hr, list_lr, name = self._scan()
            if not args.test_only:
                list_hr = list_hr[:5]
                list_lr = list_lr[:5]
            self.name = name

        self.images_hr, self.images_lr = list_hr, list_lr
        self.repeat = 1
        
    def loadData(self):
        patch_size = self.args.patch_size
        X, Y, X_val, Y_val = loadData(self.dir_data)  # (18468, 128, 128, 1) (18468, 256, 256, 1)

        N, height, width, c = X.shape
        X1 = []
        Y1 = []
        for n in range(len(X)):
            for i in range(0, width, patch_size):
                for j in range(0, height, patch_size):
                    if j + patch_size >= height and i + patch_size >= width:
                        X1.append(X[n][height-patch_size:, width - patch_size:, :])
                        Y1.append(Y[n][height * 2 - patch_size * 2:, width * 2 - patch_size * 2:, :])
                    elif j + patch_size >= height:
                        X1.append(X[n][height - patch_size:, i:i + patch_size, :])
                        Y1.append(Y[n][height * 2 - patch_size * 2:, i * 2:i * 2 + patch_size * 2, :])
                    elif i + patch_size >= width:
                        X1.append(X[n][j:j + patch_size, width - patch_size:, :])
                        Y1.append(Y[n][j * 2:j * 2 + patch_size * 2:, width * 2 - patch_size * 2:, :])
                    else:
                        X1.append(X[n][j:j + patch_size, i:i + patch_size, :])
                        Y1.append(Y[n][j * 2:j * 2 + patch_size * 2, i * 2:i * 2 + patch_size * 2, :])
        
        return X1, Y1, X_val, Y_val   # return np.array(X1), np.array(Y1), X_val, Y_val

    def _scan(self):
        
        list_hr, list_lr, nm = [], [], []
        for fi in self.filenames:
            hr = np.array(Image.open(fi.replace('LR', 'GT')))
            lr = np.array(Image.open(fi))
            nm.append(fi[len(self.dir_demo):])
            list_hr.append(np.expand_dims(hr, -1))
            list_lr.append(np.expand_dims(lr, -1))
        return list_hr, list_lr, nm

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        if self.train:
            lr, hr, filename = self.images_lr[idx], self.images_hr[idx], ''
        else:
            lr, hr, filename = self.images_lr[idx], self.images_hr[idx], self.name[idx]
        
        hr = normalize(hr, datamin, datamax, clip=True) * self.args.rgb_range
        lr = normalize(lr, datamin, datamax, clip=True) * self.args.rgb_range
        pair = (lr, hr)   # (128, 128, 1) (256, 256, 1)
        pair_t = np2Tensor(*pair)
        
        return pair_t[0], pair_t[1], filename

    def __len__(self):
        # print('len(self.images_hr)', len(self.images_hr))
        return len(self.images_hr) * self.repeat
        
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx


class Flourescenedenoise(data.Dataset):
    def __init__(self, args, istrain=True, c=1):
        self.args = args
        self.batch = 1
        self.datamin, self.datamax = args.datamin, args.datamax
        self.istrain = istrain
        
        if self.args.data_test:
            self.denoisegt = [self.args.data_test]
        else:
            self.denoisegt = ['Denoising_Planaria', 'Denoising_Tribolium'               ]
        
        if istrain:
            self._scandenoisenpy()
        else:
            self._scandenoisetif(c)
        
        self.lenthdenoise = len(self.nm_lrdenoise)
        self.lenth = self.lenthdenoise // self.batch
        
        if istrain:
            print('++ ++ ++ ++ ++ ++ length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')
    
    def _scandenoisenpy(self):
        hr = []
        lr = []
        datapath = f'{CSB_path}/DataSet/'
        for i in self.denoisegt:
            # Planaria: X/Y  (17005, 16, 64, 64, 1)(895, 16, 64, 64, 1)  float32
            # Tr  (14725, 16, 64, 64, 1) (775, 16, 64, 64, 1)
            X, Y, X_val, Y_val = loadData(datapath + i + '/train_data/data_label.npz', axes='SCZYX')
            
            print('Dataset:', i, 'np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
            print('X.shape, Y.shape, X_val.shape, Y_val.shape = ', X.shape, Y.shape, X_val.shape, Y_val.shape)
            height, width = X.shape[-3:-1]
            X = np.reshape(X, [-1, height, width, 1])
            Y = np.reshape(Y, [-1, height, width, 1])
            assert len(X) == len(Y)
            hr.extend(Y)
            lr.extend(X)
        self.nm_hrdenoise, self.nm_lrdenoise = hr, lr
        assert len(hr) == len(lr)
    
    def _scandenoisetif(self, c=1):
        lr = []
        datapath = f'{CSB_path}/DataSet/'
        lr.extend(sorted(glob.glob(datapath + '%s/test_data/condition_%d/*.tif' % (self.denoisegt[0], c))))
        self.hrpath = datapath + '%s/test_data/GT/' % self.denoisegt[0]
        lr.sort()
        self.nm_lrdenoise = lr
    
    def __getitem__(self, idx):
        idx = idx % self.lenth
        if self.istrain:
            lr, hr, filename = self._load_file_denoise_npy(idx + self.args.inputchannel//2)
        else:
            lr, hr, filename, d = self._load_file_denoise(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.args.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.args.rgb_range)).float()
        return lr, hr, filename
    
    def __len__(self):
        if self.istrain:
            return self.lenth - 2 * (self.args.inputchannel//2)
        else:
            return self.lenth
    
    def _load_file_denoise(self, idn):
        filename, fmt = os.path.splitext(os.path.basename(self.nm_lrdenoise[idn]))

        rgb = np.float32(imread(self.hrpath + filename + fmt))  # / 65535
        rgblr = np.float32(imread(self.nm_lrdenoise[idn]))
        # print('Test Denoise, ----> rgblr.max/min', rgblr.max(), rgblr.min(), rgblr.shape)
        return rgblr, rgb, filename, self.denoisegt[0]

    def _load_file_denoise_npy(self, idx):
        lr = []
        hr = []
        if self.args.inputchannel > 1:
            for i in range(self.batch):
                idn = (idx + i) % self.lenthdenoise
                hr.extend(self.nm_hrdenoise[idn:idn + 1])
                lr.extend(self.nm_lrdenoise[idn - self.args.inputchannel // 2:idn + self.args.inputchannel // 2 + 1])
            rgb = np.concatenate(hr, -1)  # 0~4.548696  [B, 64, 64, 1]
            rgblr = np.squeeze(np.concatenate(lr, -1))  # 0~87.93965  [B, 64, 64, 5]
            rgb = np.transpose(np.float32(rgb), (2, 0, 1))  # [5, 256, 256]
            rgblr = np.transpose(np.float32(rgblr), (2, 0, 1))
        else:
            for i in range(self.batch):
                idn = (idx + i) % self.lenthdenoise
                hr.append(self.nm_hrdenoise[idn])
                lr.append(self.nm_lrdenoise[idn])
    
            rgb = np.squeeze(np.concatenate(hr, -1))
            rgblr = np.squeeze(np.concatenate(lr, -1))
            rgb = np.transpose(np.float32(rgb), (2, 0, 1))  # [1, 256, 256]
            rgblr = np.transpose(np.float32(rgblr), (2, 0, 1))
        
        return rgblr, rgb, ''

class Flouresceneiso_Agents(data.Dataset):
    def __init__(self, args, istrain=True, repeat=1, shuffle=False):
        self.args = args
        self.train = istrain
        self.dir_data = ['dataset/%s/%s' % ('train' if istrain else 'test', s) for s in args.data_test.split('+')]
        self.images_hr = []
        for path in self.dir_data:
            files = glob.glob(path+'/HR*.tif')
            self.images_hr.extend(files)
        self.repeat = repeat
        if shuffle:
            np.random.shuffle(self.images_hr)
        self.args.degrade = self.args.degrade.split('+') 

    def __getitem__(self, idx):
        idx = self._get_index(idx)

        hr_file = self.images_hr[idx]

        # print('hr_file', hr_file)
        hr = imread(hr_file).astype(np.uint8)
        # print('hr.shape', hr.shape, hr.max(), hr.min())

        lr = hr
        for degrade in self.args.degrade:
            if degrade == 'None':
                continue
            elif 'i' in degrade:
                lr = lr[:,::int(degrade[1:])]
            elif 'b' in degrade:
                lr[0] = cv2.GaussianBlur(lr[0], (int(degrade[1:]), int(degrade[1:])), 0)
            else:
                lr = add_gaussian_noise(lr, int(degrade[1:]))
        
        hr = normalize(hr, datamin, datamax, clip=True) * self.args.rgb_range
        lr = normalize(lr, datamin, datamax, clip=True) * self.args.rgb_range

        lr = torch.from_numpy(np.ascontiguousarray(lr * self.args.rgb_range)).float().unsqueeze(0)
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.args.rgb_range)).float().unsqueeze(0)

        if lr.shape != hr.shape:
            lr = torch.nn.functional.interpolate(lr.unsqueeze(0), hr.shape[-3:], mode='nearest').squeeze(0)

        lr = lr.squeeze(0)
        hr = hr.squeeze(0)

        if self.train:
            lr, hr = augment(lr, hr)
        
        return lr, hr, os.path.basename(hr_file)

    def __len__(self):
        # print('len(self.images_hr)', len(self.images_hr))
        return int(len(self.images_hr) * self.repeat)
        
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

class Flouresceneiso(data.Dataset):
    def __init__(self, args, istrain=True):
        self.args = args
        self.batch = 1
        self.datamin, self.datamax = 0, 100
        self.istrain = istrain

        self.iso = ['Isotropic_Liver']
        if istrain:
            self._scanisonpy()
        else:
            self._scaniso()
            
        if istrain:
            print('++ ++ ++ ++ ++ ++ length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        
    def _scanisonpy(self):
        hr = []
        lr = []
        patch_size = self.args.patch_size
    
        datapath = f'{CSB_path}/DataSet/Isotropic/'
        for i in self.iso:
            # Liver X/Y (3872, 128, 128, 1)
            X, Y, _, _ = loadData(datapath + '%s/train_data/data_label.npz' % i, axes='SCYX', validation_split=0.0)
            
            print('Dataset:', i, 'np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
            print('X.shape, Y.shape = ', X.shape, Y.shape)
            height, width = X.shape[1:3]
            assert len(X) == len(Y)
            
            if patch_size < height:
                X1 = []
                Y1 = []
                for n in range(len(X)):
                    for i in range(0, width, patch_size):
                        for j in range(0, height, patch_size):
                            X1.append(X[n][j:j + patch_size, i:i + patch_size, :])
                            Y1.append(Y[n][j:j + patch_size, i:i + patch_size, :])
                hr.extend(Y1)
                lr.extend(X1)
            else:
                hr.extend(Y)
                lr.extend(X)

        self.nm_hriso, self.nm_lriso = hr, lr
        self.lenth = len(self.nm_lriso)

    def _scaniso(self):
        hr = []
        lr = []
        for i in self.iso:
            self.dir_lr = f'{CSB_path}/DataSet/Isotropic/%s/test_data/' % i
            hr.append(self.dir_lr + 'input_subsample_1_groundtruth.tif')  # Liver [301, 752, 752]
            lr.append(self.dir_lr + 'input_subsample_8.tif')
            
        self.nm_hr, self.nm_lr = hr, lr
        self.lenth = len(self.nm_lr)

    def __getitem__(self, idx):
        idx = idx % self.lenth
        if self.istrain:
            lr, hr, filename = self._load_file_iso_npy(idx)
        else:
            lr, hr, filename = self._load_file_isotest(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.args.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.args.rgb_range)).float()
        
        return lr, hr, filename
    
    def __len__(self):
        return self.lenth

    def _load_file_iso_npy(self, idx):
        lr = []
        hr = []
        for i in range(self.batch):
            idn = (idx + i) % self.lenth
            hr.append(self.nm_hriso[idn])
            lr.append(self.nm_lriso[idn])
    
        rgb = np.concatenate(hr, -1)
        rgblr = np.concatenate(lr, -1)
        rgb = np.transpose(np.float32(rgb), (2, 0, 1))  # [1, 128, 128]
        rgblr = np.transpose(np.float32(rgblr), (2, 0, 1))
    
        return rgblr, rgb, ''

    def _load_file_isotest(self, idx):
        filename, i = os.path.splitext(os.path.basename(self.nm_lr[idx]))
        rgblr = np.float32(imread(self.nm_lr[idx]))

        # Liver [301, 752, 752]
        hrp = self.nm_lr[idx].replace('_8.tif', '_1_groundtruth.tif')
        try:
            rgb = np.float32(imread(hrp))
        except:
            rgb = np.ones([301, 752, 752])
        
        return rgblr, rgb, filename

class Flouresceneproj_Agents(data.Dataset):
    def __init__(self, args, train=True, repeat=1, shuffle=False):
        self.args = args
        self.train = train
        self.dir_data = ['dataset/%s/%s' % ('train' if train else 'test', s) for s in args.data_test.split('+')]
        self.images_hr = []
        for path in self.dir_data:
            files = glob.glob(path+'/HR*.tif')
            self.images_hr.extend(files)
        self.repeat = repeat
        if shuffle:
            np.random.shuffle(self.images_hr)
        self.args.degrade = self.args.degrade.split('+') 

    def __getitem__(self, idx):
        idx = self._get_index(idx)

        hr_file = self.images_hr[idx]

        # print('hr_file', hr_file)
        hr = imread(hr_file).astype(np.uint8)
        # print('hr.shape', hr.shape, hr.max(), hr.min())

        lr = imread(hr_file.replace('/HR', '/'))
        # print('lr.shape', lr.shape, lr.max(), lr.min())
        for degrade in self.args.degrade:
            if degrade == 'None':
                continue
            elif 'b' in degrade:
                lr = add_gaussian_blur(lr, int(degrade[1:]))
            else:
                lr = add_gaussian_noise(lr, int(degrade[1:]))
        
        hr = normalize(hr, datamin, datamax, clip=True) * self.args.rgb_range
        lr = normalize(lr, datamin, datamax, clip=True) * self.args.rgb_range

        lr = torch.from_numpy(np.ascontiguousarray(lr * self.args.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.args.rgb_range)).float()

        if self.train:
            lr, hr = augment(lr, hr)
        
        return lr, hr, hr_file

    def __len__(self):
        # print('len(self.images_hr)', len(self.images_hr))
        return int(len(self.images_hr) * self.repeat)
        
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

class Flouresceneproj(data.Dataset):
    def __init__(self, args, istrain=True, condition=0):
        self.args = args
        self.batch = 1
        self.istrain = istrain
        self.conv2d = True
        self.iso = ['Projection_Flywing']  #
        if istrain:
            self._scannpy()
        else:
            self._scan(condition)
            
        if istrain:
            print('++ ++ ++ ++ ++ ++ length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')

    def load_training_data(self, file, axes=None, n_images=None, verbose=False):
        print('Begin np.load(file)', file)
        f = np.load(file)
        X, Y = f['X'], f['Y']
        Y = np.expand_dims(Y, 2)
        print(Y.ndim, Y.shape)
        if axes is None:
            axes = f['axes']  # 'SCZYX'
        axes = axes_check_and_normalize(axes)
    
        assert X.ndim == Y.ndim
        assert len(axes) == X.ndim
        assert 'C' in axes
        if n_images is None:
            n_images = X.shape[0]
        assert X.shape[0] == Y.shape[0]
        assert 0 < n_images <= X.shape[0]
    
        X, Y = X[:n_images], Y[:n_images]
        channel = axes_dict(axes)['C']
        
        X = move_channel_for_backend(X, channel=channel)
        Y = move_channel_for_backend(Y, channel=channel)
    
        axes = axes.replace('C', '')  # remove channel
        if backend_channels_last():
            axes = axes + 'C'
        else:
            axes = axes[:1] + 'C' + axes[1:]
    
        if verbose:
            ax = axes_dict(axes)
            n_train, n_val = len(X), 0
            image_size = tuple(X.shape[ax[a]] for a in axes if a in 'TZYX')
            n_dim = len(image_size)
            n_channel_in, n_channel_out = X.shape[ax['C']], Y.shape[ax['C']]
        
            print('number of training images:\t', n_train)
            print('number of validation images:\t', n_val)
            print('image size (%dD):\t\t' % n_dim, image_size)
            print('axes:\t\t\t\t', axes)
            print('channels in / out:\t\t', n_channel_in, '/', n_channel_out)
    
        return X, Y
    
    def _scannpy(self):
        patch_size = self.args.patch_size
        mytraindata = 0  # 2  #  1  # 
        
        if mytraindata == 1:
            datapath = f'{CSB_path}/DataSet/%s/train_data/my_training_data.npz' % \
                        self.iso[0]
            X, Y, _, _ = loadData(datapath, axes=None, validation_split=0.0)
        elif mytraindata == 2:
            datapath = f'{CSB_path}/DataSet/%s/train_data/my_training_data.npz' % \
                        self.iso[0]
            datapath2 = f'{CSB_path}/DataSet/%s/train_data/data_label.npz' % \
                        self.iso[0]
            X1, Y1, _, _ = loadData(datapath, axes=None, validation_split=0.0)
            X1l = []
            Y1l = []
            for n in range(len(X1)):
                for i in range(0, 128, 64):
                    for j in range(0, 128, 64):
                        X1l.append(X1[n][:, j:j + 64, i:i + 64, :])
                        Y1l.append(Y1[n][:, j:j + 64, i:i + 64, :])
            X1 = np.array(X1l)  # [3136, 50, 64, 64, 1]
            Y1 = np.array(Y1l)
            
            X2, Y2 = self.load_training_data(datapath2, axes='SCZYX', verbose=True)  # 0~38.15789, 0~4.73316
            X = np.concatenate([X1, X2], 0)  # [20916, 50, 64, 64, 1]
            Y = np.concatenate([Y1, Y2], 0)
        else:
            datapath = f'{CSB_path}/DataSet/%s/train_data/data_label.npz' % self.iso[0]
            X, Y = self.load_training_data(datapath, axes='SCZYX', verbose=True)  # 0~38.15789, 0~4.73316
            
        print('Dataset:', self.iso[0], 'np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
        print('X.shape, Y.shape = ', X.shape, Y.shape)
        height, width = X.shape[2:4]
        assert len(X) == len(Y)

        if patch_size < height:
            X1 = []
            Y1 = []
            for n in range(len(X)):
                for i in range(0, width, patch_size):
                    for j in range(0, height, patch_size):
                        X1.append(X[n][:, j:j + patch_size, i:i + patch_size, :])
                        Y1.append(Y[n][:, j:j + patch_size, i:i + patch_size, :])
        else:
            Y1 = Y
            X1 = X
        self.nm_hr, self.nm_lr = Y1, X1
        self.lenth = len(self.nm_lr)
    
    def _scan(self, condition):
        hr = []
        lr = []
        for i in self.iso:
            self.dir_lr = f'{CSB_path}/DataSet/%s/test_data/' % i
            
            lr.extend(glob.glob(self.dir_lr + 'Input/C%d/*.tif' % condition))  # [50, 520, 692]
            hr.extend(glob.glob(self.dir_lr + 'GT/C%d/*.tif' % condition))  # [692, 520]
        hr.sort()  # proj_C2_T026.tif
        lr.sort()  # C1_T026.tif  #
        
        self.nm_hr, self.nm_lr = hr, lr
        self.lenth = len(self.nm_lr)
    
    def __getitem__(self, idx):
        idx = idx % self.lenth
        if self.istrain:
            lr, hr, filename = self._load_file_npy(idx)
        else:
            lr, hr, filename = self._load_file_test(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.args.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.args.rgb_range)).float()
        
        return lr, hr, filename
    
    def __len__(self):
        return self.lenth
    
    def _load_file_npy(self, idn):
        rgb = np.float32(self.nm_hr[idn])  # [1, 128, 128, 1]
        rgblr = np.float32(self.nm_lr[idn])  # [50, 128, 128, 1]

        rgb = np.squeeze(rgb, -1)
        rgblr = np.squeeze(rgblr)
        if not self.conv2d:
            rgblr = np.expand_dims(rgblr, 0)  # 1, 50, 128, 128
        
        return rgblr, rgb, ''
    
    def _load_file_test(self, idx):
        filename, i = os.path.splitext(os.path.basename(self.nm_lr[idx]))
        
        rgblr = np.float32(imread(self.nm_lr[idx]))  # [50, 520, 692] 0~310
        rgb = np.expand_dims(np.float32(imread(self.nm_hr[idx])), 0)  # [1, 520, 692] 0~147
        _, h, w = rgblr.shape
        h = h // 16 * 16
        w = w // 16 * 16
        rgb = rgb[:, :h, :w]
        rgblr = rgblr[:, :h, :w]
        
        if not self.conv2d:
            rgblr = np.expand_dims(rgblr, 0)  # 1, 50,256,256

        return rgblr, rgb, filename


import imageio

class FlouresceneVCD_Agents(data.Dataset):
    def __init__(self, args, train=True, repeat=1):
        self.args = args
        self.train = train
        self.dir_data = ['dataset/%s/%s' % ('train' if train else 'test', s) for s in args.data_test.split('+')]
        self.images_hr = []
        for path in self.dir_data:
            files = glob.glob(path+'/HR*.tif')
            self.images_hr.extend(files)
        self.repeat = repeat
        self.args.degrade = self.args.degrade.split('+') 

    def __getitem__(self, idx):
        idx = self._get_index(idx)

        hr_file = self.images_hr[idx]

        # print('hr_file', hr_file)
        hr = imread(hr_file).astype(np.uint8)
        # print('hr.shape', hr.shape, hr.max(), hr.min())

        lr = imread(hr_file.replace('/HR', '/'))
        # print('lr.shape', lr.shape, lr.max(), lr.min())
        for degrade in self.args.degrade:
            if degrade == 'None':
                continue
            elif 'b' in degrade:
                lr = add_gaussian_blur(lr, int(degrade[1:]))
            else:
                lr = add_gaussian_noise(lr, int(degrade[1:]))
        
        hr = hr.astype(np.float) / 255
        lr = lr.astype(np.float) / 255

        lr = torch.from_numpy(np.ascontiguousarray(lr * self.args.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.args.rgb_range)).float()

        if self.train:
            lr, hr = augment(lr, hr)
        
        return lr, hr, hr_file

    def __len__(self):
        # print('len(self.images_hr)', len(self.images_hr))
        return len(self.images_hr) * self.repeat
        
    def _get_index(self, idx):
        return idx % len(self.images_hr)

class FlouresceneVCD:
    def __init__(self, args, istrain=True, subtestset='to_predict'):
        self.path = f'{VCD_path}/vcdnet/vcd-example-data/data/'
        self.istrain = istrain
        self.args = args
        self.lf2d_base_size = args.patch_size // 11
        self.n_slices = 61
        self.n_num = 11
        self.shuffle = True
        if args.test_only:
            if subtestset == 'to_predict':
                self.nm_lr2d = sorted(glob.glob(self.path + '%s/*.tif' % subtestset))
                self.nm_hr3d = sorted(glob.glob(f'{VCD_path}/vcdnet/results/VCD_tubulin/*.tif'))
        else:
            self.nm_hr3d = sorted(glob.glob(self.path + 'train/WF/*.tif'))
            self.nm_lr2d = sorted(glob.glob(self.path + 'train/LF/*.tif'))
            if not istrain:  # valid
                self.nm_lr2d = self.nm_lr2d[:1]
                self.nm_hr3d = self.nm_hr3d[:1]
        assert len(self.nm_hr3d) == len(self.nm_lr2d)
        
        self.lenth = len(self.nm_lr2d)

        if istrain:
            print('++ ++ ++ ++ ++ ++ length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')

    def _load_dataset(self, idx):
        def rearrange3d_fn(image):
            """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
            """
        
            image = np.squeeze(image)  # remove channels dimension
            # print('reshape : ' + str(image.shape))
            depth, height, width = image.shape
            image_re = np.zeros([height, width, depth])
            for d in range(depth):
                image_re[:, :, d] = image[d, :, :]
            return image_re
    
        def lf_extract_fn(lf2d, n_num=11, mode='toChannel', padding=False):
            """
            Extract different views from a single LF projection

            Params:
                -lf2d: numpy.array, 2-D light field projection in shape of [height, width, channels=1]
                -mode - 'toDepth' -- extract views to depth dimension (output format [depth=multi-slices, h, w, c=1])
                        'toChannel' -- extract views to channel dimension (output format [h, w, c=multi-slices])
                -padding -   True : keep extracted views the same size as lf2d by padding zeros between valid pixels
                             False : shrink size of extracted views to (lf2d.shape / Nnum);
            Returns:
                ndarray [height, width, channels=n_num^2] if mode is 'toChannel'
                        or [depth=n_num^2, height, width, channels=1] if mode is 'toDepth'
            """
            n = n_num
            h, w, c = lf2d.shape
            if padding:
                if mode == 'toDepth':
                    lf_extra = np.zeros([n * n, h, w, c])  # [depth, h, w, c]
                
                    d = 0
                    for i in range(n):
                        for j in range(n):
                            lf_extra[d, i: h: n, j: w: n, :] = lf2d[i: h: n, j: w: n, :]
                            d += 1
                elif mode == 'toChannel':
                    lf2d = np.squeeze(lf2d)
                    lf_extra = np.zeros([h, w, n * n])
                    
                    d = 0
                    for i in range(n):
                        for j in range(n):
                            lf_extra[i: h: n, j: w: n, d] = lf2d[i: h: n, j: w: n]
                            d += 1
                else:
                    raise Exception('unknown mode : %s' % mode)
            else:
                new_h = int(np.ceil(h / n))
                new_w = int(np.ceil(w / n))
            
                if mode == 'toChannel':
                    lf2d = np.squeeze(lf2d)
                    lf_extra = np.zeros([new_h, new_w, n * n])
                
                    d = 0
                    for i in range(n):
                        for j in range(n):
                            lf_extra[:, :, d] = lf2d[i: h: n, j: w: n]
                            d += 1
                elif mode == 'toDepth':
                    lf_extra = np.zeros([n * n, new_h, new_w, c])  # [depth, h, w, c]
                    d = 0
                    for i in range(n):
                        for j in range(n):
                            lf_extra[d, :, :, :] = lf2d[i: h: n, j: w: n, :]
                            d += 1
                else:
                    raise Exception('unknown mode : %s' % mode)
        
            return lf_extra
    
        def normalize(x):
            max_ = np.max(x) * 1.1
            x = x / (max_ / 2.)
            x = x - 1
            return x
    
        def _load_imgs(img_file, t2d=True):
            if t2d:
                image = imageio.imread(img_file)
                if image.ndim == 2:
                    image = image[:, :, np.newaxis]  # uint8 0~48 (176,176,1) (649, 649,1)
                img = normalize(image)  # float64 -1~1 (176,176,1)
                img = lf_extract_fn(img, n_num=self.n_num, padding=False)  # (16, 16, 121) (59, 59, 121)
            else:
                image = imageio.volread(img_file)  # uint8 0~132  [61,176,176]
                img = normalize(image)  # float64 -1~1 (61,176,176)
                img = rearrange3d_fn(img)  # (176,176,61)
    
            img = img.astype(np.float32, casting='unsafe')
            # print('\r%s : %s' % (img_file, str(img.shape)), end='')
            return img
        
        training_data_lf2d = _load_imgs(self.nm_lr2d[idx], True)  # (16, 16, 121)
        X = np.transpose(training_data_lf2d, (2, 0, 1))
        if self.args.test_only:
            training_data_hr3d = _load_imgs(self.nm_hr3d[idx], False)  # (176, 176, 61)
            name = os.path.basename(self.nm_hr3d[idx])[:-4]
            Y = np.transpose(training_data_hr3d, (2, 0, 1))
        else:
            training_data_hr3d = _load_imgs(self.nm_hr3d[idx], False)
            Y = np.transpose(training_data_hr3d, (2, 0, 1))
            name = ''
        return Y, X, name

    def __getitem__(self, idx):
        idx = idx % self.lenth
        hr, lr, filename = self._load_dataset(idx)
        
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.args.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.args.rgb_range)).float()

        return lr, hr, filename

    def __len__(self):
        return self.lenth

    
# Inheritted from CARE
class PercentileNormalizer(object):
    def __init__(self, pmin=2, pmax=99.8, do_after=True, dtype=torch.float32, **kwargs):
        if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100):
            raise ValueError
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.dtype = dtype
        self.kwargs = kwargs
    
    def before(self, img, axes):
        if len(axes) != img.ndim:
            raise ValueError
        channel = None if axes.find('C') == -1 else axes.find('C')
        axes = None if channel is None else tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img.detach().cpu().numpy(), self.pmin, axis=axes, keepdims=True).astype(np.float32, copy=False)
        self.ma = np.percentile(img.detach().cpu().numpy(), self.pmax, axis=axes, keepdims=True).astype(np.float32, copy=False)
        return (img - self.mi) / (self.ma - self.mi + 1e-20)
    
    def after(self, img):
        if not self.do_after():
            raise ValueError
        alpha = self.ma - self.mi
        beta = self.mi
        return (alpha * img + beta).astype(np.float32, copy=False)
    
    def do_after(self):
        return self._do_after

if __name__ == '__main__':
    class Args:
        rgb_range = 1
        patch_size = 128
        test_only = True

    args = Args()
    dataset = FlouresceneVCD(args, istrain=False, subtestset='to_predict')
    print(len(dataset))

    name = 'VCD'

    # import time
    # start = time.time()
    # data = dataset[0]
    # print('time', time.time() - start)
    # print(data[0].shape, data[1].shape, data[2])

    for i in tqdm(range(len(dataset))):
        lr, hr, filename = dataset[i]
        savepath = os.path.join("dataset", 
                                'train' if dataset.istrain else 'test',
                                name, 
                                "%08d.tif"%i)
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        if not os.path.exists(savepath):
            lr = normalize(lr.numpy(), 0, 100, clip=True) * 255
            imsave(savepath, lr)

        savepath = os.path.join("dataset", 
                                'train' if dataset.istrain else 'test',
                                name, 
                                "HR%08d.tif"%i)
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        if not os.path.exists(savepath):
            hr = normalize(hr.numpy(), 0, 100, clip=True) * 255
            imsave(savepath, hr)