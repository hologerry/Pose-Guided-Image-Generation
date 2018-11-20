
#%%
# http://pytorch.org/
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = get_ipython().getoutput("ldconfig -p|grep cudart.so|sed -e 's/.*\\.\\([0-9]*\\)\\.\\([0-9]*\\)$/cu\\1\\2/'")
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

get_ipython().system('pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision')

import torch
import torch.nn as nn
import math


#%%
from PIL import Image

def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions):
    for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions


#%%
from google.colab import drive
drive.mount('/content/gdrive')


#%%
import math
import os
import random
import sys

import numpy as np
import scipy.io
import scipy.stats
import skimage.morphology
from scipy import misc
from skimage.morphology import square, dilation, erosion


#%%
import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
from torch import cat


#%%
def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            distance = np.sqrt(float(i**2+j**2))
            if r+i>=0 and r+i<height and c+j>=0 and c+j<width:
                if 'Solid'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])

    return indices


def _getSparsePose(peaks, height, width, channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k]
        if 0!=len(p):
            r = p[0][1]
            c = p[0][0]
            ind = _getSparseKeypoint(r, c, k, height, width, radius, var, mode)
            indices.extend(ind)

    shape = [height, width, channel]
    return indices, shape


def _oneDimSparsePose(indices, shape):
    ind_onedim = []
    for ind in indices:
        idx = ind[0]*shape[2]*shape[1] + ind[1]*shape[2] + ind[2]
        ind_onedim.append(idx)
    shape = np.prod(shape)
    return ind_onedim, shape


def _sparse2dense(indices, shape):
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        k = indices[i][2]
        dense[r,c,k] = 1
    return dense


def _getPoseMask(peaks, height, width, radius=4, var=4, mode='Solid'):
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],                          [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],                          [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]] #
    indices = []
    for limb in limbSeq:
        p0 = peaks[limb[0] -1]
        p1 = peaks[limb[1] -1]
        if 0!=len(p0) and 0!=len(p1):
            r0 = p0[0][1]
            c0 = p0[0][0]
            r1 = p1[0][1]
            c1 = p1[0][0]
            ind  = _getSparseKeypoint(r0, c0, 0, height, width, radius, var, mode)
            indices.extend(ind)
            ind = _getSparseKeypoint(r1, c1, 0, height, width, radius, var, mode)
            indices.extend(ind)

            distance = np.sqrt((r0-r1)**2 + (c0-c1)**2)
            sampleN = int(distance/radius)
            if sampleN>1:
                for i in range(1,sampleN):
                    r = r0 + (r1-r0)*i/sampleN
                    c = c0 + (c1-c0)*i/sampleN
                    ind = _getSparseKeypoint(r, c, 0, height, width, radius, var, mode)
                    indices.extend(ind)

    shape = [height, width, 1]
    ## Fill body
    dense = np.squeeze(_sparse2dense(indices, shape))
    dense = dilation(dense, square(5))
    dense = erosion(dense, square(5))
    return dense


def _get_valid_peaks(all_peaks, subsets):
    try:
        subsets = subsets.tolist()
        valid_idx = -1
        valid_score = -1
        for i, subset in enumerate(subsets):
            score = subset[-2]
            if score > valid_score:
                valid_idx = i
                valid_score = score
        if valid_idx>=0:
            return all_peaks
        else:
            return None
    except:
        return None


def _format_data(folder_path, pairs, i, all_peaks_dic, subsets_dic):
    # Read the filename:
    img_path_0 = os.path.join(folder_path, pairs[i][0])
    img_path_1 = os.path.join(folder_path, pairs[i][1])
    image_raw_0 = misc.imread(img_path_0)
    image_raw_1 = misc.imread(img_path_1)
    height, width = image_raw_0.shape[1], image_raw_0.shape[0]

    ########################## Pose 16x8 & Pose coodinate (for 128x64(Solid) 128x64(Gaussian))##########################
    if (all_peaks_dic is not None) and (pairs[i][0] in all_peaks_dic) and (pairs[i][1] in all_peaks_dic):
        ## Pose 1
        peaks = _get_valid_peaks(all_peaks_dic[pairs[i][1]], subsets_dic[pairs[i][1]])

        indices_r4_1, shape = _getSparsePose(peaks, height, width, 18, radius=4, mode='Solid')
        indices_r4_1, shape_1 = _oneDimSparsePose(indices_r4_1, shape)

        pose_mask_r4_1 = _getPoseMask(peaks, height, width, radius=4, mode='Solid')
    else:
        return None

    image_raw_0 = np.reshape(image_raw_0, (height, width, 3))
    image_raw_0 = image_raw_0.astype('float32')
    image_raw_1 = np.reshape(image_raw_1, (height, width, 3))
    image_raw_1 = image_raw_1.astype('float32')

    mask_1 = np.reshape(pose_mask_r4_1, (height, width, 1))
    mask_1 = mask_1.astype('float32')

    indices_r4_1 = np.array(indices_r4_1).astype(np.int64).flatten().tolist()
    indices_r4_1_dense = np.zeros((shape_1))
    indices_r4_1_dense[indices_r4_1] = 1
    indices_r4_1 = np.reshape(indices_r4_1_dense, (height, width, 18))
    pose_1 = indices_r4_1.astype('float32')

    image_0 = (image_raw_0 - 127.5) / 127.5
    image_1 = (image_raw_1 - 127.5) / 127.5
    pose_1 = pose_1 * 2 - 1

    image_0 = torch.from_numpy(np.transpose(image_0, (2, 0, 1)))
    image_1 = torch.from_numpy(np.transpose(image_1, (2, 0, 1)))
    mask_1 = torch.from_numpy(np.transpose(mask_1, (2, 0, 1)))
    pose_1 = torch.from_numpy(np.transpose(pose_1, (2, 0, 1)))


    return [image_0, image_1, pose_1, mask_1]


#%%
import pickle


#%%
import pickle

def _get_train_all_p_pairs(out_dir, split_name='train'):
    assert split_name in {'train', 'train_flip', 'test', 'test_samples', 'test_seq', 'all'}
    if split_name=='train_flip':
        p_pairs_path = os.path.join(out_dir, 'p_pairs_train_flip.p')
    else:
        p_pairs_path = os.path.join(out_dir, 'p_pairs_'+split_name.split('_')[0]+'.p')

    if os.path.exists(p_pairs_path):
        with open(p_pairs_path,'rb') as f:
            p_pairs = pickle.load(f)

    print('_get_train_all_pn_pairs finish ......')
    print('p_pairs length:%d' % len(p_pairs))

    return p_pairs


#%%

p_pairs = _get_train_all_p_pairs('/content/gdrive/My Drive/Colab Notebooks/data/DF_train_data/')
p_pairs_flip = _get_train_all_p_pairs('/content/gdrive/My Drive/Colab Notebooks/data/DF_train_data/', 'train_flip')


#%%
import torch.utils.data

length = 97854 + 77538
class PoseDataset(torch.utils.data.Dataset):
    """Pose dataset."""
    def __init__(self, pose_peak_path, pose_sub_path, pose_peak_path_flip, pose_sub_path_flip):
        self.folder_path = '/content/gdrive/My Drive/Colab Notebooks/data/DF_img_pose/filted_up_train/'
        self.folder_path_flip = '/content/gdrive/My Drive/Colab Notebooks/data/DF_img_pose/filted_up_train_flip/'
        self.all_peaks_dic = None
        self.subsets_dic = None
        self.all_peaks_dic_flip = None
        self.subsets_dic_flip = None

        with open(pose_peak_path, 'rb') as f:
            self.all_peaks_dic = pickle.load(f, encoding='latin1')
        with open(pose_sub_path, 'rb') as f:
            self.subsets_dic = pickle.load(f, encoding='latin1')

        with open(pose_peak_path_flip, 'rb') as f:
            self.all_peaks_dic_flip = pickle.load(f, encoding='latin1')
        with open(pose_sub_path_flip, 'rb') as f:
            self.subsets_dic_flip = pickle.load(f, encoding='latin1')

    def __len__(self):
        return length

    def __getitem__(self, index):
        while True:
            USE_FLIP = index >= 97854
            if USE_FLIP:
                example = _format_data(self.folder_path_flip, p_pairs_flip, index - 97854, self.all_peaks_dic_flip, self.subsets_dic_flip)
                if example:

                    return example
                index = (index + 1) % length
            else:
                example = _format_data(self.folder_path, p_pairs, index, self.all_peaks_dic, self.subsets_dic)
                if example:
                    return example
                index = (index + 1) % length


#%%
pose_dataset = PoseDataset('/content/gdrive/My Drive/Colab Notebooks/data/DF_img_pose/PoseFiltered/all_peaks_dic_DeepFashion.p',
                           '/content/gdrive/My Drive/Colab Notebooks/data/DF_img_pose/PoseFiltered/subsets_dic_DeepFashion.p',
                           '/content/gdrive/My Drive/Colab Notebooks/data/DF_img_pose/PoseFiltered/all_peaks_dic_DeepFashion_Flip.p',
                           '/content/gdrive/My Drive/Colab Notebooks/data/DF_img_pose/PoseFiltered/subsets_dic_DeepFashion_Flip.p')
pose_loader = torch.utils.data.DataLoader(pose_dataset, batch_size=1, shuffle=True, num_workers=2)


#%%
img_H = 256
img_W = 256
channel = 3
batch_size = 1
max_step = 80000
d_lr = 0.00002
g_lr = 0.00002
lr_update_step = 50000
data_format = 'NHWC'

beta1 = 0.5
beta2 = 0.999
gamma = 0.5
lambda_k = 0.001
z_num = 64
conv_hidden_num = 128
repeat_num = int(np.log2(img_H)) - 2 # 6
log_step = 200
keypoint_num = 18


#%%
class GeneratorCNN_Pose_UAEAfterResidual_256(nn.Module):

    def block(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel, stride, padding),
            nn.ReLU()
        )

    def block_one(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU()
        )

    def conv(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Conv2d(ch_in, ch_out, kernel, stride, padding)

    def fc(self, ch_in, ch_out):
        return nn.Linear(ch_in, ch_out)

    def __init__(self, ch_in, z_num, repeat_num, hidden_num=128):
        super(GeneratorCNN_Pose_UAEAfterResidual_256, self).__init__()
        self.min_fea_map_H = 8
        self.z_num = z_num
        self.hidden_num = hidden_num
        self.repeat_num = repeat_num

        self.block1 = self.block(self.hidden_num, 128, 3, 1)
        self.block2 = self.block(256, 256, 3, 1)
        self.block3 = self.block(384, 384, 3, 1)
        self.block4 = self.block(512, 512, 3, 1)
        self.block5 = self.block(640, 640, 3, 1)
        self.block6 = self.block(768, 768, 3, 1)

        self.block_one1 = self.block_one(128, 256, 3, 2)
        self.block_one2 = self.block_one(256, 384, 3, 2)
        self.block_one3 = self.block_one(384, 512, 3, 2)
        self.block_one4 = self.block_one(512, 640, 3, 2)
        self.block_one5 = self.block_one(640, 768, 3, 2)

        self.fc1 = self.fc(self.min_fea_map_H * self.min_fea_map_H * 768, self.z_num)
        self.fc2 = self.fc(self.z_num, self.min_fea_map_H * self.min_fea_map_H * self.hidden_num)

        self.block7 = self.block(896, 896, 3, 1)
        self.block8 = self.block(1280, 1280, 3, 1)
        self.block9 = self.block(1024, 1024, 3, 1)
        self.block10 = self.block(768, 768, 3, 1)
        self.block11 = self.block(512, 512, 3, 1)
        self.block12 = self.block(256, 256, 3, 1)

        self.block_one6 = self.block_one(896, 640, 1, 1, padding=0)
        self.block_one7 = self.block_one(1280, 512, 1, 1, padding=0)
        self.block_one8 = self.block_one(1024, 384, 1, 1, padding=0)
        self.block_one9 = self.block_one(768, 256, 1, 1, padding=0)
        self.block_one10 = self.block_one(512, 128, 1, 1, padding=0)

        self.conv_last = self.conv(256, 3, 3, 1)

        self.block_1 = nn.Sequential(
            nn.Conv2d(ch_in, self.hidden_num, 3, 1, padding=1),
            nn.ReLU()
        )

        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        encoder_layer_list = []

        x = self.block_1(x) # x: [1, 256, 256, 21]

        # 1
        res = x
        x = self.block1(x)
        x = x + res
        encoder_layer_list.append(x)
        x = self.block_one1(x)
        # 2
        res = x
        x = self.block2(x)
        x = x + res
        encoder_layer_list.append(x)
        x = self.block_one2(x)
        # 3
        res = x
        x = self.block3(x)
        x = x + res
        encoder_layer_list.append(x)
        x = self.block_one3(x)
        # 4
        res = x
        x = self.block4(x)
        x = x + res
        encoder_layer_list.append(x)
        x = self.block_one4(x)
        # 5
        res = x
        x = self.block5(x)
        x = x + res
        encoder_layer_list.append(x)
        x = self.block_one5(x)
        # 6
        res = x
        x = self.block6(x)
        x = x + res
        encoder_layer_list.append(x)

        x = x.view(-1, self.min_fea_map_H * self.min_fea_map_H * 768)
        x = self.fc1(x)
        z = x

        x = self.fc2(z)
        x = x.view(-1, self.hidden_num, self.min_fea_map_H, self.min_fea_map_H) # x: [1, 8, 8, 128]

        # 1
        x = torch.cat([x, encoder_layer_list[5]], dim=1)
        res = x
        x = self.block7(x)
        x = x + res
        x = self.upscale(x)
        x = self.block_one6(x)
        # 2
        x = torch.cat([x, encoder_layer_list[4]], dim=1)
        res = x
        x = self.block8(x)
        x = x + res
        x = self.upscale(x)
        x = self.block_one7(x)
        # 3
        x = torch.cat([x, encoder_layer_list[3]], dim=1)
        res = x
        x = self.block9(x)
        x = x + res
        x = self.upscale(x)
        x = self.block_one8(x)
        # 4
        x = torch.cat([x, encoder_layer_list[2]], dim=1)
        res = x
        x = self.block10(x)
        x = x + res
        x = self.upscale(x)
        x = self.block_one9(x)
        # 5
        x = torch.cat([x, encoder_layer_list[1]], dim=1)
        res = x
        x = self.block11(x)
        x = x + res
        x = self.upscale(x)
        x = self.block_one10(x)
        # 6
        x = torch.cat([x, encoder_layer_list[0]], dim=1)
        res = x
        x = self.block12(x)
        x = x + res

        output = self.conv_last(x)
        return output


#%%
class UAE_noFC_AfterNoise(nn.Module):

    def block(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel, stride, padding),
            nn.ReLU()
        )

    def block_one(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU()
        )

    def conv(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Conv2d(ch_in, ch_out, kernel, stride, padding)

    def __init__(self, ch_in, repeat_num, hidden_num=128):
        super(UAE_noFC_AfterNoise, self).__init__()
        self.hidden_num = hidden_num
        self.repeat_num = repeat_num

        self.block_1 = nn.Sequential(
            nn.Conv2d(ch_in, self.hidden_num, 3, 1, padding=1),
            nn.ReLU()
        )

        self.block1 = self.block(self.hidden_num, 128, 3, 1)
        self.block2 = self.block(128, 256, 3, 1)
        self.block3 = self.block(256, 384, 3, 1)
        self.block4 = self.block(384, 512, 3, 1)

        self.block_one1 = self.block_one(128, 128, 3, 2)
        self.block_one2 = self.block_one(256, 256, 3, 2)
        self.block_one3 = self.block_one(384, 384, 3, 2)

        self.block5 = self.block(1024, 128, 3, 1)
        self.block6 = self.block(512, 128, 3, 1)
        self.block7 = self.block(384, 128, 3, 1)
        self.block8 = self.block(256, 128, 3, 1)

        self.conv_last = self.conv(128, 3, 3, 1)

        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        encoder_layer_list = []

        x = self.block_1(x) # x: [256, 256, 6]

        # 1
        x = self.block1(x)
        encoder_layer_list.append(x)
        x = self.block_one1(x)
        # 2
        x = self.block2(x)
        encoder_layer_list.append(x)
        x = self.block_one2(x)
        # 3
        x = self.block3(x)
        encoder_layer_list.append(x)
        x = self.block_one3(x)
        # 4
        x = self.block4(x)
        encoder_layer_list.append(x)

        # 1
        x = torch.cat([x, encoder_layer_list[-1]], dim=1)
        x = self.block5(x)
        x = self.upscale(x)
        # 2
        x = torch.cat([x, encoder_layer_list[-2]], dim=1)
        x = self.block6(x)
        x = self.upscale(x)
        # 3
        x = torch.cat([x, encoder_layer_list[-3]], dim=1)
        x = self.block7(x)
        x = self.upscale(x)
        # 4
        x = torch.cat([x, encoder_layer_list[-4]], dim=1)
        x = self.block8(x)

        output = self.conv_last(x)
        return output


#%%
class DCGANDiscriminator_256(nn.Module):
    def uniform(self, stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')

    def LeakyReLU(self, x, alpha=0.2):
        return torch.max(alpha*x, x)

    def conv2d(self, x, input_dim, filter_size, output_dim, gain=1, stride=1, padding=2):
        filter_values = self.uniform(
                self._weights_stdev,
                (output_dim, input_dim, filter_size, filter_size)
            )
        filter_values *= gain
        filters = torch.from_numpy(filter_values).cuda()
        biases = torch.from_numpy(np.zeros(output_dim, dtype='float32')).cuda()
        result = nn.functional.conv2d(x, filters, biases, stride, padding)
        return result

    def LayerNorm(self, ch):
        return nn.BatchNorm2d(ch)

    def __init__(self, bn=True, input_dim=3, dim=64, _weights_stdev=0.02):
        super(DCGANDiscriminator_256, self).__init__()
        self.bn = bn
        self.input_dim = input_dim
        self.dim = dim
        self._weights_stdev = _weights_stdev

        self.bn1 = self.LayerNorm(2*self.dim)
        self.bn2 = self.LayerNorm(4*self.dim)
        self.bn3 = self.LayerNorm(8*self.dim)
        self.bn4 = self.LayerNorm(8*self.dim)

        self.fc1 = nn.Linear(8*8*8*self.dim, 1)

    def forward(self, x):
        output = x

        output = self.conv2d(output, self.input_dim, 5, self.dim, stride=2)
        output = self.LeakyReLU(output)

        output = self.conv2d(output, self.dim, 5, 2*self.dim, stride=2)
        if self.bn:
            output = self.bn1(output)
        output = self.LeakyReLU(output)

        output = self.conv2d(output, 2*self.dim, 5, 4*self.dim, stride=2)
        if self.bn:
            output = self.bn2(output)
        output = self.LeakyReLU(output)

        output = self.conv2d(output, 4*self.dim, 5, 8*self.dim, stride=2)
        if self.bn:
            output = self.bn3(output)
        output = self.LeakyReLU(output)

        output = self.conv2d(output, 8*self.dim, 5, 8*self.dim, stride=2)
        if self.bn:
            output = self.bn4(output)
        output = self.LeakyReLU(output)

        output = output.view(-1, 8*8*8*self.dim)
        output = self.fc1(output)
        return output


#%%
generator_one = GeneratorCNN_Pose_UAEAfterResidual_256(21, z_num, repeat_num)
generator_two = UAE_noFC_AfterNoise(6, repeat_num-2)
discriminator = DCGANDiscriminator_256()

generator_one.cuda()
generator_two.cuda()
discriminator.cuda()


#%%
L1_criterion = nn.L1Loss()
BCE_criterion = nn.BCELoss()

gen_train_op1 = optim.Adam(generator_one.parameters(), lr=2e-5, betas=(0.5, 0.999))
gen_train_op2 = optim.Adam(generator_two.parameters(), lr=2e-5, betas=(0.5, 0.999))
dis_train_op1 = optim.Adam(discriminator.parameters(), lr=2e-5, betas=(0.5, 0.999))


#%%
def train():
    for epoch in range(10):
        for step, example in enumerate(pose_loader):
            [x, x_target, pose_target, mask_target] = example
            x = Variable(x.cuda())
            x_target = Variable(x_target.cuda())
            pose_target = Variable(pose_target.cuda())
            mask_target = Variable(mask_target.cuda())

            G1 = generator_one(torch.cat([x, pose_target], dim=1))
            if step < 22000:
                PoseMaskLoss1 = L1_criterion(G1 * mask_target, x_target * mask_target)
                g_loss_1 = L1_criterion(G1, x_target) + PoseMaskLoss1
                gen_train_op1.zero_grad()
                g_loss_1.backward()
                gen_train_op1.step()
                print('Epoch: %d, Step: %d, g_loss1: %0.05f' %(epoch+1, step+1, g_loss_1))
                if step % 1000 == 999:
                    torch.save(generator_one.state_dict(), '/content/gdrive/My Drive/Colab Notebooks/data/train_generator_one')
                continue

            DiffMap = generator_two(torch.cat([G1, x], dim=1))
            G2 = G1 + DiffMap
            triplet = torch.cat([x_target, G2, x], dim=0)
            D_z = discriminator(triplet)
            D_z = torch.clamp(D_z, 0.0, 1.0)
            D_z_pos_x_target, D_z_neg_g2, D_z_neg_x = D_z[0], D_z[1], D_z[2]
            D_z_pos = D_z_pos_x_target
            D_z_neg = torch.cat([D_z_neg_g2, D_z_neg_x], 0)

            PoseMaskLoss1 = L1_criterion(G1 * mask_target, x_target * mask_target)
            g_loss_1 = L1_criterion(G1, x_target) + PoseMaskLoss1

            g_loss_2 = BCE_criterion(D_z_neg, torch.ones((2)).cuda())
            PoseMaskLoss2 = L1_criterion(G2 * mask_target, x_target * mask_target)
            L1Loss2 = L1_criterion(G2, x_target) + PoseMaskLoss2
            g_loss_2 += 50*L1Loss2

            gen_train_op2.zero_grad()
            g_loss_2.backward(retain_graph=True)
            gen_train_op2.step()

            d_loss = BCE_criterion(D_z_pos, torch.ones((1)).cuda())
            d_loss += BCE_criterion(D_z_neg, torch.zeros((2)).cuda())
            d_loss /= 2

            dis_train_op1.zero_grad()
            d_loss.backward()
            dis_train_op1.step()

            print('Epoch: %d, Step: %d, g_loss1: %0.05f, g_loss2: %0.05f, d_loss: %0.05f' %(epoch+1, step+1, g_loss_1, g_loss_2, d_loss))
            if step % 100 == 99:
                torch.save(generator_one.state_dict(), '/content/gdrive/My Drive/Colab Notebooks/data/train_generator_one')
                torch.save(generator_two.state_dict(), '/content/gdrive/My Drive/Colab Notebooks/data/train_generator_two')
                torch.save(discriminator.state_dict(), '/content/gdrive/My Drive/Colab Notebooks/data/train_discriminator')


#%%
train()


