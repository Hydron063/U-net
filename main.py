import os
import numpy as np


import pydicom as dicom
import dicom_numpy as dn
import SimpleITK as sitk
from mayavi import mlab

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from torchvision import datasets, transforms

start_neurons = 16
num_epochs = 10
batch_size = 1
lr = 0.00005
do = 0.0


class Down(nn.Module):
    def __init__(self, k1, k2):
        super(Down, self).__init__()

        self.down_block = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Dropout3d(do),
            nn.Conv3d(k1, k2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv3d(k2, k2, 3, padding=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.down_block(x)


class Up(nn.Module):
    def __init__(self, k1, k2):
        super(Up, self).__init__()

        self.deconv = nn.ConvTranspose3d(k1, k2, 2, stride=2)
        self.double_conv = nn.Sequential(
            nn.Dropout3d(do),
            nn.Conv3d(k1, k2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv3d(k2, k2, 3, padding=1),
            nn.ReLU(True),
        )

    @staticmethod
    def crop_centre(layer, target_size, diff):
        diff //= 2
        return layer[:, :, diff[0]: diff[0] + target_size[0], diff[1]: diff[1] + target_size[1],
               diff[2]: diff[2] + target_size[2]]

    @staticmethod
    def add_padding(layer, diff):
        return F.pad(layer, [diff[-1] // 2, diff[-1] - diff[-1] // 2, diff[-2] // 2, diff[-2] - diff[-2] // 2,
                             diff[-3] // 2, diff[-3] - diff[-3] // 2])

    # [N, C, Z, Y, X]; N - number of batches, C - number of channels
    # Two options for concatenation - crop x2 or add padding to x1
    def forward(self, x1, x2, concat='crop'):
        x1 = self.deconv(x1)
        x1_size = np.array(x1.size()[-3:])
        x2_size = np.array(x2.size()[-3:])
        diff = (x2_size - x1_size)

        if concat == 'crop':
            x2 = self.crop_centre(x2, x1_size, diff)
        else:
            x1 = self.add_padding(x1, diff)

        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x


# The depth is 4. The first block contains 2 convolutions. Its result is used to concatenate then (grey arrow).
# The following 4 blocks are used to descend, and the results of the first three among them are used to concatenate
# (gray arrows) during the ascent. The fourth of them is used for ascending (green arrow, etc.)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.start = nn.Sequential(
            nn.Conv3d(1, start_neurons * 1, 3, padding=1),
            nn.Conv3d(start_neurons * 1, start_neurons * 1, 3, padding=1),
        )
        self.down1 = Down(start_neurons * 1, start_neurons * 2)
        self.down2 = Down(start_neurons * 2, start_neurons * 4)
        self.down3 = Down(start_neurons * 4, start_neurons * 8)
        self.down4 = Down(start_neurons * 8, start_neurons * 16)

        self.up4 = Up(start_neurons * 16, start_neurons * 8)
        self.up3 = Up(start_neurons * 8, start_neurons * 4)
        self.up2 = Up(start_neurons * 4, start_neurons * 2)
        self.up1 = Up(start_neurons * 2, start_neurons * 1)
        self.final = nn.Sequential(
            nn.Dropout3d(do),
            nn.Conv3d(start_neurons * 1, 1, 1),
            nn.Sigmoid()
        )

    #
    def forward(self, x):
        x1 = self.start(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        return self.final(x)


def extract_voxel_data(DCM_files):
    datasets = [dicom.read_file(f) for f in DCM_files]
    try:
        voxel_ndarray, ijk_to_xyz = dn.combine_slices(datasets)
    except dn.DicomImportException as e:
        raise e
    return voxel_ndarray


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


# load image (DICOM -> numpy)
PathDicom = "./DICOM/1/"
DCM_files = []
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():
            DCM_files.append(os.path.join(dirName, filename))

PathLabels = "./Label/"
label_files = []
for dirName, subdirList, fileList in os.walk(PathLabels):
    for filename in fileList:
        if ".mhd" in filename.lower():
            label_files.append(os.path.join(dirName, filename))

train_x = [extract_voxel_data(DCM_files)]

# each element of train_y represents 1 cube of 256 * 256 * 136 + metadata
train_y = [load_itk(label)[0] for label in label_files]

# Complement y with zeros
# train_y = np.array(
#     [np.concatenate((i, np.zeros((i.shape[0], i.shape[1], train_x.shape[2] - i.shape[2]))), axis=2) for i in train_y])
train_y = np.array(train_y)
train_x = np.array(train_x)

# # Demonstration of the pixels of each label
# point_dict = {}
# for z_i, z in enumerate(train_y[0]):
#     for y_i, y in enumerate(z):
#         for x_i, x in enumerate(y):
#             if x != 0:
#                 if x not in point_dict:
#                     point_dict[x] = [[x_i, y_i, z_i]]
#                 else:
#                     point_dict[x] += [[x_i, y_i, z_i]]
#
# for value, point in point_dict.items():
#     print(value, point)



# train_x = train_x[:, :, :, 58:186]
# train_y = train_y[:, :, :, 4:132]
train_x = train_x[:, 128:192, 128:192, 70:134]
# train_x = np.concatenate((train_x, train_x), axis=0)
train_y = train_y[:, 128:192, 128:192, 70:134]
# train_y = np.concatenate((train_y, train_y), axis=0)

train_y = np.array(list(map(lambda x: 0 if x == 0 else 1, train_y.flatten())))

# Creates a fifth fictional dimension for the number of channels (constraint of keras and pytorch)
# For pytorch the requested size is [N, C, Z, Y, X], for keras - [N, X, Y, Z, C]

# pytorch
# The size of 'y' is expected to be [N, Z, Y, X] (for nn.CrossEntropyLoss), so there is no need
# to add a fictional dimension, but you have to resize 'y' anyway to change the order of X, Y and Z.
# And in F.binary_cross_entropy not to add the dimension is called 'deprecated' (can you finally decide, damn?!)
enter_shape_x = (train_x.shape[0], 1) + train_x.shape[-3:][::-1]
enter_shape_y = (train_x.shape[0], 1) + train_x.shape[-3:][::-1]
train_x = train_x.reshape(*enter_shape_x)
train_y = train_y.reshape(*enter_shape_y)

# Visualisation
# train_y = np.array(train_y[0])
# print(train_x.shape)
# mlab.contour3d(train_y)
# mlab.savefig('surface.obj')
# input()
#
tensor_x = torch.stack([torch.tensor(i, dtype=torch.float32) for i in train_x])
# God knows why this damn thing asks LongTensor for y in criterion, but that's the way it must be done
# It seems to be a limitation of the entropy formula. A bit more on the question here (use VPN):
# https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542/4
tensor_y = torch.stack([torch.tensor(i, dtype=torch.float32) for i in train_y])

train_dataset = data.TensorDataset(tensor_x, tensor_y)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True)

test_dataset = data.TensorDataset(tensor_x, tensor_y)
test_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=lr)
# We could use nn.CrossEntropyLoss() if all our y were in {0; 1}
criterion = F.binary_cross_entropy
print(net)
iteration_num = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Passage through the network
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculation of precision
        # It's good to use when it comes to classification and we have a few layers at the output each of which
        # represents the probability of a class for the corresponding pixel
        # total = labels.size(0)
        # _, predicted = torch.max(outputs.data, 1)
        # correct = (predicted == labels).sum().item()
        # acc_list.append(correct / total)

        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, iteration_num, loss.item()))
