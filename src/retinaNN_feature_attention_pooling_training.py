###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################
import numpy as np
import configparser as ConfigParser
import os, time
'''
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model as plot
'''
from utils import calc_gradient_penalty
from torch.autograd import Variable
from torchviz import make_dot
import matplotlib.pyplot as plt
import itertools
import tqdm
import pickle
import imageio
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys

sys.path.insert(0, './lib/')
from help_functions import *
from feature_attention_residual_block import Feature_attentionBlock2D
from pyramid_pooling_block import SPPblock
# function to obtain data for training/testing (validation)
from extract_patches import get_data_training


# Define the neural network
def get_unet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv5)
    conv6 = core.Reshape((2, patch_height * patch_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Define the neural network gnet
# you need change function call "get_unet" to "get_gnet" in line 166 before use this network
def get_gnet(n_ch, patch_height, patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    up1 = UpSampling2D(size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)
    #
    up3 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)
    #
    up4 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up4)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv8)
    #
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    #
    conv10 = Convolution2D(2, 1, 1, activation='relu', border_mode='same')(conv9)
    conv10 = core.Reshape((2, patch_height * patch_width))(conv10)
    conv10 = core.Permute((2, 1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.non_local1 = Feature_attentionBlock2D(in_channels=1, inter_channels=32)
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.non_local2=Feature_attentionBlock2D(in_channels=32,inter_channels=64)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.non_local3 = Feature_attentionBlock2D(in_channels=64,inter_channels=128)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.pyramid_pooling = SPPblock(128)
        self.conv11 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv11_bn = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv12_bn = nn.BatchNorm2d(128)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv1_bn = nn.BatchNorm2d(64)
        self.non_local4 = Feature_attentionBlock2D(in_channels=128,inter_channels=64)
        self.conv7 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv8_bn = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv2_bn = nn.BatchNorm2d(32)
        self.non_local5 = Feature_attentionBlock2D(in_channels=64,inter_channels=32)
        self.conv9 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv9_bn = nn.BatchNorm2d(32)
        self.conv10 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv10_bn = nn.BatchNorm2d(32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        conv1 = F.relu(self.conv1_bn(self.conv1(input)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv2= conv2 + self.non_local1(input,conv2)
        down1 = self.pool1(conv2)

        conv3 = F.relu(self.conv3_bn(self.conv3(down1)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv4 = conv4 + self.non_local2(down1, conv4)
        down2 = self.pool2(conv4)

        conv5 = F.relu(self.conv5_bn(self.conv5(down2)))
        conv6 = F.relu(self.conv6_bn(self.conv6(conv5)))
        conv6 = self.pyramid_pooling(conv6)
        conv11 = F.relu(self.conv11_bn(self.conv11(conv6)))
        conv12 = F.relu(self.conv12_bn(self.conv12(conv11)))
        conv12 = conv12 + self.non_local3(down2, conv12)
        up1 = F.relu(self.deconv1_bn(self.deconv1(conv12)))

        up1 = torch.cat((conv4, up1), dim=1)
        conv7 = F.relu(self.conv7_bn(self.conv7(up1)))
        conv8 = F.relu(self.conv8_bn(self.conv8(conv7)))
        conv8 = conv8 + self.non_local4(up1, conv8)
        up2 = F.relu(self.deconv2_bn(self.deconv2(conv8)))

        up2 = torch.cat((conv2, up2), dim=1)
        conv9 = F.relu(self.conv9_bn(self.conv9(up2)))
        conv10 = F.relu(self.conv10_bn(self.conv10(conv9)))
        conv10 = conv10 + self.non_local5(up2, conv10)
        out = F.sigmoid(self.out(conv10))

        return out


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=32):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(2, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 3, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# ========= Load settings from Config file
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')
# patch to the datasets
path_data = config.get('data paths', 'path_local')
# Experiment name
name_experiment = config.get('experiment name', 'name')
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

# ============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original=path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth=path_data + config.get('data paths', 'train_groundTruth'),  # masks
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV')
    # select the patches only inside the FOV  (default == True)
)

# Save a sample of what you're feeding to the neural network
N_sample = min(patches_imgs_train.shape[0], 40)
visualize(group_images(patches_imgs_train[0:N_sample, :, :, :], 5),
          './' + name_experiment + '/' + "sample_input_imgs")  # .show()
visualize(group_images(patches_masks_train[0:N_sample, :, :, :], 5),
          './' + name_experiment + '/' + "sample_input_masks")  # .show()

split = 0.9
N = int(patches_imgs_train.shape[0] * split)
print("%d for training" % (N))
train_dataset = torch.utils.data.TensorDataset(torch.Tensor(patches_imgs_train[0:N, :, :, :]),
                                               torch.Tensor(patches_masks_train[0:N, :, :, :]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(torch.Tensor(patches_imgs_train[N:, :, :, :]),
                                             torch.Tensor(patches_masks_train[N:, :, :, :]))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# =========== Construct and save the model arcitecture =====
# construct the model
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]

G = generator()
D = discriminator(32)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
#G = nn.DataParallel(G)
#D = nn.DataParallel(D)
G.cuda()
D.cuda()
# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()
CROSS_loss = nn.CrossEntropyLoss()
# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
# model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
print("Check: final output of the network:")
t = torch.Tensor(1, 1, patch_height, patch_width).cuda()
#t = torch.Tensor(1, 1, patch_height, patch_width)
print((G(t)).shape)
#viz1_graph = make_dot(G(t), params=dict(G.named_parameters()))
#viz1_graph.view()
t2 = torch.Tensor(1, 2, patch_height, patch_width).cuda()
#t2 = torch.Tensor(1, 2, patch_height, patch_width)
print((D(t2)).shape)
#viz2_graph = make_dot(D(t2), params=dict(D.named_parameters()))
#viz2_graph.view()

# check how the model looks like
# plt.save('./'+name_experiment+'/'+name_experiment + '_Dmodel.png')

# json_stringG = G.to_json()
# json_stringD = D.to_json()
# open('./'+name_experiment+'/'+name_experiment +'_Garchitecture.json', 'w').write(json_stringG)
# open('./'+name_experiment+'/'+name_experiment +'_Darchitecture.json', 'w').write(json_stringD)


# ============  Training ==================================
# checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)


# model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


# ========== Save and test the last model ===================
# model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
# test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

gan_loss_percent = 0.03
one = torch.FloatTensor([1])
mone = one * -1
moneg = one * -1 * gan_loss_percent

one = one.cuda()
mone = mone.cuda()
moneg = moneg.cuda()



def show_train_hist(hist, show=False, save=False, path='./test/Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


for epoch in range(N_epochs):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for idx, (imgs, g_truth) in tqdm.tqdm(enumerate(train_loader)):

        mini_batch = imgs.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        imgs, g_truth, y_real_, y_fake_ = Variable(imgs.cuda()), Variable(g_truth.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        #imgs, g_truth, y_real_, y_fake_ = Variable(imgs), Variable(g_truth), Variable(
        #    y_real_), Variable(y_fake_)

        # train the descriminator
        if (idx + 1) % 1 == 0:
            D.zero_grad()
            D_optimizer.zero_grad()

            real_pair = torch.cat((imgs, g_truth), dim=1)

            d_real = D(real_pair)
            d_real = d_real.mean()
            d_real.backward(mone)

            fake_pair = torch.cat((imgs, G(imgs).detach()), dim=1)

            d_fake = D(fake_pair)
            d_fake = d_fake.mean()
            d_fake.backward(one)

            gradient_penalty = calc_gradient_penalty(D, real_pair.data, fake_pair.data)
            gradient_penalty.backward()

            D_optimizer.step()

            Wasserstein_D = d_real- d_fake
            D_losses.append(Wasserstein_D.item())
    # train the generator
    for idx, (imgs, g_truth) in tqdm.tqdm(enumerate(train_loader)):
        mini_batch = imgs.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        imgs, g_truth, y_real_, y_fake_ = Variable(imgs.cuda()), Variable(g_truth.cuda()), Variable( y_real_.cuda()), Variable(y_fake_.cuda())
        #imgs, g_truth, y_real_, y_fake_ = Variable(imgs), Variable(g_truth), Variable(
        #    y_real_), Variable(y_fake_)
        if (idx + 1) % 1 == 0:
            G.zero_grad()
            G_optimizer.zero_grad()
            G_result = G(imgs)

            Seg_Loss = BCE_loss(G_result, g_truth)
            Seg_Loss.backward(retain_graph=True)
            fake_pair = torch.cat((imgs, G_result), dim=1)
            gd_fake = D(fake_pair)
            gd_fake = gd_fake.mean()
            gd_fake.backward(moneg)

            G_optimizer.step()
            G_loss=Seg_Loss
            G_losses.append(G_loss.item())
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), N_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses))))
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    if (epoch + 1) % 1 == 0:
        total_accuracy = 0
        batch_number = 0
        for i_val, (real_imgs, real_labels) in enumerate(val_loader):
            eve_batch = real_labels.size()[0]
            std = (0.5 * torch.ones(eve_batch, 1, patch_height, patch_width)).cuda()
            #std = (0.5 * torch.ones(eve_batch, 1, patch_height, patch_width))
            batch_number += 1
            real_imgs = real_imgs.cuda()
            real_labels = real_labels.cuda()
            outputs = G(real_imgs)

            # valloss = loss_func(outputs, real_labels)
            outputs = (outputs >= std).float()
            correct = torch.sum((real_labels.eq(outputs)).float()).item()
            total = float(torch.numel(outputs))
            accuracy = 100 * (correct / total)
            total_accuracy += accuracy
        total_accuracy = total_accuracy / float(batch_number)
        print("epoch%d accuracy:%f" % ((epoch + 1), (total_accuracy)))
    if (epoch + 1) % 20 == 0:
        show_train_hist(train_hist, save=True, path='./'+name_experiment+'/GAN_%d_FP_train_hist.png' % ((epoch + 1)))
    if (epoch + 1) % 10 ==0:
        torch.save(G.state_dict(), "./"+name_experiment+"/generator_param_FP.pkl")
        torch.save(D.state_dict(), "./"+name_experiment+"/discriminator_param_FP.pkl")
        print ("\n2. Run the prediction on GPU (no nohup)")
        os.system('CUDA_VISIBLE_DEVICES=7 python ./src/retinaNN_predicttorch_FP.py')



torch.save(G.state_dict(), "./"+name_experiment+"/generator_param_FP.pkl")
torch.save(D.state_dict(), "./"+name_experiment+"/discriminator_param_FP.pkl")
#torch.save(G, "./testloss2/generator.pkl")
#torch.save(D, "./testloss2/discriminator.pkl")
show_train_hist(train_hist, save=True, path='./'+name_experiment+'/GAN_FP_train_hist.png')



