# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 22:07
# @Author  : Feng Xin
# @File    : MAHGAN.py
import time

import math
import numpy as np
import torch
from torch import nn
from core.modules.GeneratorA import GeneratorA
from core.modules.GeneratorM import GeneratorM
from core.modules.GeneratorH import GeneratorH
from core.blocks.Encoders import EncoderQuarter
from core.blocks.Decoders import Decoder4Times


class Generator(nn.Module):
    def __init__(self, input_channels, hidden_channels_list_a, hidden_channels_m, hidden_channels_h, kernel_size_list_a, kernel_size_m, depth_m, device, forget_bias_a=0.01):
        super(Generator, self).__init__()
        # self.encoder = EncoderQuarter(input_channels=1)
        self.generator_a = GeneratorA(input_channels, hidden_channels_list_a, kernel_size_list_a, forget_bias_a, device)
        self.generator_m = GeneratorM(input_channels, hidden_channels_m, depth_m, kernel_size_m, patch_size=10)
        self.generator_h = GeneratorH(64, hidden_channels=32, blocks_num=3)
        # self.generator_h = GeneratorH(input_channels, hidden_channels=32, blocks_num=3)
        # self.decoder = Decoder4Times(output_channels=1)

    def forward(self, inputs):
        # print(inputs.shape)
        inputs = torch.unsqueeze(inputs, dim=2)
        batch, sequence, channel, height, width = inputs.shape
        # inputs_list = []
        # for s in range(sequence):
        #     inputs_encoded = self.encoder(inputs[:, s, ...])
        #     inputs_list.append(inputs_encoded)
        # x = torch.stack(inputs_list, dim=1)
        x = inputs
        output_a = self.generator_a(x)
        # print(output_a.shape)
        # output_a_list = []
        # for s in range(sequence):
        #     output_a_element = self.decoder(output_a[:, s, ...])
        #     output_a_list.append(output_a_element)
        # output_a = torch.stack(output_a, dim=1)
        # accumulate_error = output_a.squeeze(2)
        # accumulate_error = torch.stack(output_a_list, dim=1)
        accumulate_error = output_a
        # print(accumulate_error.shape)
        # accumulate_error = self.decoder(output_a)
        correct_a = inputs[:, 1:, ...] - accumulate_error
        correct_a = torch.concatenate((inputs[:, 0, ...].unsqueeze(1), correct_a), dim=1)
        # correct_a_list = []
        # for s in range(sequence):
        #     x = self.encoder(correct_a[:, s, ...])
        #     correct_a_list.append(x)
        # output_m = self.generator_m(x)
        model_error_list = []
        for s in range(sequence):
            output_m = self.generator_m(correct_a[:, s, ...])
            model_error_list.append(output_m)
        model_error = torch.stack(model_error_list, dim=1)
        # model_error = self.decoder(output_m)
        correct = correct_a - model_error
        # x = self.encoder(correct)
        high_resolution_list = []
        for s in range(sequence):
            output_h = self.generator_h(correct[:, s, ...])
            high_resolution_list.append(output_h)
        # high_resolution = self.decoder(output_h)
        high_resolution = torch.stack(high_resolution_list, dim=1)
        return correct_a.squeeze(2), correct.squeeze(2), high_resolution.squeeze(2)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = []
        for i in range(len(input_shape)):
            in_channels, in_height, in_width = self.input_shape[i]
            patch_h, patch_w = math.ceil(in_height / 2 ** 4), math.ceil(in_width / 2 ** 4)
            self.output_shape.append((12, patch_h, patch_w))

        def discriminator_block(input_channels, hidden_channels):
            return nn.Sequential(
                nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        block_list = []
        input_channels = in_channels
        for i, hidden_channels in enumerate([64, 128, 256, 512]):
            block_list.extend(discriminator_block(input_channels, hidden_channels))
            input_channels = hidden_channels
        block_list.append(nn.Conv2d(hidden_channels, 12, kernel_size=3, stride=1, padding=1))
        # block_list.append(nn.Sigmoid())
        model = nn.Sequential(*block_list)
        self.model = nn.ModuleList([model for _ in range(3)])

    def forward(self, inputs):
        # output_list = []
        # for i in range(3):
        #     x = inputs[i]
        #     y_list = []
        #     batch, sequence, height, width = x.shape
        #     for s in range(sequence):
        #         y_element = self.model[i](x[:, s, ...].unsqueeze(1))
        #         y_list.append(y_element)
        #     y = torch.stack(y_list, dim=1)
        #     output_list.append(y)
        # return output_list
        # print(inputs[0].shape)
        # print(self.model[0](inputs[0]).shape)
        return self.model[0](inputs[0]), self.model[1](inputs[1]), self.model[2](inputs[2])

