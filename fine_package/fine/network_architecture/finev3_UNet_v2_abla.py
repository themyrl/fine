#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from fine.network_architecture.neural_network_ext import SegmentationNetwork
import torch.nn.functional

import fine.network_architecture.finev3 as fine


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Finev3_UNet_v2(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False, patch_size=(64,128,128), vt_map=(3,5,5), max_imsize=[218,660,660], clip=False,
                 vt_num=1, depths=[2, 2, 2, 2, 2, 2], dofine=[0, 0, 1, 1, 1, 1]):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Finev3_UNet_v2, self).__init__()
        print("-----> depths", depths)
        print("-----> dofine", dofine)
        print("-----> vt_num", vt_num)
        # print("\n\n\n\n\nThat's my model !! \n\n\n\n")
        # exit(0)
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        self.input_sizes = [patch_size]
        self.features_sizes = [output_features]

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
                self.input_sizes.append((self.input_sizes[d-1][0]//first_stride[0],
                                    self.input_sizes[d-1][1]//first_stride[1],
                                    self.input_sizes[d-1][2]//first_stride[2]))
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)
            self.features_sizes.append(output_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
            self.input_sizes.append((self.input_sizes[-1][0]//first_stride[0],
                                    self.input_sizes[-1][1]//first_stride[1],
                                    self.input_sizes[-1][2]//first_stride[2]))
        else:
            first_stride = None
            self.input_sizes.append(self.input_sizes[-1])

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip


            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))


        # Fine module
        self.vt_map = vt_map
        self.max_imsize = max_imsize
        self.imsize = patch_size
        #num_heads=[6, 12, 24, 80]
        num_heads=[1, 2, 4, 8, 10, 10]
        # num_heads=[6, 12, 24, 48]
        # depths=[2, 2, 2, 2, 2, 2]
        # depths=[d, 2, 2, 2, 2, 2]
        # depths=[3, 3, 3, 3, 3, 3]
        dpr = [x.item() for x in torch.linspace(0, 0.2, sum(depths))]  # stochastic depth decay rule


        # self.do_fine = [False, False, True, True, True, True]
        
        self.do_fine = [i==1 for i in dofine]
        self.fine_module_list = []
        for ii in range(len(self.input_sizes)):
            if self.do_fine[ii]:
                self.fine_module_list.append( fine.BasicLayer(
                        dim=self.features_sizes[ii],
                        input_resolution=self.input_sizes[ii],
                        depth=depths[ii],
                        num_heads=num_heads[ii],
                        window_size=4,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        qk_scale=None,
                        drop=0.,
                        attn_drop=0.,
                        drop_path=dpr[sum(depths[:ii]):sum(depths[:ii + 1])],
                        # drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                        norm_layer=nn.LayerNorm,
                        downsample=None,
                        use_checkpoint=False, gt_num=8, id_layer=ii, vt_map=vt_map,vt_num=vt_num, clip=clip))
            else:
                self.fine_module_list.append(None)
        self.fine_module_list = nn.ModuleList(self.fine_module_list)
        ###


        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        self.vt_check = torch.nn.Parameter(torch.zeros(vt_map[0]*vt_map[1]*vt_map[2],1))
        self.vt_check.requires_grad = False
        self.pos_grid, self.show_grid, self.vt_map = self.filled_grid()


        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x, pos=None):
        B = x.shape[0]
        vt_pos = []
        
        tmp_check = torch.zeros(B, self.vt_map[0]*self.vt_map[1]*self.vt_map[2],1, device=self.vt_check.device)

        for b, p in enumerate(pos):
            tmp = self.get_tokens_idx(p)
            # print("b, tmp pos", b, tmp)
            tmp_check[b,...] = self.vt_check[...]
            tmp_check[b, tmp, :] += 1
            vt_pos.append(tmp + b*np.array(self.vt_map).prod())
        vt_pos = np.array(vt_pos).flatten()
        
        # print("------> pos", pos)
        # print("------> vt_pos", vt_pos)

        ## encoder

        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)


            ## FINE here
            if self.do_fine[d]:
                Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
                x = x.flatten(2).transpose(1, 2)
                x_out, S, H, W, x, Ws, Wh, Ww = self.fine_module_list[d](x, Ws, Wh, Ww, vt_pos, self.vt_check >= 1)
                x = x_out.view(-1, S, H, W, self.features_sizes[d]).permute(0, 4, 1, 2, 3).contiguous()



            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)



        if self.do_fine[-1]:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x_out, S, H, W, x, Ws, Wh, Ww = self.fine_module_list[-1](x, Ws, Wh, Ww, vt_pos, self.vt_check >= 1)
            x = x_out.view(-1, S, H, W, self.features_sizes[-1]).permute(0, 4, 1, 2, 3).contiguous()

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        self.vt_check += tmp_check.sum(dim=0)


        if self._deep_supervision and self.do_ds:
            ret = tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
            
            return ret
        else:
            return seg_outputs[-1]


    def filled_grid(self):
        cd, ch, cw = self.imsize
        d, h, w = self.max_imsize
        grid = np.zeros(self.max_imsize, dtype = int)
        nd, nh, nw = d//cd, h//ch, w//cw
        
        vtm = [nd, nh, nw]
        
        pd, ph, pw = d%cd, h%ch, w%cw
        
        show = ""

        for i in range(nd):
            for j in range(nh):
                for k in range(nw):
                    show += str(nh*nw*i + nw*j + k)+" "
                    tmp_pd_0 = (pd//2)*(0**(i==0))
                    tmp_pd_1 = pd//2 + (pd//2)*(0**(i!=nd-1)) + pd%2

                    tmp_ph_0 = (ph//2)*(0**(j==0))
                    tmp_ph_1 = ph//2 + (ph//2)*(0**(j!=nh-1)) + ph%2
                    
                    tmp_pw_0 = (pw//2)*(0**(k==0))
                    tmp_pw_1 = pw//2 + (pw//2)*(0**(k!=nw-1)) + pw%2

                    grid[i*cd+tmp_pd_0:(i+1)*cd+tmp_pd_1, j*ch+tmp_ph_0:(j+1)*ch+tmp_ph_1, k*cw+tmp_pw_0:(k+1)*cw+tmp_pw_1] = nh*nw*i + nw*j + k
                    
                show += "\n"
            show += "\n"
               
        return grid, show, vtm

    def border_check(self, pos):
        ret = [i for i in pos]
        size = self.max_imsize
        crop_size = self.imsize
        for i in range(len(ret)):
            pad_i = (size[i]%crop_size[i])//2
            
            check_if_pos_on_frontiere = (pos[i]-pad_i)%crop_size[i] == 0
            check_if_pos_over_border_marge = (pos[i] + crop_size[i] >= size[i] -  pad_i)
            check_if_pos_under_border_marge = (pos[i] <= (size[i]%crop_size[i])//2)
            
            if check_if_pos_on_frontiere or check_if_pos_over_border_marge or check_if_pos_under_border_marge:
                if pos[i] < size[i]//2:
                    ret[i] += 1 + pad_i
                else:
                    tmp = 0
                    if pos[i]+crop_size[i] > size[i]:
                        tmp = pos[i]+crop_size[i] - size[i]
                    ret[i] -= 1 + pad_i + tmp + (size[i]%crop_size[i])%2
        return ret

    def get_tokens_idx(self, pos):
        # Myr : We put the crop in the bigger image referential
        z, x, y = [int(pos[i] + self.max_imsize[i]//2) for i in range(3)]
        pos = (z, x, y)
        pos = self.border_check(pos)
        z, x, y = pos
        cd, ch, cw = self.imsize
        tmp = self.pos_grid[z:z+cd, x:x+ch, y:y+cw]
        idx = np.unique(tmp)
        return idx


















    
