import torch

class ConvBlock2D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock2D, self).__init__()
        self.conv_block_2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block_2d(x)


class DeConvBlock3D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(DeConvBlock3D, self).__init__()
        self.deconv_block_3d = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.ELU()
        )

    def forward(self, x):
        return self.deconv_block_3d(x)


class ConvBlock3D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = torch.nn.Sequential(
            torch.nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderBlock, self).__init__()
        self.conv_eb = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        out = self.conv_eb(x)
        return out


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor):
        super(DecoderBlock, self).__init__()
        self.conv_db = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=scale_factor),
            torch.nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,
                                     padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1,
                                     padding=1),
            torch.nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = self.conv_db(x)
        return out


class TSM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input, n_frame=4, fold_div=3):
        n_frame = 4
        B, C, H, W = input.shape
        input = input.view(-1, n_frame, H, W, C)
        fold = C // fold_div
        last_fold = C - (fold_div - 1) * fold
        out1, out2, out3 = torch.split(input, [fold, fold, last_fold], -1)

        padding1 = torch.zeros_like(out1)
        padding1 = padding1[:, -1, :, :, :]
        padding1 = torch.unsqueeze(padding1, 1)
        _, out1 = torch.split(out1, [1, n_frame - 1], 1)
        out1 = torch.cat((out1, padding1), 1)

        padding2 = torch.zeros_like(out2)
        padding2 = padding2[:, 0, :, :, :]
        padding2 = torch.unsqueeze(padding2, 1)
        out2, _ = torch.split(out2, [n_frame - 1, 1], 1)
        out2 = torch.cat((padding2, out2), 1)

        out = torch.cat((out1, out2, out3), -1)
        out = out.view([-1, C, H, W])

        return out


class TSM_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,padding):
        super().__init__()
        self.tsm1 = TSM()
        self.t_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       padding=padding)

    def forward(self, input, n_frame=2, fold_div=3):
        t = self.tsm1(input, n_frame, fold_div)
        t = self.t_conv1(t)
        return t
