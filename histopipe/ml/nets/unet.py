import torch


class UNet(torch.nn.Sequential):
    """UNet implementation.

    Implementation based on: https://github.com/milesial/Pytorch-UNet
    """

    in_channels: int
    out_classes: int
    upscale_block: str
    upsample_mode: str

    def __init__(
        self,
        in_channels,
        out_classes,
        upscale_block="upsample",
        upsample_mode="bilinear",
    ):
        super().__init__()

        self.input_block = UNetBaseBlock(in_channels, 64)
        self.down_block_1 = UNetDownBlock(64, 128)
        self.down_block_2 = UNetDownBlock(128, 256)
        self.down_block_3 = UNetDownBlock(256, 512)
        self.down_block_4 = UNetDownBlock(512, 1024)
        self.up_block_1 = UNetUpBlock(1024, 512, upscale_block, upsample_mode)
        self.up_block_2 = UNetUpBlock(512, 256, upscale_block, upsample_mode)
        self.up_block_3 = UNetUpBlock(256, 128, upscale_block, upsample_mode)
        self.up_block_4 = UNetUpBlock(128, 64, upscale_block, upsample_mode)
        self.output_block = torch.nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.input_block(x)
        x2 = self.down_block_1(x1)
        x3 = self.down_block_2(x2)
        x4 = self.down_block_3(x3)
        x5 = self.down_block_4(x4)
        x = self.up_block_1(x5, [x4])
        x = self.up_block_2(x, [x3])
        x = self.up_block_3(x, [x2])
        x = self.up_block_4(x, [x1])
        x = self.output_block(x)
        return x


class UNetPlusPlus(torch.nn.Sequential):
    """UNet++ implementatiom.

    Implementatiom based on: https://github.com/keng000/pytorch_unet_plus_plus
    """

    in_channels: int
    out_classes: int
    deep_supervision: bool
    upscale_block: str
    upsample_mode: str

    def __init__(
        self,
        in_channels,
        out_classes,
        deep_supervision=True,
        upscale_block="transpose",
        upsample_mode="bilinear",
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [2 ** (6 + i) for i in range(5)]

        # layer = 1
        self.x00 = UNetBaseBlock(in_channels, filters[0])
        self.x10 = UNetDownBlock(filters[0], filters[1])
        self.x20 = UNetDownBlock(filters[1], filters[2])
        self.x30 = UNetDownBlock(filters[2], filters[3])
        self.x40 = UNetDownBlock(filters[3], filters[4])

        # layer = 2
        self.x01 = UNetUpBlock(
            filters[1], filters[0], upscale_block, upsample_mode, num_skips=1
        )
        self.x11 = UNetUpBlock(
            filters[2], filters[1], upscale_block, upsample_mode, num_skips=1
        )
        self.x21 = UNetUpBlock(
            filters[3], filters[2], upscale_block, upsample_mode, num_skips=1
        )
        self.x31 = UNetUpBlock(
            filters[4], filters[3], upscale_block, upsample_mode, num_skips=1
        )

        # layer = 3
        self.x02 = UNetUpBlock(
            filters[1], filters[0], upscale_block, upsample_mode, num_skips=2
        )
        self.x12 = UNetUpBlock(
            filters[2], filters[1], upscale_block, upsample_mode, num_skips=2
        )
        self.x22 = UNetUpBlock(
            filters[3], filters[2], upscale_block, upsample_mode, num_skips=2
        )

        # layer = 4
        self.x03 = UNetUpBlock(
            filters[1], filters[0], upscale_block, upsample_mode, num_skips=3
        )
        self.x13 = UNetUpBlock(
            filters[2], filters[1], upscale_block, upsample_mode, num_skips=3
        )

        # layer = final
        self.x04 = UNetUpBlock(
            filters[1], filters[0], upscale_block, upsample_mode, num_skips=4
        )

        self.output_block_01 = torch.nn.Conv2d(filters[0], out_classes, kernel_size=1)
        self.output_block_02 = torch.nn.Conv2d(filters[0], out_classes, kernel_size=1)
        self.output_block_03 = torch.nn.Conv2d(filters[0], out_classes, kernel_size=1)
        self.output_block_04 = torch.nn.Conv2d(filters[0], out_classes, kernel_size=1)

    def forward(self, x, prune=4):
        if not (1 <= prune <= 4):
            raise ValueError(
                f"Prune must be a value 1 <= prune <= 4. Found: '{prune}'."
            )

        x00 = self.x00(x)
        x10 = self.x10(x00)
        x01 = self.x01(x10, [x00])
        y = self.output_block_01(x01)

        if prune == 1:
            return y

        x20 = self.x20(x10)
        x11 = self.x11(x20, [x10])
        x02 = self.x02(x11, [x00, x01])
        y_new = self.output_block_02(x02)

        y = y + y_new if self.deep_supervision else y_new
        if prune == 2:
            return y / prune if self.deep_supervision else y

        x30 = self.x30(x20)
        x21 = self.x21(x30, [x20])
        x12 = self.x12(x21, [x10, x11])
        x03 = self.x03(x12, [x00, x01, x02])
        y_new = self.output_block_03(x03)

        y = y + y_new if self.deep_supervision else y_new
        if prune == 3:
            return y / prune if self.deep_supervision else y

        x40 = self.x40(x30)
        x31 = self.x31(x40, [x30])
        x22 = self.x22(x31, [x20, x21])
        x13 = self.x13(x22, [x10, x11, x12])
        x04 = self.x04(x13, [x00, x01, x02, x03])
        y_new = self.output_block_04(x04)

        y = y + y_new if self.deep_supervision else y_new
        if prune == 4:
            return y / prune if self.deep_supervision else y


class UNetBaseBlock(torch.nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int]
    stride: int | tuple[int]
    padding: str | int | tuple[int]

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv_2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.batchnorm_1 = torch.nn.BatchNorm2d(out_channels)
        self.batchnorm_2 = torch.nn.BatchNorm2d(out_channels)
        self.relu_1 = torch.nn.ReLU(inplace=True)
        self.relu_2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batchnorm_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.batchnorm_2(x)
        x = self.relu_2(x)
        return x


class UNetDownBlock(UNetBaseBlock):
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int]
    stride: int | tuple[int]
    padding: str | int | tuple[int]
    pooling: int | tuple[int]

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=2
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.maxpool = torch.nn.MaxPool2d(pooling)

    def forward(self, x):
        x = self.maxpool(x)
        x = super().forward(x)
        return x


class UNetUpBlock(UNetBaseBlock):
    in_channels: int
    out_channels: int
    upscale_block: str
    upsample_mode: str
    kernel_size: int | tuple[int]
    stride: int | tuple[int]
    num_skips: int
    upsample_scale_factor: int

    def __init__(
        self,
        in_channels,
        out_channels,
        upscale_block="upsample",
        upsample_mode="bilinear",
        kernel_size=2,
        stride=2,
        num_skips=1,
        upsample_scale_factor=2,
    ):
        super().__init__(in_channels // 2 * (num_skips + 1), out_channels)
        if upscale_block == "upsample":
            # NN does not support align_corners flag
            align_corners = True
            if upsample_mode == "nearest":
                align_corners = False

            # Determine reflection padding
            l_pad = r_pad = kernel_size // 2
            if not kernel_size % 2:
                r_pad -= 1

            self.upscale = torch.nn.Sequential(
                torch.nn.Upsample(
                    scale_factor=upsample_scale_factor,
                    mode=upsample_mode,
                    align_corners=align_corners,
                ),
                torch.nn.ReflectionPad2d((l_pad, r_pad, l_pad, r_pad)),
                torch.nn.Conv2d(in_channels, out_channels, kernel_size),
            )
        elif upscale_block == "transpose":
            self.upscale = torch.nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride
            )
        else:
            raise ValueError(
                f"Upscale mode '{upscale_block}' not recognized. Must be one of: 'upsample' or 'transpose'."
            )

    def forward(self, x, skip):
        x = self.upscale(x)
        delta_width = skip[0].size()[2] - x.size()[2]
        delta_height = skip[0].size()[3] - x.size()[3]
        x = torch.nn.functional.pad(
            x,
            [
                delta_height // 2,
                delta_height - (delta_height // 2),
                delta_width // 2,
                delta_width - (delta_width // 2),
            ],
        )
        x = torch.cat([*skip, x], dim=1)
        x = super().forward(x)
        return x
