import torch
import torch.nn.functional as F


class SEModule(torch.nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128):
        super(SEModule, self).__init__()
        self.se = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.se(x)
        return x * h


class ConvReLUBN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:

        super().__init__()

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.relu = torch.nn.ReLU(inplace=True)

        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.relu(self.conv(x)))


class Res2Block(torch.nn.Module):
    def __init__(self, inplane: int, planes: int, kernel_size: int = 3, dilation: int = 2, scale: int = 8) -> None:
        super().__init__()

        width = planes // scale
        padding = kernel_size // 2 * dilation

        self.conv1 = ConvReLUBN(inplane, width * scale, kernel_size=1, padding=0)

        # Res2
        self.res2convs = torch.nn.ModuleList()
        self.res2bns = torch.nn.ModuleList()

        for i in range(max(1, scale - 1)):
            self.res2convs.append(
                torch.nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=padding)
            )

            self.res2bns.append(torch.nn.BatchNorm1d(width))

        self.conv3 = ConvReLUBN(width * scale, planes, kernel_size=1, padding=0)
        self.se = SEModule(planes)

        self.relu = torch.nn.ReLU(inplace=True)

        self.width = width
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)

        xs = torch.split(x, self.width, 1)
        x = 0
        y = xs[-1]

        for i in range(max(1, self.scale - 1)):
            x = x + xs[i]
            x = self.res2bns[i](self.relu(self.res2convs[i](x)))
            y = torch.cat((y, x), 1)

        y = self.conv3(y)
        y = self.se(y)

        return y + residual


class ECAPATDNN(torch.nn.Module):
    def __init__(self, infeats=80, planes=1024, outfeats=192):
        super().__init__()

        self.conv1 = ConvReLUBN(infeats, planes, kernel_size=5, stride=1, padding=2)

        self.layer1 = Res2Block(planes, planes, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Res2Block(planes, planes, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Res2Block(planes, planes, kernel_size=3, dilation=4, scale=8)

        self.layer4 = torch.nn.Conv1d(3 * planes, outfeats * 8, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.attention = torch.nn.Sequential(
            torch.nn.Conv1d(outfeats * 8 * 3, 256, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Conv1d(256, outfeats * 8, kernel_size=1),
            torch.nn.Softmax(dim=2),
        )

        self.bn5 = torch.nn.BatchNorm1d(outfeats * 8 * 2)
        self.fc5 = torch.nn.Linear(outfeats * 8 * 2, outfeats)
        self.bn6 = torch.nn.BatchNorm1d(outfeats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.relu(self.layer4(x))

        def __compute_stats(x, w, dim=2):
            m = torch.mean(w * x, dim=dim, keepdim=True)
            s = torch.sqrt(torch.clamp(torch.sum((x**2) * w, dim=2, keepdim=True) - m ** 2, 1e-6))
            return m, s

        mean, std = __compute_stats(x, torch.ones_like(x))

        global_stat = torch.cat((x, mean.repeat(1, 1, x.size(-1)), std.repeat(1, 1, x.size(-1))), dim=1)
        attention = self.attention(global_stat)

        mean, std = __compute_stats(x, attention)

        x = torch.cat((mean, std), 1).flatten(1)

        x = self.bn5(x)
        x = self.fc5(x)
        x = self.bn6(x)

        return x


if __name__ == '__main__':
    from torchinfo import summary
    summary(ECAPATDNN(1024), input_size=(16, 80, 150))
