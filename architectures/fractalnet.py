import torch
import torch.nn as nn

class Join(nn.Module):
    def __init__(self, drop_path, keep_p):
        super().__init__()
        self.drop_path = drop_path
        self.keep_p = keep_p

    def forward(self, x):
        if self.training and self.drop_path:
            keep = (torch.rand(len(x)) < self.keep_p).tolist()
            if not any(keep):
                keep[torch.randint(len(x), (1,)).item()] = True
            x = [output for output, k in zip(x, keep) if k]
        return sum(x) / len(x)

class FractalUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_column,
        drop_path,
        keep_p
    ):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.n_column = n_column
        self.drop_path = drop_path
        self.keep_p = keep_p
        self.join = Join(drop_path, keep_p)

        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.relu
        )

        self.right_first = None
        self.right_second = None
        if n_column > 1:
            self.right_first = FractalUnit(
                in_channels,
                out_channels,
                n_column - 1,
                drop_path,
                keep_p
            )
            self.right_second = FractalUnit(
                out_channels,
                out_channels,
                n_column - 1,
                drop_path,
                keep_p
            )

    def forward_column(self, x, column):
        if column == 0:
            return self.left(x)    
        intermediate = self.right_first(x, column=column - 1)
        return self.right_second(intermediate, column=column - 1)

    def forward(self, x, column=None):
        # global drop-path
        if column is not None:
            return self.forward_column(x, column)
        
        # local drop-path
        outputs = [self.left(x)]
        if self.right_first is not None:
            first_outputs = self.right_first(x)
            intermediate = self.join(first_outputs)
            second_outputs = self.right_second(intermediate)
            outputs.extend(second_outputs)
        return outputs

class FractalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_column,
        drop_path,
        keep_p
    ):
        super().__init__()
        self.drop_path = drop_path
        self.keep_p = keep_p
        self.fractal = FractalUnit(
            in_channels,
            out_channels,
            n_column,
            drop_path,
            keep_p
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.join = Join(drop_path, keep_p)

    def forward(self, x, column=None):
        if column is not None:
            return self.pool(self.fractal(x, column=column))
        
        column_outputs = self.fractal(x)
        pooled_outputs = [self.pool(output) for output in column_outputs]
        return self.join(pooled_outputs)

class FractalNet(nn.Module):
    def __init__(
        self,
        layers: list,
        n_column: int = 4,
        drop_path=True,
        keep_p: float = 0.85
    ):
        super().__init__()
        self.in_channels = 3
        self.n_column = n_column
        self.drop_path = drop_path

        self.stages = nn.Sequential()
        for layer in layers:
            self.stages.append(FractalBlock(
                self.in_channels,
                layer,
                n_column,
                drop_path,
                keep_p
            ))
            self.in_channels = layer
        self.flatten = nn.Flatten(1)
        self.feature_dim = self.in_channels
    
    def forward(self, x):
        column = None
        if self.training and self.drop_path:
            # global drop-path
            if (torch.rand(1) < 0.5).item():
                column = torch.randint(self.n_column, (1,)).item()

        for block in self.stages:
            x = block(x, column=column)
        return self.flatten(x)


def fractalnet20():
    return FractalNet(
        layers=[64, 128, 256, 512, 512],
        n_column=3,
        drop_path=True,
        keep_p=0.85
    )

def fractalnet40():
    return FractalNet(
        layers=[64, 128, 256, 512, 512],
        n_column=4,
        drop_path=True,
        keep_p=0.85
    )