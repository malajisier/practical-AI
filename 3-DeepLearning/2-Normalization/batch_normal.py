import torch
from torch import nn

bn = nn.BatchNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)

x = torch.rand(10, 3, 5, 5) * 100000
official_bn = bn(x)
print(x[0, 1:])

x1 = x.permute(1, 0, 2, 3).reshape(3, -1)
print(x1.size())

mu = x1.mean(dim=1).reshape(1, 3, 1, 1)
print(mu.size())

std = x1.std(dim=1, unbiased=False).reshape(1, 3, 1, 1)
print(std.size())

my_bn = (x - mu) / std
diff = (official_bn - my_bn).sum()
print(diff)