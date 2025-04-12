import numpy as np
import torch
import torchvision
from torch import nn
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from einops import rearrange

DOWNLOAD_PATH = 'data'
OUTPUT_PATH = 'vit_mnist_print.txt'
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 1000
N_EPOCHS = 15


def print_info(string):
    print(string)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        y = self.mlp_head(x)
        return y


loss_fn = nn.CrossEntropyLoss(reduction='none')


def train_epoch(model, optimizer, data_loader):
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, data_loader):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    return avg_loss


start_time = time.time()

torch.manual_seed(42)
transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def get_test_loss(n, mu):
    train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                           transform=transform_mnist)
    train_set.data = train_set.data[:n]
    train_set.targets = train_set.targets[:n]

    for i in range(len(train_set.data)):
        noise_matrix = torch.randn(28, 28)
        noise_matrix_boundary = noise_matrix
        noise_matrix_boundary[4:24,4:24] = 0
        train_set.data[i] = train_set.data[i] * mu / (1 + mu) + torch.abs(noise_matrix_boundary) * 255 / (1 + mu)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                          transform=transform_mnist)
    for i in range(len(test_set.data)):
        noise_matrix = torch.randn(28, 28)
        noise_matrix_boundary = noise_matrix
        noise_matrix_boundary[4:24,4:24] = 0
        test_set.data[i] = test_set.data[i] * mu / (1 + mu) + torch.abs(noise_matrix_boundary) * 255 / (1 + mu)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,  # 模型
                dim=64, depth=2, heads=4, mlp_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 优化器

    # 记录时间
    for epoch in range(1, N_EPOCHS + 1):
        train_epoch(model, optimizer, train_loader)  # 训练一epoch

    return evaluate(model, test_loader)

size = 10
matrix = np.zeros((size, size))
for i in range(size):
    print(i)
    for j in range(size):
        print(' ', j)
        matrix[i, j] = get_test_loss(3000 + i * 600, (j + 1) * 0.2)
        print('   ', matrix[i, j])

np.savetxt('matrix2.npy', matrix)