import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv import ChebConv

from torch_scatter import scatter_add


def Pool(x, trans: torch.Tensor, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class Enblock(nn.Module):
    def __init__(self, in_channels, out_channels, K, **kwargs):
        super(Enblock, self).__init__()
        self.conv = ChebConv(in_channels, out_channels, K, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.conv.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, down_transform):
        # print(f'in: {x.shape}')
        out = F.elu(self.conv(x, edge_index))
        # print(f'before pool: {out.shape}')
        out = Pool(out, down_transform)
        # print(f'out: {out.shape}')
        return out


class Deblock(nn.Module):
    def __init__(self, in_channels, out_channels, K, **kwargs):
        super(Deblock, self).__init__()
        self.conv = ChebConv(in_channels, out_channels, K, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.conv.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, up_transform):
        # print(f'in: {x.shape}')
        out = Pool(x, up_transform)
        # print(f'after pool: {out.shape}')
        out = F.elu(self.conv(out, edge_index))
        # print(f'out: {out.shape}')
        return out


class NEURAL_FACE(nn.Module):
    def __init__(self, in_channels, out_channels, part_latent_size, down_edge_index_list, down_transform_list, up_edge_index, up_transform, K, **kwargs):
        super(NEURAL_FACE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down_edge_index_list = down_edge_index_list
        self.down_transform_list = down_transform_list
        self.up_edge_index = up_edge_index
        self.up_transform = up_transform

        # used in the last and the first layer of encoder and decoder
        self.num_vert = self.up_transform[-1].size(1)
        self.part_count = len(down_edge_index_list)
        self.latent_size = self.part_count * part_latent_size
        self.out_channels_len = len(self.out_channels)

        # encoder
        self.en_layers = nn.ModuleList()
        for i in range(self.part_count):
            for idx in range(self.out_channels_len):
                if idx == 0:
                    self.en_layers.append(Enblock(in_channels, out_channels[idx], K, **kwargs))
                else:
                    self.en_layers.append(Enblock(out_channels[idx - 1], out_channels[idx], K, **kwargs))
            num_vert = down_transform_list[i][-1].size(0)
            self.en_layers.append(nn.Linear(num_vert * out_channels[-1], part_latent_size))
            self.en_layers.append(nn.Linear(num_vert * out_channels[-1], part_latent_size))
            # print(f'{num_vert}, {out_channels[-1]} to {part_latent_size}')

        # decoder
        self.de_layers = nn.ModuleList()
        # print(f'{self.latent_size} to {self.num_vert}, {out_channels[-1]}')
        self.de_layers.append(nn.Linear(self.latent_size, self.num_vert * out_channels[-1]))
        for idx in range(self.out_channels_len):
            if idx == 0:
                self.de_layers.append(Deblock(out_channels[-idx - 1], out_channels[-idx - 1], K, **kwargs))
            else:
                self.de_layers.append(Deblock(out_channels[-idx], out_channels[-idx - 1], K, **kwargs))
        # reconstruction
        self.de_layers.append(ChebConv(out_channels[0], in_channels, K, **kwargs))
        # print(f'{out_channels[0]}, {in_channels}, {K}')

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, data):
        all_z = []
        outs_means = []
        outs_sigmas = []

        for i in range(self.part_count):
            x = data[i + 1]
            for j in range(self.out_channels_len):
                layer_idx = i * (self.out_channels_len + 2) + j
                x = self.en_layers[layer_idx](x, self.down_edge_index_list[i][j], self.down_transform_list[i][j])

            mean = self.en_layers[layer_idx + 1](x.view(-1, x.shape[1] * x.shape[2]))
            sigma = self.en_layers[layer_idx + 2](x.view(-1, x.shape[1] * x.shape[2]))
            z = self.reparameterize(mean, sigma)

            all_z.append(z)
            outs_means.append(mean)
            outs_sigmas.append(sigma)

        out_z = torch.cat(all_z, dim=1)
        return out_z, outs_means, outs_sigmas

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_deblocks = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
                # print(f'{i}: {x.shape}')
            elif i != num_layers - 1:
                x = layer(x, self.up_edge_index[num_deblocks - i], self.up_transform[num_deblocks - i])
                # print(f'{i}: {x.shape}')
            else:
                # last layer
                x = layer(x, self.up_edge_index[0])
                # print(f'{i}: {x.shape}')
        # print(f'final: {x.shape}')
        return x

    def forward(self, data):
        z, mean, sigma = self.encoder(data)
        out = self.decoder(z)
        return out, mean, sigma
