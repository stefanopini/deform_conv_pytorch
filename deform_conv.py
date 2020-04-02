import numpy as np
import torch
from torch import nn


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.dtype
        ks = self.kernel_size
        N = offset.shape[1] // 2

        # ------------------------------------------------------------------------
        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = torch.cat([
            torch.arange(0, 2 * N, 2, requires_grad=False, dtype=x.dtype, device=x.device),
            torch.arange(1, 2 * N + 1, 2, requires_grad=False, dtype=x.dtype, device=x.device)
        ]).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.shape)
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype, device=x.device)

        # (b, h, w, 2N)
        # p = p.contiguous().permute(0, 2, 3, 1)
        p = p.permute(0, 2, 3, 1)
        q_lt = p.detach().floor()  # do we need a clone op?
        q_rb = q_lt + 1

        q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, x.shape[2] - 1),
            torch.clamp(q_lt[..., N:], 0, x.shape[3] - 1)], dim=-1).long()
        q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, x.shape[2] - 1),
            torch.clamp(q_rb[..., N:], 0, x.shape[3] - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([
            p[..., :N].lt(self.padding) + p[..., :N].gt(x.shape[2] - 1 - self.padding),
            p[..., N:].lt(self.padding) + p[..., N:].gt(x.shape[3] - 1 - self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.shape[2] - 1), torch.clamp(p[..., N:], 0, x.shape[3] - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype, device):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                   range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2 * N, 1, 1))
        p_n = torch.tensor(p_n, dtype=dtype, requires_grad=False, device=device)

        return p_n

    def _get_p_n__torch(self, N, dtype, device):
        # slower...
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1, dtype=dtype,
                         requires_grad=False),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1, dtype=dtype,
                         requires_grad=False)
        )
        # (2N, 1)
        p_n = torch.cat((p_n_x.flatten(), p_n_y.flatten()))
        p_n = torch.reshape(p_n, (1, 2 * N, 1, 1))

        return p_n.to(device)

    @staticmethod
    def _get_p_0(h, w, N, dtype, device):
        p_0_x, p_0_y = np.meshgrid(range(1, h + 1), range(1, w + 1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = torch.tensor(p_0, dtype=dtype, requires_grad=False, device=device)

        return p_0

    @staticmethod
    def _get_p_0__torch(h, w, N, dtype, device):
        # slower...
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h + 1, dtype=dtype, requires_grad=False),
            torch.arange(1, w + 1, dtype=dtype, requires_grad=False),
        )
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat_interleave(N, dim=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat_interleave(N, dim=1)
        p_0 = torch.cat((p_0_x, p_0_y), dim=1)
        # p_0 = torch.tensor(p_0, dtype=dtype, requires_grad=False, device=device)

        return p_0.to(device)

    def _get_p(self, offset, dtype, device):
        N, h, w = offset.shape[1] // 2, offset.shape[2], offset.shape[3]

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype, device=device)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype, device=device)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.shape
        padded_w = x.shape[3]
        c = x.shape[1]
        # (b, c, h*w)
        # x = x.contiguous().view(b, c, -1)
        x = x.reshape(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        # index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        index = index.unsqueeze(dim=1).expand(-1, c, -1, -1, -1).reshape(b, c, -1)

        # x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        x_offset = x.gather(dim=-1, index=index).reshape(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.shape
        # x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = torch.cat([x_offset[..., s:s + ks].reshape(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
        # x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        x_offset = x_offset.reshape(b, c, h * ks, w * ks)

        return x_offset
