import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, bias=False):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, dilation=1, dropout=0.2, causal=False):
        super().__init__()
        if causal and kernel_size > 1:
            conv = CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation, bias=False)
        else:
            conv = nn.Conv1d(in_ch, out_ch, kernel_size, bias=False)
        self.net = nn.Sequential(
            conv,
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleDilatedConv(nn.Module):
    """
    Approximation of Fig. 3 in Zhang & Wang, INTERSPEECH 2020.
    Input channels are split into sub-bands. Each band is convolved after
    concatenating the previous band output, then the same is repeated in
    the reverse direction. Left and right results are summed.
    """
    def __init__(self, channels=514, bands=8, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        base = channels // bands
        sizes = [base] * bands
        sizes[-1] += channels - sum(sizes)
        self.sizes = sizes
        self.bands = bands

        left, right = [], []
        for i, s in enumerate(sizes):
            in_ch = s if i == 0 else s + sizes[i - 1]
            left.append(ConvBNReLU(in_ch, s, kernel_size, dilation, dropout, causal=True))
        rev = list(reversed(sizes))
        for i, s in enumerate(rev):
            in_ch = s if i == 0 else s + rev[i - 1]
            right.append(ConvBNReLU(in_ch, s, kernel_size, dilation, dropout, causal=True))
        self.left = nn.ModuleList(left)
        self.right = nn.ModuleList(right)

    def _scan(self, chunks, modules):
        outs, prev = [], None
        for x, conv in zip(chunks, modules):
            z = x if prev is None else torch.cat([x, prev], dim=1)
            prev = conv(z)
            outs.append(prev)
        return outs

    def forward(self, x):
        chunks = list(torch.split(x, self.sizes, dim=1))
        left_out = self._scan(chunks, self.left)
        right_out_rev = self._scan(list(reversed(chunks)), self.right)
        right_out = list(reversed(right_out_rev))
        y_left = torch.cat(left_out, dim=1)
        y_right = torch.cat(right_out, dim=1)
        return y_left + y_right


class MSTCNResBlock(nn.Module):
    def __init__(self, dilation=1, dropout=0.2, use_multiscale=True):
        super().__init__()
        self.reduce = ConvBNReLU(1024, 257, kernel_size=1, dropout=dropout)
        if use_multiscale:
            self.temporal = MultiScaleDilatedConv(514, bands=8, kernel_size=3, dilation=dilation, dropout=dropout)
        else:
            self.temporal = ConvBNReLU(514, 514, kernel_size=3, dilation=dilation, dropout=dropout, causal=True)
        self.expand = ConvBNReLU(514, 1024, kernel_size=1, dropout=dropout)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x, stacked_lps):
        residual = x
        z = self.reduce(x)
        z = torch.cat([z, stacked_lps], dim=1)  # forward stacking of original noisy LPS
        z = self.temporal(z)
        z = self.expand(z)
        return self.out_act(z + residual)


class MSTCNSE(nn.Module):
    def __init__(self, dilations=(1, 2, 5, 7, 11), dropout=0.2, use_multiscale=True):
        super().__init__()
        self.in_dense = nn.Linear(257, 1024)
        self.blocks = nn.ModuleList([
            MSTCNResBlock(d, dropout=dropout, use_multiscale=use_multiscale)
            for d in dilations
        ])
        self.out_dense = nn.Linear(1024, 1024)
        self.lps_head = nn.Linear(1024, 257)
        self.irm_head = nn.Linear(1024, 257)

    def forward(self, noisy_lps):
        # noisy_lps: [B, T, 257]
        x = F.relu(self.in_dense(noisy_lps)).transpose(1, 2)  # [B,1024,T]
        stacked = noisy_lps.transpose(1, 2)                   # [B,257,T]
        for block in self.blocks:
            x = block(x, stacked)
        h = F.relu(self.out_dense(x.transpose(1, 2)))         # [B,T,1024]
        pred_lps = self.lps_head(h)
        pred_irm = torch.sigmoid(self.irm_head(h))
        return pred_lps, pred_irm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
