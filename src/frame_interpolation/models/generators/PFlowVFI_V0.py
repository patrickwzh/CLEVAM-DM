"""PerVFI: Soft-binary Blending for Photo-realistic Video Frame Interpolation

"""
import accelerate
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torchvision.ops import DeformConv2d

from . import thops
from .msfusion import MultiscaleFuse
from .normalizing_flow import *
from .softsplatnet import Encode, Softmetric
from .softsplatnet.softsplat import softsplat


def resize(x, size: tuple, scale: bool):
    H, W = x.shape[-2:]
    h, w = size
    scale_ = h / H
    x_ = F.interpolate(x, size, mode="bilinear", align_corners=False)
    if scale:
        return x_ * scale_
    return x_


def binary_hole(flow):
    n, _, h, w = flow.shape
    mask = softsplat(
        tenIn=torch.ones((n, 1, h, w), device=flow.device),
        tenFlow=flow,
        tenMetric=None,
        strMode="avg",
    )
    ones = torch.ones_like(mask, device=mask.device)
    zeros = torch.zeros_like(mask, device=mask.device)
    out = torch.where(mask <= 0.5, ones, zeros)
    return out


def warp_pyramid(features: list, metric, flow):
    outputs = []
    masks = []
    for lv in range(3):
        fea = features[lv]
        if lv != 0:
            h, w = fea.shape[-2:]
            metric = resize(metric, (h, w), scale=False)
            flow = resize(flow, (h, w), scale=True)
        outputs.append(softsplat(fea, flow, metric.neg().clip(-20.0, 20.0), "soft"))
        masks.append(binary_hole(flow))
    return outputs, masks


class FeaturePyramid(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netEncode = Encode()
        self.netSoftmetric = Softmetric()

    def forward(
        self,
        tenOne,
        tenTwo=None,
        tenFlows: list[Tensor] = None,
        F1t=None,
        F2t=None,
        time: float = 0.5,
    ):
        x1s = self.netEncode(tenOne)
        if tenTwo is None:  # just encode
            return x1s
        F12, F21 = tenFlows
        x2s = self.netEncode(tenTwo)
        m1t = self.netSoftmetric(x1s, x2s, F12) * 2 * time
        if F1t is None:
            F1t = time * F12
        m2t = self.netSoftmetric(x2s, x1s, F21) * 2 * (1 - time)
        if F2t is None:
            F2t = (1 - time) * F21
        Ft2 = -1 * softsplat(F2t, F2t, m2t.neg().clip(-20.0, 20.0), "soft")
        x1s, bmasks = warp_pyramid(x1s, m1t, F1t)
        return list(zip(x1s, x2s)), bmasks, Ft2


class SoftBinary(torch.nn.Module):
    def __init__(self, cin, dilate_size=7) -> None:
        super().__init__()
        channel = 64
        reduction = 8
        self.conv1 = torch.nn.Sequential(
            *[
                torch.nn.Conv2d(1, channel, dilate_size, 1, padding="same", bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(channel, channel, 3, 1, padding="same", bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(channel, channel, 1, 1, padding="same", bias=False),
            ]
        )
        self.att = torch.nn.Conv2d(cin * 2, channel, 3, 1, padding="same")
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid(),
        )
        self.conv2 = torch.nn.Conv2d(channel, 1, 1, 1, padding="same", bias=False)

    def forward(self, bmask, feaL, feaR):  # N,1,H,W
        m_fea = self.conv1(bmask)
        x = self.att(torch.cat([feaL, feaR], dim=1))
        b, c, _, _ = x.size()
        x = self.avg(x).view(b, c)
        x = self.fc(x).view(b, c, 1, 1)
        x = m_fea * x.expand_as(x)
        x = self.conv2(x)

        x = torch.tanh(torch.abs(x))
        rand_bias = (torch.rand_like(x, device=x.device) - 0.5) / 100.0
        if self.training:
            return x + rand_bias
        return x


class DCNPack(torch.nn.Module):
    def __init__(self, cin, groups, dksize):
        super().__init__()
        cout = groups * 3 * dksize**2
        self.conv_offset = torch.nn.Conv2d(cin, cout, 3, 1, 1)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.dconv = DeformConv2d(cin, cin, dksize, padding=dksize // 2)

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger.info(f"Offset abs mean is {offset_absmean}, larger than 50.")

        return self.dconv(x, offset, mask)


class DeformableAlign(torch.nn.Module):
    def __init__(self):
        super().__init__()
        channels = [35, 64, 96]
        self.offset_conv1 = torch.nn.ModuleDict()
        self.offset_conv2 = torch.nn.ModuleDict()
        self.offset_conv3 = torch.nn.ModuleDict()
        self.deform_conv = torch.nn.ModuleDict()
        self.feat_conv = torch.nn.ModuleDict()
        self.merge_conv1 = torch.nn.ModuleDict()
        self.merge_conv2 = torch.nn.ModuleDict()
        # Pyramids
        for i in range(2, -1, -1):
            level = f"l{i}"
            c = channels[i]
            # compute offsets
            self.offset_conv1[level] = torch.nn.Conv2d(c * 2 + 3, c, 3, 1, 1)
            if i == 2:
                self.offset_conv2[level] = torch.nn.Conv2d(c, c, 3, 1, 1)
            else:
                self.offset_conv2[level] = torch.nn.Conv2d(
                    c + channels[i + 1], c, 3, 1, 1
                )
                self.offset_conv3[level] = torch.nn.Conv2d(c, c, 3, 1, 1)
            # apply deform conv
            if i == 0:
                self.deform_conv[level] = DCNPack(c, 7, 3)
            else:
                self.deform_conv[level] = DCNPack(c, 8, 3)
            self.merge_conv1[level] = torch.nn.Conv2d(c + c + 1, c, 3, 1, 1)
            if i < 2:
                self.feat_conv[level] = torch.nn.Conv2d(c + channels[i + 1], c, 3, 1, 1)
                self.merge_conv2[level] = torch.nn.Conv2d(
                    c + channels[i + 1], c, 3, 1, 1
                )

        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features, bmasks, Ft2):
        outs = []

        for i in range(2, -1, -1):
            level = f"l{i}"
            feaL, feaR = features[i]
            bmask = bmasks[i]
            flow = resize(Ft2, bmask.shape[2:], scale=True)
            offset = torch.cat([feaL, feaR, bmask, flow], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 2:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(
                    self.offset_conv2[level](
                        torch.cat([offset, upsampled_offset], dim=1)
                    )
                )
                offset = self.lrelu(self.offset_conv3[level](offset))

            warped_feaR = self.deform_conv[level](feaR, offset)

            if i < 2:
                warped_feaR = self.feat_conv[level](
                    torch.cat([warped_feaR, upsampled_feaR], dim=1)
                )

            merged_feat = self.merge_conv1[level](
                torch.cat([feaL, warped_feaR, bmask], dim=1)
            )
            if i < 2:
                merged_feat = self.merge_conv2[level](
                    torch.cat([merged_feat, upsampled_merged_feat], dim=1)
                )
            outs.append(merged_feat)

            if i > 0:  # upsample offset and features
                warped_feaR = self.lrelu(warped_feaR)
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feaR = self.upsample(warped_feaR)
                upsampled_merged_feat = self.upsample(merged_feat)

        return outs


class AttentionMerge(torch.nn.Module):
    def __init__(self, dilate_size=7):
        super().__init__()
        self.softbinary = torch.nn.ModuleDict()
        channels = [35, 64, 96]
        for i in range(2, -1, -1):
            level = f"{i}"
            c = channels[i]
            self.softbinary[level] = SoftBinary(c, dilate_size)

    def forward(self, feaL, feaR, bmask):
        outs = []
        soft_masks = []
        for i in range(2, -1, -1):
            level = f"{i}"
            sm = self.softbinary[level](bmask[i], feaL[i], feaR[i])
            soft_masks.append(sm)
            x = feaL[i] * (1 - sm) + feaR[i] * sm
            outs.append(x)
        return outs, soft_masks


class Network(torch.torch.nn.Module):
    def __init__(self, dilate_size=9):
        super().__init__()
        cond_c = [35, 64, 96]
        self.featurePyramid = FeaturePyramid()
        self.deformableAlign = DeformableAlign()
        self.attentionMerge = AttentionMerge(dilate_size=dilate_size)
        self.multiscaleFuse = MultiscaleFuse(cond_c)
        self.condFLownet = CondFlowNet(cond_c, with_bn=False, train_1x1=True, K=16)

    def get_cond(self, inps: list, F1t, F2t, time: float = 0.5):
        tenOne, tenTwo, fflow, bflow = inps
        with accelerate.Accelerator().autocast():
            feas, bmasks, Ft2 = self.featurePyramid(
                tenOne, tenTwo, [fflow, bflow], F1t, F2t, time
            )
            feaR = self.deformableAlign(feas, bmasks, Ft2)[::-1]
            feaL = [feas[i][0] for i in range(3)]
            feas, smasks = self.attentionMerge(feaL, feaR, bmasks)
            # feas = [F.interpolate(x, scale_factor=0.5, mode="bilinear") for x in feas]
            feas = self.multiscaleFuse(feas[::-1])  # downscale by 2
        return feas, smasks

    def normalize(self, x, reverse=False):
        # x in [0, 1]
        if not reverse:
            return x * 2 - 1
        else:
            return (x + 1) / 2

    def forward(self, gt=None, zs=None, inps=[], F1t=None, F2t=None, time=0.5, code="decode"):
        assert code == "decode"
        return self.decode(zs, inps, F1t, F2t, time)

    def encode(self, gt, inps: list, time: float = 0.5):
        img0, img1 = [self.normalize(x) for x in inps[:2]]
        gt = self.normalize(gt)
        cond = [img0, img1] + inps[-2:]
        pixels = thops.pixels(gt)
        conds, smasks = self.get_cond(cond, time=time)

        # add random noise before normalizing flow net
        loss = 0.0
        if self.training:
            gt = gt + ((torch.rand_like(gt, device=gt.device) - 0.5) / 255.0)
            loss += -log(255.0) * pixels
        log_p, log_det, zs = self.condFLownet(gt, conds)

        loss /= float(log(2) * pixels)
        log_p /= float(log(2) * pixels)
        log_det /= float(log(2) * pixels)
        nll = -(loss + log_det + log_p)
        return nll, zs, smasks

    def decode(self, z_list: list, inps: list, F1t, F2t, time: float = 0.5):
        img0, img1 = [self.normalize(x) for x in inps[:2]]
        cond = [img0, img1] + inps[-2:]

        conds, smasks = self.get_cond(cond, F1t, F2t, time=time)
        pred = self.condFLownet(z_list, conds, reverse=True)
        pred = self.normalize(pred, reverse=True)
        return pred, smasks

    def encode_decode(self, gt, inps: list, time: float = 0.5, zs=None):
        img0, img1 = [self.normalize(x) for x in inps[:2]]
        gt = self.normalize(gt)
        cond = [img0, img1] + inps[-2:]
        pixels = thops.pixels(gt)
        conds, smasks = self.get_cond(cond, time=time)

        # encode first
        loss = 0.0
        if self.training:
            gt = gt + ((torch.rand_like(gt, device=gt.device) - 0.5) / 255.0)
            loss += -log(255.0) * pixels
        log_p, log_det, zs_gt = self.condFLownet(gt, conds)
        loss /= float(log(2) * pixels)
        log_p /= float(log(2) * pixels)
        log_det /= float(log(2) * pixels)
        nll = -(loss + log_det + log_p)

        # decode next
        if zs is None:
            heat = torch.sqrt(torch.var(torch.cat([x.flatten() for x in zs_gt])))
            zs = self.get_z(heat, img0.shape[-2:], img0.shape[0], img0.device)
        pred = self.condFLownet(zs, conds, reverse=True)
        pred = self.normalize(pred, reverse=True)
        return nll, pred, smasks

    def get_z(self, heat: float, img_size: tuple, batch: int, device: str):
        def calc_z_shapes(img_size, n_levels):
            h, w = img_size
            z_shapes = []
            channel = 3

            for _ in range(n_levels - 1):
                h //= 2
                w //= 2
                channel *= 2
                z_shapes.append((channel, h, w))
            h //= 2
            w //= 2
            z_shapes.append((channel * 4, h, w))
            return z_shapes

        z_list = []
        z_shapes = calc_z_shapes(img_size, 3)
        for z in z_shapes:
            z_new = torch.randn(batch, *z, device=device) * heat
            z_list.append(z_new)
        return z_list
