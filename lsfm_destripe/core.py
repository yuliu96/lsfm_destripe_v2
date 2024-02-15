#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from typing import Optional, Dict
import os
from types import SimpleNamespace

import numpy as np
import scipy
from aicsimageio import AICSImage
import torch
import torch.fft as fft
from torch.nn import functional as F
from tqdm import tqdm

from network import (
    DeStripeModel,
    Loss,
    GuidedFilterHR_fast,
    GuidedFilterHR,
    GuidedFilterLoss,
)

###############################################################################

log = logging.getLogger(__name__)


###############################################################################
class DeStripe:
    def __init__(
        self,
        data_path,
        sample_name,
        isVertical=True,
        angleOffset=[0],
        isImageSequence=False,
        filter_keyword=[],
        losseps=10,
        qr=0.5,
        resampleRatio=2,
        KGF=29,
        KGFh=29,
        HKs=0.5,
        sampling_in_MSEloss=2,
        isotropic_hessian=True,
        lambda_tv=1,
        lambda_hessian=1,
        inc=16,
        n_epochs=300,
        deg=29,
        Nneighbors=16,
        _display=True,
        fast_GF=False,
        require_global_correction=True,
        mask_name=None,
    ):
        """
        data_path (str): file path for volume input
        sample_name (str): volume's name, could be a .tif, .tiff, .etc (for example, destriping_sample.tiff) or image sequence, i.e., a folder name
        isVertical (boolean): direction of the stripes. True by default
        angleOffset (list): a list of angles in degree, data range for each angle is [-90, 90]. For example [-10, 0, 10] for ultramicroscope. [0] by default
        isImageSequence (boolean): True for image sequence input, False for volumetric input. False by default.
        filter_keyword (list): a list of str (if applicable). if input is image sequence, only images that contain all the filter_keyword in name are gonna be processed, [] by default
        losseps (float): eps in loss. data range [0.1, inf). 10 by default
        qr (float): a threhold. data range [0.5, inf). 0.5 by default
        resampleRatio (int): downsample ratio, data range [1, inf), 2 by default
        KGF (int): kernel size for guided filter during training. must be odd. 29 by default
        KGFh (int): kernel size for guided filter during inference. must be odd. 29 by default
        HKs (float): sigma to generate hessian kernel. data range [0.5, 1.5]. 0.5 by default
        sampling_in_MSEloss (int): downsampling when calculating MSE. data range [1, inf). 2 by default
        isotropic_hessian (boolean): True by default
        lambda_tv (float): trade-off parameter of total variation in loss. data range (0, inf). 1 by default
        lambda_hessian (float): trade-off parameter of hessian in loss. data range (0, inf). 1 by default
        inc (int): latent neuron numbers in NN. power of 2. 16 by default
        n_epochs (int): total epochs for training. data range [1, inf). 300 by default
        deg (float): angle in degree to generate wedge-like mask. data range (0, 90). 29 by default
        Nneighbors (int): data range [1, 32], 16 by default
        _display  (boolean): display result or not after processing every slice. True by default
        fast_GF (boolean): methods used for composing high-res result, False by default
        require_global_correction (boolean): True by default
        mask_name: mask's name (if applicable) could be a XX.tif, XX.tiff, .etc, None by default
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_name = os.path.join(data_path, sample_name)
        self.angleOffset, self._isVertical, self.isImageSequence, self._filter = (
            angleOffset,
            isVertical,
            isImageSequence,
            filter_keyword,
        )
        self.mask_name = (
            os.path.join(data_path, mask_name) if mask_name != None else mask_name
        )
        self.KGFh = KGFh
        self.KGF = KGF
        self._display = _display
        qr = [qr]
        self.qr = qr
        self.losseps = losseps
        self.HKs = HKs
        self.lambda_tv = lambda_tv
        self.lambda_hessian = lambda_hessian
        self.inc = inc
        self.n_epochs = n_epochs
        self.deg = deg
        self.resampleRatio = [resampleRatio, resampleRatio]
        self.Nneighbors = Nneighbors
        self.fast_GF = fast_GF
        self.f = isotropic_hessian
        self.require_global_correction = require_global_correction
        self.sampling = sampling_in_MSEloss
        if self.isImageSequence:
            self.filename = os.listdir(self.sample_name)
            tmp = []
            for f in self.filename:
                if np.prod([c in f for c in self._filter]):
                    tmp.append(f)
            if len(self._filter) != 0:
                self.filename = tmp
            self.filename.sort()
        # self.m, self.n = (
        #     self.tiffread(self.sample_name, 0).shape
        #     if self.isImageSequence == False
        #     else self.tiffread(self.sample_name + "/" + self.filename[0], 0).shape
        # )
        self.md, self.nd = (
            self.m // self.resampleRatio[0] // 2 * 2 + 1,
            self.n // self.resampleRatio[1] // 2 * 2 + 1,
        )
        angleMask = np.stack(
            [
                self.WedgeMask(
                    self.md if self._isVertical == True else self.nd,
                    self.nd if self._isVertical == True else self.md,
                    Angle=angle,
                    deg=self.deg,
                )
                for angle in self.angleOffset
            ],
            0,
        )
        angleMask = angleMask.reshape(angleMask.shape[0], -1)[
            :, : self.md * self.nd // 2
        ]
        hier_mask = np.where(angleMask == 1)[1]
        hier_ind = np.argsort(
            np.concatenate(
                [np.where(angleMask.reshape(-1) == index)[0] for index in range(2)]
            )
        )
        NI = (
            self.NeighborSampling(self.md, self.nd, k_neighbor=Nneighbors)
            if self._isVertical == True
            else self.NeighborSampling(self.nd, self.md, k_neighbor=Nneighbors)
        )
        NI = np.concatenate(
            [NI[hier_mask == 0, 1 : Nneighbors + 1].T for hier_mask in angleMask], 1
        )
        self.NI, self.hier_mask, self.hier_ind = (
            torch.from_numpy(NI).to(self.device),
            torch.from_numpy(hier_mask).to(self.device),
            torch.from_numpy(hier_ind).to(self.device),
        )
        # self.GuidedFilterLoss = GuidedFilterLoss(rx=self.KGF, ry=self.KGF, eps=losseps)
        # if fast_GF:
        #     self.GuidedFilterHR = GuidedFilterHR_fast(
        #         rx=self.KGFh, ry=0, angleList=angleOffset, eps=1e-9
        #     ).to(self.device)
        # else:
        #     self.GuidedFilterHR = GuidedFilterHR(
        #         rX=[self.KGFh * 2 + 1, self.KGFh],
        #         rY=[0, 0],
        #         m=self.m if self._isVertical == True else self.n,
        #         n=self.n if self._isVertical == True else self.m,
        #         Angle=self.angleOffset,
        #     )

    # TODO: why need to convert img==0 to 1??
    # def tiffread(self, path, frame_index, _is_image=True):
    #     img = Image.open(path)
    #     if frame_index is not None:
    #         img.seek(frame_index)
    #     img = np.array(img)
    #     if _is_image:
    #         img[img == 0] = 1
    #     return img

    @staticmethod
    def train_one_slice(
        arr_O: np.ndarray,
        arr_map: Optional[np.ndarray] = None,
        is_vertical: bool = False,
        device: str = "cpu",
        shape_param: Dict = None,
        train_param: Dict = None,
    ):
        # convert dictionary to Object
        shape_param = SimpleNamespace(**shape_param)
        train_param = SimpleNamespace(**train_param)

        # check if vertical
        if is_vertical:
            arr_O, arr_map = arr_O.T, arr_map.T

        # convert to tensor
        X = torch.from_numpy(arr_O[None, None]).float().to(device)
        map = torch.from_numpy(arr_map[None, None]).float().to(device)

        Xd = F.interpolate(
            X,
            size=(
                shape_param.md if is_vertical else shape_param.nd,
                shape_param.nd if is_vertical else shape_param.md,
            ),
            align_corners=True,
            mode="bilinear",
        )
        map = F.interpolate(
            map,
            size=(
                shape_param.md if is_vertical else shape_param.nd,
                shape_param.nd if is_vertical else shape_param.md,
            ),
            align_corners=True,
            mode="bilinear",
        )

        # TODO: add comment
        map = map > 128
        Xf = fft.fftshift(fft.fft2(Xd)).reshape(-1)[: Xd.numel() // 2, None]
        GF_loss = GuidedFilterLoss(
            rx=train_param.KGF, ry=train_param.KGF, eps=train_param.losseps
        )
        smoothedtarget = GF_loss(Xd, Xd)
        if train_param.fast_GF:
            GF_HR = GuidedFilterHR_fast(
                rx=train_param.KGFh, ry=0, angleList=train_param.angleOffset, eps=1e-9
            ).to(device)
        else:
            GF_HR = GuidedFilterHR(
                rX=[train_param.KGFh * 2 + 1, train_param.KGFh],
                rY=[0, 0],
                m=shape_param.md if is_vertical else shape_param.nd,
                n=shape_param.md if is_vertical else shape_param.nd,
                Angle=train_param.angleOffset,
            )
        model = DeStripeModel(
            Angle=train_param.angleOffset,
            hier_mask=train_param.hier_mask,
            hier_ind=train_param.hier_ind,
            NI=train_param.NI,
            m=shape_param.md if is_vertical else shape_param.nd,
            n=shape_param.md if is_vertical else shape_param.nd,
            KS=train_param.KGF,
            Nneighbors=train_param.Nneighbors,
            inc=train_param.inc,
            device=device,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss = Loss(
            train_param.HKs,
            train_param.lambda_tv,
            train_param.lambda_hessian,
            train_param.sampling,
            train_param.f,
            shape_param.md if is_vertical else shape_param.nd,
            shape_param.md if is_vertical else shape_param.nd,
            train_param.angleOffset,
            train_param.KGF,
            train_param.losseps,
            device,
        ).to(device)
        for epoch in tqdm(range(train_param.n_epochs), leave=False):
            optimizer.zero_grad()
            outputGNN, outputLR = model(Xd, Xf)
            l = loss(outputGNN, outputLR, smoothedtarget, Xd, map)  # Xd, X
            l.backward()
            optimizer.step()
        with torch.no_grad():
            m, n = X.shape[-2:]
            outputGNN = F.interpolate(
                outputGNN, size=(m, n), mode="bilinear", align_corners=True
            )
            if not train_param.fast_GF:
                for index, qr in enumerate(train_param.qr):
                    locals()["X" + str(index)] = (
                        10 ** GF_HR(X, outputGNN, r=qr).cpu().data.numpy()[0, 0]
                    )
            else:
                locals()["X" + str(0)] = (
                    10 ** GF_HR(X, outputGNN, X).cpu().data.numpy()[0, 0]
                )

            # get all results
            X_out = {
                var_name: var_value
                for var_name, var_value in locals().items()
                if var_name.startswith("X")
            }
            return X_out

    def train(self):
        # get image information and prepare local variables
        img_handle = AICSImage(self.filename)
        dim_z = img_handle.dims.Z
        dim_m = img_handle.dims.Y
        dim_n = img_handle.dims.X
        for i in range(len(self.qr)):
            locals()["result" + str(i)] = np.zeros((dim_z, dim_m, dim_n))
        for i in range(len(self.qr)):
            locals()["mean" + str(i)] = np.zeros(dim_z)

        # loop through all Z
        for idx_z in range(dim_z):
            print(f"processing {idx_z} / {dim_z} slice .")
            arr_O = img_handle.get_dask_image_data("YX", Z=idx_z, T=0, C=0).compute()
            if self.mask_name is None:
                arr_map = np.zeros(arr_O.shape)
            else:
                arr_map = img_handle.get_dask_image_data(
                    "YX", Z=idx_z, T=0, C=0
                ).compute()
            all_X = self.train_one_slice(
                arr_O,
                arr_map,
                self._isVertical,
                self.device,
                self.shape_param,
                self.train_param,
            )
            for var_name, var_value in all_X.items():
                exec(f"{var_name} = {repr(var_value)}")
            # O = (
            #     np.log10(self.tiffread(self.sample_name + "/" + s_, None))
            #     if self.isImageSequence
            #     else np.log10(self.tiffread(self.sample_name, s_))
            # )
            # if self.mask_name == None:
            #     map = np.zeros(O.shape)
            # else:
            #     map = self.tiffread(self.mask_name, fileList.index(s_), _is_image=False)
            # if self._isVertical == False:
            #     O, map = O.T, map.T
            # X = torch.from_numpy(O[None, None]).float().to(self.device)
            # map = torch.from_numpy(map[None, None]).float().to(self.device)
            # Xd = F.interpolate(
            #     X,
            #     size=(
            #         self.md if self._isVertical else self.nd,
            #         self.nd if self._isVertical else self.md,
            #     ),
            #     align_corners=True,
            #     mode="bilinear",
            # )
            # map = F.interpolate(
            #     map,
            #     size=(
            #         self.md if self._isVertical else self.nd,
            #         self.nd if self._isVertical else self.md,
            #     ),
            #     align_corners=True,
            #     mode="bilinear",
            # )
            # map = map > 128
            # Xf = fft.fftshift(fft.fft2(Xd)).reshape(-1)[: Xd.numel() // 2, None]
            # smoothedtarget = self.GuidedFilterLoss(Xd, Xd)
            # model = DeStripeModel(
            #     Angle=self.angleOffset,
            #     hier_mask=self.hier_mask,
            #     hier_ind=self.hier_ind,
            #     NI=self.NI,
            #     m=self.md if self._isVertical == True else self.nd,
            #     n=self.nd if self._isVertical == True else self.md,
            #     KS=self.KGF,
            #     Nneighbors=self.Nneighbors,
            #     inc=self.inc,
            #     device=self.device,
            # ).to(self.device)
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            # loss = Loss(
            #     self.HKs,
            #     self.lambda_tv,
            #     self.lambda_hessian,
            #     self.sampling,
            #     self.f,
            #     self.md if self._isVertical else self.nd,
            #     self.nd if self._isVertical else self.md,
            #     self.angleOffset,
            #     self.KGF,
            #     self.losseps,
            #     self.device,
            # ).to(self.device)
            # for epoch in tqdm.tqdm(range(self.n_epochs), leave=False):
            #     optimizer.zero_grad()
            #     outputGNN, outputLR = model(Xd, Xf)
            #     l = loss(outputGNN, outputLR, smoothedtarget, Xd, map)  # Xd, X
            #     l.backward()
            #     optimizer.step()
            # with torch.no_grad():
            #     m, n = X.shape[-2:]
            #     outputGNN = F.interpolate(
            #         outputGNN, size=(m, n), mode="bilinear", align_corners=True
            #     )
            #     if self.fast_GF == False:
            #         for index, qr in enumerate(self.qr):
            #             locals()["X" + str(index)] = (
            #                 10
            #                 ** self.GuidedFilterHR(X, outputGNN, r=qr)
            #                 .cpu()
            #                 .data.numpy()[0, 0]
            #             )
            #     else:
            #         locals()["X" + str(0)] = (
            #             10
            #             ** self.GuidedFilterHR(X, outputGNN, X).cpu().data.numpy()[0, 0]
            #         )
            # if self._display:
            #     fig = plt.figure(dpi=300)
            #     ax = plt.subplot(1, 2, 2)
            #     plt.imshow(
            #         locals()["X0"] if self._isVertical else locals()["X0"].T,
            #         vmin=10 ** O.min(),
            #         vmax=10 ** O.max(),
            #         cmap="gray",
            #     )
            #     ax.set_title("output", fontsize=8, pad=1)
            #     plt.axis("off")
            #     ax = plt.subplot(1, 2, 1)
            #     plt.imshow(
            #         10**O if self._isVertical else 10**O.T,
            #         vmin=10 ** O.min(),
            #         vmax=10 ** O.max(),
            #         cmap="gray",
            #     )
            #     ax.set_title("input", fontsize=8, pad=1)
            #     plt.axis("off")
            #     plt.show()
            if self._isVertical == False:
                for index in range(len(self.qr)):
                    locals()["X" + str(index)] = locals()["X" + str(index)].T
            for index in range(len(self.qr)):
                locals()["result" + str(index)][i] = locals()["X" + str(index)]
                locals()["mean" + str(index)][i] = np.mean(locals()["X" + str(index)])
        if self.require_global_correction and dim_z != 1:
            print("global correcting...")
            for i in range(len(self.qr)):
                locals()["means" + str(i)] = scipy.signal.savgol_filter(
                    locals()["mean" + str(i)],
                    min(21, len(locals()["mean" + str(i)])),
                    1,
                )
                locals()["result" + str(i)][:] = (
                    locals()["result" + str(i)]
                    - locals()["mean" + str(i)][:, None, None]
                    + locals()["means" + str(i)][:, None, None]
                )

        # gather the final result
        out_list = []
        for i in range(len(self.qr)):
            locals()["result" + str(i)] = np.clip(
                locals()["result" + str(i)], 0, 65535
            ).astype(np.uint16)
            out_list.append(locals()["result" + str(i)])

        return np.stack(out_list, axis=0)

        # name, ext = os.path.splitext(self.sample_name)
        # fpath = name.rstrip(ext) + "+RESULT"
        # if len(self._filter) != 0:
        #     subpath = ""
        #     for fi in self._filter:
        #         subpath = subpath + fi
        # else:
        #     subpath = "all"
        # if not os.path.exists(fpath):
        #     os.makedirs(fpath)
        # if not os.path.exists(fpath + "/" + subpath):
        #     os.makedirs(fpath + "/" + subpath)
        # for qr in self.qr:
        #     locals()[fpath + str(qr)] = (
        #         fpath
        #         + "/"
        #         + subpath
        #         + "/"
        #         + "GFKernel{}".format(self.KGFh)
        #         + "_GFLossKernel{}".format(self.KGF)
        #         + "_qr{}".format(qr)
        #         + "_HKs{}".format(self.HKs)
        #         + "_losseps{}".format(self.losseps)
        #         + "_lambdatv{}".format(self.lambda_tv)
        #         + "_lambdahessian{}".format(self.lambda_hessian)
        #         + (
        #             "_withGlobalCorrection"
        #             if self.require_global_correction
        #             else "_withoutGlobalCorrection"
        #         )
        #         + ("_withoutMask" if self.mask_name == None else "_withMask")
        #         + ("_fastGF" if self.fast_GF else "_nofastGF")
        #     )
        #     if not os.path.exists(locals()[fpath + str(qr)]):
        #         os.makedirs(locals()[fpath + str(qr)])
        # for i, s_ in enumerate(tqdm.tqdm(fileList, desc="saving: ")):
        #     fname = "{:03d}.tif".format(s_) if type(s_) == int else "{}.tif".format(s_)
        #     for index, qr in enumerate(self.qr):
        #         tifffile.imwrite(
        #             locals()[fpath + str(qr)] + "/" + fname,
        #             np.asarray(locals()["result" + str(index)][i]),
        #         )
        # for index, qr in enumerate(self.qr):
        #     del locals()["result" + str(index)]
