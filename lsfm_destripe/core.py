#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from typing import Optional, Dict, Union, List
from types import SimpleNamespace
from pathlib import Path

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

from utils import prepare_aux

###############################################################################

log = logging.getLogger(__name__)


###############################################################################
class DeStripe:
    def __init__(
        self,
        data_path: Union[str, Path],
        isVertical: bool = True,
        angleOffset: List = [0],
        losseps: float = 10,
        qr: float = 0.5,
        resampleRatio: int = 2,
        KGF: int = 29,
        KGFh: int = 29,
        HKs: float = 0.5,
        sampling_in_MSEloss: int = 2,
        isotropic_hessian: bool = True,
        lambda_tv: float = 1,
        lambda_hessian: float = 1,
        inc: int = 16,
        n_epochs: int = 300,
        deg: float = 29,
        Nneighbors: int = 16,
        fast_GF: bool = False,
        require_global_correction: bool = True,
        mask_name: Union[str, Path] = None,
    ):
        """
        Main class for De-striping

        Parameters:
        -----------------------
        data_path: str or Path
            file path for volume input
        isVertical: boolean
            direction of the stripes. True by default
        angleOffset: List
            a list of angles in degree, data range for each angle is
            [-90, 90]. For example [-10, 0, 10] for ultramicroscope.
            [0] by default
        losseps: float
            eps in loss. data range [0.1, inf). 10 by default
        qr: float
            TODO (add more details)
            a threhold. data range [0.5, inf). 0.5 by default
        resampleRatio: int
            downsample ratio, data range [1, inf), 2 by default
        KGF: int
            kernel size for guided filter during training. must be odd. 29 by default
        KGFh: int
            TODO (add more details)
            kernel size for guided filter during inference. must be odd. 29 by default
        HKs: float
            sigma to generate hessian kernel. data range [0.5, 1.5]. 0.5 by default
        sampling_in_MSEloss: int
            TODO (add more details)
            downsampling when calculating MSE. data range [1, inf). 2 by default
        isotropic_hessian: boolean
            True by default
        lambda_tv: float
            trade-off parameter of total variation in loss. data range (0, inf).
            1 by default
        lambda_hessian: float
            trade-off parameter of hessian in loss. data range (0, inf). 1 by default
        inc: int
            latent neuron numbers in NN. power of 2. 16 by default
        n_epochs: int
            total epochs for training. data range [1, inf). 300 by default
        deg: float
            angle in degree to generate wedge-like mask. data range (0, 90).
            29 by default
        Nneighbors: int
            data range [1, 32], 16 by default
        fast_GF: boolean
            methods used for composing high-res result, False by default
        require_global_correction: boolean
            Whether to run additional correction for whole z-stack after processing
            individual slices. True by default
        mask_name: str or Path
            path to mask file (if applicable). None by default
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._isVertical = isVertical
        self.filename = data_path
        self.mask_name = mask_name

        # TODO: Is qr always a single number or could be a list of numbers?
        qr = [qr]
        self.qr = qr

        self.train_param = {
            "fast_GF": fast_GF,
            "KGF": KGF,
            "KGFh": KGFh,
            "losseps": losseps,
            "angleOffset": angleOffset,
            "Nneighbors": Nneighbors,
            "inc": inc,
            "HKs": HKs,
            "lambda_tv": lambda_tv,
            "lambda_hessian": lambda_hessian,
            "sampling": sampling_in_MSEloss,
            "resampleRatio": resampleRatio,
            "f": isotropic_hessian,
            "n_epochs": n_epochs,
            "deg": deg,
            "hier_mask": None,
            "hier_ind": None,
            "NI": None,
        }

        self.resampleRatio = resampleRatio
        self.require_global_correction = require_global_correction

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
        qr: Union[float, List[float]] = None,
        shape_param: Dict = None,
        train_param: Dict = None,
        device: str = "cpu",
    ):
        """
        train the network on a single 2D slice

        Parameters:
        -------------
        arr_O: ndarray
            the input slice
        arr_map: ndarray
            optional mask
        is_vertical: book
            whether the stripe is vertical
        qr: float or a list of float
            threshold TODO (add more info)
        shape_param: Dict
            parameters related to shape
        train_param: Dict
            parameters related to training
        """
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
            epoch_loss = loss(outputGNN, outputLR, smoothedtarget, Xd, map)  # Xd, X
            epoch_loss.backward()
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

    @staticmethod
    def train_full_arr(
        img_arr,
        mask_arr: np.ndarray = None,
        is_vertical: bool = False,
        train_param: Dict = None,
        device: str = "CPU",
        qr: List[float] = None,
        require_global_correction: bool = True,
    ):
        # get image information and prepare local variables
        dim_z, dim_m, dim_n = img_arr.shape
        for i in range(len(qr)):
            locals()["result" + str(i)] = np.zeros((dim_z, dim_m, dim_n))
        for i in range(len(qr)):
            locals()["mean" + str(i)] = np.zeros(dim_z)

        md, nd = (
            dim_m // train_param["resampleRatio"] // 2 * 2 + 1,
            dim_n // train_param["resampleRatio"] // 2 * 2 + 1,
        )

        shape_param = {"md": md, "nd": nd}

        # prepare auxillary variables
        NI_arr, hier_mask_arr, hier_ind_arr = prepare_aux(
            md,
            nd,
            is_vertical,
            train_param["angleOffset"],
            train_param["deg"],
            train_param["Nneighbors"],
        )

        NI, hier_mask, hier_ind = (
            torch.from_numpy(NI_arr).to(device),
            torch.from_numpy(hier_mask_arr).to(device),
            torch.from_numpy(hier_ind_arr).to(device),
        )

        train_param["NI"] = NI
        train_param["hier_mask"] = hier_mask
        train_param["hier_mask"] = hier_mask

        # loop through all Z
        for idx_z in range(dim_z):
            print(f"processing {idx_z} / {dim_z} slice .")
            arr_O = img_arr[idx_z, :, :]
            if mask_arr is None:
                arr_map = np.zeros(arr_O.shape)
            else:
                arr_map = mask_arr[idx_z, :, :]
            all_X = DeStripe.train_one_slice(
                arr_O,
                arr_map,
                is_vertical,
                device,
                shape_param,
                train_param,
            )
            for var_name, var_value in all_X.items():
                exec(f"{var_name} = {repr(var_value)}")

            # rotate if it is vertical
            if not is_vertical:
                for index in range(len(qr)):
                    locals()["X" + str(index)] = locals()["X" + str(index)].T
            for index in range(len(qr)):
                locals()["result" + str(index)][i] = locals()["X" + str(index)]
                locals()["mean" + str(index)][i] = np.mean(locals()["X" + str(index)])

        # glocal correction for the whole z-stack
        if require_global_correction and dim_z != 1:
            print("global correcting...")
            for i in range(len(qr)):
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

        # gather the final result into a z-stack
        out_list = []
        for i in range(len(qr)):
            locals()["result" + str(i)] = np.clip(
                locals()["result" + str(i)], 0, 65535
            ).astype(np.uint16)
            out_list.append(locals()["result" + str(i)])

        return np.stack(out_list, axis=0)

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

        if self.mask_name is None:
            mask_handle = AICSImage(self.mask_name)

        md, nd = (
            dim_m // self.train_param["resampleRatio"] // 2 * 2 + 1,
            dim_n // self.train_param["resampleRatio"] // 2 * 2 + 1,
        )

        shape_param = {"md": md, "nd": nd}

        # prepare auxilluary variables
        NI_arr, hier_mask_arr, hier_ind_arr = prepare_aux(
            md,
            nd,
            self.is_vertical,
            self.train_param["angleOffset"],
            self.train_param["deg"],
            self.train_param["Nneighbors"],
        )

        NI, hier_mask, hier_ind = (
            torch.from_numpy(NI_arr).to(self.device),
            torch.from_numpy(hier_mask_arr).to(self.device),
            torch.from_numpy(hier_ind_arr).to(self.device),
        )

        self.train_param["NI"] = NI
        self.train_param["hier_mask"] = hier_mask
        self.train_param["hier_mask"] = hier_mask

        # loop through all Z
        for idx_z in range(dim_z):
            print(f"processing {idx_z} / {dim_z} slice .")
            arr_O = img_handle.get_dask_image_data("YX", Z=idx_z, T=0, C=0).compute()
            if self.mask_name is None:
                arr_map = np.zeros(arr_O.shape)
            else:
                arr_map = mask_handle.get_dask_image_data(
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

            # rotate if it is vertical
            if not self._isVertical:
                for index in range(len(self.qr)):
                    locals()["X" + str(index)] = locals()["X" + str(index)].T
            for index in range(len(self.qr)):
                locals()["result" + str(index)][i] = locals()["X" + str(index)]
                locals()["mean" + str(index)][i] = np.mean(locals()["X" + str(index)])

        # glocal correction for the whole z-stack
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

        # gather the final result into a z-stack
        out_list = []
        for i in range(len(self.qr)):
            locals()["result" + str(i)] = np.clip(
                locals()["result" + str(i)], 0, 65535
            ).astype(np.uint16)
            out_list.append(locals()["result" + str(i)])

        return np.stack(out_list, axis=0)
