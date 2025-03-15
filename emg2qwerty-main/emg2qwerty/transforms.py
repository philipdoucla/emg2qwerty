# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram16:
    """
    Creates log10-scaled spectrogram from an EMG signal, then aggregates the
    frequency axis into 16 log-spaced bins.
    
    Args:
        n_fft (int): Size of FFT; produces n_fft//2+1 linear bins (default: 64).
        hop_length (int): Number of samples to stride between STFT windows (default: 16).
        sample_rate (int): Sampling rate in Hz (default: 2000).
        f_min (float): Lower frequency bound for log bins (default: 20.0 Hz).
        f_max (float): Upper frequency bound for log bins (default: 1000.0 Hz).
    """
    n_fft: int = 64
    hop_length: int = 16
    sample_rate: int = 2000
    f_min: float = 20.0
    f_max: float = 1000.0

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            center=False,
        )
        # Compute the linear frequency bin values.
        n_lin = self.n_fft // 2 + 1
        freqs = torch.linspace(0, self.sample_rate / 2, n_lin)
        # Create 17 boundaries for 16 bins (log-spaced between f_min and f_max).
        boundaries = torch.logspace(torch.log10(torch.tensor(self.f_min)),
                                    torch.log10(torch.tensor(self.f_max)),
                                    steps=17)
        # Map boundaries to the indices of the linear bins.
        self.bin_edges = torch.bucketize(boundaries, freqs).tolist()
        # Ensure proper ordering (e.g. if two boundaries map to the same index).
        for i in range(1, len(self.bin_edges)):
            if self.bin_edges[i] <= self.bin_edges[i-1]:
                self.bin_edges[i] = self.bin_edges[i-1] + 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)         # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)          # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)    # (..., C, freq, T)
        logspec = logspec.movedim(-1, 0)      # (T, ..., C, freq)
        # Aggregate frequency bins into 16 log-spaced bins.
        bins = []
        start = self.bin_edges[0]
        for end in self.bin_edges[1:]:
            # Sum over the selected linear bins.
            bins.append(logspec[..., start:end].sum(dim=-1, keepdim=True))
            start = end
        out = torch.cat(bins, dim=-1)        # (T, ..., C, 16)
        return out


@dataclass
class LogSpectrogram6:
    """
    Creates log10-scaled spectrogram from an EMG signal, then aggregates the
    frequency axis into 6 log-spaced bins.
    
    Args:
        n_fft (int): Size of FFT; produces n_fft//2+1 linear bins (default: 64).
        hop_length (int): Number of samples to stride between STFT windows (default: 16).
        sample_rate (int): Sampling rate in Hz (default: 2000).
        f_min (float): Lower frequency bound for log bins (default: 20.0 Hz).
        f_max (float): Upper frequency bound for log bins (default: 1000.0 Hz).
    """
    n_fft: int = 64
    hop_length: int = 16
    sample_rate: int = 2000
    f_min: float = 20.0
    f_max: float = 1000.0

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            center=False,
        )
        # Compute the linear frequency bin values.
        n_lin = self.n_fft // 2 + 1
        freqs = torch.linspace(0, self.sample_rate / 2, n_lin)
        # Create 7 boundaries for 6 bins (log-spaced between f_min and f_max).
        boundaries = torch.logspace(torch.log10(torch.tensor(self.f_min)),
                                    torch.log10(torch.tensor(self.f_max)),
                                    steps=7)
        # Map boundaries to the indices of the linear bins.
        self.bin_edges = torch.bucketize(boundaries, freqs).tolist()
        # Ensure proper ordering (e.g. if two boundaries map to the same index).
        for i in range(1, len(self.bin_edges)):
            if self.bin_edges[i] <= self.bin_edges[i-1]:
                self.bin_edges[i] = self.bin_edges[i-1] + 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)         # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)          # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)    # (..., C, freq, T)
        logspec = logspec.movedim(-1, 0)      # (T, ..., C, freq)
        # Aggregate frequency bins into 6 log-spaced bins.
        bins = []
        start = self.bin_edges[0]
        for end in self.bin_edges[1:]:
            # Sum over the selected linear bins.
            bins.append(logspec[..., start:end].sum(dim=-1, keepdim=True))
            start = end
        out = torch.cat(bins, dim=-1)        # (T, ..., C, 6)
        return out


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)

@dataclass
class RandomCropPercentage:
    """Applies a random crop along the time dimension (T) of the input tensor
    based on a given percentage of the total length of the time dimension.

    Args:
        crop_percentage (float): The percentage of the input signal to crop (0.0 - 100.0).
        time_dim (int): The dimension along which to apply the crop. (default: 0)
    """

    crop_percentage: float
    time_dim: int = 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert 0 <= self.crop_percentage <= 100, "crop_percentage must be between 0 and 100"
        
        time_size = tensor.shape[self.time_dim]

        crop_size = int(time_size * (self.crop_percentage / 100))

        crop_size = max(1, crop_size)
        
        assert time_size >= crop_size, (
            f"Input tensor must have at least {crop_size} elements in the time dimension"
        )

        max_start_idx = time_size - crop_size
        start_idx = np.random.randint(0, max_start_idx + 1)

        slices = [slice(None)] * tensor.ndimension()  # Create a list of full slices
        slices[self.time_dim] = slice(start_idx, start_idx + crop_size)  # Update the time dimension slice

        return tensor[tuple(slices)]
