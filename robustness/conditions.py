"""Experimental conditions and their application to one image."""
from dataclasses import asdict, dataclass
from typing import Optional

from .degradations import (NOISE_STREAM, PLACEMENT_STREAMS, PLACEMENTS, SpotConfig, bright_spots,
                           degradation_rng, gaussian_noise, noise_field, placement_regions, select_spot_centres,
                           spot_alpha, spot_candidates, spot_mass_fractions)

KINDS = ('clean', 'noise', 'spots')


@dataclass(frozen=True)
class Condition:
    kind: str                        # clean | noise | spots
    param: float = 0.0               # noise sigma (image range [0, 1]) or spot radius (px at model input size)
    placement: Optional[str] = None  # spots only: random | boundary | interior | background
    match: Optional[str] = None      # quality metric the param was calibrated to: ssim | psnr
    target: Optional[float] = None   # calibration target value
    level: Optional[int] = None      # index of the target in the calibration list

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'unknown condition kind {self.kind!r}')
        if (self.kind == 'spots') != (self.placement is not None):
            raise ValueError('placement must be given for spots and only for spots')
        if self.placement is not None and self.placement not in PLACEMENTS:
            raise ValueError(f'unknown placement {self.placement!r}')

    @property
    def name(self):
        if self.kind == 'clean':
            return 'clean'
        base = 'noise' if self.kind == 'noise' else f'spots-{self.placement}'
        if self.match is None:
            return f'{base}_p{self.param:.6g}'
        return f'{base}_{self.match}{self.target:.6g}'

    def as_row(self):
        row = asdict(self)
        row['condition'] = self.name
        return row


class ImageDegrader:
    """Applies conditions to one image, caching its placement regions, spot candidates and noise field.

    image01: (C, H, W) tensor in [0, 1]; lesion: (H, W) bool array at the same resolution;
    click_xy: prompt click (x, y) in pixels; split_id / index identify the image for seeding.
    """

    def __init__(self, image01, lesion, click_xy, split_id, index, seed, spot_cfg=SpotConfig()):
        self.image = image01
        self.lesion = lesion
        self.click_xy = (float(click_xy[0]), float(click_xy[1]))
        self.split_id = int(split_id)
        self.index = int(index)
        self.seed = int(seed)
        self.spot_cfg = spot_cfg
        self._regions = None
        self._candidates = {}
        self._noise = None

    @property
    def regions(self):
        if self._regions is None:
            self._regions = placement_regions(self.lesion, self.spot_cfg.band_px)
        return self._regions

    def _rng(self, stream):
        return degradation_rng(self.seed, self.split_id, self.index, stream)

    def candidates(self, placement):
        if placement not in self._candidates:
            rng = self._rng(PLACEMENT_STREAMS[placement])
            self._candidates[placement] = spot_candidates(self.regions[placement], rng, self.spot_cfg.max_candidates)
        return self._candidates[placement]

    def noise(self):
        if self._noise is None:
            self._noise = noise_field(tuple(self.image.shape), self._rng(NOISE_STREAM)).to(self.image.device)
        return self._noise

    def apply(self, condition):
        """Return (degraded image in [0, 1], info dict)."""
        if condition.kind == 'clean':
            return self.image, {}
        if condition.kind == 'noise':
            return gaussian_noise(self.image, condition.param, self.noise()), {}
        _, height, width = self.image.shape
        centres = select_spot_centres(self.candidates(condition.placement), self.click_xy, condition.param, self.spot_cfg)
        alpha = spot_alpha(centres, condition.param, height, width, device=self.image.device)
        info = {'n_spots': len(centres)}
        info.update(spot_mass_fractions(alpha, self.regions))
        return bright_spots(self.image, alpha), info
