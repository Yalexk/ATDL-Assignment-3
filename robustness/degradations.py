"""Diffuse (Gaussian noise) and concentrated (bright spot) degradations.

Images are float tensors in [0, 1] with shape (C, H, W). All randomness is drawn from
NumPy generators keyed by (seed, split, image index, stream), so a given image gets the
same noise field / spot candidates on every run, device and DataLoader worker layout, and
the strength parameter (sigma or radius) is the only thing that changes between levels.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

PLACEMENTS = ('random', 'boundary', 'interior', 'background')
# Stream 0 is the noise field; each placement has its own spot-candidate stream.
NOISE_STREAM = 0
PLACEMENT_STREAMS = {name: i + 1 for i, name in enumerate(PLACEMENTS)}


@dataclass(frozen=True)
class SpotConfig:
    count: int = 5             # K blobs per image
    band_px: float = 10.0      # half-width of the lesion boundary band
    click_margin: float = 3.0  # centres stay >= click_margin * radius from the prompt click
    min_separation: float = 2.0  # preferred centre spacing, in radii (relaxed if impossible)
    max_candidates: int = 20000  # candidate centres kept per image and placement


def degradation_rng(seed, split_id, index, stream):
    return np.random.default_rng([seed, split_id, index, stream])


def noise_field(shape, rng):
    """Standard-normal field; the degraded image is clip(x + sigma * z)."""
    return torch.from_numpy(rng.standard_normal(shape, dtype=np.float32))


def gaussian_noise(x, sigma, z):
    return (x + sigma * z.to(device=x.device, dtype=x.dtype)).clamp(0.0, 1.0)


def placement_regions(lesion, band_px):
    """Boolean region masks (H, W) for spot centres, from a binary lesion mask.

    boundary: within band_px of the lesion edge on either side;
    interior / background: inside / outside the lesion and farther than band_px from the edge.
    """
    lesion = np.asarray(lesion, dtype=bool)
    everywhere = np.ones_like(lesion)
    if not lesion.any() or lesion.all():
        # No edge exists, so there is no boundary band.
        nothing = np.zeros_like(lesion)
        return {'random': everywhere, 'boundary': nothing,
                'interior': lesion.copy(), 'background': ~lesion}
    dist_inside = ndimage.distance_transform_edt(lesion)    # lesion pixel -> nearest background pixel
    dist_outside = ndimage.distance_transform_edt(~lesion)  # background pixel -> nearest lesion pixel
    boundary = (lesion & (dist_inside <= band_px)) | (~lesion & (dist_outside <= band_px))
    return {
        'random': everywhere,
        'boundary': boundary,
        'interior': lesion & (dist_inside > band_px),
        'background': ~lesion & (dist_outside > band_px),
    }


def spot_candidates(region, rng, max_candidates):
    """Random, radius-independent ordering of candidate centres as (rows, cols) int arrays."""
    flat = np.flatnonzero(region)
    if len(flat) > max_candidates:
        flat = rng.choice(flat, size=max_candidates, replace=False)
    else:
        flat = rng.permutation(flat)
    rows, cols = np.divmod(flat, region.shape[1])
    return rows, cols


def select_spot_centres(candidates, click_xy, radius, cfg):
    """Pick up to cfg.count centres (row, col) in candidate order.

    Centres closer than click_margin * radius to the click are never used. Centres are first
    chosen at least min_separation * radius apart; if the region is too small for that, the
    remaining slots are filled in candidate order without the spacing rule.
    """
    rows, cols = candidates
    click_x, click_y = float(click_xy[0]), float(click_xy[1])
    allowed = (rows - click_y) ** 2 + (cols - click_x) ** 2 >= (cfg.click_margin * radius) ** 2
    rows, cols = rows[allowed], cols[allowed]

    chosen = []
    min_sq = (cfg.min_separation * radius) ** 2
    for r, c in zip(rows, cols):
        if all((r - cr) ** 2 + (c - cc) ** 2 >= min_sq for cr, cc in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) == cfg.count:
                return chosen
    taken = set(chosen)
    for r, c in zip(rows, cols):
        if len(chosen) == cfg.count:
            break
        if (int(r), int(c)) not in taken:
            chosen.append((int(r), int(c)))
    return chosen


def spot_alpha(centres, radius, height, width, device=None, dtype=torch.float32):
    """Blend weight min(1, sum_k G_k), with G_k an isotropic Gaussian of std `radius` peaking at 1."""
    alpha = torch.zeros(height, width, device=device, dtype=dtype)
    if not centres:
        return alpha
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    two_r2 = 2.0 * float(radius) ** 2
    for row, col in centres:
        alpha += torch.outer(torch.exp(-(ys - row) ** 2 / two_r2), torch.exp(-(xs - col) ** 2 / two_r2))
    return alpha.clamp_(max=1.0)


def bright_spots(x, alpha):
    """Blend toward white: x' = x + (1 - x) * alpha."""
    return x + (1.0 - x) * alpha.to(device=x.device, dtype=x.dtype)


def spot_mass_fractions(alpha, regions):
    """Share of the total blend weight that falls in each placement region (excluding 'random')."""
    total = float(alpha.sum())
    out = {}
    for name in ('boundary', 'interior', 'background'):
        mask = torch.from_numpy(regions[name]).to(alpha.device)
        out[f'spot_frac_{name}'] = float(alpha[mask].sum()) / total if total > 0 else float('nan')
    out['spot_area_frac'] = float((alpha > 0.5).float().mean())
    return out
