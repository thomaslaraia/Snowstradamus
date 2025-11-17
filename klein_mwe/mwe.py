import rioxarray
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from rasterio.enums import Resampling
import os

# ---------------- SETTINGS ----------------
SR_SCALE = 0.0000275
SR_OFFSET = -0.2

SCENE_PATH = "Landsat8_klein100_LC08_L2SP_037023_20210216_20210301_02_T1.tif"
MASK_FOLDER = "."
CAM = "klein"   # used to find the corresponding LandCover file

# ---------------- HELPERS ----------------
def find_landcover_mask(folder, cam):
    for f in os.listdir(folder):
        if cam in f and "LandCover" in f:
            return os.path.join(folder, f)
    raise FileNotFoundError("No LandCover file found for camera:", cam)

def index(b1, b2):
    denom = b1 + b2
    return xr.where(denom != 0, (b1 - b2) / denom, np.nan)

def finite_mean(da):
    return float(da.where(np.isfinite(da)).mean().values)

def normalize_rgb(img):
    # img: (y, x, band)
    p2  = np.nanpercentile(img, 2,  axis=(0, 1))
    p98 = np.nanpercentile(img, 98, axis=(0, 1))
    return np.clip((img - p2) / (p98 - p2), 0, 1)

# ---------------- LOAD + SCALE L8 SR ----------------
landsat = rioxarray.open_rasterio(SCENE_PATH, masked=True).squeeze()
landsat.name = "landsat"

# SR_B1..SR_B7 are first 7 bands in this stack
sr = landsat.isel(band=slice(0, 7)) * SR_SCALE + SR_OFFSET
sr = sr.clip(0, 1)
landsat.loc[dict(band=landsat.band.values[:7])] = sr

print(
    "Scaled SR (B1–B7) mean/min/max:",
    float(sr.mean().values),
    float(sr.min().values),
    float(sr.max().values),
)

# ---------------- LOAD LANDCOVER + MASK ----------------
lc_path = find_landcover_mask(MASK_FOLDER, CAM)
landcover = rioxarray.open_rasterio(lc_path, masked=True).squeeze()
landcover.name = "corine_landcover"

# Reproject to Landsat grid (nearest is fine for classes)
landcover = landcover.rio.reproject_match(landsat, resampling=Resampling.nearest)

# Forest land-cover mask (111–126 as in your code)
lc_mask = (landcover >= 111) & (landcover <= 126)

# ---------------- EXTRACT CORE BANDS (LS8) ----------------
# ('SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7', ...)
blue  = landsat.isel(band=1)  # SR_B2
green = landsat.isel(band=2)  # SR_B3
red   = landsat.isel(band=3)  # SR_B4
nir   = landsat.isel(band=4)  # SR_B5
swir1 = landsat.isel(band=5)  # SR_B6 (1.6 µm)

ndsi = index(green, swir1)
ndvi = index(nir, red)

# ---------------- DOZIER ----------------
cond1 = (ndsi > 0.1) & (ndsi < 0.4)
cond2 = (ndsi >= 0.4) & (nir > 0.11)
dozier = xr.where(cond1 | cond2, 1, 0).where(~np.isnan(ndsi))

# ---------------- KLEIN ----------------
klein = xr.zeros_like(ndsi)
klein = xr.where((ndsi >= 0.4) & (nir > 0.11), 1, klein)

ndsi_vals, ndvi_vals = ndsi.values, ndvi.values
mask_poly = np.zeros(ndsi.shape, dtype=bool)

region_coords = [
    (0.4, 1),
    (0.33, 0.9),
    (0.26, 0.75),
    (0.2, 0.59),
    (0.1, 0.24),
    (0.2, 0.2),
    (0.4, 0.1),
]
polygon = Polygon(region_coords)
for i in range(ndsi.shape[0]):
    for j in range(ndsi.shape[1]):
        x = ndsi_vals[i, j]
        y = ndvi_vals[i, j]
        if np.isfinite(x) and np.isfinite(y):
            if polygon.contains(Point(x, y)):
                mask_poly[i, j] = True

klein = xr.where(
    xr.DataArray(mask_poly, dims=ndsi.dims, coords=ndsi.coords),
    1,
    klein,
)

# dark-green veto
klein = xr.where(green <= 0.1, 0, klein)
klein = klein.where(~np.isnan(ndsi) & ~np.isnan(ndvi))

# ---------------- SALOMONSON & APPEL ----------------
fsc = 0.06 + 1.21 * ndsi
fsc = fsc.clip(0, 1).where(~np.isnan(ndsi))

# ---------------- APPLY LANDCOVER MASK ----------------
dozier_masked = dozier.where(lc_mask)
klein_masked  = klein.where(lc_mask)
fsc_masked    = fsc.where(lc_mask)

print("Dozier mean (forest only):", finite_mean(dozier_masked))
print("Klein mean (forest only): ", finite_mean(klein_masked))
print("Salomonson mean FSC (forest only):", finite_mean(fsc_masked))

# ---------------- RGB + PLOTS ----------------
rgb = xr.concat([red, green, blue], dim="band").transpose("y", "x", "band")
rgb_np = normalize_rgb(rgb.values)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(rgb_np)
axes[0].set_title("RGB Composite")
axes[0].axis("off")

dozier_masked.plot(ax=axes[1])
axes[1].set_title("Dozier (LandCover-masked)")
axes[1].axis("off")

klein_masked.plot(ax=axes[2])
axes[2].set_title("Klein (LandCover-masked)")
axes[2].axis("off")

plt.tight_layout()
plt.show()
