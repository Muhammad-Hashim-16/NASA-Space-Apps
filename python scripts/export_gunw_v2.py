import sys, os, h5py, numpy as np, rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
G = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/"
LAM = 0.2384                      # m, approx L-band
COH_MIN = 0.3
EPI_LON, EPI_LAT = 93.780, 32.381
EXCLUDE_KM = 100                  # ignore pixels this close to the epicenter when fitting
DEGREE = 1                        # 1 = plane
os.makedirs("out_v2", exist_ok=True)

with h5py.File(path, "r") as f:
    phase = f[G+"unwrappedPhase"][()].astype("float64")
    coh   = f[G+"coherenceMagnitude"][()].astype("float64")
    cc    = f[G+"connectedComponents"][()]
    x     = f[G+"xCoordinates"][()]
    y     = f[G+"yCoordinates"][()]
    epsg  = int(f[G+"projection"][()])

if y[1] > y[0]:
    phase, coh, cc, y = phase[::-1], coh[::-1], cc[::-1], y[::-1]
dx, dy = float(x[1]-x[0]), float(y[1]-y[0])
transform = from_origin(x[0]-dx/2, y[0]-dy/2, dx, -dy)

good = np.isfinite(phase) & (phase != 0) & (coh >= COH_MIN) & (cc > 0)

ex, ey = warp_transform("EPSG:4326", f"EPSG:{epsg}", [EPI_LON], [EPI_LAT])
xk = (x - ex[0]) / 1000.0         # km east of epicenter
yk = (y - ey[0]) / 1000.0         # km north of epicenter

# fit the ramp on a thinned grid, far-field only
s = 10
XX, YY = np.meshgrid(xk[::s], yk[::s])
fit = good[::s, ::s] & (np.hypot(XX, YY) > EXCLUDE_KM)
cols = [np.ones_like(XX), XX, YY]
if DEGREE >= 2:
    cols += [XX*XX, XX*YY, YY*YY]
A = np.column_stack([c[fit] for c in cols])
coef, *_ = np.linalg.lstsq(A, phase[::s, ::s][fit], rcond=None)

ramp = coef[0] + coef[1]*xk[None, :] + coef[2]*yk[:, None]
if DEGREE >= 2:
    ramp = ramp + coef[3]*(xk**2)[None, :] + coef[4]*np.outer(yk, xk) + coef[5]*(yk**2)[:, None]

res = phase - ramp
res[~good] = np.nan
cm = -res * LAM / (4*np.pi) * 100
fringes = np.angle(np.exp(1j*res))
coh_out = np.where(np.isfinite(coh), coh, np.nan)

def save(name, arr):
    with rasterio.open(f"out_v2/{name}", "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
        crs=f"EPSG:{epsg}", transform=transform, nodata=np.nan,
        compress="deflate") as dst:
        dst.write(arr.astype("float32"), 1)

save("displacement_cm.tif", cm)
save("fringes_rad.tif", fringes)
save("coherence.tif", coh_out)

far = np.hypot(*np.meshgrid(xk, yk)) > EXCLUDE_KM if False else None
print("EPSG:", epsg, "| valid pixels:", int(good.sum()))
print("cm percentiles 1/2/50/98/99:", np.nanpercentile(cm, [1, 2, 50, 98, 99]))
print("cm min/max:", np.nanmin(cm), np.nanmax(cm))
print("far-field noise (std of cm, thinned):",
      np.nanstd(np.where(fit, cm[::s, ::s], np.nan)))

lim = np.nanpercentile(np.abs(cm[np.isfinite(cm)]), 98)
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
im = ax[0].imshow(cm, cmap="RdBu_r", vmin=-lim, vmax=lim); ax[0].set_title("displacement cm, ramp removed")
plt.colorbar(im, ax=ax[0], fraction=0.04)
im = ax[1].imshow(fringes, cmap="turbo", vmin=-np.pi, vmax=np.pi); ax[1].set_title("fringes, ramp removed")
plt.colorbar(im, ax=ax[1], fraction=0.04)
plt.tight_layout(); plt.savefig("quicklook2.png", dpi=80)
print("saved quicklook2.png")