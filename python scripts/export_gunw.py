import sys, os, h5py, numpy as np, rasterio
from rasterio.transform import from_origin

path = sys.argv[1]
WAVELENGTH_M = 0.2384   # approx NISAR L-band; verify in product spec
COH_MIN = 0.3           # rule of thumb, not official
os.makedirs("out", exist_ok=True)

def pick(f, name, prefer="unwrappedInterferogram"):
    hits = []
    def visit(n, o):
        if isinstance(o, h5py.Dataset) and n.split("/")[-1] == name:
            hits.append(n)
    f.visititems(visit)
    if not hits:
        raise SystemExit(f"Not found: {name}")
    best = [h for h in hits if prefer in h] or hits
    print(f"{name} -> {best[0]}")
    return best[0]

with h5py.File(path, "r") as f:
    phase = f[pick(f, "unwrappedPhase")][()].astype("float64")
    coh   = f[pick(f, "coherenceMagnitude")][()].astype("float64")
    cc    = f[pick(f, "connectedComponents")][()]
    x     = f[pick(f, "xCoordinates")][()]
    y     = f[pick(f, "yCoordinates")][()]
    epsg  = int(f[pick(f, "projection")][()])

assert phase.shape == coh.shape == cc.shape == (len(y), len(x)), "shape mismatch"

if y[1] > y[0]:   # make north-up
    phase, coh, cc, y = phase[::-1], coh[::-1], cc[::-1], y[::-1]
dx, dy = float(x[1]-x[0]), float(y[1]-y[0])      # dy negative
transform = from_origin(x[0]-dx/2, y[0]-dy/2, dx, -dy)

valid = np.isfinite(phase) & (phase != 0) & (coh >= COH_MIN) & (cc > 0)

disp = -phase * WAVELENGTH_M / (4*np.pi) * 100   # cm along line of sight
disp[~valid] = np.nan
disp -= np.nanmedian(disp)                       # relative: background = 0

fringes = np.angle(np.exp(1j*phase))             # wrapped, -pi..pi (rainbow look)
fringes[~valid] = np.nan
coh_out = np.where(np.isfinite(coh), coh, np.nan)

def save(name, arr):
    with rasterio.open(f"out/{name}", "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
        crs=f"EPSG:{epsg}", transform=transform, nodata=np.nan,
        compress="deflate") as dst:
        dst.write(arr.astype("float32"), 1)

save("displacement_cm.tif", disp)
save("fringes_rad.tif", fringes)
save("coherence.tif", coh_out)

print("EPSG:", epsg, "| valid pixels:", int(valid.sum()), "of", valid.size)
print("Displacement cm 2/50/98 percentiles:", np.nanpercentile(disp, [2, 50, 98]))