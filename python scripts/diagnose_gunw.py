import sys, h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
G = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/"
P = "science/LSAR/GUNW/metadata/processingInformation/parameters/geocoding/"
LAM = 0.2384  # m, approx L-band

with h5py.File(path, "r") as f:
    phase = f[G+"unwrappedPhase"][()].astype("float64")
    coh   = f[G+"coherenceMagnitude"][()].astype("float64")
    cc    = f[G+"connectedComponents"][()]
    iono  = f[G+"ionospherePhaseScreen"][()].astype("float64")
    for k in ["rangeIonosphericCorrectionApplied", "azimuthIonosphericCorrectionApplied",
              "hydrostaticTroposphericCorrectionApplied", "wetTroposphericCorrectionApplied"]:
        print(k, "=", f[P+k][()])

finite = np.isfinite(phase) & (phase != 0)
print("finite phase fraction of grid:", round(finite.mean(), 3))
print("coherence >= 0.3 (of finite):", round((coh[finite] >= 0.3).mean(), 3))
labels, counts = np.unique(cc[finite], return_counts=True)
for l, c in sorted(zip(labels, counts), key=lambda t: -t[1])[:8]:
    print("component", l, ":", c, "pixels", f"({100*c/finite.sum():.1f}%)")
print("raw phase rad  2/50/98:", np.nanpercentile(phase[finite], [2, 50, 98]))
fi = np.isfinite(iono)
print("iono screen rad 2/50/98:", np.nanpercentile(iono[fi], [2, 50, 98]))

good = finite & (coh >= 0.3) & (cc > 0)
def cm(p): return np.where(good, -p * LAM / (4*np.pi) * 100, np.nan)
raw_cm  = cm(phase)
corr_cm = cm(phase - iono)
raw_cm  -= np.nanmedian(raw_cm)
corr_cm -= np.nanmedian(corr_cm)

fig, ax = plt.subplots(2, 3, figsize=(20, 12))
def show(a, arr, title, cmap, lim=None):
    v = lim or np.nanpercentile(arr[np.isfinite(arr)], [2, 98])
    im = a.imshow(arr, cmap=cmap, vmin=v[0], vmax=v[1]); a.set_title(title)
    plt.colorbar(im, ax=a, fraction=0.04)
show(ax[0,0], np.where(good, coh, np.nan), "coherence (masked)", "gray", (0, 1))
show(ax[0,1], np.where(finite, cc.astype(float), np.nan), "connected components", "tab20")
show(ax[0,2], np.where(fi, iono, np.nan), "ionosphere phase screen (rad)", "RdBu_r")
show(ax[1,0], raw_cm,  "RAW displacement cm", "RdBu_r")
show(ax[1,1], corr_cm, "iono-corrected displacement cm", "RdBu_r")
show(ax[1,2], np.where(good, np.angle(np.exp(1j*(phase - iono))), np.nan),
     "fringes after iono (rad)", "turbo", (-np.pi, np.pi))
plt.tight_layout(); plt.savefig("quicklook.png", dpi=80)
print("saved quicklook.png")