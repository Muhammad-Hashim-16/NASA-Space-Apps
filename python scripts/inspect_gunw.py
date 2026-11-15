import sys, h5py

path = sys.argv[1]
with h5py.File(path, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {name}  shape={obj.shape} dtype={obj.dtype}")
            if "identification" in name and obj.shape == ():
                print("     value:", obj[()])
        else:
            print(f"[group]   {name}")
    f.visititems(show)