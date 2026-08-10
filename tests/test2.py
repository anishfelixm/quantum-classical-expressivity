import numpy as np, medmnist, os
os.makedirs('data_cache', exist_ok=True) 
for ds in ["breastmnist","pneumoniamnist","bloodmnist","pathmnist"]:
    info = medmnist.INFO[ds]
    Cls = getattr(medmnist, info['python_class'])
    for split in ["train","val"]:
        d = Cls(split=split, download=True, root='data_cache')
        print(ds, split, "n_classes:", len(info['label']),
              "total:", len(d), "per-class:", np.bincount(np.array(d.labels).ravel()))
