from collections import defaultdict
import glob
import os


def check_bbox_sizes(labels_dir):
    areas = defaultdict(list)
    for p in glob.glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True):
        with open(p) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id, w, h = int(parts[0]), float(parts[3]), float(parts[4])
                    areas[cls_id].append(w * h)
    return {c: sum(a) / len(a) for c, a in areas.items() if a}

origin = r"C:\Users\Workstation\PycharmProjects\ProgettoVEDI_DL\data\UNICT-VEDI_Dataset_bbox\Training"
for folder in os.listdir(origin):
    # Verifica se in train l'area media di certe classi è ~1.0
    print(folder)
    print(check_bbox_sizes(os.path.join(origin, folder)))