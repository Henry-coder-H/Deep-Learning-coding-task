import os
from pathlib import Path

PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
             "桂","琼","川","贵","云","藏","陕","甘","青","宁","新","警","学","O"]
ALPHABETS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ","")) + ["O"]
ADS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ","")) + list("0123456789") + ["O"]

def decode_ccpd_plate_from_name(img_name: str):
    stem = Path(img_name).stem
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    code = parts[4]
    idx = list(map(int, code.split("_")))
    text = PROVINCES[idx[0]] + ALPHABETS[idx[1]] + "".join(ADS[i] for i in idx[2:])
    return text.rstrip("O")

def build(split_txt, dataset_root, out_gt_txt):
    dataset_root = Path(dataset_root)
    out_gt_txt = Path(out_gt_txt)
    out_gt_txt.parent.mkdir(parents=True, exist_ok=True)

    chars = set()
    n = 0
    with open(split_txt, "r", encoding="utf-8") as f, open(out_gt_txt, "w", encoding="utf-8") as w:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            img_path = dataset_root / rel  # /datasets/ccpd_weather/xxx.jpg
            label = decode_ccpd_plate_from_name(Path(rel).name)
            if label is None:
                continue
            w.write(str(img_path) + "\t" + label + "\n")
            for ch in label:
                chars.add(ch)
            n += 1
    return n, chars

def write_dict(chars, dict_path):
    dict_path = Path(dict_path)
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    # 固定排序，保证可复现
    with open(dict_path, "w", encoding="utf-8") as f:
        for ch in sorted(chars):
            f.write(ch + "\n")

if __name__ == "__main__":
    DATASET_ROOT = "/home/zwm/datasets"  # 数据集位置
    TRAIN_SPLIT = "/home/zwm/datasets/splits/all_train.txt"  
    TEST_SPLIT  = "/home/zwm/datasets/splits/all_test.txt"   

    OUT_DIR = "train_data/rec/ccpd"
    n_tr, ch_tr = build(TRAIN_SPLIT, DATASET_ROOT, f"{OUT_DIR}/rec_gt_train.txt")
    n_te, ch_te = build(TEST_SPLIT,  DATASET_ROOT, f"{OUT_DIR}/rec_gt_test.txt")
    write_dict(ch_tr | ch_te, f"{OUT_DIR}/plate_dict.txt")

    print("train samples:", n_tr)
    print("test samples:", n_te)
    print("done:", OUT_DIR)