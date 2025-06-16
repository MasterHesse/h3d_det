#!/usr/bin/env python
"""
百度飞桨 WIDER FACE label.txt → YOLO
支持格式：路径行 + 任意数量的 bbox 行（每行前 4 个数字为 x y w h）
cls_id = 2 (face)
"""

import argparse, pathlib, shutil, tqdm, cv2

CLS_ID = 2

def parse_label(txt_path):
    with open(txt_path) as f:
        img_rel, boxes = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ".jpg" in line or ".jpeg" in line or ".png" in line:  # 新图片路径
                if img_rel is not None:
                    yield img_rel, boxes
                img_rel, boxes = line, []
            else:  # bbox 行
                nums = line.split()
                if len(nums) >= 4:
                    x, y, w, h = map(float, nums[:4])
                    boxes.append([int(x), int(y), int(w), int(h)])
        # 最后一张
        if img_rel is not None:
            yield img_rel, boxes

def main(wider_root, out_root, split):
    split_dir = pathlib.Path(wider_root) / split
    img_root  = split_dir / "images"
    txt_path  = split_dir / "label.txt"

    out_img = pathlib.Path(out_root) / "images" / split
    out_lab = pathlib.Path(out_root) / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    for img_rel, boxes in tqdm.tqdm(parse_label(txt_path), desc=f"WIDER_bd {split}"):
        src_img = img_root / img_rel
        if not src_img.exists():
            # 个别数据路径大小写不一致可在此处理
            continue
        dst_img = out_img / src_img.name
        shutil.copy(src_img, dst_img)

        lab_path = out_lab / dst_img.with_suffix(".txt").name
        if boxes:
            im = cv2.imread(str(src_img))
            if im is None:
                continue
            h_img, w_img = im.shape[:2]
            with open(lab_path, "w", encoding="utf-8") as f:
                for x, y, w, h in boxes:
                    xc, yc = x + w/2, y + h/2
                    f.write(f"{CLS_ID} {xc/w_img:.6f} {yc/h_img:.6f} {w/w_img:.6f} {h/h_img:.6f}\n")
        else:
            lab_path.touch()  # 保持图片/标签一一对应

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wider_root", required=True, help="data/raw/widerface_bd")
    ap.add_argument("--out_root",   required=True, help="data/yolo")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    args = ap.parse_args()
    main(args.wider_root, args.out_root, args.split)