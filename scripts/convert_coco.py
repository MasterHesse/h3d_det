#!/usr/bin/env python
"""
将 COCO person 标注转换为 YOLO txt
cls_id = 0  (person)
用法示例：
python scripts/convert_coco.py \
    --coco_json data/raw/coco2017/annotations/instances_val2017.json \
    --img_dir   data/raw/coco2017/val2017 \
    --out_root  data/yolo --split val
"""
import argparse, json, pathlib, tqdm, shutil

CLS_ID = 0          # person

def coco2yolo(box, w, h):
    x, y, bw, bh = box
    return (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h

def main(args):
    out_img = pathlib.Path(args.out_root) / "images" / args.split
    out_lab = pathlib.Path(args.out_root) / "labels" / args.split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    coco = json.load(open(args.coco_json))
    id2img = {img["id"]: img for img in coco["images"]}

    copied = set()   # 记录已复制图片名，防止重复 copy

    for ann in tqdm.tqdm(coco["annotations"], desc="COCO"):
        if ann["category_id"] != 1:       # 1 = person
            continue
        img_info = id2img[ann["image_id"]]
        w, h = img_info["width"], img_info["height"]
        x_c, y_c, bw, bh = coco2yolo(ann["bbox"], w, h)

        src_img = pathlib.Path(args.img_dir) / img_info["file_name"]
        dst_img = out_img / src_img.name
        if src_img.name not in copied:
            shutil.copy(src_img, dst_img)
            copied.add(src_img.name)

        lab_path = out_lab / dst_img.with_suffix(".txt").name
        with open(lab_path, "a", encoding="utf-8") as f:
            f.write(f"{CLS_ID} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", required=True)
    parser.add_argument("--img_dir",   required=True)
    parser.add_argument("--out_root",  required=True)
    parser.add_argument("--split",     choices=["train", "val"], default="train")
    main(parser.parse_args())