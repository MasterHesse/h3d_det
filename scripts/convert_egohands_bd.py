#!/usr/bin/env python
import argparse, pathlib, tqdm, cv2, numpy as np, shutil
CLS_ID = 1

def mask2bbox(msk):
    ys, xs = np.where(msk > 0)
    if xs.size == 0: return None
    x1,y1,x2,y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return x1, y1, x2-x1, y2-y1

def main(root, out_root):
    img_dir = pathlib.Path(root)/'images'
    out_img = pathlib.Path(out_root)/'images'/'train'
    out_lab = pathlib.Path(out_root)/'labels'/'train'
    out_img.mkdir(parents=True, exist_ok=True); out_lab.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm.tqdm(img_dir.rglob('*.[jp][pn]g'), desc='EgoHands_bd'):
        msk_path = pathlib.Path(str(img_path).replace('/images/', '/masks/')).with_suffix('.jpg')
        if not msk_path.exists(): continue
        img = cv2.imread(str(img_path));  h, w = img.shape[:2]
        msk = cv2.imread(str(msk_path), 0)
        bbox = mask2bbox(msk);           0
        if bbox is None: continue
        x,y,bw,bh = bbox; x_c,y_c = x+bw/2, y+bh/2
        line = f"{CLS_ID} {x_c/w:.6f} {y_c/h:.6f} {bw/w:.6f} {bh/h:.6f}\n"

        dst = out_img/img_path.name
        shutil.copy(img_path, dst)
        (out_lab/dst.with_suffix('.txt').name).write_text(line, encoding='utf-8')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--egohands_root', required=True)
    ap.add_argument('--out_root',      required=True)
    args = ap.parse_args()
    main(args.egohands_root, args.out_root)