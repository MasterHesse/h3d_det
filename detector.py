#!/usr/bin/env python
"""
轻量 YOLOv11 检测器封装
用法:
    det = Detector(weights="weights/h3d_v11s_best.pt", conf=0.25)
    boxes = det(img)   # img: numpy BGR/HWC, torch.Tensor CHW, 或图片路径
返回:
    list[list]  # 每张图 [[x1,y1,x2,y2,conf,cls], ...]，坐标为 int
"""

from ultralytics import YOLO
import numpy as np, cv2, torch, pathlib

class Detector:
    def __init__(self, weights:str, conf:float=0.25, iou:float=0.5, device:str="auto"):
        self.model = YOLO(weights)
        self.model.fuse()
        self.model.overrides['conf'] = conf
        self.model.overrides['iou']  = iou
        self.device = device

    def _prepare(self, x):
        if isinstance(x, (str, pathlib.Path)):
            return [str(x)]
        if isinstance(x, np.ndarray):
            return [x]
        if torch.is_tensor(x):
            return [x]
        if isinstance(x, list):
            return x
        raise TypeError("Unsupported input type")

    @torch.no_grad()
    def __call__(self, img):
        results = self.model(self._prepare(img), device=self.device, verbose=False)
        outs = []
        for r in results:
            if r.boxes.is_empty:       # 无目标
                outs.append([])
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            outs.append(np.concatenate([xyxy, conf[:,None], cls[:,None]], axis=1).tolist())
        return outs

# ---------- demo ----------
if __name__ == "__main__":
    import glob, time
    det = Detector("yolov11s.pt")      # 先用官方权重试跑
    path = glob.glob("data/yolo/images/val/*.jpg")[0]
    t0 = time.time()
    boxes = det(path)
    print(f"Inference OK, {len(boxes[0])} objects, time {time.time()-t0:.3f}s")