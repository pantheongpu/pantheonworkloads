"""Vision payloads of the ort-* workloads: pretrained ONNX vision models (OpenCV Zoo, ONNX model zoo) on onnxruntime.

Loaded by tools/ort_tasks.py (`register(ort_tasks_module)` returns the task table); not run directly.
Tasks: yunet (face detection), sface (face embedding), pphumanseg (person segmentation), nanodet and
yolox (COCO object detection), crnn (English word recognition), shufflenet, efficientnet (ImageNet
classification, ONNX zoo test tensors), ssdmobilenet (COCO detection, built-in NMS).

Pre- and post-processing is re-written here from the models' documented usage (opencv_zoo demos and
ONNX zoo READMEs), in numpy and OpenCV image functions only (OpenCV's dnn module is never used, so the
same onnxruntime session runs on every target). Test images are the scikit-image sample images
(public domain / CC0), fetched at a pinned commit; the text image for CRNN is drawn in code.
"""
import os

# --------------------------------------------------------------------------- pure helpers

COCO80 = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
          "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
          "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
          "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
          "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote "
          "keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear "
          "hair_drier toothbrush").split()

# 5-point template of 112x112 aligned faces (ArcFace convention, as used by SFace): eyes, nose, mouth corners.
ARCFACE_112 = ((38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041))

# YuNet landmarks (x, y in astronaut.png pixels) of the one face in the 512x512 astronaut image, from
# face_detection_yunet_2023mar.onnx on onnxruntime CPU. SFace is fed these fixed points so that its workload
# does not depend on a second model's output: right eye, left eye, nose, right and left mouth corner.
ASTRONAUT_FACE_LANDMARKS = ((201.5, 105.0), (245.2, 104.9), (220.7, 127.1), (203.9, 145.2), (242.4, 145.0))

CRNN_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"   # CRNN_EN: index 0 is the CTC blank, i >= 1 is CRNN_CHARSET[i - 1]
CRNN_WORDS = ("hello", "world", "pantheon", "gpu2024", "street", "model")


def iou_xyxy(a, b):
    """Intersection over union of two [x1, y1, x2, y2] boxes (0 when either is empty)."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_threshold, classes=None):
    """Greedy non-maximum suppression. boxes: [x1, y1, x2, y2] rows; classes (optional): boxes only
    suppress boxes of the same class. Returns kept indices, best score first (ties: lower index)."""
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    keep = []
    for i in order:
        if all((classes is not None and classes[i] != classes[j]) or iou_xyxy(boxes[i], boxes[j]) <= iou_threshold
               for j in keep):
            keep.append(i)
    return keep


def similarity_transform(src, dst):
    """2x3 matrix of the least-squares similarity transform (rotation, uniform scale, translation)
    taking the points `src` onto `dst` (Umeyama 1991, closed form, deterministic)."""
    import numpy as np
    src, dst = np.asarray(src, dtype="float64"), np.asarray(dst, dtype="float64")
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    cov = d0.T @ s0 / len(src)
    u, sv, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    r = u @ sign @ vt
    scale = (sv * np.diag(sign)).sum() / s0.var(0).sum()
    return np.hstack([scale * r, (md - scale * r @ ms)[:, None]])


def align_face(img, landmarks):
    """The 112x112 ArcFace-aligned crop of `img` given five landmarks (eyes, nose, mouth corners)."""
    import cv2
    m = similarity_transform(landmarks, ARCFACE_112)
    return cv2.warpAffine(img, m, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def cosine(a, b):
    import numpy as np
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def softmax(x, axis=-1):
    import numpy as np
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def ctc_greedy(probs, charset=CRNN_CHARSET):
    """Greedy CTC decoding of per-step probabilities [T, n_classes] (class 0 = blank): argmax per step,
    repeated characters collapsed, blanks dropped. Returns (text, mean probability of the kept steps)."""
    import numpy as np
    ids = probs.argmax(axis=1)
    text, kept, prev = [], [], 0
    for t, c in enumerate(ids):
        if c != 0 and c != prev:
            text.append(charset[c - 1])
            kept.append(float(probs[t, c]))
        prev = c
    return "".join(text), (sum(kept) / len(kept) if kept else 0.0)


def letterbox_topleft(img, size, pad_value):
    """Resize keeping the aspect ratio so the image fits `size` x `size`, paste at the top left on a
    canvas of `pad_value`. Returns (canvas float32 HWC, ratio)."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = int(h * ratio), int(w * ratio)
    canvas = np.full((size, size, 3), pad_value, dtype="float32")
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR).astype("float32")
    return canvas, ratio


def tf91_name(class_id):
    """Name of an id of the 91-id COCO label map used by TensorFlow detection models (ids 1..90 with gaps)."""
    gaps = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83)
    if class_id < 1 or class_id > 90 or class_id in gaps:
        return f"class{class_id}"
    return COCO80[class_id - 1 - sum(g < class_id for g in gaps)]


def r1(values, nd=1):
    return [round(float(v), nd) for v in values]


# --------------------------------------------------------------------------- the tasks

def register(T):
    """Build the task table with the helpers (assets, session, med, Skip) of the ort_tasks module `T`."""
    import time

    def imread(name):
        import cv2
        (path,) = T.assets(name)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)   # BGR uint8
        if img is None:
            raise T.Skip(f"cannot decode {name}")
        return img

    def to_chw(x):
        return x.astype("float32").transpose(2, 0, 1)[None]

    def finish(run, bench, n_per_run=1):
        """Benchmark metrics for a zero-argument callable `run` (one model call): warm-up then the median."""
        if not bench:
            return {}
        for _ in range(5):
            run()
        times = []
        for _ in range(int(os.environ.get("PW_ORT_REPEATS", "30"))):
            t = time.perf_counter()
            run()
            times.append(time.perf_counter() - t)
        return {"images_per_s": n_per_run / T.med(times), "latency_ms": T.med(times) * 1000}

    # ---- YuNet face detection (OpenCV Zoo, MIT) ----
    def yunet_detect(sess, img, conf=0.6, nms_thr=0.3):
        import numpy as np
        h, w = img.shape[:2]
        import cv2
        x = to_chw(cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR))   # BGR, 0..255, no normalisation
        outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {sess.get_inputs()[0].name: x})))
        boxes, scores, lms = [], [], []
        for st in (8, 16, 32):
            n = 640 // st
            idx = np.arange(n * n)
            r, c = idx // n, idx % n
            cls = np.clip(outs[f"cls_{st}"][0, :, 0], 0, 1)
            obj = np.clip(outs[f"obj_{st}"][0, :, 0], 0, 1)
            bb, kp = outs[f"bbox_{st}"][0], outs[f"kps_{st}"][0]
            cx, cy = (c + bb[:, 0]) * st, (r + bb[:, 1]) * st
            bw, bh = np.exp(bb[:, 2]) * st, np.exp(bb[:, 3]) * st
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1))
            scores.append(np.sqrt(cls * obj))
            lms.append(np.stack([np.stack([(kp[:, 2 * k] + c) * st, (kp[:, 2 * k + 1] + r) * st], 1) for k in range(5)], 1))
        boxes, scores, lms = np.concatenate(boxes), np.concatenate(scores), np.concatenate(lms)
        sel = np.nonzero(scores >= conf)[0]
        keep = [sel[i] for i in nms(boxes[sel].tolist(), scores[sel].tolist(), nms_thr)]
        sx, sy = w / 640, h / 640
        faces = []
        for i in keep:
            b = boxes[i] * [sx, sy, sx, sy]
            faces.append({"box": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": float(scores[i]),
                          "landmarks": (lms[i] * [sx, sy]).ravel().tolist()})
        return faces

    def task_yunet(bench):
        sess, prov = T.session(T.assets("face_detection_yunet_2023mar.onnx")[0])
        out = {}
        for im in ("astronaut", "chelsea"):
            faces = yunet_detect(sess, imread(f"{im}.png"))
            out[f"n_faces_{im}"] = len(faces)
            if faces:
                f = faces[0]
                out[f"box_{im}"], out[f"score_{im}"], out[f"landmarks_{im}"] = r1(f["box"]), round(f["score"], 3), r1(f["landmarks"])
        img = imread("astronaut.png")
        metrics = finish(lambda: yunet_detect(sess, img), bench)
        return out, f"{out['n_faces_astronaut']} face in the astronaut image, {out['n_faces_chelsea']} in the cat image; {prov}", metrics

    # ---- SFace face embedding (OpenCV Zoo, Apache-2.0) ----
    def sface_embed(sess, aligned_bgr):
        import cv2
        x = to_chw(cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB))   # RGB, 0..255, no normalisation
        (e,) = sess.run(None, {sess.get_inputs()[0].name: x})
        return e[0]

    def task_sface(bench):
        import cv2
        import numpy as np
        sess, prov = T.session(T.assets("face_recognition_sface_2021dec.onnx")[0])
        img = imread("astronaut.png")
        face = align_face(img, ASTRONAUT_FACE_LANDMARKS)
        e = sface_embed(sess, face)
        e_flip = sface_embed(sess, face[:, ::-1].copy())                       # the same face mirrored
        e_cat = sface_embed(sess, cv2.resize(imread("chelsea.png"), (112, 112), interpolation=cv2.INTER_AREA))   # not a face
        out = {"embedding_dim": int(e.shape[0]), "embedding_norm": round(float(np.linalg.norm(e)), 3),
               "embedding_head": [round(float(v), 3) for v in e[:8]],
               "cosine_face_vs_mirrored": round(cosine(e, e_flip), 3), "cosine_face_vs_cat": round(cosine(e, e_cat), 3)}
        metrics = finish(lambda: sface_embed(sess, face), bench)
        return out, f"128-d embedding; cosine to the mirrored face {out['cosine_face_vs_mirrored']}, to a cat image {out['cosine_face_vs_cat']}; {prov}", metrics

    # ---- PP-HumanSeg (OpenCV Zoo, Apache-2.0) ----
    def humanseg_mask(sess, img):
        """Foreground mask (192x192, 1 = person) from a BGR image."""
        import cv2
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (192, 192), interpolation=cv2.INTER_LINEAR)
        x = (x.astype("float32") / 255.0 - 0.5) / 0.5
        (y,) = sess.run(None, {sess.get_inputs()[0].name: to_chw(x)})
        return y[0].argmax(axis=0)

    def task_pphumanseg(bench):
        sess, prov = T.session(T.assets("human_segmentation_pphumanseg_2023mar.onnx")[0])
        out = {}
        for im in ("astronaut", "chelsea"):
            m = humanseg_mask(sess, imread(f"{im}.png"))
            out[f"foreground_fraction_{im}"] = round(float(m.mean()), 3)
            if im == "astronaut":
                out["foreground_grid_8x8_astronaut"] = [round(float(v), 2) for v in m.reshape(8, 24, 8, 24).mean(axis=(1, 3)).ravel()]
        img = imread("astronaut.png")
        metrics = finish(lambda: humanseg_mask(sess, img), bench)
        return out, f"foreground covers {out['foreground_fraction_astronaut']:.0%} of the astronaut image, {out['foreground_fraction_chelsea']:.0%} of the cat image; {prov}", metrics

    # ---- detectors: shared output formatting ----
    def det_output(per_image, ndigits=1):
        out = {}
        for im, dets in per_image.items():   # dets: [(x1, y1, x2, y2, score, class_id)] best first
            out[f"n_{im}"] = len(dets)
            out[f"classes_{im}"] = " ".join(str(d[5]) for d in dets)
            out[f"scores_{im}"] = [round(float(d[4]), 3) for d in dets]
            out[f"boxes_{im}"] = [round(float(v), ndigits) for d in dets for v in d[:4]]
        return out

    def det_detail(per_image, prov, name=lambda c: COCO80[c]):
        parts = [f"{im}: " + (", ".join(name(d[5]) for d in dets) or "nothing") for im, dets in per_image.items()]
        return "; ".join(parts) + f"; {prov}"

    DET_IMAGES = ("astronaut", "chelsea", "coffee")

    # ---- NanoDet-Plus-m 416 (OpenCV Zoo, Apache-2.0) ----
    def nanodet_detect(sess, img, conf=0.35, nms_thr=0.6):
        import cv2
        import numpy as np
        h, w = img.shape[:2]
        # as the zoo demo: RGB, aspect-preserving resize (INTER_AREA) centred on a black 416x416 canvas
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ratio = min(416 / h, 416 / w)
        nh, nw = int(round(h * ratio)), int(round(w * ratio))
        top, left = (416 - nh) // 2, (416 - nw) // 2
        canvas = np.zeros((416, 416, 3), dtype="float32")
        canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        mean = np.array([103.53, 116.28, 123.675], dtype="float32")
        std = np.array([57.375, 57.12, 58.395], dtype="float32")
        outs = sess.run(None, {sess.get_inputs()[0].name: to_chw((canvas - mean) / std)})
        cls_out = sorted((o for o in outs if o.shape[-1] == 80), key=lambda o: -o.shape[1])   # strides 8, 16, 32, 64
        reg_out = sorted((o for o in outs if o.shape[-1] == 32), key=lambda o: -o.shape[1])
        boxes, scores = [], []
        for stride, cls, reg in zip((8, 16, 32, 64), cls_out, reg_out):
            n = 416 // stride
            idx = np.arange(n * n)
            ax = (idx % n) * stride + 0.5 * (stride - 1)
            ay = (idx // n) * stride + 0.5 * (stride - 1)
            dist = softmax(reg[0].reshape(-1, 4, 8), axis=-1) @ np.arange(8) * stride   # distribution focal loss, reg_max 7
            boxes.append(np.stack([np.clip(ax - dist[:, 0], 0, 416), np.clip(ay - dist[:, 1], 0, 416),
                                   np.clip(ax + dist[:, 2], 0, 416), np.clip(ay + dist[:, 3], 0, 416)], 1))
            scores.append(cls[0])
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        cid, conf_v = scores.argmax(1), scores.max(1)
        sel = np.nonzero(conf_v >= conf)[0]
        keep = [sel[i] for i in nms(boxes[sel].tolist(), conf_v[sel].tolist(), nms_thr, classes=cid[sel].tolist())]
        dets = []
        for i in keep:
            b = boxes[i]
            x1, x2 = (b[0] - left) / ratio, (b[2] - left) / ratio
            y1, y2 = (b[1] - top) / ratio, (b[3] - top) / ratio
            dets.append((max(x1, 0), max(y1, 0), min(x2, w), min(y2, h), float(conf_v[i]), int(cid[i])))
        return dets

    def task_nanodet(bench):
        sess, prov = T.session(T.assets("object_detection_nanodet_2022nov.onnx")[0])
        # only the best detection per image: NanoDet-m's scores sit close to its 0.35 threshold on these photos, so
        # the number of weaker (often duplicate) boxes is not stable across numerics; the best one is
        per = {im: nanodet_detect(sess, imread(f"{im}.png"))[:1] for im in DET_IMAGES}
        img = imread("astronaut.png")
        return det_output(per), det_detail(per, prov), finish(lambda: nanodet_detect(sess, img), bench)

    # ---- YOLOX-S 640 (OpenCV Zoo, Apache-2.0) ----
    def yolox_detect(sess, img, conf=0.35, nms_thr=0.5):
        import cv2
        import numpy as np
        h, w = img.shape[:2]
        canvas, ratio = letterbox_topleft(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 640, 114.0)   # RGB, 0..255, pad 114
        (y,) = sess.run(None, {sess.get_inputs()[0].name: to_chw(canvas)})
        d = y[0].copy()
        grids, strides = [], []
        for st in (8, 16, 32):
            n = 640 // st
            gx, gy = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack([gx, gy], 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), st))
        grids, strides = np.concatenate(grids), np.concatenate(strides)
        d[:, :2] = (d[:, :2] + grids) * strides
        d[:, 2:4] = np.exp(d[:, 2:4]) * strides
        scores = d[:, 4:5] * d[:, 5:]                     # objectness x class probability
        cid, conf_v = scores.argmax(1), scores.max(1)
        sel = np.nonzero(conf_v >= conf)[0]
        xyxy = np.stack([d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2, d[:, 0] + d[:, 2] / 2, d[:, 1] + d[:, 3] / 2], 1) / ratio
        keep = [sel[i] for i in nms(xyxy[sel].tolist(), conf_v[sel].tolist(), nms_thr, classes=cid[sel].tolist())]
        return [(max(xyxy[i][0], 0), max(xyxy[i][1], 0), min(xyxy[i][2], w), min(xyxy[i][3], h), float(conf_v[i]), int(cid[i])) for i in keep]

    def task_yolox(bench):
        sess, prov = T.session(T.assets("object_detection_yolox_2022nov.onnx")[0])
        per = {im: yolox_detect(sess, imread(f"{im}.png")) for im in DET_IMAGES}
        img = imread("astronaut.png")
        return det_output(per), det_detail(per, prov), finish(lambda: yolox_detect(sess, img), bench)

    # ---- SSD MobileNet v1 12 (ONNX zoo; NMS is inside the graph) ----
    def ssd_detect(sess, img, conf=0.5):
        import cv2
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[None]                  # uint8 NHWC, any size
        boxes, classes, scores, n = sess.run(None, {sess.get_inputs()[0].name: rgb})
        dets = []
        for i in range(int(n[0])):
            if scores[0, i] >= conf:
                y1, x1, y2, x2 = boxes[0, i]                              # normalised [ymin, xmin, ymax, xmax]
                dets.append((x1 * w, y1 * h, x2 * w, y2 * h, float(scores[0, i]), int(classes[0, i])))   # ids of the 91-id COCO map, see tf91_name
        return dets

    def task_ssdmobilenet(bench):
        sess, prov = T.session(T.assets("ssd_mobilenet_v1_12.onnx")[0])
        per = {im: ssd_detect(sess, imread(f"{im}.png")) for im in DET_IMAGES}
        img = imread("astronaut.png")
        return det_output(per), det_detail(per, prov, tf91_name), finish(lambda: ssd_detect(sess, img), bench)

    # ---- CRNN English word recognition (OpenCV Zoo, Apache-2.0) ----
    def render_word(word):
        """Gray 100x32 image of `word` in black on white: Pillow's bundled default font at 48 px, cropped to
        the text plus an 8 px margin and squeezed to the model's input size (as the zoo's text-box warp does)."""
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.load_default(size=48)
        l, t, r, b = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), word, font=font)
        im = Image.new("L", (r - l + 16, b - t + 16), 255)
        ImageDraw.Draw(im).text((8 - l, 8 - t), word, fill=0, font=font)
        return cv2.resize(np.asarray(im), (100, 32), interpolation=cv2.INTER_AREA)

    def crnn_read(sess, gray):
        x = ((gray.astype("float32") - 127.5) / 127.5)[None, None]
        (y,) = sess.run(None, {sess.get_inputs()[0].name: x})        # [24, 1, 37]
        return ctc_greedy(softmax(y[:, 0, :]))

    def task_crnn(bench):
        sess, prov = T.session(T.assets("text_recognition_CRNN_EN_2021sep.onnx")[0])
        res = [crnn_read(sess, render_word(w)) for w in CRNN_WORDS]
        out = {"words": "|".join(CRNN_WORDS), "text": "|".join(t for t, _ in res),
               "mean_confidence": round(sum(c for _, c in res) / len(res), 3)}
        img = render_word(CRNN_WORDS[0])
        metrics = finish(lambda: crnn_read(sess, img), bench)
        return out, f"read {out['text']!r} from rendered {out['words']!r}; {prov}", metrics

    return {
        "yunet": task_yunet, "sface": task_sface, "pphumanseg": task_pphumanseg, "nanodet": task_nanodet,
        "yolox": task_yolox, "ssdmobilenet": task_ssdmobilenet, "crnn": task_crnn,
        "shufflenet": lambda b: T.task_zoo("shufflenet-v2-12", b),
        "efficientnet": lambda b: T.task_zoo("efficientnet-lite4-11", b),
    }
