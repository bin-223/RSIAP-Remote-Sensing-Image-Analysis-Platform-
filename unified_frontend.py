import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms as T

try:
    import cv2
    CV2_AVAILABLE = True
    CV2_IMPORT_ERROR = ""
except Exception as exc:
    cv2 = None
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(exc)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
    ORT_IMPORT_ERROR = ""
except Exception as exc:
    ort = None
    ORT_AVAILABLE = False
    ORT_IMPORT_ERROR = str(exc)


ROOT = Path(__file__).resolve().parent
SEG_DIR = ROOT / "segmentation_pytorch"
INFER_DIR = ROOT / "recognition_onnx"
INFER_MODEL = INFER_DIR / "models" / "deeplabv3plus.onnx"

# 报告模型路径配置
CAPTION_DIR = ROOT / "report_generator"
CAPTION_CKPT_DIR = CAPTION_DIR / "weights"
CAPTION_VOCAB = CAPTION_CKPT_DIR / "vocab.json"

_CAPTION_PKG_DIR = str(CAPTION_DIR)
if _CAPTION_PKG_DIR not in sys.path:
    sys.path.insert(0, _CAPTION_PKG_DIR)

CLASS_INFO_CSV = """class_name,r,g,b,class_id
urban_land,0,255,255,0
agriculture_land,255,255,0,1
rangeland,255,0,255,2
forest_land,0,255,0,3
water,0,0,255,4
barren_land,255,255,255,5
unknown,0,0,0,6"""

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_CLASS_NAMES = [
    "城市",
    "农田",
    "牧场",
    "森林",
    "水体",
    "裸地",
    "未知",
]
DEFAULT_PALETTE = np.array([
    [0, 255, 255],   # Urban
    [255, 255, 0],   # Agriculture
    [255, 0, 255],   # Rangeland
    [0, 255, 0],     # Forest
    [0, 0, 255],     # Water
    [255, 255, 255], # Barren
    [0, 0, 0],       # Unknown
], dtype=np.uint8)


def _ensure_seg_import() -> None:
    seg_path = str(SEG_DIR)
    if seg_path not in sys.path:
        sys.path.insert(0, seg_path)


def _parse_class_info(csv_string: str) -> List[dict]:
    rows = [line.strip() for line in csv_string.strip().splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError("类别配置至少需要表头 + 1 行类别。")

    header = [part.strip() for part in rows[0].split(",")]
    expected = ["class_name", "r", "g", "b", "class_id"]
    if header != expected:
        raise ValueError("CSV表头必须是: class_name,r,g,b,class_id")

    classes = []
    seen_ids = set()
    for idx, line in enumerate(rows[1:], start=2):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"第 {idx} 行字段数量错误，应为5列。")
        name, r, g, b, cid = parts
        class_id = int(cid)
        if class_id in seen_ids:
            raise ValueError(f"class_id 重复: {class_id}")
        seen_ids.add(class_id)

        rgb = (int(r), int(g), int(b))
        for channel in rgb:
            if channel < 0 or channel > 255:
                raise ValueError(f"第 {idx} 行颜色值必须在 0-255。")
        classes.append({
            "name": name,
            "color_rgb": rgb,
            "color_bgr": (rgb[2], rgb[1], rgb[0]),
            "id": class_id,
        })
    classes.sort(key=lambda x: x["id"])
    ids = [item["id"] for item in classes]
    expected_ids = list(range(len(classes)))
    if ids != expected_ids:
        raise ValueError(f"class_id 必须从 0 开始且连续，当前为: {ids}")
    return classes


def _class_rows_for_table(classes: List[dict]) -> List[dict]:
    rows = []
    for cls in classes:
        r, g, b = cls["color_rgb"]
        rows.append({
            "ID": cls["id"],
            "类别": cls["name"],
            "RGB": f"({r}, {g}, {b})",
        })
    return rows


def _get_seg_palette(num_classes: int) -> np.ndarray:
    if num_classes == len(DEFAULT_PALETTE):
        return DEFAULT_PALETTE
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for j in range(num_classes):
        lab = j
        for i in range(8):
            palette[j, 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j, 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j, 2] |= (((lab >> 2) & 1) << (7 - i))
            lab >>= 3
    return palette


def _decode_seg_mask(mask: np.ndarray, num_classes: int) -> Image.Image:
    mask = np.clip(mask, 0, num_classes - 1)
    palette = _get_seg_palette(num_classes)
    color = palette[mask].astype("uint8")
    return Image.fromarray(color)


def _rgb_name(rgb: np.ndarray) -> str:
    color_map = {
        (0, 255, 255): "青色",
        (255, 255, 0): "黄色",
        (255, 0, 255): "品红",
        (0, 255, 0): "绿色",
        (0, 0, 255): "蓝色",
        (255, 255, 255): "白色",
        (0, 0, 0): "黑色",
    }
    return color_map.get((int(rgb[0]), int(rgb[1]), int(rgb[2])), "其他")


def _parse_seg_class_names(raw: str, num_classes: int) -> List[str]:
    if raw.strip():
        names = [s.strip() for s in raw.split(",") if s.strip()]
        if len(names) == num_classes:
            return names
    if num_classes == len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES
    return [f"Class {i}" for i in range(num_classes)]


def _region_tag(cx: float, cy: float) -> str:
    x_tag = "左侧" if cx < 0.33 else "右侧" if cx > 0.66 else "中部"
    y_tag = "上部" if cy < 0.33 else "下部" if cy > 0.66 else "中部"
    if x_tag == "中部" and y_tag == "中部":
        return "中心"
    return f"{y_tag}-{x_tag}"


def _compactness_desc(bbox_ratio: float, fill_ratio: float) -> str:
    if bbox_ratio < 0.2 and fill_ratio > 0.4:
        return "紧凑"
    if bbox_ratio > 0.5 and fill_ratio < 0.2:
        return "分散"
    return "中等"


def _analyze_seg_mask(mask: np.ndarray, class_names: List[str]) -> Tuple[List[dict], str]:
    h, w = mask.shape
    total = h * w
    summary_lines = []
    rows = []
    num_classes = len(class_names)
    for c in range(num_classes):
        ys, xs = np.where(mask == c)
        count = ys.size
        if count == 0:
            continue
        percent = 100.0 * count / total
        cy = float(ys.mean() / h)
        cx = float(xs.mean() / w)
        region = _region_tag(cx, cy)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        bbox_area = (y1 - y0 + 1) * (x1 - x0 + 1)
        bbox_ratio = bbox_area / total
        fill_ratio = count / max(1, bbox_area)
        compact = _compactness_desc(bbox_ratio, fill_ratio)
        rows.append({
            "类别": class_names[c],
            "面积占比(%)": f"{percent:.2f}",
            "主要区域": region,
            "分布形态": compact,
        })
        summary_lines.append(
            f"{class_names[c]} 占比 {percent:.2f}% ，主要位于 {region} ，分布形态为 {compact}。"
        )
    return rows, "\n".join(summary_lines)


@st.cache_resource
def _load_seg_model(model_name: str, num_classes: int, ckpt_path: str, output_stride: int):
    _ensure_seg_import()
    import network  # type: ignore

    model = network.modeling.__dict__[model_name](
        num_classes=num_classes, output_stride=output_stride, pretrained_backbone=False
    )
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _preprocess_image(image: Image.Image, target_hw: Tuple[int, int] | None = None) -> np.ndarray:
    work_img = image
    if target_hw is not None:
        target_h, target_w = target_hw
        if target_h > 0 and target_w > 0:
            work_img = image.resize((target_w, target_h), Image.BILINEAR)

    img_rgb = np.array(work_img).astype(np.float32) / 255.0
    img_normalized = (img_rgb - MEAN) / STD
    img_transposed = img_normalized.transpose(2, 0, 1)
    return np.expand_dims(img_transposed, axis=0)


@st.cache_resource(show_spinner=False)
def _load_session(model_path: str, providers: Tuple[str, ...]):
    return ort.InferenceSession(model_path, providers=list(providers))


def _resolve_infer_providers(provider_mode: str) -> Tuple[str, ...]:
    available = set(ort.get_available_providers())
    if provider_mode == "仅CPU":
        return ("CPUExecutionProvider",)
    if provider_mode == "强制GPU":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("当前环境不可用 CUDAExecutionProvider，无法强制GPU。")
        return ("CUDAExecutionProvider",)
    if provider_mode == "TensorRT优先":
        providers = []
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return tuple(providers)

    if "CUDAExecutionProvider" in available:
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


def _run_onnx_infer(
    model_path: Path,
    image: Image.Image,
    providers: Tuple[str, ...],
) -> Tuple[np.ndarray, Tuple[str, ...], float]:
    if not ORT_AVAILABLE:
        raise RuntimeError(f"onnxruntime 未安装: {ORT_IMPORT_ERROR}")

    session = _load_session(str(model_path), providers)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape if hasattr(input_meta, "shape") else None

    target_hw = None
    if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 4:
        h = input_shape[2]
        w = input_shape[3]
        if isinstance(h, (int, np.integer)) and isinstance(w, (int, np.integer)):
            if int(h) > 0 and int(w) > 0:
                target_hw = (int(h), int(w))

    input_tensor = _preprocess_image(image, target_hw)
    start = time.perf_counter()
    output = session.run(None, {input_name: input_tensor})[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    pred = np.argmax(output, axis=1).squeeze(0)
    return pred, tuple(session.get_providers()), elapsed_ms


def _postprocess_edge_label(
    original_img_bgr: np.ndarray,
    pred_mask: np.ndarray,
    classes: List[dict],
    min_area: int = 800,
    contour_thickness: int = 2,
    font_scale: float = 0.7,
    font_thickness: int = 2,
    skip_unknown: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if not CV2_AVAILABLE:
        raise RuntimeError(f"cv2 未安装: {CV2_IMPORT_ERROR}")

    orig_h, orig_w = original_img_bgr.shape[:2]
    pred_mask_resized = cv2.resize(
        pred_mask.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )

    result_img = original_img_bgr.copy()

    for cls in classes:
        cls_name = str(cls["name"]).strip().lower()
        if skip_unknown and cls_name in {"unknown", "未知"}:
            continue
        binary_mask = np.uint8(pred_mask_resized == cls["id"])
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cv2.drawContours(
                result_img,
                contours,
                -1,
                cls["color_bgr"],
                contour_thickness,
                lineType=cv2.LINE_AA,
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area <= min_area:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                label = cls["name"]
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )

                cv2.rectangle(
                    result_img,
                    (cX - text_w // 2 - 5, cY - text_h - 10),
                    (cX + text_w // 2 + 5, cY + baseline - 10),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    result_img,
                    label,
                    (cX - text_w // 2, cY - 10),
                    font,
                    font_scale,
                    cls["color_bgr"],
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )

    return pred_mask_resized, result_img


def _safe_stem(name: str) -> str:
    raw = Path(name).stem.strip()
    if not raw:
        raw = datetime.now().strftime("image_%Y%m%d_%H%M%S")
    return "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in raw)


def _uploaded_signature(uploaded, extra: dict | None = None) -> str:
    parts = [
        str(getattr(uploaded, "name", "")),
        str(getattr(uploaded, "size", "")),
    ]
    if extra:
        for key in sorted(extra.keys()):
            parts.append(f"{key}={extra[key]}")
    return "|".join(parts)


def _pil_to_bytes(image: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def _render_infer_ui() -> None:
    st.subheader("识别")
    st.write("上传图片后直接在网页里推理与可视化（边界标注 + 灰度掩码）。")

    left, right = st.columns([1, 2])

    with left:
        st.markdown("**设置**")
        model_path = st.text_input("模型路径", str(INFER_MODEL), key="infer_model_path")
        provider_mode = st.selectbox(
            "推理后端",
            ["GPU优先(回退CPU)", "仅CPU", "强制GPU", "TensorRT优先"],
            index=0,
            key="infer_provider_mode",
        )
        min_area = st.number_input("最小标注面积(像素)", min_value=0, max_value=1000000, value=800, step=50, key="infer_min_area")
        contour_thickness = st.number_input("边界线宽", min_value=1, max_value=20, value=2, step=1, key="infer_contour_thickness")
        font_scale = st.slider("标签字体缩放", min_value=0.3, max_value=2.0, value=0.7, step=0.05, key="infer_font_scale")
        font_thickness = st.number_input("标签粗细", min_value=1, max_value=10, value=2, step=1, key="infer_font_thickness")
        skip_unknown = st.checkbox("跳过 unknown/未知 标注", value=True, key="infer_skip_unknown")
        overlay_alpha = st.slider("叠加透明度(可选)", 0.0, 1.0, 0.4, 0.05, key="infer_overlay_alpha")
        class_csv = st.text_area(
            "类别配置CSV(class_name,r,g,b,class_id)",
            CLASS_INFO_CSV,
            height=180,
            key="infer_class_csv",
        )
        run_btn = st.button("开始识别", type="primary", key="infer_run")

        if ORT_AVAILABLE:
            st.caption("可用后端: " + ", ".join(ort.get_available_providers()))

    with right:
        uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"], key="infer_upload")
        if uploaded is None:
            st.info("请先上传图片。")
            return

        if not Path(model_path).exists():
            st.error(f"未找到模型文件: {model_path}")
            return

        if not ORT_AVAILABLE:
            st.error("当前 Python 环境未安装 onnxruntime，无法进行网页识别。")
            st.code(f"导入错误: {ORT_IMPORT_ERROR}")
            st.info("建议直接运行 `run_unified_frontend.bat`，它会优先使用项目内置运行时。")
            return

        if not CV2_AVAILABLE:
            st.error("当前 Python 环境未安装 opencv-python，无法绘制边界标注。")
            st.code(f"导入错误: {CV2_IMPORT_ERROR}")
            st.info("请安装 opencv-python 后重试。")
            return

        try:
            classes = _parse_class_info(class_csv)
        except Exception as exc:
            st.error(f"类别配置错误: {exc}")
            return

        try:
            providers = _resolve_infer_providers(provider_mode)
        except RuntimeError as exc:
            st.error(str(exc))
            return

        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="输入图像", use_container_width=True)

        infer_signature = _uploaded_signature(
            uploaded,
            {
                "model_path": str(Path(model_path).resolve()),
                "provider_mode": provider_mode,
                "class_csv": class_csv,
                "min_area": int(min_area),
                "contour_thickness": int(contour_thickness),
                "font_scale": float(font_scale),
                "font_thickness": int(font_thickness),
                "skip_unknown": bool(skip_unknown),
                "overlay_alpha": float(overlay_alpha),
            },
        )

        if run_btn:
            with st.spinner("推理中..."):
                pred, active_providers, elapsed_ms = _run_onnx_infer(Path(model_path), image, providers)

            original_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray_mask, edge_label_bgr = _postprocess_edge_label(
                original_bgr,
                pred,
                classes,
                min_area=int(min_area),
                contour_thickness=int(contour_thickness),
                font_scale=float(font_scale),
                font_thickness=int(font_thickness),
                skip_unknown=bool(skip_unknown),
            )

            gray_mask_img = Image.fromarray(gray_mask)
            edge_label_rgb = cv2.cvtColor(edge_label_bgr, cv2.COLOR_BGR2RGB)
            edge_label_img = Image.fromarray(edge_label_rgb)

            if overlay_alpha > 0:
                overlay = Image.blend(image, edge_label_img, alpha=float(overlay_alpha))
            else:
                overlay = edge_label_img

            st.session_state["infer_result"] = {
                "signature": infer_signature,
                "uploaded_name": uploaded.name,
                "gray_png": _pil_to_bytes(gray_mask_img, "PNG"),
                "edge_jpg": _pil_to_bytes(edge_label_img.convert("RGB"), "JPEG"),
                "overlay_jpg": _pil_to_bytes(overlay.convert("RGB"), "JPEG"),
                "class_rows": _class_rows_for_table(classes),
                "active_providers": list(active_providers),
                "elapsed_ms": float(elapsed_ms),
            }

        infer_result = st.session_state.get("infer_result")
        if not infer_result or infer_result.get("signature") != infer_signature:
            st.info('点击"开始识别"执行推理。')
            return

        gray_bytes = infer_result["gray_png"]
        edge_bytes = infer_result["edge_jpg"]
        overlay_bytes = infer_result["overlay_jpg"]

        col1, col2 = st.columns(2)
        with col1:
            st.image(gray_bytes, caption="灰度掩码", use_container_width=True)
        with col2:
            st.image(edge_bytes, caption="边界标注图", use_container_width=True)

        st.caption("实际推理后端: " + ", ".join(infer_result["active_providers"]))
        st.caption(f"单张推理耗时: {infer_result['elapsed_ms']:.1f} ms")

        st.subheader("叠加预览")
        st.image(overlay_bytes, caption="边界标注叠加(可选)", use_container_width=True)

        st.subheader("类别配置")
        st.table(infer_result["class_rows"])

        base_stem = _safe_stem(uploaded.name)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "下载灰度掩码",
                data=gray_bytes,
                file_name=f"{base_stem}_mask.png",
                mime="image/png",
                key="infer_dl_gray",
            )
        with d2:
            st.download_button(
                "下载边界标注图",
                data=edge_bytes,
                file_name=f"{base_stem}_label.jpg",
                mime="image/jpeg",
                key="infer_dl_edge",
            )
        with d3:
            st.download_button(
                "下载叠加图",
                data=overlay_bytes,
                file_name=f"{base_stem}_overlay.jpg",
                mime="image/jpeg",
                key="infer_dl_overlay",
            )



def _resolve_ckpt_path(raw: str) -> str:
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    candidate = SEG_DIR / raw
    if candidate.exists():
        return str(candidate)
    return str(p)


def _render_seg_ui() -> None:
    st.subheader("分割")
    st.write("上传图片后直接在网页里分割与可视化。")

    left, right = st.columns([1, 2])

    with left:
        st.markdown("**设置**")
        ckpt_raw = st.text_input("模型权重路径", "weights/deeplabv3plus_model.pth", key="seg_ckpt_path")
        ckpt = _resolve_ckpt_path(ckpt_raw)

        _ensure_seg_import()
        import network  # type: ignore

        available_models = sorted(
            name for name in network.modeling.__dict__
            if name.islower()
            and not (name.startswith("__") or name.startswith("_"))
            and callable(network.modeling.__dict__[name])
        )
        default_idx = (
            available_models.index("deeplabv3plus_resnet50")
            if "deeplabv3plus_resnet50" in available_models
            else 0
        )
        model_name = st.selectbox("模型", available_models, index=default_idx, key="seg_model_name")
        num_classes = st.number_input("类别数", min_value=2, max_value=256, value=7, step=1, key="seg_num_classes")
        output_stride = st.selectbox("输出步长", [8, 16], index=1, key="seg_output_stride")
        resize = st.number_input("缩放尺寸(正方形, 0=不缩放)", min_value=0, max_value=2048, value=513, step=1, key="seg_resize")
        default_names = ",".join(DEFAULT_CLASS_NAMES) if int(num_classes) == len(DEFAULT_CLASS_NAMES) else ""
        class_names_raw = st.text_input("类别名称(逗号分隔)", default_names, key="seg_class_names")
        overlay_alpha = st.slider("叠加透明度", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="seg_overlay_alpha")
        use_amp = st.checkbox("使用 AMP(仅 CUDA)", value=True, key="seg_use_amp")

        run_infer = st.button("开始分割", type="primary", key="seg_run")

    with right:
        uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"], key="seg_upload")
        if uploaded is None:
            st.info("请先上传图片。")
            return

        if not os.path.isfile(ckpt):
            st.error(f"未找到模型权重文件: {ckpt}")
            return

        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="输入图像", use_container_width=True)

        seg_signature = _uploaded_signature(
            uploaded,
            {
                "ckpt": str(Path(ckpt).resolve()),
                "model_name": model_name,
                "num_classes": int(num_classes),
                "output_stride": int(output_stride),
                "resize": int(resize),
                "class_names_raw": class_names_raw,
                "overlay_alpha": float(overlay_alpha),
                "use_amp": bool(use_amp),
            },
        )

        if run_infer:
            tfs = []
            if resize and resize > 0:
                tfs.append(T.Resize((resize, resize)))
            tfs.extend([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            transform = T.Compose(tfs)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = _load_seg_model(model_name, int(num_classes), ckpt, output_stride).to(device)

            img_tensor = transform(image).unsqueeze(0).to(device)
            amp_enabled = use_amp and device.type == "cuda"
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    outputs = model(img_tensor)
            pred = outputs.argmax(1).squeeze(0).cpu().numpy()

            color_mask = _decode_seg_mask(pred, int(num_classes))
            if resize and resize > 0:
                color_mask = color_mask.resize(image.size, resample=Image.NEAREST)

            overlay = Image.blend(image, color_mask.convert("RGB"), alpha=float(overlay_alpha))
            class_names = _parse_seg_class_names(class_names_raw, int(num_classes))
            rows, summary = _analyze_seg_mask(pred, class_names)

            palette = _get_seg_palette(int(num_classes))
            legend_rows = []
            for idx, name in enumerate(class_names):
                rgb = palette[idx] if idx < len(palette) else np.array([0, 0, 0], dtype=np.uint8)
                legend_rows.append({
                    "类别": name,
                    "颜色": _rgb_name(rgb),
                })

            st.session_state["seg_result"] = {
                "signature": seg_signature,
                "uploaded_name": uploaded.name,
                "color_mask_png": _pil_to_bytes(color_mask.convert("RGB"), "PNG"),
                "overlay_png": _pil_to_bytes(overlay.convert("RGB"), "PNG"),
                "rows": rows,
                "summary": summary,
                "legend_rows": legend_rows,
            }

        seg_result = st.session_state.get("seg_result")
        if not seg_result or seg_result.get("signature") != seg_signature:
            st.info('点击"开始分割"执行推理。')
            return

        seg_mask_bytes = seg_result["color_mask_png"]
        seg_overlay_bytes = seg_result["overlay_png"]
        rows = seg_result["rows"]
        summary = seg_result["summary"]

        col1, col2 = st.columns(2)
        with col1:
            st.image(seg_mask_bytes, caption="分割结果", use_container_width=True)
        with col2:
            st.image(seg_overlay_bytes, caption="叠加可视化", use_container_width=True)

        if rows:
            st.subheader("面积占比与空间分布")
            st.table(rows)
            st.text(summary)
        else:
            st.warning("未检测到任何类别。")

        st.subheader("颜色与类别对照")
        st.table(seg_result["legend_rows"])

        seg_stem = _safe_stem(seg_result["uploaded_name"])
        stats_text = summary if summary else "未检测到任何类别。"
        stats_bytes = stats_text.encode("utf-8")

        st.subheader("下载结果")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "下载分割结果图",
                data=seg_mask_bytes,
                file_name=f"{seg_stem}_segmentation.png",
                mime="image/png",
                key="seg_dl_mask",
            )
        with d2:
            st.download_button(
                "下载叠加可视化",
                data=seg_overlay_bytes,
                file_name=f"{seg_stem}_overlay.png",
                mime="image/png",
                key="seg_dl_overlay",
            )
        with d3:
            st.download_button(
                "下载面积统计",
                data=stats_bytes,
                file_name=f"{seg_stem}_stats.txt",
                mime="text/plain",
                key="seg_dl_stats",
            )




def _render_env_status() -> None:
    with st.expander("环境状态与兼容建议", expanded=False):
        seg_ckpt = SEG_DIR / "weights" / "deeplabv3plus_model.pth"
        st.write(f"Python 解释器: `{sys.executable}`")
        st.write(f"ONNX 模型: `{INFER_MODEL}` -> {'存在' if INFER_MODEL.exists() else '缺失'}")
        st.write(f"分割权重: `{seg_ckpt}` -> {'存在' if seg_ckpt.exists() else '缺失'}")
        st.write(f"PyTorch: `{torch.__version__}`")
        st.write(f"OpenCV: `{cv2.__version__ if CV2_AVAILABLE else '未安装'}`")
        st.write(f"ONNXRuntime: `{ort.__version__ if ORT_AVAILABLE else '未安装'}`")
        if ORT_AVAILABLE:
            st.write("可用推理后端: " + ", ".join(ort.get_available_providers()))
        st.code(".\\run_unified_frontend.bat")
        st.caption("分发到其他 Windows 设备时，优先双击这个 bat，不依赖系统 Python。")
        st.code(
            ".\\deeplabv3p_infer\\deeplabv3p_infer\\python.exe -m pip install streamlit "
            "torch torchvision onnxruntime-gpu opencv-python pillow numpy"
        )


def main() -> None:
    st.set_page_config(page_title="统一前端", layout="wide")
    st.title("统一前端入口")
    st.write("请选择进入识别、分割或报告生成界面。")
    _render_env_status()

    tab_infer, tab_seg, tab_caption = st.tabs(["识别", "分割", "报告生成"])
    with tab_infer:
        _render_infer_ui()
    with tab_seg:
        _render_seg_ui()
    with tab_caption:
        _render_caption_ui()


# ==================== 字幕生成模块 ====================

@st.cache_resource
def _load_caption_model():
    """加载 new_new 统一报告生成模型（单 best.pt，包含 4 模块 prompt token）"""
    import json

    # 动态导入，确保路径正确
    import models.encoder
    import models.decoder_transformer
    import models.transformer_model

    # 加载 checkpoint
    ckpt_path = CAPTION_CKPT_DIR / "best_2_epoch11_2.66.pt"
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)

    # 从 checkpoint 恢复配置
    model_cfg = ckpt["config"]["model"]
    vocab = ckpt["vocab"]
    special_tokens = ckpt.get("special_token_ids", {})

    # 重建模型
    from models.transformer_model import ImageCaptioningTransformer, build_image_captioning_transformer
    model = build_image_captioning_transformer(
        vocab_size=len(vocab),
        encoder_name=model_cfg.get("encoder_name", "resnet101"),
        d_model=model_cfg.get("d_model", 512),
        nhead=model_cfg.get("nhead", 8),
        num_decoder_layers=model_cfg.get("num_decoder_layers", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 2048),
        dropout=model_cfg.get("dropout", 0.25),
        max_len=model_cfg.get("max_len", 128),
        pad_idx=vocab.get("<PAD>", 0),
        pretrained_encoder=False,
        train_backbone=False,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # idx2word
    idx2word = {v: k for k, v in vocab.items()}

    return {
        "model": model,
        "vocab": vocab,
        "idx2word": idx2word,
        "special_tokens": special_tokens,
        "device": torch.device("cpu"),
    }


def _tokens_to_text(token_ids, idx2word, vocab):
    """将 token id 序列转成可读文本"""
    pad_idx = vocab.get("<PAD>", 0)
    end_idx = vocab.get("<END>", 2)
    skip_tokens = {"<PAD>", "<START>", "<UNK>", "<GLOBAL>", "<DETAIL>", "<ABNORMAL>", "<CONCLUSION>", ""}
    words = []
    for idx in token_ids:
        if idx == end_idx or idx == pad_idx:
            break
        word = idx2word.get(int(idx), "")
        if word not in skip_tokens:
            words.append(word)

    text = " ".join(words)
    # 去掉句尾可能残留的零碎标点（如 </s> 解码出来的杂字符），再补上句号
    text = text.rstrip()
    if text and text[-1] not in "。.!?！？":
        text += " ."
    return text


def _generate_report(model, image_tensor, vocab, idx2word, device, max_len, use_beam):
    """调用统一模型的 generate_all_modules，一次性生成4个模块文本"""
    start_token = vocab.get("<START>", 1)
    end_token = vocab.get("<END>", 2)
    module_token_ids = {
        "global": vocab.get("<GLOBAL>", 4),
        "detail": vocab.get("<DETAIL>", 5),
        "abnormal": vocab.get("<ABNORMAL>", 6),
        "conclusion": vocab.get("<CONCLUSION>", 7),
    }

    strategy = "beam" if use_beam else "greedy"
    results = model.generate_all_modules(
        image_tensor,
        module_token_ids=module_token_ids,
        start_token=start_token,
        end_token=end_token,
        max_len=max_len,
        strategy=strategy,
        beam_size=5,
        length_penalty=0.7,
    )

    module_display_names = {
        "global": "全局描述 (Global Description)",
        "detail": "详细描述 (Detailed Description)",
        "abnormal": "异常检测 (Abnormality)",
        "conclusion": "综合结论 (Comprehensive Conclusion)",
    }

    captions = {}
    for key, token_ids in results.items():
        # Greedy -> torch.Tensor [1, T] -> [T]
        # Beam    -> list[torch.Tensor]   -> [T]
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids[0].tolist()
        elif isinstance(token_ids, list):
            # Beam Search 返回 list，取第一个 batch 结果
            token_ids = token_ids[0].tolist()
        captions[module_display_names[key]] = _tokens_to_text(token_ids, idx2word, vocab)

    return captions


def _render_caption_ui() -> None:
    """报告生成 UI - 基于 new_new 统一模型"""
    st.subheader("遥感图像报告生成")
    st.write("上传图片生成分析报告（全局描述、详细描述、异常检测、综合结论）")

    left, right = st.columns([1, 2])

    with left:
        st.markdown("**设置**")
        use_beam = st.checkbox("使用 Beam Search（较慢但质量更高）", value=True, key="caption_beam")
        max_len = st.slider("最大生成长度", 20, 50, 40, key="caption_max_len")
        run_btn = st.button("生成报告", type="primary", key="caption_run")

        st.markdown("---")
        st.markdown("**模型信息**")
        st.caption(f"模型目录: {CAPTION_DIR}")
        st.caption(f"检查点: {CAPTION_CKPT_DIR}")

        best_pt = CAPTION_CKPT_DIR / "best_2_epoch11_2.66.pt"
        vocab_file = CAPTION_VOCAB
        st.caption(f"best_2_epoch11: {'✅' if best_pt.exists() else '❌'}")
        st.caption(f"vocab.json: {'✅' if vocab_file.exists() else '❌'}")

    with right:
        uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"], key="caption_upload")
        if uploaded is None:
            st.info("请先上传图片。")
            return

        if not (best_pt.exists() and vocab_file.exists()):
            st.error("报告模型文件缺失，请确保 checkpoints/best_2_epoch11_2.66.pt 和 checkpoints/vocab.json 存在。")
            return

        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="输入图像", use_container_width=True)

        caption_signature = _uploaded_signature(uploaded, {"max_len": max_len, "beam": use_beam})

        if run_btn:
            with st.spinner("加载模型并生成报告..."):
                try:
                    model_data = _load_caption_model()
                    model = model_data["model"]
                    vocab = model_data["vocab"]
                    idx2word = model_data["idx2word"]
                    device = model_data["device"]

                    # 图片预处理（与训练一致：224x224 + ImageNet 归一化）
                    transform = T.Compose([
                        T.Resize((224, 224)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    image_tensor = transform(image).unsqueeze(0).to(device)

                    # 一次生成全部 4 个模块
                    captions = _generate_report(
                        model, image_tensor, vocab, idx2word, device,
                        max_len=max_len, use_beam=use_beam
                    )

                    st.session_state["caption_result"] = {
                        "signature": caption_signature,
                        "captions": captions,
                        "image_name": uploaded.name,
                        "image_bytes": _pil_to_bytes(image, "PNG"),
                    }

                except Exception as e:
                    st.error(f"生成失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        # 显示结果
        caption_result = st.session_state.get("caption_result")
        if not caption_result or caption_result.get("signature") != caption_signature:
            st.info("点击「生成报告」开始生成分析报告。")
            return

        captions = caption_result["captions"]

        st.markdown("---")
        st.subheader("生成的分析报告")

        for module_name, caption_text in captions.items():
            st.markdown(f"**{module_name}**")
            st.text_area("", caption_text, height=80, key=f"result_{module_name}", disabled=True)
            st.markdown("---")

        # 下载功能
        base_stem = _safe_stem(uploaded.name)
        image_bytes = caption_result.get("image_bytes", b"")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        module_keys = list(captions.keys())

        # Markdown 完整报告
        md_lines = [
            f"# 遥感图像分析报告\n",
            f"\n",
            f"**图像:** {uploaded.name}  \n",
            f"**生成时间:** {timestamp}\n",
            f"\n---\n",
        ]
        for mn in module_keys:
            md_lines.append(f"\n## {mn}\n\n{captions[mn]}\n")
        md_report = "".join(md_lines)

        # PDF 完整报告
        def _make_pdf_bytes(img_png_bytes: bytes, cap_dict: dict, fname: str, ts: str) -> bytes:
            import io
            from fpdf import FPDF
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_font("CJK", fname="C:/Windows/Fonts/msyh.ttc")

            pdf.add_page()
            pdf.set_font("CJK", "", 16)
            pdf.cell(0, 10, "Remote Sensing Image Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(3)
            pdf.set_font("CJK", "", 9)
            pdf.cell(0, 6, f"Image: {fname}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 6, f"Generated: {ts}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)

            if img_png_bytes:
                img_io = io.BytesIO(img_png_bytes)
                pdf.image(img_io, w=160)
                pdf.ln(5)

            for mn, text in cap_dict.items():
                pdf.ln(3)
                pdf.set_font("CJK", "", 12)
                pdf.cell(0, 7, mn, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("CJK", "", 10)
                pdf.multi_cell(0, 5, text)

            return bytes(pdf.output())

        st.subheader("下载结果")
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "下载完整报告 (Markdown)",
                data=md_report,
                file_name="rs_report.md",
                mime="text/plain",
                key="caption_dl_md",
            )

        with col2:
            if image_bytes:
                pdf_bytes = _make_pdf_bytes(image_bytes, captions, uploaded.name, timestamp)
                st.download_button(
                    "下载完整报告 (PDF)",
                    data=pdf_bytes,
                    file_name="rs_report.pdf",
                    mime="application/pdf",
                    key="caption_dl_pdf",
                )
            else:
                st.button("下载完整报告 (PDF)", disabled=True)
                st.caption("图片未加载，无法生成 PDF")


if __name__ == "__main__":
    main()
