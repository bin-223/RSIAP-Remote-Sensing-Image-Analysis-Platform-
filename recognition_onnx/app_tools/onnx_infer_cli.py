import os
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from pathlib import Path

# ====================== 配置区域 ======================
BASE_DIR = Path(__file__).resolve().parents[1]
ONNX_MODEL_PATH = str(BASE_DIR / "models" / "deeplabv3plus.onnx")
INPUT_FOLDER = str(BASE_DIR / "input")
OUTPUT_FOLDER = str(BASE_DIR / "output")

os.makedirs(INPUT_FOLDER, exist_ok=True)

# 类别定义
CLASS_INFO_CSV = """class_name,r,g,b,class_id
urban_land,0,255,255,0
agriculture_land,255,255,0,1
rangeland,255,0,255,2
forest_land,0,255,0,3
water,0,0,255,4
barren_land,255,255,255,5
unknown,0,0,0,6"""
# =======================================================


def parse_class_info(csv_string):
    """解析类别信息，注意OpenCV使用BGR顺序"""
    lines = csv_string.strip().split('\n')[1:]
    classes = []
    for line in lines:
        parts = line.split(',')
        # CSV里是RGB，OpenCV需要BGR，所以这里反转一下
        classes.append({
            'name': parts[0],
            'color': (int(parts[3]), int(parts[2]), int(parts[1])), # (B, G, R)
            'id': int(parts[4])
        })
    return classes


def create_output_dirs():
    """创建输出文件夹"""
    output_gray = os.path.join(OUTPUT_FOLDER, "gray_mask")
    output_edge = os.path.join(OUTPUT_FOLDER, "edge_label")
    os.makedirs(output_gray, exist_ok=True)
    os.makedirs(output_edge, exist_ok=True)
    return output_gray, output_edge


def get_inference_session():
    """优先使用GPU推理，若不可用则自动回退CPU"""
    available_providers = ort.get_available_providers()
    print(f"当前可用后端: {available_providers}")

    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("✅ 使用 GPU (CUDA) 推理")
    else:
        providers = ['CPUExecutionProvider']
        print("⚠️ 未检测到 CUDA，回退到 CPU 推理")

    return ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)


def preprocess_image(image_path):
    """预处理图片"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    
    original_h, original_w = img.shape[:2]
    
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 归一化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_rgb.astype(np.float32) / 255.0 - mean) / std
    
    # 维度变换
    img_transposed = img_normalized.transpose(2, 0, 1)
    input_tensor = np.expand_dims(img_transposed, axis=0)
    
    return img, input_tensor, (original_h, original_w)


def postprocess_and_save(original_img, pred_mask, classes, output_gray_path, output_edge_path, original_shape):
    """
    后处理：
    1. 保存灰度图
    2. 保存仅含边界和文字标注的原图（不覆盖/不叠加蒙版）
    """
    orig_h, orig_w = original_shape
    
    # Resize 回原图尺寸
    pred_mask_resized = cv2.resize(
        pred_mask.astype(np.uint8), 
        (orig_w, orig_h), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # --------------------------
    # 1. 保存灰度图
    # --------------------------
    cv2.imwrite(output_gray_path, pred_mask_resized)
    
    # --------------------------
    # 2. 保存边界标注图 (直接在原图副本上绘制，不做半透明叠加)
    # --------------------------
    # 复制一份原图，我们只在副本上画线和写字，完全保留原图底色
    result_img = original_img.copy()
    
    for cls in classes:
        # 跳过 unknown
        if cls['name'] == 'unknown':
            continue
            
        # 生成二值图
        binary_mask = np.uint8(pred_mask_resized == cls['id'])
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 1. 绘制边界线 (线条粗细设为 2)
            cv2.drawContours(result_img, contours, -1, cls['color'], 2, lineType=cv2.LINE_AA)
            
            # 2. 绘制文字标注 (只标注面积足够大的区域)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 800: # 过滤掉过小的噪点
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        label = cls['name']
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        
                        # 获取文字大小
                        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # 画一个黑色背景框让文字更清晰
                        cv2.rectangle(
                            result_img, 
                            (cX - text_w//2 - 5, cY - text_h - 10), 
                            (cX + text_w//2 + 5, cY + baseline - 10), 
                            (0, 0, 0), 
                            -1
                        )
                        
                        # 画文字 (颜色使用类别对应的颜色)
                        cv2.putText(
                            result_img, 
                            label, 
                            (cX - text_w//2, cY - 10), 
                            font, 
                            font_scale, 
                            cls['color'], 
                            thickness,
                            lineType=cv2.LINE_AA
                        )

    # 保存结果
    cv2.imwrite(output_edge_path, result_img)


def main():
    print("="*60)
    print("DeepLabV3+ ONNX 推理 (GPU优先, 自动CPU回退 / 仅边界标注)")
    print("="*60)
    
    # 初始化
    classes = parse_class_info(CLASS_INFO_CSV)
    output_gray_dir, output_edge_dir = create_output_dirs()
    
    session = get_inference_session()

    # 获取图片列表
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not image_files:
        print(f"❌ 输入文件夹为空: {INPUT_FOLDER}")
        print(f"   请将图片放入该文件夹后重新运行。")
        input("按回车键退出...")
        return
        
    print(f"发现 {len(image_files)} 张图片\n")
    
    # 处理循环
    pbar = tqdm(image_files, desc="处理进度")
    for filename in pbar:
        input_path = os.path.join(INPUT_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        
        out_gray_path = os.path.join(output_gray_dir, f"{base_name}_mask.png")
        out_edge_path = os.path.join(output_edge_dir, f"{base_name}_label.jpg")
        
        # 预处理
        original_img, input_tensor, orig_shape = preprocess_image(input_path)
        if original_img is None:
            continue
            
        # 推理
        output = session.run(None, {'input': input_tensor})[0]
        
        # 后处理
        pred_mask = np.argmax(output, axis=1).squeeze(0)
        
        # 保存
        postprocess_and_save(original_img, pred_mask, classes, out_gray_path, out_edge_path, orig_shape)
    
    print("\n" + "="*60)
    print("✅ 全部完成！")
    print(f"📂 灰度图: {output_gray_dir}")
    print(f"📂 边界标注图: {output_edge_dir}")
    print("="*60)


if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"请将图片放入: {INPUT_FOLDER}")
    else:
        main()
