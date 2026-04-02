import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm
import network
import utils
import os
import argparse
import numpy as np

from torchvision import transforms as T

import torch
import torch.nn as nn

from PIL import Image
from glob import glob

def _get_pascal_palette(num_classes):
    # Deterministic color map (VOC-style) for visualization.
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for j in range(num_classes):
        lab = j
        for i in range(8):
            palette[j, 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j, 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j, 2] |= (((lab >> 2) & 1) << (7 - i))
            lab >>= 3
    return palette

def _decode_custom(mask, num_classes):
    mask = np.clip(mask, 0, num_classes - 1)
    palette = _get_pascal_palette(num_classes)
    return palette[mask]

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='custom',
                        choices=['custom'], help='Name of training set')
    parser.add_argument("--num_classes", type=int, default=7,
                        help="number of classes for custom dataset")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Inference Options
    parser.add_argument("--output", default="predictions",
                        help="directory to save predicted masks")
    parser.add_argument("--save_val_results_to", default=None,
                        help="(deprecated) same as --output")
    parser.add_argument("--resize", type=int, default=513,
                        help="resize input to square size, set 0 to disable")
    parser.add_argument("--save_index_mask", action='store_true', default=False,
                        help="also save raw class index mask as .png")

    
    parser.add_argument("--ckpt", default="weights/deeplabv3plus_model.pth", type=str,
                        help="path to model weights")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    decode_fn = lambda m: _decode_custom(m, opts.num_classes)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Resolve output directory
    output_dir = opts.save_val_results_to or opts.output
    os.makedirs(output_dir, exist_ok=True)

    # Collect inputs
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    else:
        raise FileNotFoundError(f"Input not found: {opts.input}")
    if len(image_files) == 0:
        raise FileNotFoundError("No images found in input.")
    
    # Set up model (all models are constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {opts.ckpt}")
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    print("Loaded model from %s" % opts.ckpt)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    tfs = []
    if opts.resize and opts.resize > 0:
        tfs.append(T.Resize((opts.resize, opts.resize)))
    tfs.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    transform = T.Compose(tfs)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            colorized_preds.save(os.path.join(output_dir, img_name + '_pred.png'))
            if opts.save_index_mask:
                Image.fromarray(pred.astype('uint8'), mode='L').save(
                    os.path.join(output_dir, img_name + '_mask.png')
                )

if __name__ == '__main__':
    main()
