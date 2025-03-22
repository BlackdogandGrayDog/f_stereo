import os
import sys
import cv2
import torch
import imageio
import numpy as np
import logging
import time

#  Local imports
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

#  Wrapper to silence all irrelevant logging/prints/warnings
def suppress_external_warnings():
    import warnings
    warnings.filterwarnings("ignore")

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    class SuppressStd:
        def __enter__(self):
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        def __exit__(self, *args):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._stdout
            sys.stderr = self._stderr

    return SuppressStd()

#  Run stereo inference
def run_stereo_inference(args_dict):
    # Set up logging and environment
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args_dict['out_dir'], exist_ok=True)

    # Load config and merge with args
    cfg = OmegaConf.load(f"{os.path.dirname(args_dict['ckpt_dir'])}/cfg.yaml")
    for k, v in args_dict.items():
        cfg[k] = v
    args = OmegaConf.create(cfg)

    # logging.info(f"📦 Using pretrained model from: {args.ckpt_dir}")

    # Initialize model + load weights (silencing noise)
    with suppress_external_warnings():
        model = FoundationStereo(args)
        ckpt = torch.load(args.ckpt_dir)

    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()

    # Load input stereo pair
    img0 = imageio.imread(args.left_file)
    img1 = imageio.imread(args.right_file)
    scale = 1.0
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H, W = img0.shape[:2]
    img0_ori = img0.copy()

    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    # Inference with timing
    logging.info(f"🚀 Running inference for {args.left_file} ...")
    start_time = time.time()
    with torch.amp.autocast("cuda"):
        disp = model.forward(img0, img1, iters=32, test_mode=True)
    end_time = time.time()

    elapsed_ms = (end_time - start_time) * 1000
    fps = 1000 / elapsed_ms
    logging.info(f"✅ Inference completed! (⏱️ {elapsed_ms:.2f} ms | ⚡ {fps:.2f} FPS)")

    # Postprocess & Save
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)

    base_name = args.out_name if "out_name" in args else "disp"
    np.save(f"{args.out_dir}/{base_name}.npy", disp)

    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    os.makedirs(f"{args.out_dir}/visualisation", exist_ok=True)
    imageio.imwrite(f"{args.out_dir}/visualisation/{base_name}.png", vis)

    logging.info(f"💾 Disparity saved: {args.out_dir}/{base_name}.npy")
    logging.info(f"🖼️ Visualization saved: {args.out_dir}/visualisation/{base_name}.png")
    logging.info("-------------------------------- \n")


#  Run all stereo pairs in a directory
def run_all_stereo_pairs(base_dir):
    left_dir = os.path.join(base_dir, "image01")
    right_dir = os.path.join(base_dir, "image02")
    out_dir = os.path.join(base_dir, "disparity_map")
    ckpt_path = "./pretrained_models/model_best_bp2.pth"

    os.makedirs(out_dir, exist_ok=True)
    left_files = sorted(os.listdir(left_dir))

    for fname in left_files:
        left_path = os.path.join(left_dir, fname)
        right_path = os.path.join(right_dir, fname)

        if not os.path.exists(right_path):
            logging.warning(f"⚠️ Skipping {fname}: Right image not found.")
            continue

        args_dict = {
            'left_file': left_path,
            'right_file': right_path,
            'ckpt_dir': ckpt_path,
            'out_dir': out_dir,
            'out_name': os.path.splitext(fname)[0]
        }

        run_stereo_inference(args_dict)



if __name__ == "__main__":
    base_dir = "./datasets/rectified04"
    run_all_stereo_pairs(base_dir)
