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

    logging.info(f"📦 Using pretrained model from: {args.ckpt_dir}")

    # Initialize model + load weights (silencing noise)
    with suppress_external_warnings():
        model = FoundationStereo(args)
        ckpt = torch.load(args.ckpt_dir)

    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()

    # Load input stereo pair
    img0 = imageio.v2.imread(args.left_file)
    img1 = imageio.v2.imread(args.right_file)
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
    logging.info("🚀 Running inference...")
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
    np.save(f"{args.out_dir}/disp.npy", disp)
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    imageio.imwrite(f"{args.out_dir}/vis.png", vis)
    logging.info(f"💾 Disparity saved: {args.out_dir}/disp.npy")
    logging.info(f"🖼️ Visualization saved: {args.out_dir}/vis.png")


if __name__ == "__main__":
    args_dict = {
        'left_file': "./simulator/trj_6/L_00.png",
        'right_file': "./simulator/trj_6/R_00.png",
        'intrinsic_file': "./simulator/trj_6/K.txt",
        'ckpt_dir': "./pretrained_models/model_best_bp2.pth",
        'out_dir': "./simulator_outputs/trj_6"
    }
    run_stereo_inference(args_dict)
