import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
# from PIL import Image # <-- CHANGE: No longer needed for loading
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import glob # <-- ADDED: To find all .npy files

class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image_path: Path,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        args = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_path_to_tensor(image_path).to(self.device) # <-- CHANGE: This now calls our new .npy loader

        self.num_points = num_points
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}")
        
        if model_name == "GaussianImage_Cholesky":
            from gaussianimage_audio_v2 import GaussianImage_Cholesky # <-- CHANGE: Import our modified model
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device)

        elif model_name == "GaussianImage_RS":
            # (Original code, unchanged)
            from gaussianimage_rs import GaussianImage_RS
            self.gaussian_model = GaussianImage_RS(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device) 

        elif model_name == "3DGS":
            # (Original code, unchanged)
            from gaussiansplatting_3d import Gaussian3D
            self.gaussian_model = Gaussian3D(loss_type="Fusion2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, sh_degree=args.sh_degree, lr=args.lr).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def train(self):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_image)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    # NOTE: PSNR is not a reliable metric here, but loss is.
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item()) # NOTE: PSNR is not reliable.
        
        # WARNING: ms_ssim will likely FAIL or give incorrect results for 2-channel, non-normalized data.
        # It's better to rely on MSE/L2 loss as the primary metric.
        try:
            ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        except:
            ms_ssim_value = 0.0 # Set to 0 if it fails
            
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        
        # --- CHANGE: Save output as .npy instead of .png ---
        if self.save_imgs:
            # 1. Get tensor, remove batch dim, move to CPU
            output_tensor = out["render"].float().squeeze(0).cpu() # Shape: [2, H, W]
            # 2. Permute back to (H, W, 2) for easier analysis
            output_tensor = output_tensor.permute(1, 2, 0) # Shape: [H, W, 2]
            # 3. Convert to NumPy array
            output_numpy = output_tensor.numpy()
            # 4. Save as .npy file
            name = self.image_name + "_fitting.npy" 
            np.save(str(self.log_dir / name), output_numpy)
            
        return psnr, ms_ssim_value

# --- CHANGE: Replaced entire function to load .npy files ---
def image_path_to_tensor(image_path: Path):
    """
    Loads a spectrogram from a .npy file.
    Expected .npy file shape: (H, W, 2)
    Output tensor shape: (1, 2, H, W)
    """
    # 1. Load the numpy file
    spec_data = np.load(image_path)
    
    # 2. Convert to PyTorch tensor
    #    Expected shape: (H, W, 2)
    img_tensor = torch.from_numpy(spec_data).float()
    
    # 3. Permute dimensions to match model input: (C, H, W)
    #    (H, W, C) -> (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)
    
    # 4. Add batch dimension: (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0) # [1, 2, H, W]
    return img_tensor
# --- End of function change ---

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    # ... (rest of args are fine) ...
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0

    # --- CHANGE: Simplified the data loading loop to find all .npy files ---
    #    The old loop was hardcoded for kodak/DIV2K file names.
    #    This now finds all files ending in .npy in the provided dataset directory.
    print(f"Searching for .npy files in: {args.dataset}")
    image_paths = sorted(glob.glob(os.path.join(args.dataset, '*.npy')))
    print(f"Found {len(image_paths)} files.")
    
    if len(image_paths) == 0:
        print(f"Error: No .npy files found in {args.dataset}. Did you run preprocess.py?")
        return

    for image_path in image_paths:
        image_path = Path(image_path) # Convert string to Path object
        
        trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
            iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path)
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps))
    # --- End of loop change ---

    if len(image_paths) > 0:
        avg_psnr = torch.tensor(psnrs).mean().item()
        avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
        avg_training_time = torch.tensor(training_times).mean().item()
        avg_eval_time = torch.tensor(eval_times).mean().item()
        avg_eval_fps = torch.tensor(eval_fpses).mean().item()
        avg_h = image_h // len(image_paths)
        avg_w = image_w // len(image_paths)

        logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps))    
    else:
        logwriter.write("No files were processed.")

if __name__ == "__main__":
    main(sys.argv[1:])