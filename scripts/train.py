import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
# Make sure this import is correct (e.g., from utils import *)
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import glob

# --- 1. SimpleTrainer2d Class ---
class SimpleTrainer2d:
    """Trains random 2d gaussians to fit a single spectrogram."""
    def __init__(
        self,
        image_path: Path,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        args = None,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Loading Ground Truth Spectrogram: {image_path}")
        self.gt_image = image_path_to_tensor(image_path).to(self.device) # This is [1, 2, H, W]

        self.num_points = num_points
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        print(f"Spectrogram dimensions: H={self.H}, W={self.W}")
        
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}")
        
        if model_name == "GaussianImage_Cholesky":
            # Import our 3-channel (fixed) model
            from gaussianimage_audio_v2 import GaussianImage_Cholesky 
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device)
        else:
            raise ValueError(f"Model name '{model_name}' not supported in this script. Only GaussianImage_Cholesky is configured.")

        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            # (This part is unchanged)
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def train(self):     
        # (This function is unchanged, it correctly calls the fixed train_iter)
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc=f"Training on {self.image_name}")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_image)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test() # Calls the fixed test() func below
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
            out = self.gaussian_model() # out["render"] is [1, 3, H, W]
        
        # --- CUDA FIX: Slice the 3-channel output to match the 2-channel ground truth ---
        # gt_image is [1, 2, H, W]
        render_sliced = out["render"].float()[:, :self.gaussian_model.true_feature_dim, :, :] # Slice to [1, 2, H, W]
        # --- END OF FIX ---

        # Now, compare the [1, 2, H, W] slice to the [1, 2, H, W] ground truth
        mse_loss = F.mse_loss(render_sliced, self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item()) # Note: PSNR is not a reliable metric
        
        try:
            # This should also use the sliced render
            ms_ssim_value = ms_ssim(render_sliced, self.gt_image.float(), data_range=1, size_average=True).item()
        except:
            ms_ssim_value = 0.0 # Will likely fail anyway
            
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        
        # Save the output .npy
        if self.save_imgs:
            print(f"Saving output spectrogram to {self.log_dir / (self.image_name + '_fitting.npy')}")
            
            # --- CUDA FIX: Save the 2-channel sliced version, not the 3-channel one ---
            output_tensor = render_sliced.squeeze(0).cpu() # Shape: [2, H, W]
            # --- END OF FIX ---
            
            output_tensor = output_tensor.permute(1, 2, 0) # Shape: [H, W, 2]
            output_numpy = output_tensor.numpy()
            name = self.image_name + "_fitting.npy" 
            np.save(str(self.log_dir / name), output_numpy)
            
        return psnr, ms_ssim_value

# --- 2. Data Loading Function (Unchanged) ---
def image_path_to_tensor(image_path: Path):
    """
    Loads a spectrogram from a .npy file.
    Expected .npy file shape: (H, W, 2)
    Output tensor shape: (1, 2, H, W)
    """
    spec_data = np.load(image_path)
    img_tensor = torch.from_numpy(spec_data).float()
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0) # [1, 2, H, W]
    return img_tensor

# --- 3. Arg Parsing Function (Unchanged) ---
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Single spectrogram training script.")
    
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to a single .npy spectrogram file"
    )
    parser.add_argument(
        "--data_name", type=str, default='ljspeech_test', help="Name for logging directory (e.g., ljspeech_test)"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training iterations (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save output .npy spectrogram")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

# --- 4. Main Function (Unchanged) ---
def main(argv):
    args = parse_args(argv)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    image_path = Path(args.input)
    if not image_path.exists():
        print(f"Error: Input file not found at {image_path}")
        return

    print(f"--- Starting Training for Single File ---")
    print(f"File: {image_path}")
    print(f"Iterations: {args.iterations}")
    print(f"Num Points: {args.num_points}")
    print(f"Learning Rate: {args.lr}")
    print(f"-------------------------------------------")

    trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
        iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path)
    
    psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
    
    print(f"--- Training Finished ---")
    print(f"Results for {image_path.stem}:")
    print(f"  Final PSNR: {psnr:.4f} (Note: L2 Loss is more reliable)")
    print(f"  Final MS-SSIM: {ms_ssim:.4f} (Note: Likely 0.0)")
    print(f"  Training Time: {training_time:.4f}s")
    print(f"  Rendering FPS: {eval_fps:.4f}")
    print(f"  Output saved in: {trainer.log_dir}")
    print(f"-------------------------------------------")

if __name__ == "__main__":
    main(sys.argv[1:])