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
from utils import * # Make sure this import is correct
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import glob # Needed to find .npy files
import os # Needed to join paths

# Define the number of files for our subset
SUBSET_SIZE = 3  # Using 3 for debugging

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
        self.gt_image = image_path_to_tensor(image_path).to(self.device) # This is [1, 2, H, W]

        self.num_points = num_points
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        # Create log directory name based on whether using gps or fixed num_points
        if hasattr(args, 'gps_rate') and args.gps_rate is not None:
            # Convert to int for cleaner directory names
            gps_rate_int = int(args.gps_rate)
            self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_{args.iterations}_{gps_rate_int}gps/{self.image_name}")
        else:
            self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}")
        
        if model_name == "GaussianImage_Cholesky":
            # Import our 3-channel (fixed) model
            from gaussianimage_audio_v2 import GaussianImage_Cholesky
            # Check if quantization is enabled
            use_quantize = args.quantize if hasattr(args, 'quantize') else False
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                device=self.device, lr=args.lr, quantize=use_quantize).to(self.device)
        else:
            raise ValueError(f"Model name '{model_name}' not supported. Only GaussianImage_Cholesky is configured.")

        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"טוען מודל שמור מ: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def train(self):     
        progress_bar = tqdm(range(1, self.iterations+1), desc=f"מאמן את {self.image_name}", leave=False) # 'leave=False' makes the bar disappear after
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_image)
            with torch.no_grad():
                if iter % 100 == 0: # Update less often to speed things up
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(100)
        end_time = time.time() - start_time
        progress_bar.close()
        
        psnr_value, ms_ssim_value = self.test() # Calls the fixed test() func below
        
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write(f"אימון הושלם תוך {end_time:.4f} שניות, זמן רינדור: {test_end_time:.8f} שניות, FPS:{1/test_end_time:.4f}")
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        
        # We don't need to save training.npy for a subset test
        # np.save(...)
        
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model() # out["render"] is [1, 3, H, W]
        
        # --- CUDA FIX: Slice the 3-channel output to match the 2-channel ground truth ---
        render_sliced = out["render"].float()[:, :self.gaussian_model.true_feature_dim, :, :] # Slice to [1, 2, H, W]
        # --- END OF FIX ---

        mse_loss = F.mse_loss(render_sliced, self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item()) # Note: PSNR is not a reliable metric
        
        try:
            ms_ssim_value = ms_ssim(render_sliced, self.gt_image.float(), data_range=1, size_average=True).item()
        except:
            ms_ssim_value = 0.0 # Will likely fail anyway
            
        self.logwriter.write(f"Test PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}")
        
        # Save the output .npy
        if self.save_imgs:
            output_tensor = render_sliced.squeeze(0).cpu() # Shape: [2, H, W]
            output_tensor = output_tensor.permute(1, 2, 0) # Shape: [H, W, 2]
            output_numpy = output_tensor.numpy()
            name = self.image_name + "_fitting.npy" 
            np.save(str(self.log_dir / name), output_numpy)
            
        return psnr, ms_ssim_value

# --- Data Loading Function (Unchanged) ---
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

# --- Helper function to calculate num_points from gaussians_per_second ---
def calculate_num_points_from_rate(spec_width: int, gaussians_per_second: float,
                                   hop_length: int = 256, sample_rate: int = 22050):
    """
    Calculate the number of Gaussian points based on gaussians per second rate.

    Args:
        spec_width: Width of the spectrogram (number of time frames)
        gaussians_per_second: Desired rate of Gaussians per second
        hop_length: STFT hop length (default: 256)
        sample_rate: Audio sample rate (default: 22050)

    Returns:
        num_points: Total number of Gaussian points for this spectrogram
    """
    duration_seconds = float(spec_width * hop_length) / float(sample_rate)
    num_points = int(float(gaussians_per_second) * duration_seconds)
    # Ensure it's a Python int, not numpy int
    return int(num_points)

# --- Arg Parsing Function ---
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Subset training script for audio spectrograms.")
    
    parser.add_argument(
        "-d", "--dataset", 
        type=str, 
        default='./dataset/ljspeech_spectrograms/', 
        help="Path to the directory containing .npy spectrogram files."
    )
    parser.add_argument(
        "--data_name", 
        type=str, 
        default='ljspeech_subset_test', 
        help="Name for logging directory (e.g., ljspeech_subset_test)"
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=10000, # Lowered iterations for faster testing
        help="number of training iterations (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=None,
        help="2D GS points (default: %(default)s). If --gaussians_per_second is set, this is ignored.",
    )
    parser.add_argument(
        "--gaussians_per_second",
        type=float,
        default=None,
        help="Rate of Gaussians per second of audio. If set, overrides --num_points.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save output .npy spectrograms")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization for compression")
    
    # We remove these as they are not needed for this script
    # parser.add_argument("--model_name", ... )
    # parser.add_argument("--model_path", ... )
    
    args = parser.parse_args(argv)
    return args

# --- Main Function ---
def main(argv):
    args = parse_args(argv)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    # Setup the main logger for the whole run based on whether using gps or fixed num_points
    if args.gaussians_per_second is not None:
        gps_rate_int = int(args.gaussians_per_second)  # Convert to int for clean naming
        args.gps_rate = gps_rate_int  # Store as int for later use
        logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.iterations}_{gps_rate_int}gps"))
    else:
        args.gps_rate = None
        num_pts = args.num_points if args.num_points is not None else 50000
        logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.iterations}_{num_pts}"))
    
    # Find all .npy files in the dataset directory
    print(f"מחפש קבצי .npy בנתיב: {args.dataset}")
    all_image_paths = sorted(glob.glob(os.path.join(args.dataset, '*.npy')))
    
    if len(all_image_paths) == 0:
        print(f"שגיאה: לא נמצאו קבצי .npy ב- {args.dataset}")
        print("אנא ודא שהרצת את preprocess.py והנתיב נכון.")
        return

    # --- THIS IS THE SUBSET LOGIC ---
    # Select only the first SUBSET_SIZE files
    subset_paths = all_image_paths[:SUBSET_SIZE]
    print(f"נמצאו {len(all_image_paths)} קבצים. רץ על תת-קבוצה של {len(subset_paths)} קבצים.")
    # --- END OF SUBSET LOGIC ---

    psnrs, ms_ssims, training_times, eval_fpses = [], [], [], []
    
    # Use tqdm for the outer loop as well, to see progress through files
    for image_path_str in tqdm(subset_paths, desc="מעבד קבצים בתת-הקבוצה"):
        image_path = Path(image_path_str)
        image_name = image_path.stem
        logwriter.write(f"\n--- מתחיל אימון עבור: {image_name} ---")

        # Calculate num_points based on gaussians_per_second if specified
        if args.gaussians_per_second is not None:
            # Load the spectrogram to get its width
            spec_data = np.load(image_path)
            spec_width = int(spec_data.shape[1])  # W dimension (time frames)
            num_points = calculate_num_points_from_rate(
                spec_width,
                float(args.gaussians_per_second),
                hop_length=256,
                sample_rate=22050
            )
            # Ensure num_points is a Python int
            num_points = int(num_points)
            logwriter.write(f"Using {args.gaussians_per_second} gaussians/sec -> {num_points} total points (duration: {spec_width * 256 / 22050:.2f}s)")
        else:
            num_points = int(args.num_points) if args.num_points is not None else 50000
            logwriter.write(f"Using fixed num_points: {num_points}")

        trainer = SimpleTrainer2d(
            image_path=image_path,
            num_points=num_points,
            iterations=args.iterations,
            model_name="GaussianImage_Cholesky", # Hardcoded for this script
            args=args,
            model_path=None # We always train from scratch
        )
        
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
        
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_fpses.append(eval_fps)
        
        logwriter.write(f"--- סיום עבור: {image_name} ---")
        logwriter.write(f"PSNR (לא אמין): {psnr:.4f}, MS-SSIM: {ms_ssim:.4f}, זמן אימון: {training_time:.4f} שניות, FPS: {eval_fps:.4f}")

    # Calculate and log averages
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()

    logwriter.write(f"\n--- סיכום תת-הקבוצה ({len(subset_paths)} קבצים) ---")
    logwriter.write(f"PSNR ממוצע (לא אמין): {avg_psnr:.4f}")
    logwriter.write(f"MS-SSIM ממוצע: {avg_ms_ssim:.4f}")
    logwriter.write(f"זמן אימון ממוצע לקובץ: {avg_training_time:.4f} שניות")
    logwriter.write(f"FPS ממוצע לרינדור: {avg_eval_fps:.4f}")

if __name__ == "__main__":
    main(sys.argv[1:])