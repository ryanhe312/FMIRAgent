import os
import cv2
import glob
import argparse
import tqdm
import numpy as np

from tifffile import imread, imwrite as imsave
from piq import psnr, ssim

from function import *

# seed everything
set_seed(42)

def calculate_metrics(file1, file2):
    # Read the TIFF files
    img1 = imread(file1)
    img2 = normalize(imread(file2),0,100)

    # Ensure the images have the same shape
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Calculate PSNR and SSIM
    img1 = torch.tensor(img1).cuda()
    img2 = torch.tensor(img2).cuda()
    if img1.ndim == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.ndim == 3:
        img1 = img1.permute(2,0,1).unsqueeze(0)
        img2 = img2.permute(2,0,1).unsqueeze(0)

    psnr_value = psnr(img1, img2).item()
    ssim_value = ssim(img1, img2).item()

    return psnr_value, ssim_value

def process_single_image(img_input, gt_path, save_path, args, objective_options='None'):
    
    if args.force_plan is not None:
        plan_message = args.force_plan
    else:
        plan_message = bot_streaming(img_input, objective_options, args=args)
    output_file, _ = run_plan(img_input, plan_message)

    # Save the output image
    image = imread(output_file)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # print(f"Saving restored image to {save_path}")
    imsave(save_path, image)

    # Calculate the PSNR and SSIM
    psnr_value, ssim_value = calculate_metrics(output_file, gt_path)
    return psnr_value, ssim_value, plan_message

def get_dataset(path, output_path,name):
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist.")
    
    if name == "DeepBacs":
        # Get the paths for input, ground truth, and save directories
        input_path = sorted(glob.glob(os.path.join(path, "*", "test", "low_SNR","*.tif")) + glob.glob(os.path.join(path, "*", "test", "WF","*.tif")))
        gt_path = sorted(glob.glob(os.path.join(path, "*", "test", "high_SNR","*.tif")) + glob.glob(os.path.join(path, "*", "test", "SIM","*.tif")))
        save_path = [p.replace("low_SNR", "SR").replace('WF','SR').replace(path,os.path.join(output_path,name)) for p in input_path]
    
    elif name == "Shareloc":
        # Get the paths for input, ground truth, and save directories
        input_path = sorted(glob.glob(os.path.join(path, "*", "LR.tif")))
        gt_path = sorted(glob.glob(os.path.join(path, "*", "HR.tif")))
        save_path = [p.replace("LR", "SR").replace(path,os.path.join(output_path,name)) for p in input_path]

    else:
        # Get the paths for input, ground truth, and save directories
        input_path = sorted(glob.glob(os.path.join(path, "000000*.tif")))
        gt_path = sorted(glob.glob(os.path.join(path, "HR000000*.tif")))
        save_path = [p.replace("HR", "SR").replace(path,os.path.join(output_path,name)) for p in gt_path]

    return input_path, gt_path, save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--force-plan", type=str, default=None)
    parser.add_argument("--unseen-dataset", action="store_true")
    args = parser.parse_args()

    # Load the model
    load_models("CUDA", "Yes", "float16", args=args)

    # Set the paths for datasets
    if not args.unseen_dataset:
        paths = ["dataset/test/DIV2K_Agents_0314",'dataset/test/Flouresceneiso_Agents_New','dataset/test/Flouresceneproj_Agents_New','dataset/test/FlouresceneVCD_Agents_New']
        names = ["Normal","Isotropic","Projection","Volumetric"]
    else:
        paths = ["Shareloc","DeepBacs"]
        names = ["Shareloc","DeepBacs"]

    # Iterate over each dataset
    for path, name in zip(paths, names):
        input_path, gt_path, save_path = get_dataset(path, args.output_path, name)
        print(f"Input Path: {input_path}")

        # Check if the number of input images matches the number of ground truth images
        if len(input_path) != len(gt_path):
            raise ValueError("The number of input images and ground truth images must match.")
        
        # run the model on each image
        psnr_values = []
        ssim_values = []
        plan_messages = []
        pbar = tqdm.tqdm(total=len(input_path), desc=f"Processing {name} Dataset")
        for i in range(len(input_path)):
            psnr_value, ssim_value, plan_message = process_single_image(input_path[i], gt_path[i], save_path[i], args, objective_options=name)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            plan_messages.append(plan_message)

            pbar.update(1)
            pbar.set_postfix({"PSNR": f"{psnr_value:.2f}", "SSIM": f"{ssim_value:.4f}"})

        # Calculate the average PSNR and SSIM
        avg_psnr, std_psnr = np.mean(psnr_values), np.std(psnr_values)
        avg_ssim, std_ssim = np.mean(ssim_values), np.std(ssim_values)
        print(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")

        # Save the results to a text file
        result_path = os.path.join(args.output_path, f"results_{name}.txt")
        with open(result_path, "w") as f:
            for i in range(len(input_path)):
                f.write(f"Image {i+1}: PSNR={psnr_values[i]:.2f}, SSIM={ssim_values[i]:.4f} PLAN={plan_messages[i]}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")

        print(f"Results saved to {result_path}")