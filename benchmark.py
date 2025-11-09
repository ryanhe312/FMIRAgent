import os
import cv2
import time
import glob
import argparse
import tqdm
import math
import numpy as np

from tifffile import imread, imwrite as imsave
from piq import psnr, ssim, LPIPS

from function import *

# seed everything
set_seed(42)

def calculate_metrics(file1, file2, lpips_metric):
    # Read the TIFF files
    img1 = imread(file1)
    img2 = normalize(imread(file2), 0, 100, clip=True)

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
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)

    psnr_value = psnr(img1, img2).item()
    ssim_value = ssim(img1, img2).item()
    
    lpips_value = [lpips_metric(img1[i:i+1], img2[i:i+1]).item() for i in range(img1.shape[0])]
    lpips_value = np.mean(lpips_value)

    return psnr_value, ssim_value, lpips_value

def process_batch_image(img_inputs, gt_paths, save_paths, args, lpips_metric, objective_options='None'):
    st = time.time()
    if args.force_plan is not None:
        plan_messages = [args.force_plan] * len(img_inputs)
    else:
        plan_messages = bot_streaming(img_inputs, objective_options, args=args)
    plan_time = time.time() - st

    psnr_values = []
    ssim_values = []
    lpips_values = []
    nrmse_values = []

    from tqdm import tqdm
    for img_input, plan_message, gt_path, save_path in tqdm(zip(img_inputs, plan_messages,gt_paths,save_paths)):
        # st = time.time()
        # Run the plan
        output_file, error = run_plan(img_input, plan_message)

        # print(f"Processing time: {time.time() - st:.2f}s")
        # st = time.time()
        if output_file is None:
            print(f"Failed to process image {img_input} with error {error}")
            continue

        # Save the output image
        image = imread(output_file)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # print(f"Saving restored image to {save_path}")
        imsave(save_path, image)

        # Calculate the PSNR and SSIM
        psnr_value, ssim_value, lpips_value = calculate_metrics(output_file, gt_path, lpips_metric)
        nrmse_value = 10 ** (-psnr_value / 20.0)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)
        nrmse_values.append(nrmse_value)
        # print(f"Evaluation time: {time.time() - st:.2f}s")
    return psnr_values, ssim_values, lpips_values, nrmse_values, plan_messages, plan_time/len(psnr_values)

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
    parser.add_argument("--model-path", type=str, default="experiment/FMIRAgent")
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force-plan", type=str, default=None)
    parser.add_argument("--unseen-dataset", action="store_true")
    args = parser.parse_args()

    # Load the model
    load_models("Paralleled CUDA", "Yes", "float16", args=args)
    lpips_metric = LPIPS(reduction='mean').to(args.device)

    # Set the paths for datasets
    if not args.unseen_dataset:
        paths = ["dataset/test_split/DIV2K_Agents_0314",'dataset/test_split/Flouresceneiso_Agents_New','dataset/test_split/Flouresceneproj_Agents_New','dataset/test_split/FlouresceneVCD_Agents_New']
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
        lpips_values = []
        nrmse_values = []
        plan_messages = []
        plan_times = []
        batch_size = args.batch_size
        pbar = tqdm.tqdm(total=math.ceil(len(input_path)/batch_size) , desc=f"Processing {name} Dataset")
        for i in range(0,len(input_path),batch_size):
            psnr_value, ssim_value, lpips_value, nrmse_value, plan_message, plan_time = process_batch_image(
                input_path[i:i+batch_size], 
                gt_path[i:i+batch_size], 
                save_path[i:i+batch_size], 
                args,
                lpips_metric,
                objective_options=name
            )

            psnr_values += psnr_value
            ssim_values += ssim_value
            lpips_values += lpips_value
            nrmse_values += nrmse_value
            plan_messages += plan_message
            plan_times.append(plan_time)

            pbar.update(1)
            pbar.set_postfix({"PSNR": f"{np.mean(psnr_value):.2f}", "SSIM": f"{np.mean(ssim_value):.4f}", "LPIPS": f"{np.mean(lpips_value):.4f}", "NRMSE": f"{np.mean(nrmse_value):.4f}", "Time": f"{plan_time:.2f}s"})

        # Calculate the average PSNR and SSIM
        avg_psnr, std_psnr = np.mean(psnr_values), np.std(psnr_values)
        avg_ssim, std_ssim = np.mean(ssim_values), np.std(ssim_values)
        avg_lpips, std_lpips = np.mean(lpips_values), np.std(lpips_values)
        avg_nrmse, std_nrmse = np.mean(nrmse_values), np.std(nrmse_values)
        print(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f} ± {std_lpips:.4f}")
        print(f"Average NRMSE: {avg_nrmse:.4f} ± {std_nrmse:.4f}")

        # Save the results to a text file
        result_path = os.path.join(args.output_path, f"results_{name}.txt")
        with open(result_path, "w") as f:
            for i in range(len(psnr_values)):
                f.write(f"Image {i+1}: PSNR={psnr_values[i]:.2f}, SSIM={ssim_values[i]:.4f}, LPIPS={lpips_values[i]:.4f}, NRMSE={nrmse_values[i]:.4f} PLAN={plan_messages[i]} TIME={plan_times[i//batch_size]:.2f}s\n")
            f.write(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")
            f.write(f"Average LPIPS: {avg_lpips:.4f} ± {std_lpips:.4f}\n")
            f.write(f"Average NRMSE: {avg_nrmse:.4f} ± {std_nrmse:.4f}\n")

        print(f"Results saved to {result_path}")