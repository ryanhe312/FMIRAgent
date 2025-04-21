import argparse
from threading import Thread
import gradio as gr
from PIL import Image
from qwen2.src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from transformers import TextIteratorStreamer
from functools import partial
import warnings
import json
import os
import re
import time
import random
import datasets
from tqdm import tqdm
import numpy as np
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

warnings.filterwarnings("ignore")

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def bot_streaming(message, history, generation_args):
    # Initialize variables
    images = []
    videos = []

    if message["files"]:
        for file_item in message["files"]:
            if isinstance(file_item, dict):
                file_path = file_item["path"]
            else:
                file_path = file_item
            if isinstance(file_path, str) and is_video_file(file_path):
                videos.append(file_path)
            else:
                images.append(file_path)

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_turn, assistant_turn in history:
        user_content = []
        if isinstance(user_turn, tuple):
            file_paths = user_turn[0]
            user_text = user_turn[1]
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            for file_path in file_paths:
                if isinstance(file_path, str) and is_video_file(file_path):
                    user_content.append({"type": "video", "video": file_path, "fps":1.0})
                else:
                    user_content.append({"type": "image", "image": file_path})
            if user_text:
                user_content.append({"type": "text", "text": user_text})
        else:
            user_content.append({"type": "text", "text": user_turn})
        conversation.append({"role": "user", "content": user_content})

        if assistant_turn is not None:
            assistant_content = [{"type": "text", "text": assistant_turn}]
            conversation.append({"role": "assistant", "content": assistant_content})

    user_content = []
    for image in images:
        user_content.append({"type": "image", "image": image})
    for video in videos:
        user_content.append({"type": "video", "video": video, "fps":1.0})
    user_text = message['text']
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    conversation.append({"role": "user", "content": user_content})

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device) 

    streamer = TextIteratorStreamer(processor.tokenizer, **{"skip_special_tokens": True, "skip_prompt": True, 'clean_up_tokenization_spaces':False,}) 
    generation_kwargs = dict(inputs, streamer=streamer, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
    
    thread.join()
    return buffer

def main(args):

    global processor, model, device

    device = args.device
    
    disable_torch_init()

    use_flash_attn = True
    
    model_name = get_model_name_from_path(args.model_path)
    
    if args.disable_flash_attention:
        use_flash_attn = False

    processor, model = load_pretrained_model(model_base = args.model_base, model_path = args.model_path, 
                                                device_map=args.device, model_name=model_name, 
                                                load_4bit=args.load_4bit, load_8bit=args.load_8bit,
                                                device=args.device, use_flash_attn=use_flash_attn
    )
    
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }

    dataset = datasets.load_from_disk(args.dataset_path)
    ref_dataset = json.load(open(args.dataset_path.replace('hf','json'), "r"))

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "dists": [],
        "type": [],
        "plan": [],
        "path": [],
        "response": [],
        "nrmse": []
    }

    invalid_plans = 0
    for example, ref in tqdm(list(zip(dataset,ref_dataset))):
        data_image = example["image"]
        data_conversations = QUESTION_TEMPLATE.format(Question=example["problem"])
        message = {
            "text": data_conversations,
            "files": data_image if isinstance(data_image, list) else [data_image]
        }
        content = bot_streaming(message, [], generation_args)
        metrics["response"].append(content)
        
        try: 
            plan = content.split("<answer>")[1].split("</answer>")[0].strip().replace("  ", " ")
        except:
            plan = content

        metric_results = "plans"+os.path.basename(ref["image"][0]).split(".")[0]+".json"
        metric_results = os.path.join(os.path.dirname(ref["image"][0]), metric_results)
        metric_results = json.load(open(metric_results, "r"))

        if "origin_plan" in metric_results.keys():
            metric_results.pop("origin_plan")

        if plan in list(metric_results.keys()):
            metrics["type"].append(3 if "Volumetric" in plan else (2 if "Projection" in plan else (1 if "Isotropic" in plan else 0)))

            results = metric_results[plan]
            for metric in results:
                metrics[metric].append(results[metric])
                if metric == "psnr":
                    nrmse = 1 / 10 ** (results[metric] / 20)
                    metrics["nrmse"].append(nrmse)

            metrics["plan"].append(content.replace("\n"," "))
            metrics["path"].append(ref["image"][0])
        elif "Isotropic_None "+ plan in list(metric_results.keys()):
            metrics["type"].append(1)

            results = metric_results["Isotropic_None "+ plan]
            for metric in results:
                metrics[metric].append(results[metric])
                if metric == "psnr":
                    nrmse = 1 / 10 ** (results[metric] / 20)
                    metrics["nrmse"].append(nrmse)

            metrics["plan"].append(content.replace("\n"," "))
            metrics["path"].append(ref["image"][0])
        elif "Projection "+ plan in list(metric_results.keys()):
            metrics["type"].append(2)

            results = metric_results["Projection "+ plan]
            for metric in results:
                metrics[metric].append(results[metric])
                if metric == "psnr":
                    nrmse = 1 / 10 ** (results[metric] / 20)
                    metrics["nrmse"].append(nrmse)

            metrics["plan"].append(content.replace("\n"," "))
            metrics["path"].append(ref["image"][0])
        elif "Volumetric "+ plan in list(metric_results.keys()):
            metrics["type"].append(3)

            results = metric_results["Volumetric "+ plan]
            for metric in results:
                metrics[metric].append(results[metric])
                if metric == "psnr":
                    nrmse = 1 / 10 ** (results[metric] / 20)
                    metrics["nrmse"].append(nrmse)

            metrics["plan"].append(content.replace("\n"," "))
            metrics["path"].append(ref["image"][0])
        else:
            invalid_plans += 1

    print("Invalid plans: ", invalid_plans, "/", len(dataset))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-path", type=str, default="metrics.csv")
    parser.add_argument("--random", type=int, default=None)
    args = parser.parse_args()
    metrics = main(args)

    f = open(args.output_path, "w")
    f_std = open(args.output_path.replace(".csv", ".std.csv"), "w")
    print("Type, Num, PSNR, SSIM, LPIPS, DISTS, NRMSE", file=f)
    print("Type, Num, PSNR, SSIM, LPIPS, DISTS, NRMSE", file=f_std)

    print(f'Total, 0, {sum(metrics["psnr"])/len(metrics["psnr"]):.2f}, {sum(metrics["ssim"])/len(metrics["ssim"]):.2f}, {sum(metrics["lpips"])/len(metrics["lpips"]):.4f}, {sum(metrics["dists"])/len(metrics["dists"]):.4f}, {sum(metrics["nrmse"])/len(metrics["nrmse"]):.4f}', file=f)
    print(f"Total, 0, {np.std(metrics['psnr']):.2f}, {np.std(metrics['ssim']):.2f}, {np.std(metrics['lpips']):.4f}, {np.std(metrics['dists']):.4f}, {np.std(metrics['nrmse']):.4f}", file=f_std)

    for type in range(4):
        psnr_per_type = [metrics["psnr"][i] for i in range(len(metrics["type"])) if metrics["type"][i] == type]
        ssim_per_type = [metrics["ssim"][i] for i in range(len(metrics["type"])) if metrics["type"][i] == type]
        lpips_per_type = [metrics["lpips"][i] for i in range(len(metrics["type"])) if metrics["type"][i] == type]
        dists_per_type = [metrics["dists"][i] for i in range(len(metrics["type"])) if metrics["type"][i] == type]
        nrmse_per_type = [metrics["nrmse"][i] for i in range(len(metrics["type"])) if metrics["type"][i] == type]

        # calculate average
        print(f"{type}, {metrics['type'].count(type)}, {sum(psnr_per_type)/len(psnr_per_type):.2f}, {sum(ssim_per_type)/len(ssim_per_type):.2f}, {sum(lpips_per_type)/len(lpips_per_type):.4f}, {sum(dists_per_type)/len(dists_per_type):.4f}, {sum(nrmse_per_type)/len(nrmse_per_type):.4f}", file=f)

        # calculate standard deviation
        print(f"{type}, {metrics['type'].count(type)}, {np.std(psnr_per_type):.2f}, {np.std(ssim_per_type):.2f}, {np.std(lpips_per_type):.4f}, {np.std(dists_per_type):.4f}, {np.std(nrmse_per_type):.4f}", file=f_std)

    f.close()
    f_std.close()
