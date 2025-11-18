import os
import torch
import numpy as np
import cv2
import argparse
import tqdm
import json
from tifffile import imread
from scipy import spatial, stats, special
from sklearn import metrics
import matplotlib.pyplot as plt

# 导入 function.py 中的工具
from function import *

# 禁用梯度计算
torch.set_grad_enabled(False)

# ==========================================
# 核心指标计算函数 (基于提供的参考代码)
# ==========================================

def cc_shap_score(ratios_prediction, ratios_explanation):
    """
    计算两个贡献分布之间的一致性指标。
    Ref: On Measuring Faithfulness or Self-consistency of Natural Language Explanations
    """
    # 防止全零向量导致的计算错误
    if np.all(ratios_prediction == 0) or np.all(ratios_explanation == 0):
        return 1.0, 1.0, 0.0, 0.0, 0.0, 0.0

    # Cosine Distance (1 - Cosine Similarity)
    # 注意：scipy.spatial.distance.cosine 计算的是 distance (0表示完全相同，2表示相反)
    # 论文 Figure 1 中 CC-SHAP Eq.4 倾向于 consistency score (higher is better)
    # 但这里我们严格遵循提供的代码返回 distance
    cosine_dist = spatial.distance.cosine(ratios_prediction, ratios_explanation)
    
    distance_correlation = spatial.distance.correlation(ratios_prediction, ratios_explanation)
    mse = metrics.mean_squared_error(ratios_prediction, ratios_explanation)
    var = np.sum(((ratios_prediction - ratios_explanation)**2 - mse)**2) / ratios_prediction.shape[0]
    
    # KL Div 和 JS Div 需要概率分布 (Softmax)
    # how many bits does one need to encode P using a code optimised for Q
    kl_div = stats.entropy(special.softmax(ratios_explanation), special.softmax(ratios_prediction))
    js_div = spatial.distance.jensenshannon(special.softmax(ratios_prediction), special.softmax(ratios_explanation))

    return cosine_dist, distance_correlation, mse, var, kl_div, js_div

def normalize_and_aggregate(phi_values):
    """
    将原始贡献值转换为 Ratios (百分比)。
    """
    # Avoid division by zero
    total_contribution = np.sum(np.abs(phi_values))
    if total_contribution == 0:
        return np.zeros_like(phi_values)
    
    # Ratios calculation: value / sum(|values|) * 100
    ratios = phi_values / total_contribution * 100
    return ratios

# ==========================================
# VLM 采样与处理工具
# ==========================================

def get_image_patches_mask(image_shape, patch_size=16):
    h, w = image_shape[:2]
    h_patches = h // patch_size
    w_patches = w // patch_size
    num_patches = h_patches * w_patches
    return num_patches, (h_patches, w_patches)

def apply_mask(image, mask_indices, patch_size=16, h_patches=None, w_patches=None):
    masked_img = image.copy()
    mask_layer = np.zeros_like(image)
    idx = 0
    for i in range(h_patches):
        for j in range(w_patches):
            if idx in mask_indices:
                y1, y2 = i * patch_size, (i + 1) * patch_size
                x1, x2 = j * patch_size, (j + 1) * patch_size
                mask_layer[y1:y2, x1:x2] = 1
            idx += 1
    return masked_img * mask_layer

def get_log_probs(model, inputs, target_ids):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return -loss.sum().item()

def calculate_cc_shap_score(model, processor, image_path, objective, args, n_samples=20, patch_size=16):
    """
    计算 CC-SHAP。
    使用 Monte Carlo 采样近似 SHAP value，并应用论文中的 Normalization 和 Metric。
    """
    
    # 1. 准备 Spectrum 和 Prompt
    spectrum_img = generate_spectrum(image_path)
    temp_spec_path = "temp_shap_spec.png"
    cv2.imwrite(temp_spec_path, spectrum_img)
    user_text = construct_prompt(objective)

    # 2. 准备输入
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": temp_spec_path}, {"type": "text", "text": user_text}]}
    ]
    text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(args.device)

    # 3. Full Generation (获取 Prediction 和 Explanation 文本)
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": processor.tokenizer.eos_token_id
    }
    generated_ids = model.generate(**inputs, **generation_args)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 4. 解析 Think 和 Answer
    try:
        if "<think>" in response_text and "</think>" in response_text:
            think_content = response_text.split("<think>")[1].split("</think>")[0].strip()
        else:
            return None, response_text 
        if "<answer>" in response_text and "</answer>" in response_text:
            answer_content = response_text.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            return None, response_text
    except IndexError:
        return None, response_text
    
    print("Generated Think:", think_content)
    print("Generated Answer:", answer_content)

    think_ids = processor.tokenizer(think_content, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
    answer_ids = processor.tokenizer(answer_content, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)

    if think_ids.shape[1] == 0 or answer_ids.shape[1] == 0:
        return None, response_text

    # 5. Monte Carlo Sampling (获取原始 SHAP 值 / Phi)
    num_patches, (hp, wp) = get_image_patches_mask(spectrum_img.shape, patch_size)
    
    samples_log_probs_think = []
    samples_log_probs_answer = []
    samples_indices = []

    for _ in range(n_samples):
        perm = np.random.permutation(num_patches)
        n_keep = np.random.randint(1, num_patches)
        keep_indices = perm[:n_keep]
        samples_indices.append(keep_indices)
        
        masked_spec = apply_mask(spectrum_img, keep_indices, patch_size, hp, wp)
        cv2.imwrite(temp_spec_path, masked_spec)

        masked_conversation = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": [{"type": "image", "image": temp_spec_path}, {"type": "text", "text": user_text}]}]
        masked_image_inputs, _ = process_vision_info(masked_conversation)
        masked_inputs = processor(text=[text_prompt], images=masked_image_inputs, padding=True, return_tensors="pt").to(args.device)

        # Think Log Prob
        input_ids_think = torch.cat([masked_inputs.input_ids, think_ids], dim=1)
        masked_inputs_think = {
            "input_ids": input_ids_think,
            "attention_mask": torch.cat([masked_inputs.attention_mask, torch.ones_like(think_ids)], dim=1),
            "pixel_values": masked_inputs.pixel_values,
            "image_grid_thw": masked_inputs.image_grid_thw
        }
        samples_log_probs_think.append(get_log_probs(model, masked_inputs_think, input_ids_think))

        # Answer Log Prob
        input_ids_answer = torch.cat([masked_inputs.input_ids, answer_ids], dim=1)
        masked_inputs_answer = {
            "input_ids": input_ids_answer,
            "attention_mask": torch.cat([masked_inputs.attention_mask, torch.ones_like(answer_ids)], dim=1),
            "pixel_values": masked_inputs.pixel_values,
            "image_grid_thw": masked_inputs.image_grid_thw
        }
        samples_log_probs_answer.append(get_log_probs(model, masked_inputs_answer, input_ids_answer))

    # 6. 原始 Shapley 值估算 (Centering)
    phi_think = np.zeros(num_patches)
    phi_answer = np.zeros(num_patches)
    mean_think = np.mean(samples_log_probs_think)
    mean_answer = np.mean(samples_log_probs_answer)
    
    for i in range(n_samples):
        indices = samples_indices[i]
        val_think = samples_log_probs_think[i] - mean_think
        val_answer = samples_log_probs_answer[i] - mean_answer
        phi_think[indices] += (val_think / len(indices))
        phi_answer[indices] += (val_answer / len(indices))

    # 7. Normalization (Eq. 2 & 3 from Figure/Code)
    # 将 Shapley 值转换为 Ratios
    ratios_explanation = normalize_and_aggregate(phi_think)
    ratios_prediction = normalize_and_aggregate(phi_answer)

    # 8. 计算 Metric
    cosine_dist, dist_correl, mse, var, kl_div, js_div = cc_shap_score(ratios_prediction, ratios_explanation)
    
    # 构造返回结果
    metrics_result = {
        "cosine_distance": cosine_dist,
        "correlation": dist_correl,
        "mse": mse,
        "var": var,
        "kl_div": kl_div,
        "js_div": js_div
    }

    if os.path.exists(temp_spec_path):
        os.remove(temp_spec_path)

    return metrics_result, response_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="experiment/FMIRAgent")
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset-path", type=str, default="dataset/test_split/DIV2K_Agents_0314")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--n-shap-samples", type=int, default=20)
    parser.add_argument("--objective", type=str, default="SR")
    args = parser.parse_args()

    set_seed(42)

    print("Loading Language Model...")
    from qwen2.src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(model_base=args.model_base, model_path=args.model_path, model_name=model_name, device=args.device)
    
    input_paths = sorted(os.listdir(args.dataset_path))
    input_paths = [os.path.join(args.dataset_path, p) for p in input_paths if p.endswith(('.tif'))][:args.samples]

    print(f"Evaluating CC-SHAP on {len(input_paths)} images with objective '{args.objective}'...")
    
    # 存储所有结果以便计算平均值
    agg_metrics = {
        "cosine_distance": [],
        "correlation": [],
        "mse": [],
        "var": [],
        "kl_div": [],
        "js_div": []
    }
    
    for img_path in tqdm.tqdm(input_paths):
        try:
            metrics_res, response = calculate_cc_shap_score(
                model, processor, img_path, args.objective, args, n_samples=args.n_shap_samples
            )
            
            if metrics_res is not None:
                for k, v in metrics_res.items():
                    agg_metrics[k].append(v)
                # 可选：打印当前样本的 Cosine Distance
                # print(f" {os.path.basename(img_path)} | Cosine Dist: {metrics_res['cosine_distance']:.4f}")
            else:
                print(f"Skipping {os.path.basename(img_path)}: Failed to generate/parse.")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\nAggregated Results for {args.model_path}:")
    for k, v in list(agg_metrics.items()):
        if len(v) > 0:
            mean_val = np.mean(v)
            std_val = np.std(v)
            agg_metrics[f"{k}_mean"] = mean_val
            agg_metrics[f"{k}_std"] = std_val
            print(f"Average {k}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"Average {k}: N/A")

    # 存储结果
    output_file = f"cc_shap_results_{args.model_path.replace('/','_')}.json"
    with open(output_file, "w") as f:
        json.dump(agg_metrics, f, indent=4)