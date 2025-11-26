import os
import torch
import numpy as np
import argparse
import tqdm
import json
from scipy import special
import cv2
from openai import OpenAI  # Required for DeepSeek API

# Import tools from function.py consistent with analysis.py
from function import *

# Disable gradient calculation to save memory
torch.set_grad_enabled(False)

def get_answer_log_prob(model, inputs, prompt_length):
    """
    Calculate the log probability of the generated part (Answer), ignoring the Prompt part.
    """
    input_ids = inputs["input_ids"]
    target_ids = input_ids.clone()
    
    # Mask the prompt part so it doesn't contribute to the loss
    target_ids[:, :prompt_length] = -100

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        
        # Calculate CrossEntropyLoss without reduction (per token)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reshape back to batch size
        loss = loss.view(target_ids.shape[0], -1)
        
        # Sum valid losses (where label != -100)
        valid_mask = (shift_labels != -100).float()
        sum_log_prob = - (loss * valid_mask).sum(dim=1).item()
        
        # Calculate sequence length (number of valid tokens)
        seq_len = valid_mask.sum(dim=1).item()
        
    return sum_log_prob, seq_len

def check_entailment_deepseek(client, question, answer1, answer2):
    """
    Use Online DeepSeek API to check if answer1 entails answer2.
    """
    # NLI Prompt from the paper
    nli_text = (
        f"We are evaluating answers to the question: {question}\n"
        f"Here are two possible answers:\n"
        f"Possible Answer 1: {answer1}\n"
        f"Possible Answer 2: {answer2}\n"
        f"Does Possible Answer 1 semantically entail Possible Answer 2? "
        f"Respond directly with entailment, contradiction, or neutral."
    )
    
    for i in range(3):  # Retry up to 3 times
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for Natural Language Inference."},
                    {"role": "user", "content": nli_text}
                ],
                temperature=0.0, # Deterministic
                max_tokens=10
            )
            content = response.choices[0].message.content.strip().lower()
            return "entailment" in content
            
        except Exception as e:
            print(f"  [Warning] DeepSeek API Call Failed: {e}")

    # Fallback to False (assume different) to be safe
    return False

def cluster_answers(answers, question, client):
    """
    Cluster answers based on bidirectional entailment using DeepSeek.
    """
    clusters = [] # Each cluster is a list of indices in 'answers'
    
    print(f"  - Clustering {len(answers)} answers with DeepSeek...")
    
    for i, ans in enumerate(answers):
        matched_cluster = None
        
        # Try to find an existing compatible cluster
        for cluster in clusters:
            # Pick the first element of the cluster as representative
            rep_idx = cluster[0]
            rep_ans = answers[rep_idx]
            
            # Bidirectional Entailment Check via DeepSeek
            # 1. Does New entail Rep?
            entails_rep = check_entailment_deepseek(client, question, ans, rep_ans)
            
            if entails_rep:
                # 2. Does Rep entail New?
                entails_new = check_entailment_deepseek(client, question, rep_ans, ans)
                
                if entails_new:
                    matched_cluster = cluster
                    break
        
        if matched_cluster:
            matched_cluster.append(i)
        else:
            clusters.append([i])
            
    return clusters

def calculate_semantic_entropy(model, processor, deepseek_client, image_path, objective, args, n_samples=10):
    """
    Calculate Semantic Entropy (SE) with DeepSeek clustering.
    """
    # 1. Prepare Inputs
    spectrum_img = generate_spectrum(image_path) # from function.py
    temp_spec_path = "temp_se_spec.png"
    cv2.imwrite(temp_spec_path, spectrum_img)
    user_text = construct_prompt(objective) # from function.py

    # Base conversation for generating answers
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": temp_spec_path}, {"type": "text", "text": user_text}]}
    ]
    text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    
    # Pre-process inputs once
    base_inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(args.device)
    prompt_len = base_inputs.input_ids.shape[1]

    # 2. Generate Multiple Samples (Sampling with Local VLM)
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 1.0,       # High temperature for diversity
        "do_sample": True,        # Enable sampling
        "top_k": 50,              # Limit to top 50 tokens (prevent tail risk)
        "top_p": 0.95,            # Nucleus sampling
        "repetition_penalty": 1.0,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id
    }

    generated_texts = []
    log_probs = []
    
    print(f"Sampling {n_samples} answers for {os.path.basename(image_path)}...")
    for i in range(n_samples):
        # Change Seed for Diversity
        set_seed(i + 1000)
        print(f"  - Generating sample with new seed {i + 1000}...")

        # Generate
        outputs = model.generate(**base_inputs, **generation_args)
        
        generated_ids = outputs[0][prompt_len:]
        response_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        try:
            if "<answer>" in response_text and "</answer>" in response_text:
                answer_content = response_text.split("<answer>")[1].split("</answer>")[0].strip()
            else:
                answer_content = response_text.strip()
        except:
            answer_content = response_text.strip()
            
        if not answer_content:
            answer_content = "Empty"
            
        # Debug print to verify diversity
        if i < 3: 
            print(f"    Sample {i}: {answer_content[:50]}...")

        generated_texts.append(answer_content)

        # Calculate Log Probability
        gen_input = {
            "input_ids": outputs,
            "attention_mask": torch.ones_like(outputs),
            "pixel_values": base_inputs.pixel_values,
            "image_grid_thw": base_inputs.image_grid_thw
        }
        log_prob_sum, seq_len = get_answer_log_prob(model, gen_input, prompt_len)
        
        # Length Normalization
        if seq_len > 0:
            norm_log_prob = log_prob_sum / seq_len
        else:
            norm_log_prob = -1e9
        log_probs.append(norm_log_prob)

    # 3. Cluster Answers (Using DeepSeek API)
    clusters = cluster_answers(generated_texts, user_text, deepseek_client)
    print(f"  - Formed {len(clusters)} clusters.")
    for idx, cluster in enumerate(clusters):
        print(f"    Cluster {idx+1}:")
        for ans_idx in cluster:
            print(f"      - {generated_texts[ans_idx]}")
    
    # 4. Calculate Semantic Entropy (Same logic)
    log_probs = np.array(log_probs)
    cluster_log_probs = []
    
    for cluster_indices in clusters:
        c_log_probs = log_probs[cluster_indices]
        cluster_log_prob = special.logsumexp(c_log_probs)
        cluster_log_probs.append(cluster_log_prob)
    
    cluster_log_probs = np.array(cluster_log_probs)
    total_log_prob = special.logsumexp(cluster_log_probs)
    normalized_cluster_probs = np.exp(cluster_log_probs - total_log_prob)
    
    semantic_entropy = -np.sum(normalized_cluster_probs * np.log(normalized_cluster_probs + 1e-10))
    
    # Discrete Entropy (Baseline)
    cluster_counts = np.array([len(c) for c in clusters])
    discrete_probs = cluster_counts / n_samples
    discrete_entropy = -np.sum(discrete_probs * np.log(discrete_probs + 1e-10))

    print(f"  > Semantic Entropy: {semantic_entropy:.4f}")
    print(f"  > Discrete Entropy: {discrete_entropy:.4f}")
    print(f"  > Num Clusters: {len(clusters)}")

    if os.path.exists(temp_spec_path):
        os.remove(temp_spec_path)
        
    return {
        "semantic_entropy": semantic_entropy,
        "discrete_entropy": discrete_entropy,
        "num_clusters": len(clusters),
        "generated_answers": generated_texts
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="experiment/FMIRAgent")
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset-path", type=str, default="dataset/test_split/DIV2K_Agents_0314")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--n-entropy-samples", type=int, default=5, help="Number of generations for entropy estimation")
    parser.add_argument("--objective", type=str, default="SR")
    args = parser.parse_args()

    # --- Initialize DeepSeek Client ---
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Please set DEEPSEEK_API_KEY environment variable.")
    
    print("Initializing DeepSeek Client...")
    deepseek_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("Loading Local VLM (Qwen)...")
    from utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(model_base=args.model_base, model_path=args.model_path, model_name=model_name, device=args.device)
    
    input_paths = sorted(os.listdir(args.dataset_path))
    input_paths = [os.path.join(args.dataset_path, p) for p in input_paths if p.endswith(('.tif'))]
    input_paths = input_paths[:args.samples]

    results = []
    
    for img_path in tqdm.tqdm(input_paths):
        try:
            res = calculate_semantic_entropy(
                model, processor, deepseek_client, img_path, args.objective, args, n_samples=args.n_entropy_samples
            )
            results.append(res)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Aggregate Results

    metrics = {
        "avg_semantic_entropy": np.mean([r["semantic_entropy"] for r in results]),
        "avg_discrete_entropy": np.mean([r["discrete_entropy"] for r in results]),
        "avg_num_clusters": np.mean([r["num_clusters"] for r in results])
    }
    print(f"\nAggregated Results for {args.model_path}: {metrics}")
    metrics["details"] = results

    # Save results
    output_file = f"semantic_entropy_results_{args.model_path.replace('/','_')}.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)


