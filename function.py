import os
import cv2
import torch
import json
import utility
import numpy as np
import gradio as gr
import numpy as np
import model
import random

from functools import partial
from div2k import normalize
from transformers import TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from tifffile import imread, imwrite as imsave
from qwen2.src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init

import warnings
warnings.filterwarnings("ignore")

# define global constants
DEVICES = ['CPU','CUDA','Paralleled CUDA']
QUANT = ['float32','float16',]
TASKS = ['SR_5','SR_7','SR_9','Isotropic_2','Isotropic_4','Isotropic_6','Projection','Denoising_10','Denoising_20','Denoising_30','Volumetric']
INPUTS = ['SR', 'Denoising', 'Isotropic', 'Projection', 'Volumetric']

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

processing_options = {
    "Projection": ["Projection"],
    "Volumetric": ["Volumetric"],
    "Isotropic": ["Isotropic_None", "Isotropic_2", "Isotropic_4", "Isotropic_6"],
    "SR": ["SR_None", "SR_5", "SR_7", "SR_9"],
    "Denoising": ["Denoising_None", "Denoising_10", "Denoising_20", "Denoising_30"]
}

objectives = ["Projection", "Volumetric", "Isotropic", "SR/Denoise"]

# define unifmir args
class Args:
    model = 'SwinIR'
    test_only = True
    resume = 0
    modelpath = None
    save = None
    task = None
    dir_data = None
    dir_demo = None
    data_test = None

    epoch = 1000
    batch_size = 16
    patch_size = None
    rgb_range = 1
    n_colors = 1
    inch = None
    datamin = 0
    datamax = 100
    
    cpu = False
    print_every = 1000
    test_every = 2000
    load=''
    lr = 0.00005
    n_GPUs = 1
    n_resblocks = 8
    n_feats = 32
    save_models = True
    save_results = True
    save_gt = False

    debug = False
    scale = None
    chunk_size = 144
    n_hashes = 4
    chop = False
    self_ensemble = False
    no_augment = False
    inputchannel = None

    act = 'relu'
    extend = '.'
    res_scale = 0.1
    shift_mean = True
    dilation = False
    precision = 'single'

    seed = 1
    local_rank = 0
    n_threads = 0
    reset = False
    split_batch = 1
    gan_k = 1

def load_model(type, device='CUDA', chop=False, quantization='float32', skip='No'):
    ARGS = Args()

    if quantization == 'float16':
        ARGS.precision = 'half'

    if chop == 'Yes':
        ARGS.chop = True

    if device == 'CPU':
        ARGS.cpu = True
    elif device == 'CUDA':
        ARGS.cpu = False
        ARGS.n_GPUs = 1
    elif device == 'Paralleled CUDA':
        ARGS.cpu = False
        ARGS.n_GPUs = torch.cuda.device_count()
    else:
        print("Device not found!")
        return "Device not found"

    if 'SR' in type:
        ARGS.task = 1
        ARGS.patch_size = 128
        ARGS.scale = '1'
        ARGS.inch = 1
        # ARGS.chop = False

        if type == 'SR_5':
            ARGS.save = 'SwinIR'
            ARGS.modelpath = 'experiment/SwinIR-degrade-b5/model/model_best.pt'
        elif type == 'SR_7':
            ARGS.save = 'SwinIR'
            ARGS.modelpath = 'experiment/SwinIR-degrade-b7/model/model_best.pt'
        elif type == 'SR_9':
            ARGS.save = 'SwinIR'
            ARGS.modelpath = 'experiment/SwinIR-degrade-b9/model/model_best.pt'
        else:
            print("Model not found!")
            return "Model not found"
        

    elif 'Denoising' in type:
        ARGS.task = 2
        ARGS.patch_size = 128
        ARGS.scale = '1'
        
        if type == 'Denoising_10':
            ARGS.save = 'SwinIR'
            ARGS.modelpath = 'experiment/SwinIR-degrade-n10/model/model_best.pt'
        elif type == 'Denoising_20':
            ARGS.save = 'SwinIR'
            ARGS.modelpath = 'experiment/SwinIR-degrade-n20/model/model_best.pt'
        elif type == 'Denoising_30':
            ARGS.save = 'SwinIR'
            ARGS.modelpath = 'experiment/SwinIR-degrade-n30/model/model_best.pt'
        else:
            print("Model not found!")
            return "Model not found"

    elif 'Isotropic' in type:
        ARGS.task = 3
        ARGS.patch_size = 128
        ARGS.scale = '1'

        if type == 'Isotropic_2':
            ARGS.save = 'SwinIR'
            ARGS.modelpath = 'experiment/SwinIR-degrade-i2/model/model_best.pt'
        elif type == 'Isotropic_4':
            ARGS.save = 'SwinIR'
            ARGS.modelpath = 'experiment/SwinIR-degrade-i4/model/model_best.pt'
        elif type == 'Isotropic_6':
            ARGS.save = 'SwinIR'
            ARGS.modelpath = 'experiment/SwinIR-degrade-i6/model/model_best.pt'
        else:
            print("Model not found!")
            return "Model not found"

    elif 'Projection' in type:
        ARGS.task = 4
        ARGS.patch_size = 128
        ARGS.scale = '1'
        ARGS.inch = 50

        ARGS.model = 'SwinIRproj2stg_enlcn_2npz'
        ARGS.save = 'SwinIRproj2stg_enlcn_2npzProjection_Flywing'
        ARGS.resume = -6
        ARGS.modelpath = './experiment/SwinIRproj2stg_enlcn_2npzProjection_Flywing/model_best6.pt'

    elif 'Volumetric' in type:
        ARGS.task = 5
        ARGS.patch_size = 176
        ARGS.scale = '1'

        ARGS.model = 'SwinIR2t3_stage2'
        ARGS.save = 'SwinIR2t3_stage2VCD'
        ARGS.resume = -17
        ARGS.modelpath = './experiment/SwinIR2t3_stage2VCD/model_best17.pt'
    else:
        print("Task not found!")
        return "Task not found"
    
    ARGS.scale = list(map(lambda x: int(x), ARGS.scale.split('+')))

    checkpoint = utility.checkpoint(ARGS)
    MODEL = model.Model(ARGS, checkpoint)
    MODEL.eval()

    if skip == 'Yes' and ARGS.n_GPUs <= 1:
        if 'Projection' in type:
            MODEL.model.denoise.layers[1].prune()
        else:
            MODEL.model.layers[1].prune()

    return MODEL

def visualize(img_input, progress=gr.Progress()):
    print(f'Opening {img_input.name}...')
    if not img_input.name.endswith('.tif'):
        gr.Error("Image must be a tiff file!")
        return None
    
    image = imread(img_input.name)
    shape = image.shape
    print(f'Image shape: {shape}')

    if len(shape) == 2:
        image = utility.savecolorim(None, image, norm=True)
        return [[image], f'2D image loaded with shape {shape}']
    elif len(shape) == 3:
        clips = []
        # show middle 10 slices
        image = image[max(shape[0]//2-5, 0):min(shape[0]//2+5, shape[0])]
        for clip in image:
            clips.append(utility.savecolorim(None, clip, norm=True))
        return [clips, f'3D image loaded with shape {shape}']
    else:
        gr.Error("Image must be 2 or 3 dimensional!")
        return None
    

def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""
    
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    # print('minmax: ', mi, ma)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)
    
    x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)
    
    return x
    
def rearrange3d_fn(image):
    """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
    """

    image = np.squeeze(image)  # remove channels dimension
    # print('reshape : ' + str(image.shape))
    depth, height, width = image.shape
    image_re = np.zeros([height, width, depth])
    for d in range(depth):
        image_re[:, :, d] = image[d, :, :]
    return image_re

def lf_extract_fn(lf2d, n_num=11, mode='toChannel', padding=False):
    """
    Extract different views from a single LF projection

    Params:
        -lf2d: numpy.array, 2-D light field projection in shape of [height, width, channels=1]
        -mode - 'toDepth' -- extract views to depth dimension (output format [depth=multi-slices, h, w, c=1])
                'toChannel' -- extract views to channel dimension (output format [h, w, c=multi-slices])
        -padding -   True : keep extracted views the same size as lf2d by padding zeros between valid pixels
                        False : shrink size of extracted views to (lf2d.shape / Nnum);
    Returns:
        ndarray [height, width, channels=n_num^2] if mode is 'toChannel'
                or [depth=n_num^2, height, width, channels=1] if mode is 'toDepth'
    """
    n = n_num
    h, w, c = lf2d.shape
    if padding:
        if mode == 'toDepth':
            lf_extra = np.zeros([n * n, h, w, c])  # [depth, h, w, c]
        
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, i: h: n, j: w: n, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        elif mode == 'toChannel':
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([h, w, n * n])
            
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[i: h: n, j: w: n, d] = lf2d[i: h: n, j: w: n]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)
    else:
        new_h = int(np.ceil(h / n))
        new_w = int(np.ceil(w / n))
    
        if mode == 'toChannel':
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([new_h, new_w, n * n])
        
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[:, :, d] = lf2d[i: h: n, j: w: n]
                    d += 1
        elif mode == 'toDepth':
            lf_extra = np.zeros([n * n, new_h, new_w, c])  # [depth, h, w, c]
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, :, :, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)

    return lf_extra

def load_models(device, chop, quantization, args, progress=gr.Progress()):
    global sr_models, noise_models, proj_models, iso_models, vol_models, processor, language_model

    load_fn = partial(load_model, device=device, chop=chop, quantization=quantization)

    # load restoration models
    sr_models = {'SR_None': (lambda x,y: x), 'SR_5': load_fn('SR_5'), 'SR_7': load_fn('SR_7'), 'SR_9': load_fn('SR_9')}
    progress(0.25)
    noise_models = {'Denoising_None': (lambda x,y: x), 'Denoising_10': load_fn('Denoising_10'), 'Denoising_20': load_fn('Denoising_20'), 'Denoising_30': load_fn('Denoising_30')}
    progress(0.5)
    iso_models = {'Isotropic_None': (lambda x,y: x), 'Isotropic_2': load_fn('Isotropic_2'), 'Isotropic_4': load_fn('Isotropic_4'), 'Isotropic_6': load_fn('Isotropic_6')}
    progress(0.75)
    proj_models = {'Projection': load_fn('Projection') }
    vol_models = {'Volumetric': load_fn('Volumetric') }
    
    # load language model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    processor, language_model = load_pretrained_model(model_base = args.model_base, model_path = args.model_path, model_name=model_name, device=args.device)
    language_model = torch.compile(language_model)

    return f"Models loaded! device={device}, chop={chop}, quantization={quantization}"

def get_plan(img_input, plan):
    plan_path = os.path.join(os.path.dirname(img_input),f'plans{os.path.basename(img_input)[:-4]}.json')
    if not os.path.exists(plan_path):
        return None, None, None, None
    try:
        plan = plan.split("<answer>")[-1].split("</answer>")[0].strip()
        with open(plan_path,'r') as f:
            plan_details = json.load(f)[plan]
        return plan_details['psnr'], plan_details['ssim'], plan_details['lpips'], plan_details['dists']
    except Exception as e:
        print(f"Error in loading plan details: {str(e)}")
        return None, None, None, None

@torch.no_grad()
def run_plan(data_image, plan, progress=gr.Progress()):
    try: 
        plan = plan.split("<answer>")[-1].split("</answer>")[0].strip()
    except:
        plan = "SR_None Denoising_None"

    image = imread(data_image)
    # print(image.shape, image.max(), image.min())
    image = normalize(image, 0, 100, clip=True)
    lrs = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0).cuda()

    image = lrs.clone()
    # print(f"Plan: {plan}")

    plan = plan.replace("SR_?", np.random.choice(list(sr_models.keys())))
    plan = plan.replace("Denoising_?", np.random.choice(list(noise_models.keys())))
    plan = plan.replace("Isotropic_?", np.random.choice(list(iso_models.keys())))

    try:
        if "Volumetric" in plan:
            vol_model = vol_models[plan.split()[0]]
            sr_model = sr_models[plan.split()[1]] 
            noise_model = noise_models[plan.split()[2]]
        elif "Projection" in plan:
            proj_model = proj_models[plan.split()[0]]
            sr_model = sr_models[plan.split()[1]]
            noise_model = noise_models[plan.split()[2]]
        elif "Isotropic" in plan:
            iso_model = iso_models[plan.split()[0]]
            sr_model = sr_models[plan.split()[1]]
            noise_model = noise_models[plan.split()[2]]
            image = image.unsqueeze(0)
        else:
            sr_model = sr_models[plan.split()[0]]
            noise_model = noise_models[plan.split()[1]]
            image = image.unsqueeze(0)

        # 1chw to c1hw
        image = image.permute(1,0,2,3).contiguous()
        image = sr_model(noise_model(image, 0).clamp(0, 1), 0).clamp(0, 1)
        image = image.permute(1,0,2,3).contiguous()

        if "Volumetric" in plan:
            image = image * 2 - 1
            image = vol_model(image, 0)
            image = image[1] if image is tuple else image
            image = (image.clamp(-1, 1) + 1) / 2 
        elif "Projection" in plan:
            image = proj_model(image, 0)
            image = image[1] if image is tuple else image
        elif "Isotropic" in plan:
            image = iso_model(image, 0)
    except Exception as e:
        return [None, f"Error in running plan: {str(e)}"]

    image = image.squeeze().cpu().numpy().clip(0,1)
    axes = "YX" if not "Volumetric" in plan else "ZYX"
    utility.save_tiff_imagej_compatible('output.tif', image, axes)
    return ['output.tif', "Output Successfully Saved!"]

def set_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If you are using a GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def bot_streaming(data_images, objective, args, progress=gr.Progress()):
    problem = 'You are a microscopy expert. Given the Fourier spectrum of a microscopy image. How to enhance the quality of this image for <metric>? You can use the tools: <operation><isotropic1>SR_(None/5/7/9) Denoising_(None/10/20/30). Please analyze the image and give the plan. For example, "<think> The spectrum shows ? which indicates that the image has <isotropic2>a blur kernel of ? and a noise of level ?. </think> <answer> <operation><isotropic3>SR_? Denoising_? </answer>".'

    is_single = False
    if type(data_images) is not list:
        is_single = True
        print("Single image mode")
        data_images = [data_images]

    input_texts = []
    input_images = []

    for data_image in data_images:
        # check if spectrum image exists
        if not os.path.exists(f"{data_image}_spectrum_new.png"):
            # generate spectrum image
            image = imread(data_image)
            image = normalize(image, 0, 100, clip=True)
            if len(image.shape) == 3:
                image = image[0]
            if image.shape[0] > 128 and image.shape[1] > 128:
                # center crop
                x = image.shape[0]//2 - 64
                y = image.shape[1]//2 - 64
                image = image[x:x+128, y:y+128]
            else:
                image = cv2.resize(image, (128, 128))
            dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
            cv2.imwrite(f"{data_image}_spectrum_new.png", magnitude_spectrum)

        # generate conversation
        operation = "Projection " if "Projection" in objective else ("Volumetric " if "Volumetric" in objective else "")
        problem = problem.replace("<metric>", 'psnr')
        problem = problem.replace("<operation>", operation)
        problem = problem.replace("<isotropic1>", "Isotropic_(None/2/4/6) " if "Isotropic" in objective else "")
        problem = problem.replace("<isotropic2>", "a resolution ratio of ?, " if "Isotropic" in objective else "")
        problem = problem.replace("<isotropic3>", "Isotropic_? " if "Isotropic" in objective else "")

        QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
        message = {
            "text": QUESTION_TEMPLATE.format(Question=problem),
            "files": [f"{data_image}_spectrum_new.png"]
        }

        # Initialize variables
        images = []
        if message["files"]:
            for file_item in message["files"]:
                if isinstance(file_item, dict):
                    file_path = file_item["path"]
                else:
                    file_path = file_item
                
                images.append(file_path)

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_content = []
        for image in images:
            user_content.append({"type": "image", "image": image})
        user_text = message['text']
        if user_text:
            user_content.append({"type": "text", "text": user_text})
        conversation.append({"role": "user", "content": user_content})

        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conversation)

        input_texts += [prompt]
        input_images += image_inputs

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": processor.tokenizer.eos_token_id
    }

    # offload language model to GPU
    language_model.to(args.device)
    
    inputs = processor(text=input_texts, images=input_images, padding=True, return_tensors="pt").to(args.device) 
    outputs = language_model.generate(**inputs, **generation_args)
    answers = processor.batch_decode(outputs, skip_special_tokens=True)

    # offload language model to CPU
    language_model.to('cpu')

    return answers[0] if is_single else answers