import argparse
import gradio as gr
import numpy as np

from function import *

# seed everything
set_seed(42)

def bot_streaming_wrap(img_input, objective_options, progress=gr.Progress()):
    return bot_streaming(img_input, objective_options, args, progress)

def load_models_wrap(device, chop, quantization, progress=gr.Progress()):
    return load_models(device, chop, quantization, args, progress)

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="experiment/FMIRAgent-7B")
parser.add_argument("--model-base", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--no-reasoning", action="store_true")
parser.add_argument("--use-ft", action="store_true", help="Use fine-tuned versions of SR and Denoising models")
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()

with gr.Blocks(title="FMIRAgents Web Demo") as demo:

    gr.Markdown("# Self-Explained Thinking Agent for Autonomous Microscopy Restoration")

    with gr.Row():
        with gr.Column():

            gr.Markdown("## How to Use This Demo")
            gr.Markdown("### Step 1: Upload Your Image")
            gr.Markdown("• Select a microscopy image file in **TIFF format** (.tif or .tiff)")
            gr.Markdown("• Supported formats: 2D grayscale images or 3D image stacks")
            gr.Markdown("• Click **'Check Input'** to preview your uploaded image")
            
            gr.Markdown("### Step 2: Configure Processing Settings")
            gr.Markdown("• **Device**: Choose computational backend (CPU, CUDA, or Parallel CUDA)")
            gr.Markdown("• **Quantization**: Select precision (float16 for faster processing, float32 for higher quality)")
            gr.Markdown("• **Chop**: Enable memory-efficient processing for large images (recommended: Yes)")
            gr.Markdown("• Click **'Load Model'** to initialize the AI models (this may take a few moments)")

        with gr.Column():

            gr.Markdown("### Step 3: Generate Enhancement Plan")
            gr.Markdown("• **Operation**: Select the primary objective for your image:")
            gr.Markdown("  - **SR/Denoise**: General enhancement (super-resolution + denoising)")
            gr.Markdown("  - **Projection**: Light field microscopy projection enhancement")
            gr.Markdown("  - **Volumetric**: 3D volume reconstruction and enhancement")
            gr.Markdown("  - **Isotropic**: Convert anisotropic to isotropic resolution")
            gr.Markdown("• Click **'Generate Plan'** to let the AI analyze your image and create an optimal enhancement strategy")
    
        with gr.Column():

            gr.Markdown("### Step 4: Execute Enhancement")
            gr.Markdown("• Click **'Restore Image'** to apply the generated plan to your image")
            gr.Markdown("• Processing time varies based on image size and selected operations")
            
            gr.Markdown("### Step 5: Review Results")
            gr.Markdown("• Click **'Check Output'** to preview the enhanced image")
            gr.Markdown("• Download the enhanced image from the 'Output File' section")
            gr.Markdown("• The output maintains the same format and bit depth as your input")
            
            gr.Markdown("---")
            gr.Markdown("💡 **Tips**: For best results, ensure your input image has good contrast and is properly focused. Large images may require chop mode for memory efficiency.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Upload Image")
            img_input = gr.File(label="Input File", interactive=True)
            img_visual = gr.Gallery(label="Input Visualization", interactive=False)
            input_message = gr.Textbox(label="Image Information", value="Image not loaded")
            check_input = gr.Button("Check Input") 

        with gr.Column():
            gr.Markdown("## Plan Generation")
            device = gr.Dropdown(label="Device", choices=DEVICES, value="CUDA")
            quantization = gr.Dropdown(label="Quantization", choices=QUANT, value="float16")
            chop = gr.Dropdown(label="Chop", choices=['Yes','No'], value="Yes")
            load_progress = gr.Textbox(label="Model Information", value="Model not loaded")
            load_btn = gr.Button("Load Model")
            objective_options = gr.Dropdown(label="Operation", choices=objectives, value="SR/Denoise")
            plan_message = gr.Textbox(label="Plan Information", value="No plan generated")
            plan_btn = gr.Button("Generate Plan")

        with gr.Column():
            gr.Markdown("## Restore Image")
            output_file = gr.File(label="Output File", interactive=False)
            img_output = gr.Gallery(label="Output Visualization")
            output_message = gr.Textbox(label="Output Information", value="Image not loaded")
            run_btn = gr.Button("Restore Image")
            display_btn = gr.Button("Check Output")

    check_input.click(visualize, inputs=img_input, outputs=[img_visual, input_message], queue=True)
    display_btn.click(visualize, inputs=output_file, outputs=[img_output, output_message], queue=True)
    load_btn.click(load_models_wrap,inputs=[device, chop, quantization],outputs=load_progress, queue=True)
    plan_btn.click(bot_streaming_wrap, inputs=[img_input, objective_options], outputs=plan_message, queue=True)
    run_btn.click(run_plan, inputs=[img_input, plan_message], outputs=[output_file, output_message], queue=True)

demo.queue().launch(server_name='0.0.0.0', server_port=8989)