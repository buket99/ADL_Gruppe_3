from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

# Define the base directory as the script's location
BASE_DIR = Path(__file__).parent.resolve()

# Model options with descriptions
MODELS = {
    "stable-diffusion-2-1-base (fast)": "stabilityai/stable-diffusion-2-1-base",
    "stable-diffusion-v1-5 (mid)": "runwayml/stable-diffusion-v1-5",
    "ldm-text2im-large-256 (slow)": "CompVis/ldm-text2im-large-256",
}

def get_available_diffusion_models():
    """
    Returns a dictionary of available diffusion models.
    """
    return MODELS

def generate_image(model_key: str, prompt: str, output_filename: str = "output_image"):
    """
    Generate an image using the specified model.

    Parameters:
        model_key (str): One of fast, mid, or slow.
        prompt (str): The text prompt for image generation.
        output_filename (str): The base filename for the output image (without extension).
    """
    if model_key not in MODELS:
        raise ValueError(
            f"Invalid model key: {model_key}. Choose from {list(MODELS.keys())}."
        )

    model_id = MODELS[model_key]
    print(f"Loading model: {model_id} ({model_key})")

    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # Check for MPS availability and fallback to CUDA or CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    pipe.to(device)

    # Generate the image
    print(f"Generating image for prompt: {prompt} on device: {device}")
    image = pipe(prompt).images[0]

    # Save the image
    output_path = BASE_DIR / "output" / f"{output_filename}_{model_key}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Image saved as {output_path}")

    return str(output_path) # Return the path to the generated image