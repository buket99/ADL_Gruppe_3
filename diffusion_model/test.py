from diffusers import StableDiffusionPipeline
import torch
import re
from pathlib import Path

# Define the base directory as the script's location
base_dir = Path(__file__).parent.resolve()

# Extract image description from output.txt
def extract_image_description(file_path: Path) -> str:
    file_path = base_dir / file_path
    with file_path.open("r") as f:
        content = f.read()

    match = re.search(r"Image Description:\n(.*(?:\n.*)*)", content)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Image description not found in the file.")

# Main function
def main():
    # Extract the prompt from the file
    prompt_file = base_dir / "output.txt"
    prompt = extract_image_description(prompt_file)
    print(f"Generating image for prompt: {prompt}")

    # Load the model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("mps")  # Use "cuda" on GPU-enabled systems or "cpu" if GPU is unavailable

    # Generate and save the image
    output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output" / "output_small_from_text.png"

    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"Image saved as {output_path}")

if __name__ == "__main__":
    main()
