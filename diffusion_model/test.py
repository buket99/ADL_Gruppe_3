from diffusers import StableDiffusionPipeline
import torch
import re


# Extract image description from output.txt
def extract_image_description(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    match = re.search(r"Image Description:\n(.*(?:\n.*)*)", content)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Image description not found in the file.")


# Main function
def main():
    # Extract the prompt from the file
    prompt = extract_image_description("output.txt")
    print(f"Generating image for prompt: {prompt}")

    # Load the model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("mps")  # Use "cuda" on GPU-enabled systems or "cpu" if GPU is unavailable

    # Generate and save the image
    image = pipe(prompt).images[0]
    image.save("output_small_from_text.png")
    print("Image saved as output_small.png")


if __name__ == "__main__":
    main()
