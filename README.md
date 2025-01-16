# ADL_Gruppe_3

This project is an application developed by Group 3 for the Advanced Deep Learning course.

## Team Members and Responsibilities

-   **Nico:** Image Classifier
    -   Developed the `classify_image` function to identify bottle types from images.
-   **Buket:** Article Agent, Article Assembler
    -   Created the `generate_article_content` function to produce article text and image descriptions using an LLM, based on the bottle classification.
    -   Developed the `assemble_article` function to combine the article text and image descriptions into a final PDF using a Markdown template and Pandoc.
-   **Moritz:** Article Assembler, Diffusion Model, and Main Application
    -   Integrated the `diffusion_model` function to generate images based on textual prompts.
    -   Built the main application (`main.py`) with a GUI using PyQt6, integrating all modules.

## Modules

-   **Image Classifier:** Classifies the type of bottle in an input image.
-   **Article Agent:** Generates article content (text and image descriptions) based on the bottle classification, using an LLM (e.g., GPT-4) and tools like Wikipedia and DuckDuckGO for information retrieval.
-   **Article Assembler:** Combines the generated article text and image descriptions into a PDF document using a Markdown template and Pandoc.
-   **Diffusion Model:** Generates images based on textual prompts provided by the Article Agent.
-   **Main:** The main application with a GUI that ties all modules together, allowing users to capture images, classify them, generate articles, and view the results.

## Installation and Usage

Please refer to the `documentation/manual.qmd` file for detailed instructions on how to install and use the application. The manual provides step-by-step guidance, including setting up the environment using Docker, configuring your OpenAI API key, and using the GUI to generate articles.

## Project Structure

```
ADL_Gruppe_3/
├── agents/             # Article Agent module
├── assemblers/         # Article Assembler module
├── captured_images/    # Directory for captured images
├── classifiers/        # Image Classifier module
├── diffusion_model/    # Diffusion Model module
├── documentation/      # Documentation files (manual, architecture, etc.)
├── logs/               # Log files
├── resources/          # Resource files (e.g., HM logo)
├── main.py             # Main application script
├── Dockerfile          # Docker configuration file
├── requirements.txt    # Python dependencies
└── ...
```

