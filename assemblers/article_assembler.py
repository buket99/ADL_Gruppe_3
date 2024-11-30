import subprocess
import time

from agents.article_agent import generate_article_content, generate_image_description


# Function to create Markdown file
def create_markdown_file(topic, article, image_descriptions, output_file):
    markdown_content = f"""---
title: "Generated Article"
author: "LangChain Automation"
date: "{time.strftime('%Y-%m-%d')}"
---

# Article: {topic}

{article}

---

## Image Descriptions to be added

"""
    with open(output_file, "w") as md_file:
        md_file.write(markdown_content)
    print(f"Markdown file created: {output_file}")


# Function to convert Markdown to PDF
def markdown_to_pdf(markdown_file, output_pdf):
    try:
        subprocess.run(
            ["pandoc", markdown_file, "-o", output_pdf, "--pdf-engine=xelatex"],
            check=True
        )
        print(f"PDF created successfully: {output_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"Error during PDF conversion: {e}")
    except FileNotFoundError:
        print("Pandoc is not installed or not found in the system PATH.")


if __name__ == "__main__":
    topic = "Wine Bottle" # TODO: here pClassification will be added

    article = generate_article_content(topic)
    image_descriptions = generate_image_description(topic)

    # Output files
    markdown_file = "output.md"
    pdf_file = "output.pdf"

    # Create Markdown file
    create_markdown_file(topic, article, image_descriptions, markdown_file)

    # Convert Markdown to PDF
    markdown_to_pdf(markdown_file, pdf_file)

    print("All tasks completed successfully!")
