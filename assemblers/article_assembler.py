import subprocess
import time

from article_agent import generate_article_content, generate_image_description


# Function to create Markdown file
def create_markdown_file_from_template(template_file, output_file, replacements):
    with open(template_file, "r") as file:
        template_content = file.read()

    markdown_content = template_content.format(**replacements)

    with open(output_file, "w") as md_file:
        md_file.write(markdown_content)
    print(f"Markdown file created: {output_file}")


def convert_markdown_to_pdf(markdown_file, pdf_file):
    try:
        subprocess.run(["pandoc", markdown_file, "-o", pdf_file], check=True)
        print(f"PDF successfully created: {pdf_file}")
    except FileNotFoundError:
        print("Pandoc is not installed. Please install Pandoc and try again.")
    except subprocess.CalledProcessError:
        print("An error occurred while converting the Markdown file to PDF.")


if __name__ == "__main__":
    topic = "Wine Bottle"
    article = generate_article_content(topic)
    # image_description = generate_image_description(topic)

    replacements = {
        "title": "Generated Article",
        "author": "LangChain Automation",
        "date": time.strftime("%Y-%m-%d"),
        "topic": topic,
        "article": article,
        "image_descriptions": "to be added" # TODO
    }

    markdown_file = "output.md"
    pdf_file = "output.pdf"
    create_markdown_file_from_template("template.md", markdown_file, replacements)
    convert_markdown_to_pdf(markdown_file, pdf_file)
