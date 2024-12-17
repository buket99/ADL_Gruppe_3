import subprocess
import time

from agents.article_agent import generate_article_content, generate_image_description


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
    topic = "Coke" # TODO: here is topic/classification
    print("start generated paragraphs..")
    article_paragraphs = generate_article_content(topic)
    image_descriptions = [
        generate_image_description(topic, article_paragraphs[i], i + 1) for i in range(len(article_paragraphs))
    ]
    replacements = {
        "title": f"Article about {topic}",
        "author": "ADL Gruppe 3",
        "date": time.strftime("%Y-%m-%d"),
        "topic": topic,
        "article_paragraph_1": article_paragraphs[0],
        "article_paragraph_2": article_paragraphs[1],
        "article_paragraph_3": article_paragraphs[2],
        "article_paragraph_4": article_paragraphs[3],
        "image_description_1": image_descriptions[0],  # TODO: image has to be created or handed over
        "image_description_2": image_descriptions[1],  # TODO: image has to be created or handed over
        "image_description_3": image_descriptions[2],  # TODO: image has to be created or handed over
        "image_description_4": image_descriptions[3],  # TODO: image has to be created or handed over
    }

    markdown_file = "output.md"
    pdf_file = "output.pdf"
    create_markdown_file_from_template("template.md", markdown_file, replacements)
    convert_markdown_to_pdf(markdown_file, pdf_file)
    print("done")
