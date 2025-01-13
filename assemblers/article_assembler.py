#article_assembler.py
import subprocess
import time
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
This module provides functionalities to assemble an article from generated content 
and convert it into a PDF. It uses a Markdown template and Pandoc for the conversion.
"""


class ArticleAssembler:
    """
    Assembles the article from generated text and images and converts it to PDF.
    """

    def __init__(self):
        """
        Initializes the ArticleAssembler.
        """
        self.base_dir = Path(__file__).resolve().parent.parent
        self.template_path = self.base_dir / "assemblers" / "template.md"

    def create_markdown_file_from_template(self, output_file, replacements):
        """
        Creates a Markdown file from a template, replacing placeholders with actual content.

        Args:
            output_file (str or Path): The path to the output Markdown file.
            replacements (dict): A dictionary of replacements, where keys are placeholders
                                 in the template and values are the content to replace them with.
        """
        with open(self.template_path, "r") as file:
            template_content = file.read()

        markdown_content = template_content.format(**replacements)

        with open(output_file, "w") as md_file:
            md_file.write(markdown_content)
        print(f"Markdown file created: {output_file}")

    def convert_markdown_to_pdf(self, markdown_file, pdf_file):
        """
        Converts a Markdown file to a PDF using Pandoc.

        Args:
            markdown_file (str or Path): The path to the input Markdown file.
            pdf_file (str or Path): The path to the output PDF file.
        """
        try:
            subprocess.run(
                ["pandoc", str(markdown_file), "-o", str(pdf_file)], check=True
            )
            print(f"PDF successfully created: {pdf_file}")
        except FileNotFoundError:
            print("Pandoc is not installed. Please install Pandoc and try again.")
        except subprocess.CalledProcessError:
            print("An error occurred while converting the Markdown file to PDF.")

    def assemble_article(self, topic, paragraphs, image_paths):
        """
        Assembles the article content into a Markdown file and converts it to PDF.

        Args:
            topic (str): The topic of the article.
            paragraphs (list): List of paragraphs for the article.
            image_paths (list): List of paths to the generated images.
        """
        replacements = {
            "title": f"Article about {topic}",
            "author": "ADL Gruppe 3",
            "date": time.strftime("%Y-%m-%d"),
            "topic": topic,
            "article_paragraph_1": paragraphs[0],
            "article_paragraph_2": paragraphs[1],
            "article_paragraph_3": paragraphs[2],
            "article_paragraph_4": paragraphs[3],
            "image_path_1": image_paths[0],
            "image_path_2": image_paths[1],
            "image_path_3": image_paths[2],
            "image_path_4": image_paths[3],
        }

        markdown_file = self.base_dir / "output.md"
        pdf_file = self.base_dir / "output.pdf"
        self.create_markdown_file_from_template(markdown_file, replacements)
        self.convert_markdown_to_pdf(markdown_file, pdf_file)
        print("done")
