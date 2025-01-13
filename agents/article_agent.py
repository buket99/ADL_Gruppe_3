#article_agent.py
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
import os
from dotenv import load_dotenv
import warnings
from pathlib import Path
from openai import OpenAI
from diffusion_model.diffusion_model import generate_image

"""
This module defines the ArticleAgent class, responsible for generating article content 
based on a given classification using a Large Language Model (LLM). It utilizes tools like 
Wikipedia and DuckDuckGo for information retrieval.
"""


class ArticleAgent:
    """
    Generates article content based on image classification using an LLM and external tools.
    """

    def __init__(
        self,
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=4000,
        timeout=120,
        max_retries=2,
    ):
        """
        Initializes the ArticleAgent.

        Args:
            model (str): The language model to use.
            temperature (float): The temperature for the LLM, controlling randomness.
            max_tokens (int): The maximum number of tokens in the generated response.
            timeout (int): The timeout in seconds for the LLM.
            max_retries (int): The maximum number of retries for the LLM.
        """
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="langchain"
        )

        load_dotenv()
        self.API_KEY = os.getenv("OPEN_AI_API")
        self.base_dir = Path(__file__).resolve().parent.parent
        self.assembler_dir = self.base_dir / "assemblers"
        self.diffusion_model_dir = self.base_dir / "diffusion_model"

        # Initialize the language model
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            api_key=self.API_KEY,
        )

        # Define LangChain tools
        wikipedia_tool = WikipediaAPIWrapper()
        duckduckgo_tool = DuckDuckGoSearchRun()
        self.tools = [
            Tool(
                name="Wikipedia",
                func=wikipedia_tool.run,
                description="Fetch summaries and details about topics from Wikipedia.",
            ),
            Tool(
                name="DuckDuckGo",
                func=duckduckgo_tool.run,
                description="Perform web searches to gather information.",
            ),
        ]

        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def generate_paragraph(self, pClassification, section):
        """
        Generates a detailed paragraph about a specific section of the article.

        Args:
            pClassification (str): The classification of the image.
            section (str): The section to generate a paragraph for.

        Returns:
            str: The generated paragraph.
        """
        message = [
            SystemMessage(content="You are an expert article writer."),
            HumanMessage(
                content=f"Write a detailed paragraph (at least 200 words) about '{pClassification}', focusing on "
                f"'{section}'. Use tools like Wikipedia and DuckDuckGo for research. At the end of each "
                f"paragraph write the tool in brackets that you have used."
            ),
        ]
        response = self.llm.invoke(message)
        return response.content

    def generate_image_description(self, pClassification, paragraph, image_number):
        """
        Generates a detailed description for an image related to the article.

        Args:
            pClassification (str): The classification of the image.
            paragraph (str): The paragraph related to the image.
            image_number (int): The number of the image in the article.

        Returns:
            str: The generated image description.
        """
        message = [
            SystemMessage(
                content="You are an expert in creating detailed image descriptions."
            ),
            HumanMessage(
                content=f"""
                    Imagine a visually engaging image related to "{pClassification}". This image should also complement the 
                    following paragraph:
                    {paragraph}
                    Describe the image in detail as if it were included in the article. This is the description for image 
                    number {image_number}.
                    Be specific about the following aspects:
                    - What is in the foreground?
                    - What is in the background?
                    - What are the colors, textures, and lighting like?
                    - If there are any objects, describe their arrangement and details.
                    Ensure the description is vivid and helps the reader visualize the image while aligning with the 
                    paragraph content.
                    """
            ),
        ]
        response = self.llm.invoke(message)
        return response.content

    def generate_article_content(self, pClassification, selected_diffusion_model):
        """
        Generates the content of the article, consisting of multiple paragraphs.

        Args:
            pClassification (str): The classification of the image.
            selected_diffusion_model (str): The selected diffusion model for image generation.

        Returns:
            tuple: (list, list) A tuple containing the list of generated paragraphs and a list of image paths.
        """
        # Define the sections for the article
        sections = [
            "composition, origins, and production",
            "cultural, historical, and practical Importance",
            "Sensory Experience and Effects",
            "Global Influence and Trade",
        ]

        # Generate each paragraph
        paragraphs = []
        image_paths = []
        for i, section in enumerate(sections):
            print(f"Generating paragraph for: {section}")
            paragraph = self.generate_paragraph(pClassification, section)
            paragraphs.append(paragraph)

            # Generate image description and create image
            image_description = self.generate_image_description(
                pClassification, paragraph, i + 1
            )
            image_path = generate_image(
                selected_diffusion_model,
                image_description,
                f"image_{i+1}_for_{pClassification}",
            )
            image_paths.append(image_path)

        # Check total word count and expand if necessary
        min_words = 1000
        total_word_count = sum(
            len(p.split()) for p in paragraphs
        )  # Calculate total words
        while total_word_count < min_words:
            print(
                f"Word count ({total_word_count}) is below the minimum ({min_words}). Expanding content..."
            )
            additional_message = HumanMessage(
                content=f"The article is currently {total_word_count} words long. Add a new paragraph with a fun fact about "
                f"'{pClassification}' to help reach at least {min_words} words."
            )
            response = self.llm.invoke(
                [
                    SystemMessage(content="Expand the article further."),
                    additional_message,
                ]
            )
            print(f"LLM response: {response}")
            paragraphs.append(response.content)
            total_word_count = sum(
                len(p.split()) for p in paragraphs
            )  # Recalculate total words

        return paragraphs, image_paths

    def get_available_models(self):
        """
        Returns a list of available agent models from the OpenAI API.

        Returns:
            list: A list of available agent models.
        """
        try:
            client = OpenAI(api_key=self.API_KEY)
            response = client.models.list()
            available_models = [
                model.id for model in response.data if "gpt" in model.id
            ]
            print(available_models)
            return available_models
        except Exception as e:
            print(f"Error fetching available models: {e}")
            return ["gpt-3.5-turbo"]  # Fallback to default model
