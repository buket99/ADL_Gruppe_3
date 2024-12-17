from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
import time
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPEN_AI_API")

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # TODO: here you decide the model
    temperature=0.5,  # reduced for more consistent output
    max_tokens=4000,  # increased for longer responses
    timeout=120,
    max_retries=2,
    api_key=API_KEY # TODO: here is API_Key for OpenAI
)

# Define LangChain tools
wikipedia_tool = WikipediaAPIWrapper()
duckduckgo_tool = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia_tool.run,
        description="Fetch summaries and details about topics from Wikipedia."
    ),
    Tool(
        name="DuckDuckGo",
        func=duckduckgo_tool.run,
        description="Perform web searches to gather information."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


def generate_paragraph(pClassification, section):
    message = [
        SystemMessage(content="You are an expert article writer."),
        HumanMessage(content=f"Write a detailed paragraph (at least 200 words) about '{pClassification}', focusing on "
                             f"'{section}'. Use tools like Wikipedia and DuckDuckGo for research. At the end of each "
                             f"paragraph write the tool in brackets that you have used.")
    ]
    response = llm.invoke(message)
    return response.content


def generate_image_description(pClassification, paragraph, image_number):
    message = [
        SystemMessage(content="You are an expert in creating detailed image descriptions."),
        HumanMessage(content=f"""
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
                """)
    ]
    response = llm.invoke(message)
    return response.content


# Function to combine paragraphs into an article
def generate_article_content(pClassification):
    # Define the sections for the article
    sections = [
        "composition, origins, and production",
        "cultural, historical, and practical Importance",
        "Sensory Experience and Effects",
        "Global Influence and Trade"
    ]

    # Generate each paragraph
    paragraphs = []
    for section in sections:
        print(f"Generating paragraph for: {section}")
        paragraph = generate_paragraph(pClassification, section)
        paragraphs.append(paragraph)

    # Check total word count and expand if necessary
    min_words = 1000
    total_word_count = sum(len(p.split()) for p in paragraphs)  # Calculate total words
    while total_word_count < min_words:
        print(f"Word count ({total_word_count}) is below the minimum ({min_words}). Expanding content...")
        additional_message = HumanMessage(
            content=f"The article is currently {total_word_count} words long. Add a new paragraph with a fun fact about "
                    f"'{pClassification}' to help reach at least {min_words} words."
        )
        response = llm.invoke([SystemMessage(content="Expand the article further."), additional_message])
        print(f"LLM response: {response}")
        paragraphs.append(response.content)
        total_word_count = sum(len(p.split()) for p in paragraphs)  # Recalculate total words

    return paragraphs
