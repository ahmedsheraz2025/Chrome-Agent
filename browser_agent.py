from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent as BrowserAgent
from dotenv import load_dotenv
import asyncio
import os


# Load environment variables from .env file
load_dotenv()


# Retrieve Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")


# Initialize external client for Gemini API using AsyncOpenAI wrapper
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# Configure the Gemini model for the agent system
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)
# Define run configuration for agents
config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)
# Initialize LangChain Google Generative AI for browser tasks
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY
)



# Define a tool for browser-based tasks (searching or playing videos)
@function_tool
async def search_tool(query: str) -> None:
    """
    Executes a browser task based on the provided query.
    Can perform searches using BrowserAgent.
    
    Args:
        query (str): The task or search query to execute.
    """
    browser_task_agent = BrowserAgent(
        task=query,
        llm=llm_gemini
    )
    await browser_task_agent.run()

# Browser Agent: Handles browser-related tasks like searching
browser_agent = Agent(
    name="Browser Agent",
    handoff_description="Handles browser-based tasks such as searches and YouTube playback",
    instructions="""
        You are a specialized agent for browser tasks. Your role is to interpret the user's intent
        and execute browser actions using the provided search tool. Ensure the query passed to the
        tool is clear and precise to avoid errors in the BrowserAgent execution.
    """,
    tools=[search_tool]
)

# Translator Agent: Converts user input into clear, grammatically correct English
translator_agent = Agent(
    name="Translator Agent",
    instructions="""
        You are a language translation expert. Your task is to take user input in any language
        (e.g., Hindi, English) and convert it into clear, grammatically correct, and concise English.
        Focus on clarity and simplicity to ensure the output is understandable.
    """
)

async def main() -> None:
    """
    Main function to run the agent pipeline:
    1. Takes user input.
    2. Translates it to clear English using Translator Agent.
    3. Passes the translated output to Browser Agent for task execution.
    """
    try:
        # Get user input
        user_input = input("Enter your query: ")

        # Step 1: Translate user input to clear English
        translation_result = await Runner.run(
            starting_agent=translator_agent,
            input=user_input,
            run_config=config
        )
        translated_query = translation_result.final_output
        print(f"Translated Query: {translated_query}")

        # Step 2: Execute browser task with translated query
        browser_result = await Runner.run(
            starting_agent=browser_agent,
            input=translated_query,
            run_config=config
        )
        print(f"Browser Agent Result: {browser_result.final_output}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())