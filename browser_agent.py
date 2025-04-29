from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
from openai import AsyncOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from openai.types.responses import ResponseTextDeltaEvent
from browser_use import Agent as BrowserAgent
from dotenv import load_dotenv
import chainlit as cl
import asyncio
import os

# Disable tracing for the application to reduce logging overhead
set_tracing_disabled(True)

# Load environment variables from .env file for secure configuration
load_dotenv()

# Retrieve Gemini API key from environment variables for authentication
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    # Raise an error if the API key is not found to ensure proper configuration
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize external client for Gemini API using AsyncOpenAI wrapper
# Configures the client with the API key and base URL for Gemini's generative AI
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Initialize LangChain Google Generative AI for browser-related tasks
# Uses the Gemini 1.5 Flash model for efficient processing
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY
)

# Define a tool for browser-based tasks such as searching or playing videos
@function_tool
async def search_tool(query: str) -> None:
    """
    Executes a browser task based on the provided query.
    Can perform searches using BrowserAgent.
    
    Args:
        query (str): The task or search query to execute.
    """
    # Create a BrowserAgent instance to handle the specified query
    browser_task_agent = BrowserAgent(
        task=query,
        llm=llm_gemini
    )
    # Run the browser task asynchronously
    await browser_task_agent.run()

# Browser Agent: Specialized agent for handling browser-related tasks
browser_agent = Agent(
    name="Browser Agent",
    handoff_description="Handles browser-based tasks such as searches and YouTube playback",
    instructions="""
        You are a specialized agent for browser tasks. Your role is to interpret the user's intent
        and execute browser actions using the provided search tool. Ensure the query passed to the
        tool is clear and precise to avoid errors in the BrowserAgent execution.
    """,
    tools=[search_tool],  # Assign the search tool for browser tasks
    model=OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",  # Use Gemini 1.5 Flash model for efficiency
        openai_client=external_client  # Use the configured AsyncOpenAI client
    )
)

# Translator Agent: Converts user input into clear, grammatically correct English
translator_agent = Agent(
    name="Translator Agent",
    instructions="""
        You are a language translation expert. Your task is to take user input in any language
        (e.g., Hindi, English) and convert it into clear, grammatically correct, and concise English.
        Focus on clarity and simplicity to ensure the output is understandable.
    """,
    model=OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",  # Use Gemini 1.5 Flash model for translation
        openai_client=external_client  # Use the configured AsyncOpenAI client
    )
)

# Define the main handler for incoming messages in the Chainlit application
@cl.on_message
async def main(message: cl.Message) -> None:
    """
    Main function to run the agent pipeline:
    1. Takes user input from the Chainlit message.
    2. Translates it to clear English using the Translator Agent.
    3. Passes the translated output to the Browser Agent for task execution.
    
    Args:
        message (cl.Message): The incoming user message from Chainlit.
    """
    try:
        # Step 1: Translate user input to clear English using the Translator Agent
        translation_result = await Runner.run(
            starting_agent=translator_agent,
            input=message.content
        )
        translated_query = translation_result.final_output  # Extract the translated text

        # Step 2: Execute browser task with the translated query using the Browser Agent
        browser_result = Runner.run_streamed(
            starting_agent=browser_agent,
            input=translated_query,
        )
        
        # Initialize an empty Chainlit message for streaming responses
        msg = cl.Message(content="")
        await msg.send()
        
        # Stream the browser task results and update the message content
        async for event in browser_result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                # Stream text deltas from the browser task response
                await msg.stream_token(event.data.delta)
        
        # Finalize and update the streamed message
        await msg.update()
        
    except Exception as e:
        # Log any errors that occur during the pipeline execution
        print(f"An error occurred: {str(e)}")

# Entry point: Run the async main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())