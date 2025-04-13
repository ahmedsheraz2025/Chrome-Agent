from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig

from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent as BrowserAgent
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")

external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/" 
)
model = OpenAIChatCompletionsModel(
    model = "gemini-1.5-flash",
    openai_client = external_client,
)
config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)
llm1= ChatGoogleGenerativeAI(
        model= "gemini-1.5-flash",
        api_key= gemini_api_key
    )




@function_tool
async def search(output: str):
    """
    This cool function is use for searching anything on browsser/ and play videos on youtube
    """
    agent = BrowserAgent(
        task=f"{output}",
        llm=llm1
    )
    await agent.run()
    
    
browser_agent = Agent(
    name = "Browser Agent",
    handoff_description = "Specific for browser search",
    instructions = """You are a browser agent, Your duty is to understand user
    requirement and do action, you are a seaching master , when you use the provided tool
    makesure that your prompt (which go to BrowserAgent) must be clearify perfectly 
    because BrowserAgent (For reference: in line 39) will not run if we dont clearify what I saying
    """,
    tools = [search]
)    

    
# triage_agent = Agent(
#     name = "Triage Agent",
#     instructions = """You are a triage agent. Use provided agent (sub-agent)
#     so handoffs to the other agent to answer the user
#     """,
#     handoffs = [browser_agent]
# )


translator_agent = Agent(
    name = "Translator Agent",
    instructions = """
    You are a translator Agent, your main duty is to after getting the input from
    user in any language e.g, hindi, english , Convert user input into easy, grammly
    corrected and understandable english.
    """
)

async def main():
    output = input("Enter what you want to search: ")
    
    result = await Runner.run(
        translator_agent,
        output,
        run_config = config     
    )
    print(f"Translated: {result.final_output}")
        
     
    response = await Runner.run(
        browser_agent,
        result.final_output,
        run_config = config
    )
    
    print(f"Browser Agent: {response.final_output}")
    

    
asyncio.run(main())