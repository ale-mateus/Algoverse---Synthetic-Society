import os

from autogen import AssistantAgent, UserProxyAgent
from autogen import LLMConfig
from dotenv import load_dotenv

load_dotenv()

gemini_key = os.environ.get("GEMINI_API_KEY")

def create_LLM_config(model="gemini-2.5-flash"):
    return LLMConfig({
        "model": model,
        "api_key": gemini_key,
        "api_type": "google",
        "temperature": 0.2,
        # "max_tokens": 500,
    })

def create_agent(name, system_message, llm_config=None):
    llm_config = llm_config or create_LLM_config()

    print("llm_config:", llm_config)

    return AssistantAgent(
        name=name,
        system_message=system_message,
        llm_config=llm_config
    )
