import os

from autogen import AssistantAgent, UserProxyAgent
from autogen import LLMConfig
from dotenv import load_dotenv

load_dotenv()

gemini_key = os.environ.get("GEMINI_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

def create_LLM_config(model="gemini-2.0-flash-lite", api_type="google", temperature=None, max_tokens=None):

    # Check for API key presence based on api_type
    if api_type == "google" and not gemini_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    if api_type == "openai" and not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Select the appropriate API key, raising an error if the api_type is unsupported
    if api_type == "google":
        api_key = gemini_key
    elif api_type == "openai":
        api_key = openai_key
    else:
        raise ValueError(f"Unsupported api_type: {api_type}")

    # Returns the LLM configuration specified
    return LLMConfig({
        "model": model,
        "api_key": api_key,
        "api_type": api_type,
        "temperature": temperature,
        "max_tokens": max_tokens
    })

def create_agent(name, system_message, llm_config=None):
    llm_config = llm_config or create_LLM_config()

    return AssistantAgent(
        name=name,
        system_message=system_message,
        llm_config=llm_config
    )
