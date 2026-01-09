from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import os
from dotenv import load_dotenv
load_dotenv()

def get_model_client(model_name: str):
    return AzureOpenAIChatCompletionClient(
        model=model_name,
        azure_deployment=model_name,   
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-12-01-preview",
    )

def create_agent(name: str, system_message: str, model_name: str):
    return AssistantAgent(
        name=name,
        system_message=system_message,
        model_client=get_model_client(model_name),
    )