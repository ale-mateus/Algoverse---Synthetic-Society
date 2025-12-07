from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

def get_model_client(model_name: str, provider: str):
    if provider == "openai":
        return OpenAIChatCompletionClient(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
            model_info=None
        )

    elif provider == "openrouter":
        dummy_model_info = {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "openrouter",
            "input_cost_per_1k_tokens": 0,
            "output_cost_per_1k_tokens": 0,
            "max_tokens": 32768
        }

        return OpenAIChatCompletionClient(
            model=model_name,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model_info=dummy_model_info,
            extra_body={"provider": {"fallback_models": []}}
        )

    else:
        raise ValueError(provider)

def create_agent(name: str, system_message: str, model_name: str, provider: str):
    client = get_model_client(model_name, provider)
    return AssistantAgent(
        name=name,
        system_message=system_message,
        model_client=client
    )
