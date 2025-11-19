from autogen import AssistantAgent, UserProxyAgent

def create_LLM_config(model="gpt-4.1-nano"):
    return {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 500,
    }

def create_agent(name, system_message, llm_config=None):
    llm_config = llm_config or create_LLM_config()

    return AssistantAgent(
        name=name,
        system_message=system_message,
        llm_config=llm_config
    )
