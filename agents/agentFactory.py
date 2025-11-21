import json
import os
from autogen import AssistantAgent, UserProxyAgent
from .baseAgents import create_LLM_config

def load_society(config_path):
    config = load_config(config_path)

    agents = []

    print(f"--- Loading Society: {config.get('society_name', 'Unknown')} ---")

    for agent_def in config["agents"]:
        name = agent_def["name"]
        sys_msg = agent_def["system_message"]

        override = agent_def.get("llm_config_override")
        
        if override:
            llm_config = create_LLM_config(
                model=override.get("model"),
                api_type=override.get("api_type"),
                temperature=override.get("temperature", 0.2),
                max_tokens=override.get("max_tokens")
            )
        else:
            llm_config = create_LLM_config()

        new_agent = AssistantAgent(
            name=name,
            system_message=sys_msg,
            llm_config=llm_config,
            description=agent_def.get("description")
        )
        
        agents.append(new_agent)
        print(f"Created agent: {name} using model: {llm_config['config_list'][0]['model']}")
    
    return agents, config["settings"]

def load_config(config_path):
    """
    Reads a JSON config file and returns a list of agents and society settings.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config
