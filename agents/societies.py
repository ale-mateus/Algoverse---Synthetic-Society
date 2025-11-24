import json
from agents.baseAgents import create_agent

def parse_condition(cond_str):
    if cond_str is None:
        return lambda msg: True
    cond_str = cond_str.lower().strip()
    if cond_str.startswith("contains:"):
        token = cond_str.split("contains:")[1].strip()
        return lambda msg: token in msg.to_model_text().lower()
    if cond_str.startswith("not contains:"):
        token = cond_str.split("not contains:")[1].strip()
        return lambda msg: token not in msg.to_model_text().lower()
    return lambda msg: True

def create_society_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    settings = data.get("settings", {})
    agent_defs = data.get("agents", [])
    graph_def = data.get("graph", {})

    agents = {}
    for a in agent_defs:
        agents[a["name"]] = create_agent(
            name=a["name"],
            system_message=a.get("system_message", "")
        )

    entry_point = graph_def.get("entry_point")
    edges_json = graph_def.get("edges", [])

    edges = []
    for e in edges_json:
        cond_fn = parse_condition(e.get("condition"))
        edges.append({
            "from": e["from"],
            "to": e["to"],
            "condition": cond_fn
        })

    return agents, settings, entry_point, edges
