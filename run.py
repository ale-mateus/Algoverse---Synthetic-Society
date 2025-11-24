import asyncio
import argparse

from autogen_agentchat.teams import GraphFlow, DiGraphBuilder
from autogen_agentchat.conditions import MaxMessageTermination

from agents.societies import create_society_from_json

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--task", required=False, default="Build a short plan for a mobile app.")
    args = parser.parse_args()

    agents, settings, entry_point, edges = create_society_from_json(args.json)

    builder = DiGraphBuilder()
    for agent in agents.values():
        builder.add_node(agent)

    for e in edges:
        builder.add_edge(
            agents[e["from"]],
            agents[e["to"]],
            condition=e["condition"]
        )

    builder.set_entry_point(agents[entry_point])
    graph = builder.build()

    team = GraphFlow(
        participants=list(agents.values()),
        graph=graph,
        termination_condition=MaxMessageTermination(settings.get("max_round", 20)),
    )

    convo = []
    async for event in team.run_stream(task=args.task):
        print(event)
        convo.append(event)

    with open("convo.txt", "w", encoding="utf-8") as f:
        for e in convo:
            src = getattr(e, "source", "Unknown")
            content = getattr(e, "content", str(e))
            f.write(f"{src}:\n{content}\n\n")

if __name__ == "__main__":
    asyncio.run(main())
