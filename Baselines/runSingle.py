import asyncio
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import GraphFlow, DiGraphBuilder
from agents.singleAgent import create_single_agent
from autogen_agentchat.messages import TextMessage

async def main():
    agent = create_single_agent()

    builder = DiGraphBuilder()
    builder.add_node(agent)
    builder.set_entry_point(agent)
    graph = builder.build()

    team = GraphFlow(
        participants=[agent],
        graph=graph,
        termination_condition=MaxMessageTermination(5),
    )

    log = []

    async for event in team.run_stream(task="Build a short plan for a mobile app."):
        print(event)
        if isinstance(event, TextMessage):
            log.append(f"{event.source}: {event.content}\n")

    with open("convo_single.txt", "w") as f:
        f.writelines(log)

if __name__ == "__main__":
    asyncio.run(main())
