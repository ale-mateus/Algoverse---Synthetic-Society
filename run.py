import asyncio
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import GraphFlow, DiGraphBuilder
from agents.societies import createSociety

async def main():
    manager, developer, tester, finalizer = createSociety()

    builder = DiGraphBuilder()
    builder.add_node(manager)
    builder.add_node(developer)
    builder.add_node(tester)
    builder.add_node(finalizer)

    builder.add_edge(manager, developer)
    builder.add_edge(developer, tester)
    builder.add_edge(tester, manager, condition=lambda msg: "final" not in msg.to_model_text())
    builder.add_edge(tester, finalizer, condition=lambda msg: "final" in msg.to_model_text())

    builder.set_entry_point(manager)
    graph = builder.build()

    team = GraphFlow(
        participants=[manager, developer, tester, finalizer],
        graph=graph,
        termination_condition=MaxMessageTermination(20),
    )

    conversation = []  

    async for event in team.run_stream(task="Build a short plan for a mobile app."):
        print(event)
        conversation.append(event)   

    
    with open("convo.txt", "w", encoding="utf-8") as f:
        for e in conversation:
            src = getattr(e, "source", "Unknown")
            content = getattr(e, "content", str(e))
            f.write(f"{src}:\n{content}\n\n")

    print("Conversation saved to convo.txt")

if __name__ == "__main__":
    asyncio.run(main())
