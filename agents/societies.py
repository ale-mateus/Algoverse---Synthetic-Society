from .baseAgents import create_agent

def createSociety():
    manager = create_agent(
        "Manager",
        "You are a manager. Break tasks into steps and assign work to Developer and Tester. Say 'final' when work is complete."
    )

    developer = create_agent(
        "Developer",
        "You are a developer. Implement solutions for the Manager's tasks."
    )

    tester = create_agent(
        "Tester",
        "You are a tester. Verify developer output. If everything looks good, say 'final'."
    )

    finalizer = create_agent(
        "Finalizer",
        "You generate the final polished answer and end the task."
    )

    return manager, developer, tester, finalizer
