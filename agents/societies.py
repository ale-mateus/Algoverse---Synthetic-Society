from .baseAgents import create_agent

def createSociety():
    manager = create_agent(
        name="manager",
        system_message="You are a project manager coordinating the task."
    )
    developer = create_agent(
        name="developer",
        system_message=" You are a software developer implementing the task assigned by the manager."
    )
    tester = create_agent(
        name="tester",
        system_message="You are a quality assurance tester ensuring the task meets the requirements."
    )

    return [manager, developer, tester]
