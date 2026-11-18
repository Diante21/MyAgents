"""Orchestrator: routes tasks between agents. (Roadmap phase 6 - not built yet.)

Planned shape, so you can see where it's going:

    class Orchestrator:
        def __init__(self, agents: list[BaseAgent], llm: LLMClient): ...
        def route(self, task: Task) -> BaseAgent:
            # ask Claude to pick an agent from each agent's `name` + `description`
        def dispatch(self, message: AgentMessage) -> AgentResult:
            # deliver a task from one agent to another and return the result
"""
