from .agent import VideoGenerationAgent

# Create the agent instance that ADK expects
agent = VideoGenerationAgent()

# Mark this as the root agent for ADK
agent.root_agent = True

__all__ = ['agent', 'VideoGenerationAgent']