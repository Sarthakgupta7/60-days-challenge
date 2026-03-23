from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType

llm = ChatOpenAI(model="gpt-4.1-mini")

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def get_ai_definition() -> str:
    """Return a simple definition of Artificial Intelligence."""
    return "Artificial Intelligence is the field of building systems that can perform tasks requiring human intelligence such as learning, reasoning, and decision making."

tools = [add_numbers, multiply_numbers, get_ai_definition]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)


response1 = agent.invoke("What is Artificial Intelligence?")
print("\nResponse 1:", response1)

response2 = agent.invoke("Add 12 and 8")
print("\nResponse 2:", response2)

response3 = agent.invoke("Multiply 6 and 7")
print("\nResponse 3:", response3)