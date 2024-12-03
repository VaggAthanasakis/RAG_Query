from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END , MessagesState
import random
from PIL import Image as PILImage
from io import BytesIO
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver



def display_graph(graph):
    image_data = graph.get_graph().draw_mermaid_png()
    image = PILImage.open(BytesIO(image_data))
    image.show()


def multiply(a:int, b:int) -> int:
    """
    Call this only when you are asked to multiply 2 numbers
    Multiply a and b.

    Args:
        a: first int
        b: second int

    """
    return a * b


def add(a:int, b:int) -> int:
    """
    Call this only when you are asked to add 2 numbers
    Add a and b.

    Args:
        a: first int
        b: second int

    """
    return a + b


def divide(a:int, b:int) -> int:
    """
    Call this only when you are asked to dividee 2 numbers
    Multiply a and b.

    Args:
        a: first int
        b: second int

    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOllama(model="llama3.1:8b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools=tools))

builder.add_edge(START,"assistant")
builder.add_conditional_edges(
    "assistant", tools_condition
)
builder.add_edge("tools","assistant")

react_graph = builder.compile()



display_graph(react_graph)


# messages = [HumanMessage(content=" Add 3 and 4, then multiply the output by 2 and finally divide the output by 5")]
# messages = react_graph.invoke({'messages': messages})

# for m in messages["messages"]:
#     m.pretty_print()


# Adding Memory
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

# specify a thread
# this id contains all the checkpoints
config = {"configurable": {"thread_id": "1"}}

messages = [HumanMessage(content="Add 3 and 4")]
messages = react_graph_memory.invoke({'messages': messages}, config=config)

for m in messages["messages"]:
     m.pretty_print()

messages = [HumanMessage(content="Multiply the previous output by 2")]
messages = react_graph_memory.invoke({'messages': messages}, config=config)

for m in messages["messages"]:
     m.pretty_print()





