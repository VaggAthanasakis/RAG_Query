from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END , MessagesState
import random
from PIL import Image as PILImage
from io import BytesIO
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, tools_condition



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
    return a*b



# The State schema serves as the input schema for all
# Nodes and Edges in the Graph

# This example uses a simple dictionary structure with
# a single key graph_state of type str


class State(TypedDict):
    graph_state: str

# Nodes are python functions
# The first positional argument is the state
# each node operates on the state
# By default, each node will also override the prior state value


def node_1(state):
    print("---Node 1 ---")
    return {"graph_state": state['graph_state'] + " I am"}


def node_2(state):
    print("---Node 2 ---")
    return {"graph_state": state['graph_state'] + " I happy!"}


def node_3(state):
    print("---Node 3 ---")
    return {"graph_state": state['graph_state'] + " sad!"}


# Edges connect the nodes
# Normal Edges are used if you want to alwayes go from e.g node_1 to node_2
# Conditional Edges are used if we want to optionally route between nodes.
# Conditional edge is implemented as a function that returns the next node based upon some logic

def decide_mood(state) -> Literal["node_2", "node_3"]:

    # Often we will use state to decide on the next node to visit
    user_input = state['graph_state']

    if random.random() < 0.5:
        return "node_2"
    
    return "node_3"


# Graph construction

# Build Graph 
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END) 


graph = builder.compile()

#display(Image(graph.get_graph().draw_mermaid_png()))
#display_graph(graph)


response = graph.invoke({"graph_state": "Hi, this is Vaggelis."})

#print(response)


# Using Messages As States

class MessagesState(TypedDict):
    messages: list[AnyMessage]

# As we run our graph, we want to append to the messages state key
# But each node will also override the prior state value
# Reducer functions address this
# They allow us to specify how state updates perform

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# this is prebuild

# this is exactly the same
class State(MessagesState):
    # messages is prebuild
    pass


class MessagesState(MessagesState):
    pass


llm = ChatOllama(model="llama3.1:8b" , temperature=0)
llm_with_tools = llm.bind_tools([multiply])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

graph = builder.compile()
#display_graph(graph)

messages = graph.invoke({"messages": HumanMessage(content="Say hello!")})
print(messages)



#######################################
##### Router #####

builder = StateGraph(MessagesState)

builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm", 
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # if the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition
)
builder.add_edge("tools", END)
graph = builder.compile()

display_graph(graph)

#messages = [HumanMessage(content=" Multiply 3 and 4")]
messages = [HumanMessage(content=" Hi!")]

messages = graph.invoke({'messages': messages})
for m in messages['messages']:
    m.pretty_print()




####################################################################3
################### Agent ############################













