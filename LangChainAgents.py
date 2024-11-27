from typing import List
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM



def validate_user(user_id: int , addresses: List) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id: (int) the user ID.
        addresses: Previous addresses.
    """
    return True


llm = ChatOllama(model="llama3.1:8b", temperature= 0).bind_tools([validate_user])
# load the LLM that we are going to use
#llm = OllamaLLM(model="llama3.1:8b", temperature = 0.1)#.bind_tools([validate_user])

result = llm.invoke("Could you validate user 123? they previously lived at 123 FAKE St in Boston MA and 234 Pretend Boulevard in Houston TX")

print(result.tool_calls)