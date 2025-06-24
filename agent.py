import os
import warnings
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

# Suppress the deprecation warning for convert_system_message_to_human
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# --- 1. Initialize LLM and Tools ---

# Gemini 1.5 Flash (faster and uses less quota than Pro)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# Wikipedia Tool
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

# DuckDuckGo Search Tool
duckduckgo_tool = DuckDuckGoSearchRun()

# List of tools available to the agent
tools = [wikipedia_tool, duckduckgo_tool]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# --- 2. Define Agent State ---

class AgentState(TypedDict):
    """
    Represents the state of our agent in the graph.
    Messages: A list of messages passed between nodes.
    """
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- 3. Define Nodes ---

def llm_agent_node(state: AgentState):
    """
    Node that invokes the LLM to decide the next action.
    """
    messages = state["messages"]
    
    # Add a system message to guide the LLM to use tools
    if len(messages) == 1:  # First message, add instructions
        system_prompt = """You are a research assistant with access to search tools. 

Available tools:
- Wikipedia: Use for encyclopedic information, definitions, historical facts, scientific concepts
- DuckDuckGo Search: Use for current events, recent news, company updates, breaking news, anything from 2023-2024

ALWAYS use the appropriate search tool when asked about:
- Current events or recent news
- Company updates or developments  
- Recent developments in technology
- What happened in specific years (especially 2023-2024)
- Any factual information you're not certain about

Choose the most appropriate tool based on the question type."""
        
        system_message = HumanMessage(content=system_prompt)
        messages = [system_message] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    """
    Custom tool node that shows which tool is being used.
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    tool_outputs = []
    
    for tool_call in tool_calls:
        # Get tool name and show which source is being used
        tool_name = tool_call.get('name') if isinstance(tool_call, dict) else tool_call.name
        tool_args = tool_call.get('args') if isinstance(tool_call, dict) else tool_call.args
        tool_id = tool_call.get('id') if isinstance(tool_call, dict) else tool_call.id
        
        # Display which tool is being used
        if tool_name == 'wikipedia':
            print(f"   Searching Wikipedia for: {tool_args.get('query', 'information')}")
            tool_executor = wikipedia_tool
        elif tool_name == 'duckduckgo_search':
            print(f"   Searching the web for: {tool_args.get('query', 'information')}")
            tool_executor = duckduckgo_tool
        else:
            print(f"   Using tool: {tool_name}")
            # Fallback to ToolNode for unknown tools
            tool_node_fallback = ToolNode(tools)
            return tool_node_fallback.invoke(state)
        
        try:
            output = tool_executor.invoke(tool_call)
            # Add source information to the tool message
            if tool_name == 'wikipedia':
                source_info = "\n\nSource: Wikipedia"
            elif tool_name == 'duckduckgo_search':
                source_info = "\n\nSource: Web Search (DuckDuckGo)"
            else:
                source_info = f"\n\nSource: {tool_name}"
            
            tool_outputs.append(ToolMessage(
                content=str(output) + source_info, 
                tool_call_id=tool_id
            ))
        except Exception as e:
            print(f"    Error using {tool_name}: {e}")
            tool_outputs.append(ToolMessage(
                content=f"Error: {e}", 
                tool_call_id=tool_id
            ))
    
    return {"messages": tool_outputs}

def route_agent(state: AgentState):
    """
    Router function to determine the next node based on the LLM's response.
    """
    last_message = state["messages"][-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    else:
        return END

# --- 4. Build the LangGraph Workflow ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("llm_agent_node", llm_agent_node)
workflow.add_node("tool_node", tool_node)  # Use our custom tool node

# Set entry point
workflow.set_entry_point("llm_agent_node")

# Add edges
workflow.add_conditional_edges(
    "llm_agent_node",
    route_agent,
    {
        "tool_node": "tool_node",
        END: END
    }
)
workflow.add_edge("tool_node", "llm_agent_node")

# Compile the graph
app = workflow.compile()

# --- 5. Run the Agent ---

def run_research_agent(query: str):
    """
    Runs the research agent with a given query.
    """
    print(f"\nQuestion: {query}")
    print("=" * 50)
    
    initial_state = {"messages": [HumanMessage(content=query)]}
      # Stream the execution to see the process
    step_count = 1
    max_steps = 10  # Prevent infinite loops
    
    for s in app.stream(initial_state):
        if "__end__" not in s:
            node_name = list(s.keys())[0]
            if node_name == "llm_agent_node":
                print(f"Step {step_count}: LLM is analyzing your question...")
            elif node_name == "tool_node":
                print(f"Step {step_count}: Gathering information from sources...")
            step_count += 1
            
            # Safety check to prevent infinite loops
            if step_count > max_steps:
                print("WARNING: Maximum steps reached. Stopping to prevent infinite loop.")
                break
      # Get final result
    try:
        final_state = app.invoke(initial_state)
        print("=" * 50)
        print(f"Answer: {final_state['messages'][-1].content}")
        print("=" * 50)
    except Exception as e:
        print("=" * 50)
        print(f"Error getting final answer: {e}")
        print("=" * 50)

def run_research_agent_verbose(query: str):
    """
    Runs the research agent with detailed debug output.
    """
    print(f"\n--- Running Research Agent for: '{query}' ---\n")
    initial_state = {"messages": [HumanMessage(content=query)]}
    for s in app.stream(initial_state):
        if "__end__" not in s:
            print(s)
            print("---")
    final_state = app.invoke(initial_state)
    print("\n--- FINAL RESPONSE ---")
    print(final_state["messages"][-1].content)

if __name__ == "__main__":
    print("Welcome to the Enhanced Research Agent!")
    print("Ask me anything and I'll search Wikipedia and the web for answers.")
    print("=" * 60)
    
    while True:
        try:
            # Get user input
            user_query = input("\nEnter your question (or 'quit' to exit): ").strip()
            
            # Check if user wants to quit
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Check if input is empty
            if not user_query:
                print("Please enter a question!")
                continue
            
            # Run the agent with user's query
            run_research_agent(user_query)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again with a different question.")
