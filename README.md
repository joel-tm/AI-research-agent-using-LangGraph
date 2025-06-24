# LangGraph Research Agent

A sophisticated AI-powered research agent built with LangGraph that can answer questions by intelligently searching both Wikipedia and the web using DuckDuckGo, powered by Google's Gemini AI model.

## SAMPLE OUTPUT
![Screenshot 2025-06-24 113201](https://github.com/user-attachments/assets/660d3362-a1ee-44bf-98e9-66c20f1af01f)


## Overview

This project demonstrates how to build an intelligent agent using LangGraph that can:
- Accept natural language questions from users
- Automatically decide between Wikipedia and web search based on the question type
- Query Wikipedia for encyclopedic information, definitions, and historical facts
- Search the web using DuckDuckGo for current events, recent news, and breaking developments
- Show which source provided the information (Wikipedia or Web Search)
- Process and synthesize answers using Google's Gemini 1.5 Flash model
- Provide clean, professional responses without distracting visual elements

## Architecture

The agent uses a graph-based architecture with three main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  LLM Agent Node │───▶│  Router Logic   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Final Answer   │◀───│  LLM Agent Node │◀───│   Tool Node     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Wikipedia Search│
                                              │ DuckDuckGo Web  │
                                              └─────────────────┘
```

### Core Components

1. **LLM Agent Node**: Uses Google Gemini 1.5 Flash to process queries and intelligently decide on actions
2. **Tool Node**: Executes Wikipedia or DuckDuckGo searches based on the question type
3. **Router**: Intelligently routes between nodes based on the LLM's decisions
4. **State Management**: Maintains conversation history and context

### Intelligent Tool Selection

The agent automatically chooses the most appropriate search tool based on the question:

- **Wikipedia**: For encyclopedic information, definitions, historical facts, scientific concepts
- **DuckDuckGo**: For current events, recent news, company updates, breaking news, anything from 2023-2024

## Features

- **Interactive Chat Interface**: Ask questions in natural language
- **Dual Search Capabilities**: Access both Wikipedia and web search with intelligent routing
- **Automatic Tool Selection**: LLM automatically chooses between Wikipedia and DuckDuckGo based on question type
- **Source Tracking**: Clear indicators showing whether information came from Wikipedia or Web Search
- **Loop Prevention**: Built-in safeguards against infinite loops
- **Error Handling**: Graceful error recovery and user-friendly messages
- **Clean Output**: Professional display with step-by-step process visualization
- **Virtual Environment Support**: Properly isolated Python environment
- **System Prompts**: Intelligent guidance for tool usage based on question context

## Prerequisites

- Python 3.8 or higher
- Google AI API key (Gemini)
- Internet connection for Wikipedia and web searches



## Project Structure

```
langgraph project/
│
├── agent.py           # Main agent implementation
├── requirements.txt   # Python dependencies
├── .env              # Environment variables (create this)
├── README.md         # This documentation
└── venv/             # Virtual environment (auto-created)
```

##  Usage

### Running the Agent

1. **Activate your virtual environment** (if not already active):
   ```bash
   .\venv\Scripts\Activate.ps1
   ```

2. **Run the agent**:
   ```bash
   python agent.py
   ```

3. **Interact with the agent**:
   ```
   Welcome to the Enhanced Research Agent!
   Ask me anything and I'll search Wikipedia and the web for answers.
   ============================================================
   
   Enter your question (or 'quit' to exit): What is quantum computing?
   ```

### Example Interactions

**Example 1: Encyclopedic Question (Uses Wikipedia)**
```
Question: What is quantum computing?
==================================================
Step 1: LLM is analyzing your question...
Step 2: Gathering information from sources...
   Searching Wikipedia for: quantum computing
==================================================
Answer: Quantum computing is a type of computation that harnesses...

Source: Wikipedia
==================================================
```

**Example 2: Current Events (Uses Web Search)**
```
Question: What happened with OpenAI in 2024?
==================================================
Step 1: LLM is analyzing your question...
Step 2: Gathering information from sources...
   Searching the web for: OpenAI 2024 news
==================================================
Answer: In 2024, OpenAI announced several major developments...

Source: Web Search (DuckDuckGo)
==================================================
```

### Dependencies

The project uses the following key libraries:

- **LangChain**: Framework for building LLM applications
- **LangGraph**: Graph-based workflow orchestration
- **langchain-google-genai**: Google Gemini AI integration
- **langchain-community**: Community tools including Wikipedia and DuckDuckGo
- **duckduckgo-search**: Web search capabilities
- **python-dotenv**: Environment variable management

### Agent Workflow

1. **Input Processing**: User question is converted to a `HumanMessage`
2. **System Prompt Injection**: Intelligent system prompt guides tool selection based on question type
3. **LLM Analysis**: Gemini AI analyzes the question and decides which tool to use
4. **Tool Execution**: Either Wikipedia search or DuckDuckGo web search is performed
5. **Source Attribution**: Results are tagged with their source (Wikipedia or Web Search)
6. **Result Processing**: LLM processes search results and formulates an answer
7. **Response**: Clean, formatted answer with source information is presented to the user


### Error Handling

The agent includes several safety mechanisms:
- **Loop Prevention**: Maximum step limit to prevent infinite cycles
- **Exception Handling**: Graceful error recovery
- **Input Validation**: Checks for empty or invalid inputs
- **Quota Management**: Uses efficient Gemini 1.5 Flash model

##  Configuration Options

### Model Configuration
You can modify the LLM model in `agent.py`:
```python
# Current: Fast and efficient
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# Alternative: More capable but uses more quota
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
```

### DuckDuckGo Search Settings
Configure web search behavior:
```python
duckduckgo_tool = DuckDuckGoSearchRun()
# Additional configuration options available in the DuckDuckGo library
```

### Tool Selection Logic
The agent uses an intelligent system prompt that guides tool selection:
```python
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
```

### Wikipedia Search Settings
Adjust Wikipedia search parameters:
```python
wikipedia_wrapper = WikipediaAPIWrapper(
    top_k_results=3,           # Number of search results
    doc_content_chars_max=1000 # Maximum content length
)
```

### Loop Prevention
Modify the maximum steps to prevent infinite loops:
```python
max_steps = 10  # Adjust as needed
```

### Debug Mode

For detailed debugging, use the verbose function:
```python
# In agent.py, replace run_research_agent with:
run_research_agent_verbose("your question")
```

### Modifying Agent Behavior

The agent's behavior is controlled by:
- **LLM Agent Node**: Handles reasoning and decision making
- **Router Function**: Determines workflow paths
- **Tool Node**: Executes external tools

## Performance

- **Average Response Time**: 3-7 seconds per query (depending on search complexity)
- **API Calls**: 1-3 Gemini API calls per question
- **Supported Query Types**: Encyclopedic questions, current events, definitions, explanations, recent news
- **Search Sources**: Wikipedia (encyclopedic) + DuckDuckGo (web/current events)





## Future Enhancements

Potential improvements for this project:
- Add more specialized tools (calculator, code execution, etc.)
- Implement conversation memory across sessions
- Add voice input/output capabilities
- Create a web interface
- Add support for different LLM providers
- Implement caching for repeated queries
- Add more sophisticated search result ranking
- Support for file uploads and document analysis

---


