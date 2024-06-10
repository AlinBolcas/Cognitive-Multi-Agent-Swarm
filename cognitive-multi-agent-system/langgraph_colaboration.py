import os
import sys
import traceback
import functools
import operator
import threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Sequence, TypedDict, Literal

from io import BytesIO
import networkx as nx
from dotenv import load_dotenv
from IPython.display import Image
from PIL import Image as PILImage, ImageTk
from pyvis.network import Network
from tkhtmlview import HTMLLabel

import customtkinter as ctk
from io import BytesIO
from PIL import Image as PILImage, ImageTk

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Define project root and temporary directory
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
output_path = project_root / 'output' / 'tmp'
output_path.mkdir(parents=True, exist_ok=True)

WORKING_DIRECTORY = output_path

# Initialize tools
tavily_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPLTool()

llm = ChatOpenAI(model="gpt-4o")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
        result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
        return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
        )
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

def agent_node(state, agent, name):
    result = None
    try:
        # Add status update before invoking the agent
        state['messages'].append(AIMessage(content=f"{name} is processing...", name=name))
        # Invoke the agent
        result = agent.invoke(state)
        # We convert the agent output into a format that is suitable to append to the global state
        if isinstance(result, ToolMessage):
            # Handle ToolMessage if needed
            pass
        else:
            # Convert result to AIMessage
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    except Exception as e:
        # Handle exceptions by creating an error message
        result = AIMessage(content=f"Error in {name}: {e}", name=name)
    # Return the updated state with the result
    return {
        "messages": state['messages'] + [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

# Research agent and node
research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You should provide accurate data for the chart_generator to use.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# chart_generator
chart_agent = create_agent(
    llm,
    [python_repl_tool],
    system_message="Any charts you display will be visible by the user.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

tools = [tavily_tool, python_repl_tool]
tool_node = ToolNode(tools)

def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"

workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
)

workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "chart_generator": "chart_generator",
    },
)
workflow.set_entry_point("Researcher")
graph = workflow.compile()

class AIUI(ctk.CTk):
    def __init__(self, graph, message_queue):
        super().__init__()
        self.title("AI Swarm Collaboration")
        self.geometry("800x600")
        
        self.graph = graph
        self.message_queue = message_queue
        self.create_widgets()
        self.after(100, self.process_queue)

    def create_widgets(self):
        # Frame for displaying the graph
        graph_frame = ctk.CTkFrame(self)
        graph_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        
        # Button to load the graph image
        load_graph_btn = ctk.CTkButton(graph_frame, text="Load Graph", command=self.load_graph)
        load_graph_btn.pack(pady=10)

        # Canvas to display the graph image
        self.graph_canvas = ctk.CTkLabel(graph_frame, text="")
        self.graph_canvas.pack(pady=10)

        # Frame for displaying text output
        text_frame = ctk.CTkFrame(self)
        text_frame.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)
        
        self.text_output = ctk.CTkTextbox(text_frame, wrap="word", state="disabled")
        self.text_output.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = ctk.CTkScrollbar(text_frame, command=self.text_output.yview)
        scrollbar.pack(side="right", fill="y")
        self.text_output["yscrollcommand"] = scrollbar.set
        
    def load_graph(self):
        try:
            img_bytes = self.graph.get_graph(xray=True).draw_mermaid_png()
            img = PILImage.open(BytesIO(img_bytes))
            img.save("graph.png")
            ctk_image = ImageTk.PhotoImage(img)
            self.graph_canvas.configure(image=ctk_image)
            self.graph_canvas.image = ctk_image
        except Exception as e:
            self.text_output.configure(state="normal")
            self.text_output.insert("end", f"Error loading graph: {e}\n{traceback.format_exc()}\n")
            self.text_output.configure(state="disabled")


    def display_output(self, output):
        self.text_output.configure(state="normal")
        self.text_output.insert("end", output + "\n")
        self.text_output.see("end")
        self.text_output.configure(state="disabled")

    def process_queue(self):
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.display_output(message)
        except queue.Empty:
            pass
        self.after(100, self.process_queue)

def format_dynamic_output(event):
    for node, content in event.items():
        if isinstance(content, dict) and "next" in content:
            next_role = content["next"]
            return f"{node} -> {next_role}:"
        if isinstance(content, dict) and "messages" in content:
            message = content["messages"][-1].content
            sender = content["messages"][-1].name
            return f"{node} -> {sender}: {message}"

if __name__ == "__main__":
    message_queue = queue.Queue()
    app = AIUI(graph, message_queue)

    input_nvidia = "please speculate on a general overview of the most important tech stocks predictions for each month of year 2024 Jan to 2025 Dec."

    def run_conversation():
        message_queue.put("Starting conversation...")
        for s in graph.stream(
            {
                "messages": [
                    HumanMessage(content=input_nvidia)
                ]
            },
            {"recursion_limit": 150},
        ):
            if "__end__" not in s.values():
                formatted_output = format_dynamic_output(s)
                if formatted_output:
                    message_queue.put(formatted_output)
                message_queue.put("----")
        message_queue.put("Conversation finished.")

    with ThreadPoolExecutor() as executor:
        future = executor.submit(run_conversation)
        app.mainloop()
