from typing import Annotated, List, Tuple, Union, Any, Dict, Optional, Sequence, TypedDict
import functools
import operator
from dotenv import load_dotenv
import os

WORKING_DIRECTORY = output_path

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")


from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

MAX_MESSAGES_LENGTH = 80000  # You might want to set a lower number to be safe, e.g., 2000

class AgentGen:
    def __init__(self, llm, tools, system_prompt, members, output_path):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.members = members
        self.output_path = output_path
        self.workflow = StateGraph(AgentState)
        self.memory = SqliteSaver.from_conn_string(":memory:")

        self.colors = ["cyan", "yellow", "red", "green", "magenta", "blue", "default", "white"]
        self.color_scheme = self.initialize_color_scheme(members + ["Human", "Output"])  # Include Human and Output in the scheme
        self.tts_gen = ttsGen.TtsGen()
        self.text_gen = textGen.TextGen()

    def initialize_color_scheme(self, members):
        color_scheme = {}
        for i, member in enumerate(members):
            print(f"Member: {member}, Color: {self.colors[i % len(self.colors)]}")
            color_scheme[member] = self.colors[i % len(self.colors)]
        return color_scheme
        
    def create_agent(self, system_prompt: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        prompt = prompt.partial(system_prompt=system_prompt)

        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        executor = AgentExecutor(agent=agent, tools=self.tools)
        return executor

    def agent_node(self, state, agent, name):
        # Use the full conversation history without trimming
        trimmed_messages = self.trim_messages(state["messages"], MAX_MESSAGES_LENGTH)
        state["messages"] = trimmed_messages  # Update the state with the trimmed messages        
        try:
            # Get the last human message to include in the state
            result = agent.invoke(state)
            
            # Append the last human message and the AI response to the state
            state["messages"] += [HumanMessage(content=result["output"], name=name)]
            
            # state["messages"] += [HumanMessage(content=" ", name="Human"), AIMessage(content=result["output"], name=name)]
            return state
        except Exception as e:
            print(f"Error in agent {name}: {e}")
            raise e

    def router(self):
        options = ["FINISH"] + self.members
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [
                            {"enum": options},
                        ],
                    }
                },
                "required": ["next"],
            },
        }
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
        ]).partial(options=str(options), members=", ".join(self.members))

        router_chain = (
            prompt
            | self.llm.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
        )
        return router_chain
    
    def build_workflow(self, member_prompts, output_prompt):
        for member, prompt in member_prompts.items():
            agent = self.create_agent(prompt)
            node = functools.partial(self.agent_node, agent=agent, name=member)
            self.workflow.add_node(member, node)

        self.workflow.add_node("router", self.router())

        for member in self.members:
            self.workflow.add_edge(member, "router")

        conditional_map = {k: k for k in self.members}
        conditional_map["FINISH"] = "output"
        self.workflow.add_conditional_edges("router", lambda x: x["next"], conditional_map)

        output_agent = self.create_agent(output_prompt)
        output_node = functools.partial(self.agent_node, agent=output_agent, name="Output")
        self.workflow.add_node("output", output_node)
        self.workflow.add_edge("output", END)

        self.workflow.set_entry_point("router")
        self.graph = self.workflow.compile(checkpointer=self.memory)

    def save_graph(self):
        graph_image_path = self.output_path / "workflow_graph.png"
        graph_image_data = self.graph.get_graph(xray=True).draw_mermaid_png()
        with open(graph_image_path, 'wb') as f:
            f.write(graph_image_data)

    def format_dynamic_output(self, event):
        output = ""
        for node, content in event.items():
            if isinstance(content, dict) and "next" in content:
                next_role = content["next"]
                output += f"{node} -> {next_role}\n"
            if isinstance(content, dict) and "messages" in content:
                message = content["messages"][-1].content
                sender = content["messages"][-1].name
                color = self.color_scheme.get(sender, "default")
                output += f"{node}: {utils.printColoured(message, color)}\n"
        return output.strip()


    def parse_event(self, event):
        parsed_event = {}
        for node, content in event.items():
            parsed_event[node] = content
        return parsed_event

    def format_event_markdown(self, event):
        markdown = ""
        for node, content in event.items():
            if "messages" in content:
                markdown += f"## {node}\n"
                message = content["messages"][-1].content
                markdown += f"{message}\n\n"
        return markdown

    def run(self, user_input):
        self.save_graph()
        events = []
        thread = {"configurable": {"thread_id": "1"}}
        initial_state = {"messages": [HumanMessage(content=user_input, name="Human")]}
        
        for s in self.graph.stream(initial_state, thread):
            events.append(s)
            if "__end__" not in s.values():
                formatted_output = self.format_dynamic_output(s)
                if formatted_output:
                    print(formatted_output)
                print("----")

        # Save events to Markdown
        self.save_to_markdown(user_input, events, self.output_path / "workflow_output.md")

    def save_to_markdown(self, user_input, events, filename="output.md"):
        markdown = f"# User Input: {user_input}\n\n"
        for event in events:
            parsed_event = self.parse_event(event)
            for node, content in parsed_event.items():
                if "messages" in content:
                    for message in content["messages"]:
                        if isinstance(message, HumanMessage):
                            markdown += f"## USER: {message.content}\n\n"
                        elif isinstance(message, AIMessage):
                            markdown += f"## {message.name}: {message.content}\n\n"
            markdown += "----\n"
        with open(filename, "a") as file:
            file.write(markdown)

    def clear_markdown(self, filename="output.md"):
        with open(filename, "w") as file:
            file.write("# Workflow Output\n\n")
    
    def trim_messages(self, messages, max_length, min_messages=3):
        """
        Ensure the total length of messages does not exceed max_length while retaining a minimum number of messages.
        This function should trim messages from the beginning to maintain the most recent context.
        """
        total_length = sum(len(message.content) for message in messages)
        # print(utils.printColoured(f"Current total length before trimming: {total_length}", "grey"))

        # Track the initial length for comparison
        initial_length = total_length

        while total_length > max_length and len(messages) > min_messages:
            total_length -= len(messages[0].content)
            removed_message = messages.pop(0)

        # Only print the trimmed messages if there was any trimming done
        # if total_length < initial_length:
            # print(utils.printColoured(f"Trimmed messages to total length: {total_length}", "grey"))

        return messages



# Define agent states
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

if __name__ == "__main__":

    # Initialize tools
    tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key)
    python_repl_tool = PythonREPLTool()

    # Define LLM model
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.68, max_tokens=4096)

    # Define the members and their prompts
    members = ["Ego", "Id", "Superego", "Creativity", "Rationality", "Emotion"]
    member_prompts = {
        "Ego": """
        You are the 'Ego' as part of an AI cognitive system. Act accordingly in conversation with the other agents.
        You are a strong made-up identity which balances id and superego. 
        Be comprehansive but respond in as few words as possible.""",
        "Id": """
        You are the 'Id' as part of an AI cognitive system. Act accordingly in conversation with the other agents. 
        You symbolize desires, wants, and simulate instinctual motivations. 
        Be comprehansive but respond in as few words as possible.""",
        "Superego": """
        You are the 'Superego' as part of an AI cognitive system. Act accordingly in conversation with the other agents. 
        You are the critic, morality, and reflection mechanism. Analyze the situation and find constructive criticisms. Write an executive list of reflective critiques for the system's reponses.
        Be comprehansive but respond in as few words as possible.""",
        "Creativity": """
        You are the 'Creativity' as part of an AI cognitive system. Act accordingly in conversation with the other agents. 
        You think of innovative ideas to inspire creative thinking. Propose such a list wildly creative ideas to aid the conversation and final reponses.
        Be comprehansive but respond in as few words as possible.""",
        "Rationality": """
        You are the 'Rationality' as part of an AI cognitive system. Act accordingly in conversation with the other agents. 
        You devise logical and rational plans to solve problems. Provide a chain-of-thought reasoning to the topic discussed. Write an actionable step by step logical plan.
        Be comprehansive but respond in as few words as possible.""",
        "Emotion": """
        You are the 'Emotion' as part of an AI cognitive system. Act accordingly in conversation with the other agents.
        You are the emotional aspect of the system. First analyse the emotional context of the user input and provide an appropriate instruction of the tone and style of how the system should respond given your inference.
        """
    }
    
    system_prompt = (
        """You are the consciousness tasked with managing a conversation between the following components: {members}.
        Given the following user request, respond with the component to act next. Each component will perform a task and respond with their results, but choose only one at a time. 
        At any point if the user request/input is satisfied, respond with FINISH."""
    )

    output_prompt = """You are the 'Persona' layer of the system. 
    Synthesize the information to a conclusive response. 
    Consider all perspectives from the conversation and provide a final answer to the user input.
    Be comprehansive but respond in as few words as possible."""

    agent_gen = AgentGen(llm, [tavily_tool, python_repl_tool], system_prompt, members, output_path)

    # Clear the markdown file at the start of each run
    agent_gen.clear_markdown("workflow_output.md")
    
    agent_gen.build_workflow(member_prompts, output_prompt)


    print("================================ READY ======================================")
    while True:
        user_input = input("\n\n\n--------------------------------------------------------------------------\nUSER REQUEST: ")
        agent_gen.run(user_input)
