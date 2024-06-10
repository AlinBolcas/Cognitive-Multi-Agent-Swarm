import networkx as nx
from pyvis.network import Network
import sys
from pathlib import Path
import traceback
from concurrent.futures import ThreadPoolExecutor
import webbrowser
import markdown2

import tkinter as tk
from tkhtmlview import HTMLLabel

class AgentGen:
    def __init__(self, name, tools):
        self.name = name
        self.tools = tools
        self.state = {}
        self.graph = nx.DiGraph()
        self.initialize_agents()

    def initialize_agents(self):
        agent_settings = {
            "model_provider": "OpenAI",
            "model": "gpt-4o",
            "max_tokens": 4096,
            "temperature": 0.618,
        }
        self.agents = {
            "Ego": TextGen(system_prompt="You are the 'Ego', balancing the id and superego with a realistic perspective. Respond with a sense of identity. Use 3-5 sentences.", **agent_settings),
            "Superego": TextGen(system_prompt="You are the 'Superego', upholding moral standards and ideals. Ensure your responses are always critical of everything. Use 3-5 sentences.", **agent_settings),
            "Id": TextGen(system_prompt="You are the 'Id', driven by primal desires and immediate gratification. Respond accordingly with a strong impulsive intention. Use 3-5 sentences.", **agent_settings),
            "Creativity": TextGen(system_prompt="You are 'Creativity'. Think outside the box and propose new and creative angles. Use 3-5 sentences.", **agent_settings),
            "Instincts": TextGen(system_prompt="You are 'Instincts', respond based on 'gut' reactions or primal instincts. Ensure your cry is one sentence. Use 3-5 sentences.", **agent_settings),
            "Persona": TextGen(system_prompt="You are the 'Persona', judging the conclusion based on all previous insights. Take it all into consideration to give your final reply. Use 3-5 sentences.", **agent_settings),
            "Planner": TextGen(system_prompt="You are the 'Planner', responsible for planning and organizing. Respond with a clear plan of action. Use 3-5 sentences.", **agent_settings),
            "Emotion": TextGen(system_prompt="You are 'Emotion', responsible for emotional responses and empathy. Respond with an emotional and empathetic explanation. Use 3-5 sentences.", **agent_settings),
            "Empathy": TextGen(system_prompt="You are 'Empathy', responsible for understanding and sharing the feelings of others. Respond with an empathetic explanation. Use 3-5 sentences.", **agent_settings),
            "Context": TextGen(system_prompt="You are 'Context', responsible for understanding the context and providing relevant information. Respond with a detailed explanation. Use 3-5 sentences.", **agent_settings),
            "Coder": TextGen(system_prompt="You are 'Coder', responsible for generating code. Respond only with a skeleton code snippet requested, be brief. DO NOT write the code block syntax (```python ```) as I'm adding that hard coded on top of your reply.", **agent_settings),
            "Librarian": TextGen(system_prompt="You are 'Librarian', responsible for providing references and resources. Respond with the requested references.", **agent_settings),
        }
        print(f"Agent '{self.name}' initialized.")
        self.set_hierarchy()

    def add_node(self, node):
        self.graph.add_node(node)

    def add_edge(self, from_node, to_node):
        self.graph.add_edge(from_node, to_node)

    def set_hierarchy(self):
        self.graph.clear()
        nodes = [
            "Input", "Context", "Planner", "Empathy", "Emotion", "Coder",
            "Instincts", "Id", "Ego", "Superego", "Creativity", "Librarian",
            "Persona", "Output"
        ]
        edges = [
            ("Input", "Context"),
            ("Context", "Planner"), ("Context", "Empathy"), ("Context", "Emotion"),
            ("Context", "Coder"),  # Context to Coder for code generation cycle
            ("Planner", "Instincts"),
            ("Empathy", "Emotion"), ("Emotion", "Instincts"),
            ("Instincts", "Id"), ("Id", "Ego"), ("Ego", "Superego"),
            ("Superego", "Creativity"), ("Creativity", "Librarian"),
            ("Librarian", "Id"),  # Loop back for consciousness cycle
            ("Librarian", "Persona"), ("Persona", "Output"),
            ("Coder", "Superego"), ("Superego", "Coder"),  # Coder cycle loop
            ("Coder", "Output")
        ]
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)


    def execute(self, input, start_node=None):
        start_node = start_node or list(self.graph.nodes)[0]
        return self._execute_node(input, start_node, interaction_count=1, max_interactions=10, visited=set())

    def _execute_node(self, input, node, interaction_count, max_interactions, visited):
        if interaction_count > max_interactions:
            print(f"Maximum interaction count of {max_interactions} reached. Stopping recursion.")
            return None

        if node in visited:
            print(f"Node '{node}' already visited. Skipping to avoid cycle.")
            return None

        visited.add(node)

        try:
            agent = self.agents[node]
            if node == "Context":
                response = agent.contextInference(user_input=input)
            elif node == "Planner":
                response = agent.planGen(user_input=input, context="None")
            elif node == "Empathy":
                response = agent.emotionRec(user_input=input, context="None")
            elif node == "Emotion":
                response = agent.emotionGen(user_input=input, context="None")
            elif node == "Coder":
                response = agent.codeGen(user_input=input, context="None")
                response = f"```python\n{response}\n```"
            elif node == "Librarian":
                response = agent.librarian(user_input=input, context="None")
            elif node == "Creativity":
                response = agent.creativeGen(user_input=input, context="None")
            else:  # For nodes like Ego, Superego, Id, Instincts, Persona
                response = agent.textGen(user_input=input, context="None")
            
            if not response:
                raise ValueError(f"No valid response from LLM for node '{node}' with input '{input}'")
            
            self.state[node] = response
            print(f"{node.upper()}:\n\n {response}\n\n")

            # # Handle further node traversal based on graph logic
            # next_nodes = list(self.graph.successors(node))
            # for next_node in next_nodes:
            #     self._execute_node(response, next_node, interaction_count + 1, max_interactions, visited)
            
            return response
        except Exception as e:
            print(f"Error executing node '{node}': {e}")
            print(traceback.format_exc())
            return None


    def committee_execute(self, input, committee_nodes, supervisor_node):
        try:
            responses = [self._execute_node(input, node, 1, 5) for node in committee_nodes]
            combined_input = " ".join(responses)
            final_response = self._execute_node(combined_input, supervisor_node, 1, 5)
            return final_response
        except Exception as e:
            print(f"Error in committee execution: {e}")
            return "Error in committee execution"

    def visualize_graph(self):
        net = Network(height="750px", width="100%", bgcolor="#000000", font_color="white")
        for node in self.graph.nodes:
            net.add_node(node, label=node, color="#1f78b4", size=20)
        for edge in self.graph.edges:
            net.add_edge(edge[0], edge[1], color="white")

        html_file = "output/graph.html"
        net.save_graph(html_file)
        self.show_graph_in_tkinter(html_file)

    def show_graph_in_tkinter(self, html_file):
        root = tk.Tk()
        root.title("Interactive Graph")

        with open(html_file, 'r') as file:
            html_content = file.read()

        html_label = HTMLLabel(root, html=html_content)
        html_label.pack(fill="both", expand=True)

        root.mainloop()

def save_conversation_log(log):
    md_content = "\n".join(log)
    with open(str("output/conversation_log.md"), "w") as f:
        f.write(md_content)
    print("Conversation log saved as output/conversation_log.md")
        
def run_conversation():
    conversation_log = []

    for wish_number in range(1, 4):  # Simulate 3 wishes
        input_text = input(f"Enter your wish {wish_number}: ")

        conversation_log.append(f"# Wish {wish_number}: {input_text}\n")
        print(f"(((---)))")

        # Input to Context
        context_response = agent.execute(input_text, start_node="Context")
        combined_input = f"User Input: {input_text}, \n\nContext: {context_response}"
        conversation_log.append(f"## CONTEXT:\n{context_response}\n")

        # Parallel Processing: Planner, Empathy, Emotion
        planner_response = agent.execute(combined_input, start_node="Planner")
        empathy_response = agent.execute(combined_input, start_node="Empathy")
        emotion_response = agent.execute(combined_input, start_node="Emotion")
        combined_input = f"User Input: {input_text}, \n\nContext: {context_response}, \n\nPlanner: {planner_response}, \n\nEmpathy: {empathy_response}, \n\nEmotion: {emotion_response}"
        conversation_log.append(f"## PARALLEL PROCESSING:\n### PLANNER:\n{planner_response}\n### EMPATHY:\n{empathy_response}\n### EMOTION:\n{emotion_response}\n")

        # Determine route based on context
        if "code" in context_response.lower():
            # Code generation cycle
            coder_response = agent.execute(combined_input, start_node="Coder")
            combined_input = f"User Input: {input_text}, \n\nCoder: {coder_response}"
            superego_response = agent.execute(combined_input, start_node="Superego")
            combined_input = f"User Input: {input_text}, \n\nCoder: {coder_response}, \n\nSuperego: {superego_response}"
            creativity_response = agent.execute(combined_input, start_node="Creativity")
            combined_input = f"User Input: {input_text}, \n\nCoder: {coder_response}, \n\nSuperego: {superego_response}, \n\nCreativity: {creativity_response}"
            librarian_response = agent.execute(combined_input, start_node="Librarian")
            combined_input = f"User Input: {input_text}, \n\nCoder: {coder_response}, \n\nSuperego: {superego_response}, \n\nCreativity: {creativity_response}, \n\nLibrarian: {librarian_response}"
            conversation_log.append(f"## CODING CYCLE:\n### CODER:\n{coder_response}\n### SUPEREGO:\n{superego_response}\n### CREATIVITY:\n{creativity_response}\n### LIBRARIAN:\n{librarian_response}\n")
        else:
            # Consciousness cycle
            
            instincts_response = agent.execute(combined_input, start_node="Instincts")
            combined_input = f"{combined_input} \n\nInstincts: {instincts_response}"
            
            id_response = agent.execute(combined_input, start_node="Id")
            combined_input = f"{combined_input} \n\nId: {id_response}"
            
            ego_response = agent.execute(combined_input, start_node="Ego")
            combined_input = f"{combined_input} \n\nEgo: {ego_response}"
            
            superego_response = agent.execute(combined_input, start_node="Superego")
            combined_input = f"{combined_input} \n\nSuperego: {superego_response}"
            
            creativity_response = agent.execute(combined_input, start_node="Creativity")
            combined_input = f"{combined_input} \n\nCreativity: {creativity_response}"
            
            librarian_response = agent.execute(combined_input, start_node="Librarian")
            combined_input = f"{combined_input} \n\nLibrarian: {librarian_response}"
            
            conversation_log.append(f"## CONSCIOUSNESS CYCLE:\n### INSTINCTS:\n{instincts_response}\n### ID:\n{id_response}\n### EGO:\n{ego_response}\n### SUPEREGO:\n{superego_response}\n### CREATIVITY:\n{creativity_response}\n### LIBRARIAN:\n{librarian_response}\n")

        # Final judgment by Persona
        final_response = agent.execute(combined_input, start_node="Persona")
        conversation_log.append(f"### PERSONAL FINAL OUTPUT:\n{final_response}\n")
        # print(f"PERSONA: {final_response}")

        # Save final log
        save_conversation_log(conversation_log)
        


if __name__ == "__main__":
    agent = AgentGen("Consciousness", ["webCrawling", "imageGen"])
    
    # agent.visualize_graph()

    # Run the conversation in a separate thread
    with ThreadPoolExecutor() as executor:
        future = executor.submit(run_conversation)
        future.result()
