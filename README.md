# Cognitive Agents Swarm

## Overview

This repository contains a Python implementation of a cognitive agents swarm, inspired by psychoanalytic theory. The system integrates various cognitive components such as Ego, Id, Superego, Creativity, Rationality, and Emotion to simulate complex interactions and generate responses to user inputs. The goal is to manage conversations between these components to provide comprehensive and balanced outputs.

## Key Features

- **Initialization and Setup**: Imports necessary libraries and modules, sets up environment variables and paths, and initializes tools and the large language model (LLM) from OpenAI.
- **AgentGen Class**: Manages the AI components, memory, and color schemes for message display. Includes methods for creating agents, managing interactions, and building workflows.
- **Conversation Management**: Defines the behavior and responses of cognitive components, ensuring modularity and flexibility.
- **Output Handling**: Saves conversation logs in markdown format and generates text-to-speech (TTS) outputs.
- **Dynamic Interaction**: Continuously processes user inputs to simulate ongoing conversations.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries: `dotenv`, `IPython`, `langchain`, `pathlib`, `random`, `sys`, `functools`, `operator`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AlinBolcas/Cognitive-Multi-Agent-Swarm.git
    cd cognitive-multi-agent-swarm
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add your OpenAI and Tavily API keys:
      ```
      OPENAI_API_KEY=your_openai_api_key
      TAVILY_API_KEY=your_tavily_api_key
      ```

### Running the System

1. Navigate to the project root directory.
2. Run the main script:
    ```bash
    python path/to/your_main_script.py
    ```
3. Interact with the system by providing user inputs.

## TODO

- Add a way to interrupt the conversation.
- Remove redundancy of the output node if only one agent node is used.
- Make the system more modular by declaring all agents together and removing TTS and text generation functionalities from the class.

## Code Overview

### AgentGen Class

- **Initialization**: Sets up AI components, memory, color schemes, and initializes tools.
- **create_agent**: Constructs agents using specified system prompts.
- **agent_node**: Manages the interaction between the state and agents.
- **router**: Routes the conversation to the appropriate agent based on context.
- **build_workflow**: Defines the workflow for the conversation between agents.
- **run**: Executes the workflow with user input, generating responses and outputs.

### Conversation Management

- **Prompts and Members**: Defines the behavior and responses of cognitive components.
- **Modularity**: Ensures components can be easily added or modified for different conversation needs.

### Output Handling

- **Markdown and TTS**: Saves conversation logs in markdown format and generates text-to-speech (TTS) outputs.
- **Dynamic Interaction**: Continuously processes user inputs to simulate ongoing conversations.
