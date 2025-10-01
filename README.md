**MazeNavigationRL\_Thesis**

**Note:** This documentation is based on the gymnasium branch of the repository, which contains the core RL implementation and its integration with the JavaScript game. 

**Maze Navigation using Reinforcement Learning** This repository contains the source code for a thesis project on training an agent to navigate mazes using Reinforcement Learning \(RL\). The project leverages the Gymnasium library to implement a standard maze navigation environment and explores the generalization capabilities of RL agents in increasingly complex scenarios. 

**Project Overview**

The central goal of this project is to explore and implement RL algorithms for a maze navigation task. The entire implementation is self-contained within the Python environment, using Gymnasium for the environment and agent interaction. The core algorithm used is Proximal Policy Optimization \(PPO\), implemented via the Stable-Baselines3 library. 

**RL Framework:** The agent's logic, training loops, and algorithm implementations are built in Python using the Gymnasium framework. 

**Standard Environment:** The project uses a custom, yet standard, Gymnasium environment for maze navigation. For training and evaluation, there is no external game server or WebSocket communication involved; the environment is fully defined and runs within the Python script. 

**Integration:** A key feature of this project is the successful integration of the trained Python-based agent with an external JavaScript game, demonstrating the transition from a simulated training environment to a practical, interactive application. 

**Technical Stack**

**Core RL & Environment:** Python, Gymnasium, Stable-Baselines3

**Deep Learning Backend:** PyTorch

**Dependencies:** The specific Python libraries required \(e.g., PyTorch, Gymnasium, Stable-Baselines3\) can be found in requirements.txt . 



**Getting Started**

To run this project, you only need to set up the Python environment. 

**Prerequisites**

Python \(3.x recommended\) and the pip package installer. 

**Setup Instructions**

1. Clone the repository:

git clone https://github.com/VasileiosKokki/MazeNavigationRL_Thesis.git 

2. Navigate to the project directory:

cd MazeNavigationRL\_Thesis

3. Set up the Python Environment \(it is highly recommended to use a virtual environment\):

\# Create a virtual environment

python -m venv venv

\# Activate the virtual environment

\# On Windows:

venv\\Scripts\\activate

\# On macOS/Linux:

source venv/bin/activate

\# Install Python dependencies

pip install -r requirements.txt

The repository also contains a requirements\_mac.txt file. If you are on a macOS

system and encounter issues, you may need to use this file instead. 

**How to Run an Experiment**

The main script for interacting with the agent is python/agent.py . It accepts arguments to control its behavior, such as training, testing, or evaluating. 

1. Activate the Python virtual environment. 

2. Run the agent script:



Navigate to the root project directory and execute the agent.py script with the desired arguments. The script supports the following modes:

--train : Starts a new training session. 

--test : Runs the agent in testing mode, often to visualize its behavior. 

--eval : Runs a formal evaluation of a trained model. 

You must also specify the experiment folder, which contains the configuration and saved models. 

**Example \(running a test\):**

python python/agent.py --folder experiment7 --test **Example \(starting training\):**

python python/agent.py --folder experiment7 --train 3. Monitor Training:

The project uses TensorBoard. You can launch it to view training progress: tensorboard --logdir logs

**Tankio: A Multiplayer Game Engine**

This project is a lightweight, real-time multiplayer game engine built with Node.js and JavaScript. It follows a Model-View-Controller \(MVC\) thin-client architecture, where the server holds the complete game state and logic, while clients are responsible only for rendering. This engine serves as the interactive environment for the Maze Navigation Reinforcement Learning Thesis project. 

**Architectural Overview**

The engine is designed around a clear separation of concerns between the server and the client:

**Server \(The "Model" & "Controller"\):** A Node.js application responsible for: Managing WebSocket connections from multiple clients. 

Processing player actions and inputs. 





Continuously updating the game's model, which represents the entire game world \(players, obstacles, etc.\). 

Broadcasting the updated game state to all connected clients at a regular interval \(every 40ms\). 

**Client \(The "View"\):** A browser-based application built with HTML and JavaScript that: Establishes a WebSocket connection to the server. 

Receives game state updates as an array of arrays. 

Renders the game world on an HTML <canvas> element based on the received state. 

Captures user input and sends it to the server for processing. 

Utilizes a Web Worker to separate the canvas drawing logic from the main thread, which handles user input and communication with the server. This ensures a smoother user experience and prevents rendering from blocking I/O operations. 

**Getting Started**

Follow these instructions to set up and run the game server locally. 

**Prerequisites**

You must have Node.js and the npm package manager installed on your system. 

**Installation**

1. Clone the repository:

git clone https://github.com/VasileiosKokki/MazeNavigationRL_Thesis.git



2. Navigate to the project directory:

cd MazeNavigationRL\_Thesis

3. Install dependencies:

This command reads the package.json file and installs the required Node.js modules. 

npm install

**Running the Game**

1. Start the Server Hub:

The server hub manages and lists available game servers. It must be running for clients to find and join a game. Open a terminal and run: node serverhub/serverhub.js

2. Start the Game Server:

In a separate terminal, start the main game server. This server will register itself with the hub. You can specify a port or run it in evaluation mode. 

node server/server.js

The server accepts the following command-line arguments:

--port : Sets the port number. Defaults to 3001. 

--eval : Runs the server in evaluation mode for integration testing with Gymnasium. This mode includes speedup functionality and saves the visited cells for analysis. 

Example:

node server/server.js --port 3002

3. Access the client:



Once both the hub and the server are running, open your web browser and navigate to the server hub, where you can join the game. Users can only join servers through the hub. 

**Configuration**

The default configuration is set up for a local environment: **Server Hub Port:** The hub listens on port 3000 . 

**Game Server Port:** The game server \( server/server.js \) listens on port 3001 by default, but can be changed with the --port argument. 

**WebSocket Host:** The client connects to servers via the hub running on localhost . 

If you need to deploy this to a different environment, you will need to adjust these settings in the project's source files. 

**Integration and Evaluation of the RL Agent** A critical objective of this thesis was to bridge the gap between the abstract, simulated training of a Reinforcement Learning agent and its application in a practical, interactive environment. 

This section details the architecture and methodology used to integrate the Python-based Gymnasium agent with the real-time JavaScript game, and the process used to verify the correctness of this integration. 

**System Architecture**

The integration is designed to facilitate communication between two heterogeneous systems: the JavaScript game, which operates in a continuous 2D space, and the Gymnasium RL

environment, which operates in a discrete grid-based space. This is achieved through an intermediary Python script, interface.py , which acts as a translator and synchronizer between the two worlds. 





**Data Flow**

The interaction loop between the game and the RL agent follows a precise sequence of steps, orchestrated by the interface script:

1. **State Capture from Game:** The JavaScript game provides the current continuous coordinates \(x, y\) of the agent and the target. 

2. **Coordinate Transformation:** The interface.py script receives these continuous coordinates and converts them into discrete grid cell coordinates. It then updates the internal state of the custom Gymnasium environment with these new discrete positions. 

3. **Observation Generation:** The Gymnasium environment generates a formal observation \(e.g., a 3D array for a CNN policy\) based on the updated discrete state and returns it to the interface. 

4. **Action Prediction:** The interface script forwards this observation to the pre-trained PPO

model. 

5. **Action Selection:** The model predicts the next best action \(e.g., 0 for ';right', 1 for 'up'\) and sends it back to the interface. 

6. **State Update and Synchronization:** The interface script applies the selected action to calculate the agent's new continuous position and sends these updated coordinates back to the JavaScript game, which then re-renders the scene. This completes the loop. 

**Technical Implementation**

The communication channel between the Node.js environment of the game and the Python process of the agent is established by spawning the Python script from Node.js: const pythonProcess = spawn\(pythonExecutable, ...\); Data is exchanged between the two processes using their standard input \( stdin \) and standard output \( stdout \) streams, with messages formatted as JSON strings. The flow is as follows:

1. The JavaScript game serializes the state of its drawable objects \(agent, target\) into a JSON object and writes it to the Python process';s stdin . 

function sendDrawablesToPython\(drawables, pythonProcess\) \{

pythonProcess.stdin.write\(JSON.stringify\(\{type: 'drawables', data: drawables\}\) \+ '\\n

\}

2. The Python script continuously listens on its stdin . Upon receiving the 'drawables' 

data, it processes it, runs it through the RL agent to get the next action, updates the



state, and prints the new state as a JSON string to stdout . The flush=True argument is critical to ensure the data is sent immediately. 

for line in sys.stdin:

try:

data = json.loads\(line\)

... 

if data.get\('type'\) == 'drawables':

drawables = data.get\('data'\)

\# ... RL logic to get new agent state ... 

print\(json.dumps\(agents\), flush=True\)

... 

3. The JavaScript side listens to the stdout of the Python process. When a new line of data \(the updated JSON state\) is received, it parses it and updates the positions of the game objects on the canvas. 

const rl = readline.createInterface\(\{

input: pythonProcess.stdout, 

crlfDelay: Infinity

\}\); 

rl.on\('line', \(line\) => \{

try \{

const parsedData = JSON.parse\(line\); 

parsedData.forEach\(newDrawable => \{

const existingDrawable = drawables.find\(d => d.clientId === newDrawable.clien if \(existingDrawable\) \{

existingDrawable.topLeftX = newDrawable.topLeftX; existingDrawable.topLeftY = newDrawable.topLeftY; 

\}

\}\); 

\} catch \(e\) \{

console.log\(e\); 

\}

\}\); 

This cycle repeats, creating a real-time interaction loop between the game and the intelligent agent. 

**Integration Verification**

To rigorously verify that the integration was correct and did not alter the agent's deterministic behavior, a verification test was conducted. The goal was to prove that the agent's path is perfectly reproducible under the same initial conditions, whether running in the standalone Python environment or in the integrated JavaScript-Python system. 



The process was as follows:

1. **Standalone Execution:** The Gymnasium environment was run autonomously in Python for 100 episodes with eval mode and folder experiment2. A fixed random seed \( seed = 42 \) was used to ensure deterministic starting positions and behavior. The sequence of grid cells visited by the agent in each episode was logged to a file. 

2. **Integrated Execution:** The full integrated system \(JavaScript game \+ Python interface\) was run, also with seed = 42 , for 100 episodes with eval mode and folder experiment2. The visited cells were logged in the same manner. 

3. **Comparison:** The two log files were compared by running compare_visited_cells.py . The comparison showed no discrepancies, confirming that the integration layer was transparent and did not introduce any side effects. The agent's behavior remained identical and reproducible. 

The following table shows a sample of the logged data, demonstrating the deterministic paths taken by the agent in different episodes. 

**Episode**

**Visited Cells**

1

61, 51, 52, 53, 54, 44, 45

2

74, 73, 72, 62, 61, 51

3

12, 13, 23, 33, 43, 53, 63, 64, 74

4

66, 65, 55, 45, 35, 25, 24, 14

5

46, 45, 44, 34

... 

... 

95

27, 26, 25, 24, 23, 22, 21, 11

96

61, 51, 41, 31, 32, 33, 34, 35, 36, 26, 27

97

26, 36, 46, 45, 44, 43, 53

98

76, 75, 74, 73, 72, 62, 52, 42, 32, 31, 21

99

32, 33, 34, 35, 36, 46, 56, 57, 67



**Episode**

**Visited Cells**

100

73, 63, 53, 43, 44

This successful verification was a crucial step, proving the viability of deploying the trained agent in a real-world, interactive application. 

This HTML page was generated on: 2025-10-01. 

The content is derived from an analysis of the GitHub repository

VasileiosKokki/MazeNavigationRL_Thesis on the gymnasium branch. 

Information was sourced from the file list, commit history, and the uploaded thesis document provided in the user's request.



