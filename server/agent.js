import {
    drawables,
    gameBoundsDimensions,
    pathGridDimensions,
    unwalkableCellsExpanded
} from "./server.js";
import PF from "pathfinding";


function findTarget(agent, drawables) {
    // Find the first drawable target (e.g., player or other agents)
    const target = drawables.find(drawable => drawable.type === 'player');  // Adjust 'target' based on your implementation

    return target || null;  // Return the target or null if not found
}




// ------------------------------------------------------------------------------------------------

let gridBackup
let startCell;
let endCell;


const cell = {width: gameBoundsDimensions.width/pathGridDimensions.cols, height: gameBoundsDimensions.height/pathGridDimensions.rows}


const pathGrid = new PF.Grid(pathGridDimensions.cols, pathGridDimensions.rows);
unwalkableCellsExpanded.forEach(([x, y]) => {
    pathGrid.setWalkableAt(x, y, false);
});



let interval = setInterval(function(){
    // ai agent
    const agents = drawables.filter(drawable => drawable.type === 'agent');
    for (const agent of agents) {
        const target = findTarget(agent, drawables);

        if (target){
            endCell = {
                x: Math.floor((target.topLeftX + target.width / 2) / cell.width),
                y: Math.floor((target.topLeftY + target.height / 2) / cell.height)
            }
            pathGrid.setWalkableAt(endCell.x,endCell.y, true);
        }


        if (target && agent) {
            // Calculate the path to the target
            startCell = {
                x: Math.floor((agent.topLeftX + agent.width / 2) / cell.width),
                y: Math.floor((agent.topLeftY + agent.height / 2) / cell.height)
            }
            endCell = {
                x: Math.floor((target.topLeftX + target.width / 2) / cell.width),
                y: Math.floor((target.topLeftY + target.height / 2) / cell.height)
            }
            // console.log(endNode)

            if (
                (startCell.x >= 0 && startCell.x <= pathGridDimensions.cols) && (endCell.x >= 0 && endCell.x <= pathGridDimensions.cols) &&
                (startCell.y >= 0 && startCell.y <= pathGridDimensions.rows) && (endCell.y >= 0 && endCell.y <= pathGridDimensions.rows)
            ) {


                // Find the path using the A* algorithm
                const finder = new PF.AStarFinder();
                gridBackup = pathGrid.clone();
                const path = finder.findPath(startCell.x, startCell.y, endCell.x, endCell.y, gridBackup);

                if (path.length >= 1) {
                    const currentNode = path[0];
                    if (!agent.previousNode) {
                        agent.previousNode = currentNode;
                    }


                    if (JSON.stringify(agent.previousNode) != JSON.stringify(currentNode)) {
                        pathGrid.setWalkableAt(agent.previousNode[0], agent.previousNode[1], true);
                    }


                    if (JSON.stringify(agent.previousNode) == JSON.stringify(currentNode)) {
                        pathGrid.setWalkableAt(agent.previousNode[0], agent.previousNode[1], false);
                    }



                    agent.previousNode = currentNode;
                }

                // If a path is found, move the agent along the path
                if (path.length > 1) {
                    // Get the next node in the path
                    const nextNode = path[1];


                        // Calculate the target cell's pixel position
                        const targetX = nextNode[0] * cell.width + cell.width / 2;  // Convert pathGridDimensions position to pixel position
                        const targetY = nextNode[1] * cell.height + cell.height / 2;  // Convert pathGridDimensions position to pixel position

                        // Calculate the direction vector from the agent's current position to the target cell's position
                        const deltaX = targetX - (agent.topLeftX + agent.width / 2);
                        const deltaY = targetY - (agent.topLeftY + agent.height / 2);

                        // Calculate the distance to the target
                        const distance = Math.abs(deltaX) + Math.abs(deltaY);

                        // Normalize the direction vector and apply speed
                        const normalizedStepX = distance !== 0 ? deltaX / distance : 0;
                        const normalizedStepY = distance !== 0 ? deltaY / distance : 0;

                        // Update the agent's position towards the target
                        agent.topLeftX += normalizedStepX * agent.speed;
                        agent.topLeftY += normalizedStepY * agent.speed;




                    } else if (path.length == 1) {
                        let dirX = target.topLeftX - agent.topLeftX;
                        let dirY = target.topLeftY - agent.topLeftY;

                        // Calculate the distance
                        let distance = Math.sqrt(dirX * dirX + dirY * dirY);

                        // Normalize the direction vector
                        let normDirX = distance !== 0 ? dirX / distance : 0;
                        let normDirY = distance !== 0 ? dirY / distance : 0;

                        // Move the agent towards the target
                        agent.topLeftX += normDirX * agent.speed;
                        agent.topLeftY += normDirY * agent.speed;
                    }
                }
            }
        }
},5);

