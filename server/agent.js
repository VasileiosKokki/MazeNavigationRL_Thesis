import {
    drawables,
    gameBoundsDimensions,
    pathGridDimensions,
    unwalkableCellsExpanded
} from "./server.js";
import PF from "pathfinding";



const ACTIONS = {
    UP: { x: 0, y: -1 },
    DOWN: { x: 0, y: 1 },
    LEFT: { x: -1, y: 0 },
    RIGHT: { x: 1, y: 0 },
    STOP: { x: 0, y: 0 }
};

function isWithinBounds(cellX, cellY) {
    return (
        cellX >= 0 &&
        cellX < pathGridDimensions.cols &&
        cellY >= 0 &&
        cellY < pathGridDimensions.rows
    );
}

// Precompute the set of unwalkable cells once
const unwalkableCellsSet = new Set(unwalkableCellsExpanded.map(cell => `${cell[0]},${cell[1]}`));

function isObstacleBetween(drawable1, drawable2) {
    const x1 = drawable1.cellX;
    const y1 = drawable1.cellY;
    const x2 = drawable2.cellX;
    const y2 = drawable2.cellY;

    // Only check if they are aligned horizontally or vertically
    if (x1 === x2) {
        // Moving vertically
        const direction = y2 > y1 ? 1 : -1;
        // Directly check up to 3 key points in this direction
        for (let i = 1; i <= 3; i++) {
            const checkY = y1 + i * direction;
            if (unwalkableCellsSet.has(`${x1},${checkY}`)) {
                return true;
            }
            if ((direction === 1 && checkY >= y2) || (direction === -1 && checkY <= y2)) break;
        }
    } else if (y1 === y2) {
        // Moving horizontally
        const direction = x2 > x1 ? 1 : -1;
        // Directly check up to 3 key points in this direction
        for (let i = 1; i <= 3; i++) {
            const checkX = x1 + i * direction;
            if (unwalkableCellsSet.has(`${checkX},${y1}`)) {
                return true;
            }
            if ((direction === 1 && checkX >= x2) || (direction === -1 && checkX <= x2)) break;
        }
    }

    // No obstacles found in direct path
    return false;
}

// Function to check if a cell is walkable
function isWalkable(cellX, cellY) {
    // First, ensure the cell is within bounds
    if (!isWithinBounds(cellX, cellY)) {
        return false;
    }

    // Then, ensure it is not in unwalkable cells
    return !unwalkableCellsExpanded.some(
        unwalkable => unwalkable[0] === cellX && unwalkable[1] === cellY
    );
}

function calculateNewPath(path, newCell, index){
    //console.log(path)
    const cellIndex = path.findIndex(([pathX, pathY]) => pathX === newCell.x && pathY === newCell.y);

    if (cellIndex !== -1) {
        // If the cell is found, remove it
        if (index == 0) {
            path.splice(cellIndex - 1, 1);
            //console.log("The cell was in the path and has been removed.");
        } else {
            path.splice(cellIndex + 1, 1);
            //console.log("The cell was in the path and has been removed.");
        }
    } else {
        // If the cell is not found, add it to the beginning of the path
        if (index == 0) {
            path.unshift([newCell.x, newCell.y]);
            //console.log("The cell was not in the path and has been added to the beginning.");
        } else {
            path.push([newCell.x, newCell.y]);
            //console.log("The cell was not in the path and has been added to the end.");
        }
    }
    return path
}


function generateSuccessor(index, action, state, paths) {
    const move = ACTIONS[action];
    const drawable = state[index]

    if (!move) {
        throw new Error(`Invalid action: ${action}`);
    }

    const newCell = {
        x: drawable.cellX + move.x,
        y: drawable.cellY + move.y
    };

    //drawable.cell = newCell

    const newState = state.map(obj => ({ ...obj }));
    let newPaths = paths.map(innerArray => [...innerArray]);

    if (index == 0) {
        newPaths[0] = calculateNewPath(newPaths[0], newCell, index)
        newPaths[1] = calculateNewPath(newPaths[1], newCell, index)
    } else if (index == 1) {
        newPaths[0] = calculateNewPath(newPaths[0], newCell, index)
    } else {
        newPaths[1] = calculateNewPath(newPaths[1], newCell, index)
    }

    const newDrawable = {...drawable};

    newDrawable.cellX = newCell.x
    newDrawable.cellY = newCell.y


    newState[index] = newDrawable; // Replace the old drawable with the updated one

    return {newState, newPaths};
}


// Function to get legal actions
function getLegalActions(index, state) {
    const legalActions = [];
    const drawable = state[index]

    for (const [actionName, { x, y }] of Object.entries(ACTIONS)) {
        const newCell = { x: drawable.cellX + x, y: drawable.cellY + y };

        if (isWalkable(newCell.x, newCell.y)) {
            legalActions.push(actionName);
        }
    }

    return legalActions;
}

function getAction(state, paths, depth) {
    const legalActions = getLegalActions(0, state);
    let maxAction = 'STOP';
    let maxV = -Infinity;

    for (const action of legalActions) {
        const {newState, newPaths} = generateSuccessor(0, action, state, paths);
        // Pacman (root) is the max player, so new player is a min player.
        const v = value(newState, newPaths, depth, 1);
        if (v > maxV) {
            maxAction = action;
            maxV = v;
        }
    }
    return {maxAction, maxV};
}

function isLose(state) {
    const target = state[0]
    const agents = state.slice(1)

    for (const agent of agents) {
        if (target.cellX === agent.cellX && target.cellY === agent.cellY) {
            return true
        }
    }
    return false
}


function value(state, paths, depth, ghostIndex) {
    //console.log(depth);

    if (isLose(state) || depth === 0) {
        return evaluationFunction(state, paths);
    }

    if (ghostIndex === 0) {
        return maxValue(state, paths, depth);
    } else {
        return minValue(state, paths, depth, ghostIndex);
    }
}


function maxValue(state, paths, depth) {

    let v = -Infinity;
    const legalActions = getLegalActions(0, state);

    for (const action of legalActions) {
        const {newState, newPaths} = generateSuccessor(0, action, state, paths);
        v = Math.max(v, value(newState, newPaths, depth, 1));
    }
    return v;
}


function minValue(state, paths, depth, ghostIndex) {
    let v = Infinity;
    const legalActions = getLegalActions(ghostIndex, state);

    for (const action of legalActions) {
        const {newState, newPaths} = generateSuccessor(ghostIndex, action, state, paths);

        let newDepth, newAgentIndex;
        if (ghostIndex === 2) {
            // If last ghost agent, decrement depth for new cycle and reset to Pacman (index 0).
            newDepth = depth - 1;
            newAgentIndex = 0;
        } else {
            // If not last agent, continue to the new ghost.
            newDepth = depth;
            newAgentIndex = ghostIndex + 1;
        }

        v = Math.min(v, value(newState, newPaths, newDepth, newAgentIndex));
    }
    return v;
}


function manhattanDistance(drawable1, drawable2) {
    return Math.abs(drawable1.cellX - drawable2.cellX) + Math.abs(drawable1.cellY - drawable2.cellY);
}

function aStarPath(drawable1, drawable2) {
    const gridBackup = pathGrid.clone();
    const path = finder.findPath(drawable1.cellX, drawable1.cellY, drawable2.cellX, drawable2.cellY, gridBackup);
    return path
}

function evaluationFunction(state, paths) {
    const target = state[0]
    const agents = state.slice(1)
    let closestGhostDistance = Infinity;

    for (let index = 0; index < agents.length; index++) {
        //const ghostDistance = manhattanDistance(target, agent)
        const ghostDistance = paths[index].length
        //const obstacleExists = isObstacleBetween(target, agent)
        //console.log(obstacleExists)
        //const ghostDistance = aStarDistance(target, agent)
        if (ghostDistance < closestGhostDistance) {
            closestGhostDistance = ghostDistance
        }
        // if (ghostDistance < 2) {
        //     closestGhostDistance = -Infinity
        // }
    }

    const score = -closestGhostDistance
    return score
}






// ------------------------------------------------------------------------------------------------

let startCell;
let endCell;


const cell = {width: gameBoundsDimensions.width/pathGridDimensions.cols, height: gameBoundsDimensions.height/pathGridDimensions.rows}

const pathGrid = new PF.Grid(pathGridDimensions.cols, pathGridDimensions.rows);
unwalkableCellsExpanded.forEach(([x, y]) => {
    pathGrid.setWalkableAt(x, y, false);
});
const finder = new PF.AStarFinder();



let interval = setInterval(function(){
    // ai agent
    const agents = drawables.filter(drawable => drawable.type === 'agent');
    const target = drawables.find(drawable => drawable.type === 'player');
    const targetAgentsArray = [target, ...agents];
    const paths = [];



    if (target && agents) {
        endCell = {
            x: Math.floor((target.topLeftX + target.width / 2) / cell.width),
            y: Math.floor((target.topLeftY + target.height / 2) / cell.height)
        }
        target.cellX = endCell.x
        target.cellY = endCell.y


        for (const agent of agents) {
            if (agent) {
                startCell = {
                    x: Math.floor((agent.topLeftX + agent.width / 2) / cell.width),
                    y: Math.floor((agent.topLeftY + agent.height / 2) / cell.height)
                }
                agent.cellX = startCell.x
                agent.cellY = startCell.y
                //console.log(getLegalActions(agent)+"\n")
                const ghostDistance = aStarPath(target, agent)
                paths.push(ghostDistance)
            }
        }





        const {maxAction, maxV} = getAction(targetAgentsArray, paths, 2)
        const action = maxAction
        //console.log(maxV)

        //console.log(action)
        if (action != "STOP") {
            const newCell = {
                x: target.cellX + ACTIONS[action].x,
                y: target.cellY + ACTIONS[action].y
            };

            const targetX = newCell.x * cell.width + cell.width / 2;  // Convert pathGridDimensions position to pixel position
            const targetY = newCell.y * cell.height + cell.height / 2;  // Convert pathGridDimensions position to pixel position

            // Calculate the direction vector from the agent's current position to the target cell's position
            const deltaX = targetX - (target.topLeftX + target.width / 2);
            const deltaY = targetY - (target.topLeftY + target.height / 2);

            // Calculate the distance to the target
            const distance = Math.abs(deltaX) + Math.abs(deltaY);

            // Normalize the direction vector and apply speed
            const normalizedStepX = distance !== 0 ? deltaX / distance : 0;
            const normalizedStepY = distance !== 0 ? deltaY / distance : 0;

            // Update the agent's position towards the target
            target.topLeftX += ACTIONS[action].x * target.speed * 3;
            target.topLeftY += ACTIONS[action].y * target.speed * 3;
        }

    }
},40);

