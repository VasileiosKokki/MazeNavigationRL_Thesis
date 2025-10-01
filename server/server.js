import express from 'express';
import expressWs from 'express-ws'
import zlib from 'node:zlib';
import rateLimit from 'express-rate-limit';
import {
	updateLevel,
	knockbackWithDmg,
	findHighestScore,
	checkForRespawn,
	checkForDeath,
	sendServerInfo,
	createFood,
	createProjectile,
	checkCollision,
	clamp,
	isInUnwalkableCell,
	createAgent, range, generateRandomWalls, sendDrawablesToPython, sendOneTimeDataToPython
} from './functions.js';
import { router } from './routes.js'
import { spawn } from 'child_process';
import * as fs from "fs";
import path from "path";
import { fileURLToPath } from 'url';
import { ArgumentParser } from 'argparse';


const pythonExecutable = path.join(process.env.HOME, 'miniforge3/envs/tankio/bin/python');
const mainScript = 'python/interface.py';
const dirname = path.dirname(fileURLToPath(import.meta.url));

const pythonProcess = spawn(pythonExecutable, [mainScript], {
	cwd: path.resolve(dirname, '..') // set working dir to project root
});

const limiter = rateLimit({
	windowMs: 1 * 60 * 1000, // 1 minute
	max: 1000, // Number of requests allowed in the time window
	message: 'Too many requests from this IP, please try again later.',
});
const app = express();

expressWs(app)
app.use(limiter);
app.use(router);

import readline from 'readline';

const rl = readline.createInterface({
	input: pythonProcess.stdout,
	crlfDelay: Infinity
});

rl.on('line', (line) => {
	try {
		const parsedData = JSON.parse(line);

		parsedData.forEach(newDrawable => {
			const existingDrawable = drawables.find(d => d.clientId === newDrawable.clientId);
			if (existingDrawable) {
				existingDrawable.topLeftX = newDrawable.topLeftX;
				existingDrawable.topLeftY = newDrawable.topLeftY;
			}
		});
	} catch (e) {
		console.log(e);
	}
});


pythonProcess.stderr.on('data', (data) => {
	console.log(`${data}`);
});


function Drawable(clientId, name, width, height, topLeftX, topLeftY, color, speed, direction, maxHealth, currentHealth, healthRegenRate, healthRegenDelay, bodyDamage, lastDamageTime, lastDamagerId, shape, type, parentId, projectileVelocity, projectileHealth, projectileDamage, shootingAngle, isShooting, lastProjectileTime, rateOfFire, level, score, experience, isToBeRespawned){
	this.clientId = clientId;
	this.name = name;
	this.width = Math.abs(width);
	this.height = Math.abs(height);
	this.topLeftX = topLeftX;
	this.topLeftY = topLeftY;
	this.color = color;
	this.speed = speed; // Movement Logic
	this.direction = direction; // Movement Logic
	this.maxHealth = maxHealth; // Health Logic
	this.currentHealth = currentHealth; // Health Logic
	this.healthRegenRate = healthRegenRate; // Health Logic
	this.healthRegenDelay = healthRegenDelay; // Health Logic
	this.bodyDamage = bodyDamage;
	this.lastDamageTime = lastDamageTime; // Health Logic
	this.lastDamagerId = lastDamagerId;
	this.shape = shape;
	this.type = type;
	this.parentId = parentId; // Projectile Logic
	this.projectileVelocity = projectileVelocity; // Projectile Logic
	this.projectileHealth = projectileHealth; // Projectile Logic
	this.projectileDamage = projectileDamage;
	this.shootingAngle = shootingAngle; // Projectile Logic
	this.isShooting = isShooting; // Projectile Logic
	this.lastProjectileTime = lastProjectileTime; // Projectile Logic
	this.rateOfFire = rateOfFire; // Projectile Logic
	this.level = level;
	this.score = score;
	this.experience = experience;
	this.isToBeRespawned = isToBeRespawned;
};



class SpatialPartitionGrid {
    constructor(width, height, rows, cols) {
        this.width = width;
        this.height = height;
        this.rows = rows;
        this.cols = cols;
        this.cellWidth = width / cols;
        this.cellHeight = height / rows;
        this.cells = new Array(rows * cols).fill().map(() => []);
    }

    clear() {
        for (const cell of this.cells) {
            cell.length = 0;
        }
    }

    getIndex(row, col) {
        return row * this.cols + col;
    }

    insert(drawable) {
        const leftCol = Math.floor((drawable.topLeftX) / this.cellWidth);
    	const rightCol = Math.floor((drawable.topLeftX + drawable.width) / this.cellWidth);
    	const topRow = Math.floor((drawable.topLeftY) / this.cellHeight);
    	const bottomRow = Math.floor((drawable.topLeftY + drawable.height) / this.cellHeight);

    	for (let row = topRow; row <= bottomRow; row++) {
        	for (let col = leftCol; col <= rightCol; col++) {
            	const index = this.getIndex(row, col);
            	if (row >= 0 && row < this.rows && col >= 0 && col < this.cols) {
                	this.cells[index].push(drawable);
            	}
        	}
    	}
    }
}


let clients = [];
let drawables = [];
let projectilesToBeShot = [];
let drawablesToBeRewarded = [];
let arrayOfArrays = [];
let deadButConnected = [];





let clientCounter = 0;
let currentPlayers = 0;
let maxPlayers = 5;
let currentHighestScore = 0;


let domain;
const secretKey = "fd867a587aa02407952a83e59675e99e4c8de5bb6640db609eb8e7fdfb358373";
// Create parser

const parser = new ArgumentParser({
	description: 'Server configuration'
});

parser.add_argument('--port', { type: 'int', default: 3001, help: 'Port number for the server' });

parser.add_argument('--eval', { action: 'store_true', default: false, help: 'Run in evaluation mode' });

// Parse arguments
const args = parser.parse_args();

// Use CLI argument instead of env var
const port = args.port;
const evalMode = args.eval;


let cellNum = 10
let cellSize = 250
let mapSize = cellSize * cellNum

const gameBoundsDimensions = {width:mapSize,height:mapSize};  // gamebounds
const spatialGridDimensions = {rows:cellNum,cols:cellNum}
const pathGridDimensions = {rows:cellNum,cols:cellNum}

let unwalkableCells = [];

for (let i = 0; i < cellNum; i++) {
	// Top row
	unwalkableCells.push([0, i]);
	// Bottom row
	unwalkableCells.push([cellNum - 1, i]);
	// Left col
	unwalkableCells.push([i, 0]);
	// Right col
	unwalkableCells.push([i, cellNum - 1]);
}

unwalkableCells = unwalkableCells.map(([x, y]) => [x, y, x, y]);


// let unwalkableCells;
//
// try {
// 	// Read the file asynchronously
// 	const data = fs.readFileSync('obstacle_patterns.json', 'utf8');
//
// 	// Parse the JSON data
// 	unwalkableCells = JSON.parse(data);
//
// 	// Convert arrays (if necessary) into tuples (arrays of length 2)
// 	unwalkableCells = unwalkableCells.map(([x, y]) => [x, y, x, y]);  // If items are in array format like [1, 1], [2, 2]
//
// 	console.log(unwalkableCells); // Display obstacles
//
// } catch (error) {
// 	if (error.code === 'ENOENT') {
// 		console.log("The file 'obstacles.json' does not exist.");
// 	} else if (error instanceof SyntaxError) {
// 		console.log("Error: The file 'obstacles.json' contains invalid JSON.");
// 	} else {
// 		console.error("An unexpected error occurred:", error);
// 	}
// }

const unwalkableCellsExpanded = unwalkableCells.flatMap(([xStart, yStart, xEnd, yEnd]) =>
	range(xStart, yStart, xEnd, yEnd)
);

// prevent the player from being able to go inside a wall when it is placed next to the gamebounds
const unwalkableCellsOvershoot = unwalkableCells.map(row =>
	row.map((value, index) => {
		if (index < 2 && value === 0) {
			// Modify Xstart and Ystart if they are 0
			return -1;
		} else if (index === 2 && value === pathGridDimensions.cols - 1) {
			// Modify Xend if it equals pathGridDimensions.cols
			return pathGridDimensions.cols;
		} else if (index === 3 && value === pathGridDimensions.rows - 1) {
			// Modify Yend if it equals pathGridDimensions.rows
			return pathGridDimensions.rows;
		}
		return value;
	})
);



const spatialGrid = new SpatialPartitionGrid(gameBoundsDimensions.width, gameBoundsDimensions.height, spatialGridDimensions.rows, spatialGridDimensions.cols);

let thisServer = {
	name: 'Server ' + port,
    port: port,
    status: 'Online',
	playerCount: 0,
    maxCount: maxPlayers
};



sendServerInfo(thisServer, secretKey);
setInterval(() => {
	sendServerInfo(thisServer, secretKey);
}, 2000);

sendOneTimeDataToPython(gameBoundsDimensions, pathGridDimensions, unwalkableCellsExpanded, evalMode, pythonProcess)


if (!evalMode) {
	for (let i = 0; i < 100; i++) {     // up to 10000 lagless without shooting
		clientCounter = createFood(clientCounter, drawables, cellSize, cellNum);
	}
}

for (let i = 0; i < 1; i++) {
	clientCounter = createAgent(clientCounter, drawables, cellSize, cellNum, evalMode);
}


function setDrawables(value) {
	drawables = value;
}

function setClients(value) {
	clients = value;
}

function setCurrentPlayers(value) {
	currentPlayers = value;
}

function setCurrentHighestScore(value) {
	currentHighestScore = value;
}

function setClientCounter(value) {
	clientCounter = value;
}

function getDrawables() {
	return drawables;
}

function getClients() {
	return clients;
}

function getCurrentPlayers() {
	return currentPlayers;
}

function getCurrentHighestScore() {
	return currentHighestScore;
}

function getClientCounter() {
	return clientCounter;
}

function getUnwalkableCells() {
	return unwalkableCells;
}


// ----------------------------------------- Part 1: Game Logic -----------------------------------------
						
	
function GameLogic(){
	drawables.forEach(function(drawable){
		switch(drawable.direction){
			case 'up':
				drawable.topLeftY-=drawable.speed;
				break;
			case 'down':
				drawable.topLeftY+=drawable.speed;
				break;	
			case 'right':
				drawable.topLeftX+=drawable.speed;
				break;
			case 'left':
				drawable.topLeftX-=drawable.speed;
				break;
			case 'up+right':
				drawable.topLeftY-=drawable.speed/1.414;
				drawable.topLeftX+=drawable.speed/1.414;
				break;
			case 'up+left':
				drawable.topLeftY-=drawable.speed/1.414;
				drawable.topLeftX-=drawable.speed/1.414;
				break;
			case 'down+right':
				drawable.topLeftY+=drawable.speed/1.414;
				drawable.topLeftX+=drawable.speed/1.414;
				break;
			case 'down+left':
				drawable.topLeftY+=drawable.speed/1.414;
				drawable.topLeftX-=drawable.speed/1.414;
				break;
			default:
				break;
		}

		// check currentHealth and maxHealth
		if (drawable.currentHealth > drawable.maxHealth){
			drawable.currentHealth = drawable.maxHealth;
		}


		// timer for regeneration
		if ((drawable.currentHealth < drawable.maxHealth) && (Date.now() - drawable.lastDamageTime >= drawable.healthRegenDelay)){
			drawable.currentHealth += drawable.healthRegenRate;
		}

		// timer for reload speed
		if (drawable.isShooting == true && (Date.now() - drawable.lastProjectileTime) >= drawable.rateOfFire){
			projectilesToBeShot.push(drawable)
			//createProjectile(drawable);
			//drawable.lastProjectileTime = Date.now();
		}

		// projectile path
		if (drawable.type == 'projectile'){
			drawable.topLeftX += Math.cos(drawable.direction) * drawable.speed;
			drawable.topLeftY += Math.sin(drawable.direction) * drawable.speed;
		}


		// boundary respect
		drawable.topLeftX = clamp(drawable.topLeftX, 0, gameBoundsDimensions.width - drawable.width);
    	drawable.topLeftY = clamp(drawable.topLeftY, 0, gameBoundsDimensions.height - drawable.height);

		isInUnwalkableCell(drawable, unwalkableCellsOvershoot, gameBoundsDimensions, pathGridDimensions)

		// insert drawable into spatialGridDimensions
		spatialGrid.insert(drawable);

	});



	// spawn projectiles in batches
	for (const drawable of projectilesToBeShot){
		clientCounter = createProjectile(drawable, clientCounter, drawables);
		drawable.lastProjectileTime = Date.now();
	}
	projectilesToBeShot.length = 0;

	


	// give experience while avoiding O(n^2)
	drawables.forEach(function (drawable) {
		drawablesToBeRewarded.forEach(function (pair) {
			if (pair.lastDamagerId === drawable.clientId && drawable.currentHealth > 0){
		  		drawable.experience += pair.experience;
				drawable.score += pair.experience;
		  		updateLevel(drawable, clients);
				currentHighestScore = findHighestScore(drawables, clients, currentHighestScore);
			}
		});
	  });
	drawablesToBeRewarded.length = 0;

	  
	// collision logic here
	for (let row = 0; row < spatialGridDimensions.rows; row++) {
		for (let col = 0; col < spatialGridDimensions.cols; col++) {
			const cellIndex = spatialGrid.getIndex(row, col);
			const cellDrawables = spatialGrid.cells[cellIndex];
	
			// Iterate through drawables within the current cell
			for (const drawable of cellDrawables) {
				// Check for collisions with other drawables in the same cell
				for (const otherDrawable of cellDrawables) {
					if (drawable !== otherDrawable && checkCollision(drawable, otherDrawable)) {
						knockbackWithDmg(drawable, otherDrawable);
					}
				}
			}
		}
	}
	spatialGrid.clear();



	// check currentHealth for death
	drawables = checkForDeath(drawables, drawablesToBeRewarded, deadButConnected, clients, cellSize, cellNum);

	// check message from client for player respawn
	({deadButConnected, currentHighestScore} = checkForRespawn(deadButConnected, drawables, clients, currentHighestScore, domain, port, secretKey, cellSize, cellNum));
}


// ----------------------------------------- Part 2: Message Logic -----------------------------------------

function MessageLogic(){
	const reducedDrawables = drawables.map(drawable => {	// we send only the data useful for the presentation logic
		return {
			clientId: drawable.clientId,
			name: drawable.name,
			width: drawable.width,
			height: drawable.height,
			topLeftX: drawable.topLeftX,
			topLeftY: drawable.topLeftY,
			color: drawable.color,
			maxHealth: drawable.maxHealth,
			currentHealth: drawable.currentHealth,
			shape: drawable.shape,
			type: drawable.type,
			shootingAngle: drawable.shootingAngle
		};
	});


	const sortedDrawables = reducedDrawables.slice().sort((a, b) => {	// we sort the array in order to use ctx.fillstyle as few times as possible	
		// Compare based on the 'color' property
		return a.color.localeCompare(b.color);
	});

	arrayOfArrays.length = 0;
	for (let i = 0; i < sortedDrawables.length; i++) {		 // reduce message size #1
		let drawable = sortedDrawables[i];
		let attributesArray = [
			drawable.clientId,
			drawable.name,
			drawable.width,
			drawable.height,
			drawable.topLeftX,
			drawable.topLeftY,
			drawable.color,
			drawable.maxHealth,
			drawable.currentHealth,
			drawable.shape,
			drawable.type,
			drawable.shootingAngle
		];
	
		arrayOfArrays.push(attributesArray);
	}
	
	

	const message = JSON.stringify({type:'model',data:arrayOfArrays});
	const compressedData = zlib.deflateSync(message, { level: -1 });	  	  // reduce message size #2
	//console.log(`Sending message of size ${compressedData.length} bytes`);

	// Add the flag to the compressed data
	const flag = "1"; // flag for compressed data
	const dataWithFlag = Buffer.concat([compressedData, Buffer.from(flag)]);

	clients.forEach(function(client){
		if (client.connection.readyState == 1 /*&& client.clientId <= 10000*/){
			client.connection.send(dataWithFlag);
			//console.log(`Sending message of size ${compressedData.length} bytes to client ${client.clientId}`);
		}
	});
	
}


// ----------------------------------------- Main -----------------------------------------


let interval = setInterval(function(){
	GameLogic();	// the more we add the better the collision detection for projectiles with high velocity but worse performance for the server
	GameLogic();
	GameLogic();
	GameLogic();
	GameLogic();
	GameLogic();
	GameLogic();
	GameLogic();
	MessageLogic();
	sendDrawablesToPython(drawables, pythonProcess)
},40);







app.listen(port);



export { Drawable, setCurrentHighestScore, setCurrentPlayers, setDrawables, setClients, setClientCounter, getDrawables, getCurrentHighestScore, getClients, getCurrentPlayers, getClientCounter, getUnwalkableCells }
export { maxPlayers, port, secretKey, gameBoundsDimensions, cellSize, cellNum, spatialGridDimensions, pathGridDimensions, drawables, unwalkableCells, unwalkableCellsExpanded }
