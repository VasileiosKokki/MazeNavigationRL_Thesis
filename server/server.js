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

const pythonProcess = spawn('python', ['./python/main_petting.py']);

const limiter = rateLimit({
	windowMs: 1 * 60 * 1000, // 1 minute
	max: 1000, // Number of requests allowed in the time window
	message: 'Too many requests from this IP, please try again later.',
});
const app = express();

expressWs(app)
app.use(limiter);
app.use(router);



let dataBuffer = '';  // To accumulate incoming data

pythonProcess.stdout.on('data', (data) => {
	dataBuffer += data.toString();  // Accumulate data as string

	// Check if the buffer contains a complete message (e.g., using newline as delimiter)
	let messages = dataBuffer.split('\n');

	// Process each message except the last one, which may be incomplete
	messages.slice(0, -1).forEach(message => {
		try {
			const parsedData = JSON.parse(message);  // Try to parse each complete JSON string

			parsedData.forEach(newDrawable => {
				const index = drawables.findIndex(existingDrawable => existingDrawable.clientId === newDrawable.clientId);
				if (index !== -1) {
					drawables[index] = newDrawable;  // Update drawable
				}
			});

		} catch (e) {
			//console.error('Failed to parse JSON:', e);
			console.log(message)
		}
	});

	// Keep the last (incomplete) part in the buffer
	dataBuffer = messages[messages.length - 1];
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
let port = parseInt(process.env.PORT, 10);
const secretKey = "fd867a587aa02407952a83e59675e99e4c8de5bb6640db609eb8e7fdfb358373";




const gameBoundsDimensions = {width:3750,height:3750};  // gamebounds
const spatialGridDimensions = {rows:15,cols:15}
const pathGridDimensions = {rows:15,cols:15}



// const unwalkableCells = generateRandomWalls(0);
// const unwalkableCells = [
// 	[6,5], [5,2], [5,0], [6,2], [4, 0], [4, 9], [5, 1], [2, 2], [0, 14],
// 	[11, 14], [15, 5], [1, 15], [18, 10], [4, 2], [5, 3], [8, 2], [14, 15],
// 	[17, 14], [11, 7], [2, 4], [13, 1], [15, 7], [6, 4], [5, 5], [5, 14],
// 	[9, 12], [15, 0], [6, 6], [7, 5], [17, 9], [1, 3], [13, 5], [1, 12],
// 	[15, 11], [7, 7], [18, 16], [14, 3], [17, 2], [14, 12], [11, 4], [9, 16],
// 	[13, 7], [15, 4], [6, 1], [18, 0], [13, 16], [15, 13], [2, 13], [18, 18],
// 	[12, 17], [10, 1], [7, 2], [13, 18], [11, 18], [16, 14], [7, 11], [18, 11],
// 	[3, 16], [1, 0], [1, 9], [11, 11], [16, 16], [6, 17], [18, 8], [4, 17],
// 	[8, 8], [13, 4], [8, 17], [11, 13], [2, 10], [0, 13], [3, 2], [3, 11],
// 	[4, 10], [17, 13], [10, 7], [1, 4], [1, 13], [6, 3], [3, 4], [10, 0],
// 	[8, 12], [15, 8], [18, 13], [10, 2], [10, 11], [2, 16], [15, 10], [18, 15],
// 	[4, 16], [17, 10], [11, 3], [10, 13], [9, 15], [0, 12], [2, 9], [2, 18],
// 	[6, 0], [15, 12], [7, 8]
// ].map(([x, y]) => [x, y, x, y]);

// const unwalkableCells = [
// 	[0, 2], [9, 2],
// 	[1, 2], [1, 3], [1, 5], [1, 6],
// 	[2, 1], [2, 3], [2, 6], [2, 8],
// 	[3, 1], [3, 8],
// 	[4, 3], [4, 5], [4, 6],
// 	[5, 1], [5, 3], [5, 6], [5, 8],
// 	[6, 2], [6, 3], [6, 5], [6, 7],
// 	[7, 4],
// 	[8, 2], [8, 3], [8, 6], [8, 7]
// ].map(([x, y]) => [x, y, x, y]);

let unwalkableCells;

try {
	// Read the file asynchronously
	const data = fs.readFileSync('obstacles.json', 'utf8');

	// Parse the JSON data
	unwalkableCells = JSON.parse(data);
	unwalkableCells = [[3,4]];

	// Convert arrays (if necessary) into tuples (arrays of length 2)
	unwalkableCells = unwalkableCells.map(([x, y]) => [x, y, x, y]);  // If items are in array format like [1, 1], [2, 2]

	// console.log(unwalkableCells); // Display obstacles

} catch (error) {
	if (error.code === 'ENOENT') {
		console.log("The file 'obstacles.json' does not exist.");
	} else if (error instanceof SyntaxError) {
		console.log("Error: The file 'obstacles.json' contains invalid JSON.");
	} else {
		console.error("An unexpected error occurred:", error);
	}
}

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

sendOneTimeDataToPython(gameBoundsDimensions, pathGridDimensions, unwalkableCellsExpanded, pythonProcess)



for (let i = 0; i < 100; i++){     // up to 10000 lagless without shooting
	clientCounter = createFood(clientCounter, drawables, gameBoundsDimensions);
}
for (let i = 0; i < 2; i++) {
	clientCounter = createAgent(clientCounter, drawables, gameBoundsDimensions);
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
				drawable.topLeftY-=drawable.speed/1.414;  // diairoume me to 1.414 giati pame diagonia
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
	drawables = checkForDeath(drawables, drawablesToBeRewarded, deadButConnected, clients, gameBoundsDimensions);

	// check message from client for player respawn
	({deadButConnected, currentHighestScore} = checkForRespawn(deadButConnected, drawables, clients, currentHighestScore, domain, port, secretKey, gameBoundsDimensions));
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
	
	
	
	//const message = JSON.stringify({type:'model',data:sortedDrawables});
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
	// 8
	MessageLogic();
	sendDrawablesToPython(drawables, pythonProcess)
},40);







app.listen(port);



export { Drawable, setCurrentHighestScore, setCurrentPlayers, setDrawables, setClients, setClientCounter, getDrawables, getCurrentHighestScore, getClients, getCurrentPlayers, getClientCounter, getUnwalkableCells }
export { maxPlayers, port, secretKey, gameBoundsDimensions, spatialGridDimensions, pathGridDimensions, drawables, unwalkableCells, unwalkableCellsExpanded }
