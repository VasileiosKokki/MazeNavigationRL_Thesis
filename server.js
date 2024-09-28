const express = require('express');
const app = express();
var expressWs = require('express-ws')(app);
const zlib = require('node:zlib');
var rateLimit = require('ws-rate-limit')('0.05s', 5);
const rateLimit2 = require('express-rate-limit');










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




const canvas = {width:5000,height:5000};  // gamebounds
const cols = 20;  // best number for many cases
const rows = 20;  // best number for many cases
const spatialGrid = new SpatialPartitionGrid(canvas.width, canvas.height, rows, cols);



var thisServer = {
	name: 'Server '+port,
    port: port,
    status: 'Online',
    playerCount: currentPlayers,
    maxCount: maxPlayers
};



const limiter = rateLimit2({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 1000, // Number of requests allowed in the time window
    message: 'Too many requests from this IP, please try again later.',
});



sendServerInfo();
setInterval(() => {
	sendServerInfo();
}, 2000);











// ----------------------------------------- Start of Environmental Food -----------------------------------------


for (i = 0; i < 1000; i++){     // up to 10000 lagless without shooting
	createFood();
}




// ----------------------------------------- End of Environmental Food -----------------------------------------





// ----------------------------------------- Start of Route Handlers -----------------------------------------



app.use(limiter);


app.ws('/connected', function(ws, req){

	rateLimit(ws);
	if (currentPlayers >= maxPlayers) {
		ws.close(1000, 'Maximum player count reached');
	} else {
		const parts = req.headers.host.split(':');
		domain = parts[0];


		const clientId = clientCounter;
		clients.push({clientId:clientId,connection:ws});
		clientCounter++;
		currentPlayers++;
		sendPlayerCount();


		const name = req.url.split('?')[1];
		var width = 15;
		var height = 15;
		const topLeftX = Math.random()*(canvas.width-30);
		const topLeftY = Math.random()*(canvas.height-30);
		const color = '#' + (Math.random() * 0xFFFFFF << 0).toString(16).padStart(6, '0');
		var speed = 5;
		var direction = null;
		var maxHealth = 2000;
		var currentHealth = maxHealth;
		var healthRegenRate = 0.5 / 100 * maxHealth;  // change only the first part of the variable
		var healthRegenDelay = 10 * 1000;  // change only the first part of the variable (in seconds)
		var bodyDamage = 20;
		var lastDamageTime = null;
		var lastDamagerId = null;
		const shape = 'oval';
		// type
		const parentId = null;
		var projectileVelocity = 20;
		var projectileHealth = 40;
		var projectileDamage = 50;
		var shootingAngle = null;
		var isShooting = false;
		var lastProjectileTime = null;
		var rateOfFire = 0.1 * 1000;  // change only the first part of the variable (in seconds)
		var level = 1;
		var score = 0;
		var experience = 0;
		var isToBeRespawned = false; 

		
		let drawable = new Drawable(clientId, name, width, height, topLeftX, topLeftY, color, speed, direction, maxHealth, currentHealth, healthRegenRate, healthRegenDelay, bodyDamage, lastDamageTime, lastDamagerId, shape, 'player', parentId, projectileVelocity, projectileHealth, projectileDamage, shootingAngle, isShooting, lastProjectileTime, rateOfFire, level, score, experience, isToBeRespawned);
		drawables.push(drawable);
		

		addListeners(drawable, ws);
	
		

		const message = JSON.stringify({type:'connected',data:{clientId:drawable.clientId,bounds:canvas,grid:{cols,rows}}});
		const nonCompressedData = message;

		const flag = "0"; // flag for noncompressed data
		const dataWithFlag = Buffer.concat([Buffer.from(nonCompressedData), Buffer.from(flag)]);


		const message2 = JSON.stringify({type:'highScore',data:{highScore : currentHighestScore}});
		const nonCompressedData2 = message2;

		const flag2 = "0"; // flag for noncompressed data
		const dataWithFlag2 = Buffer.concat([Buffer.from(nonCompressedData2), Buffer.from(flag2)]);



		clients.forEach(function(client){
			if (client.connection.readyState == 1 && client.clientId == drawable.clientId){
				client.connection.send(dataWithFlag);
				client.connection.send(dataWithFlag2);
			}
		});

	
	}
	

});


app.get('/*', (req, res) => {
	res.redirect(`http://${req.hostname}:3000`);		
});



// ----------------------------------------- End of Route Handlers -----------------------------------------





// ----------------------------------------- Start Of Functions -----------------------------------------


function createFood(){
	const clientId = clientCounter;
	clientCounter++;

	const name = null;
	var width = 30;
	var height = 30;
	const topLeftX = Math.random()*(canvas.width-30);
	const topLeftY = Math.random()*(canvas.height-30);
	var color;
	if (clientId % 2 == 0){
		color = '#AA0000'
	} else {
		color = '#00AAAA'
	}
	var speed = 0;
	var direction = null;
	var maxHealth = 20;
	var currentHealth = maxHealth;
	var healthRegenRate = 0.5 / 100 * maxHealth;  // change only the first part of the variable
	var healthRegenDelay = 10 * 1000;  // change only the first part of the variable (in seconds)
	var bodyDamage = 20;
	var lastDamageTime = null;
	var lastDamagerId = null;
	const shape = 'oval';
	// type
	const parentId = null;
	var projectileVelocity = null;
	var projectileHealth = null;
	var projectileDamage = null;
	var shootingAngle = null;
	var isShooting = null;
	var lastProjectileTime = null;
	var rateOfFire = null;
	var level = 1;
	var score = null;
	var experience = 1000;
	var isToBeRespawned = null;

	
	let drawable = new Drawable(clientId, name, width, height, topLeftX, topLeftY, color, speed, direction, maxHealth, currentHealth, healthRegenRate, healthRegenDelay, bodyDamage, lastDamageTime, lastDamagerId, shape, 'food', parentId, projectileVelocity, projectileHealth, projectileDamage, shootingAngle, isShooting, lastProjectileTime, rateOfFire, level, score, experience, isToBeRespawned);
	drawables.push(drawable);
}



function createProjectile(parent){


	xaxis = parent.topLeftX + parent.width/2;
	yaxis = parent.topLeftY + parent.height/2;
	
		
	const clientId = clientCounter;
	clientCounter++;

	const name = null;
	var width = parent.width/5;
	var height = parent.height/5;
	const topLeftX = xaxis - width/2 + Math.cos(parent.shootingAngle) * parent.width/2;
	const topLeftY = yaxis - height/2 + Math.sin(parent.shootingAngle) * parent.height/2;
	const color = parent.color;
	var speed = parent.projectileVelocity;
	var direction = parent.shootingAngle;
	var maxHealth = parent.projectileHealth;
	var currentHealth = maxHealth;
	var healthRegenRate = null;
	var healthRegenDelay = null;
	var bodyDamage = parent.projectileDamage;
	var lastDamageTime = null;
	var lastDamagerId = null;
	const shape = 'oval';
	// type
	const parentId = parent.clientId;
	var projectileVelocity = null;
	var projectileHealth = null;
	var projectileDamage = null;
	var shootingAngle = null;
	var isShooting = null;
	var lastProjectileTime = null;
	var rateOfFire = null;
	var level = parent.level;
	var score = null;
	var experience = null;
	var isToBeRespawned = null;
	

	
	let drawable = new Drawable(clientId, name, width, height, topLeftX, topLeftY, color, speed, direction, maxHealth, currentHealth, healthRegenRate, healthRegenDelay, bodyDamage, lastDamageTime, lastDamagerId, shape, 'projectile', parentId, projectileVelocity, projectileHealth, projectileDamage, shootingAngle, isShooting, lastProjectileTime, rateOfFire, level, score, experience, isToBeRespawned);
	drawables.push(drawable);

	
	
	setTimeout(function() {
		drawable.currentHealth = 0;
	}, 4000);
	
	
}



function respawnPlayer(oldDrawable, ws){
	const clientId = oldDrawable.clientId;

	const name = oldDrawable.name;
	var width = 150;
	var height = 150;
	const topLeftX = Math.random()*(canvas.width-30);
	const topLeftY = Math.random()*(canvas.height-30);
	const color = '#' + (Math.random() * 0xFFFFFF << 0).toString(16).padStart(6, '0');
	var speed = 10;
	var direction = null;
	var maxHealth = 2000;
	var currentHealth = maxHealth;
	var healthRegenRate = 0.5 / 100 * maxHealth;  // change only the first part of the variable
	var healthRegenDelay = 10 * 1000;  // change only the first part of the variable (in seconds)
	var bodyDamage = 20;
	var lastDamageTime = null;
	var lastDamagerId = null;
	const shape = 'oval';
	// type
	const parentId = null;
	var projectileVelocity = 20;
	var projectileHealth = 20;
	var projectileDamage = 50;
	var shootingAngle = null;
	var isShooting = false;
	var lastProjectileTime = null;
	var rateOfFire = 0.1 * 1000;  // change only the first part of the variable (in seconds)
	var level = 1;
	var score = (turnLevelToExperience(oldDrawable.level) + oldDrawable.experience) / 2;
	var experience = score;
	var isToBeRespawned = false; 
	

	
	let drawable = new Drawable(clientId, name, width, height, topLeftX, topLeftY, color, speed, direction, maxHealth, currentHealth, healthRegenRate, healthRegenDelay, bodyDamage, lastDamageTime, lastDamagerId, shape, 'player', parentId, projectileVelocity, projectileHealth, projectileDamage, shootingAngle, isShooting, lastProjectileTime, rateOfFire, level, score, experience, isToBeRespawned);
	drawables.push(drawable);


	updateLevel(drawable);
	findHighestScore();
	addListeners(drawable, ws);

}



function addListeners(drawable, ws){
	ws.on('message', function(msg){
		const message = JSON.parse(msg);
		switch(message.type){
			case 'userAction':
				switch(message.data){
					case 'ArrowUp':
						drawable.direction = 'up';
						break;
					case 'ArrowDown':
						drawable.direction = 'down';
						break;
					case 'ArrowRight':
						drawable.direction = 'right';
						break;
					case 'ArrowLeft':
						drawable.direction = 'left';
						break;
					case 'ArrowUp+Right':
						drawable.direction = 'up+right';
						break;
					case 'ArrowUp+Left':
						drawable.direction = 'up+left';
						break;
					case 'ArrowDown+Right':
						drawable.direction = 'down+right';
						break;
					case 'ArrowDown+Left':
						drawable.direction = 'down+left';
						break;
					case 'ArrowNotClicked':
						drawable.direction = null;
						break;
				}
				break;
			case 'userProjectile':
				switch(message.shoot){
					case 'true':
						drawable.isShooting = true;				
						break;
					case 'false':
						drawable.isShooting = false;
						break;
					case 'direction':						
						drawable.shootingAngle = message.data;
						break;
				}
				break;
			case 'userRespawn':
				if (drawable.currentHealth <= 0){
					drawable.isToBeRespawned = true;
				}
				break;
		}
	});


		
	ws.on('error',function(){disconnect(drawable.clientId);});
	ws.on('close',function(){disconnect(drawable.clientId);});

}

function disconnect(clientId){
	currentPlayers--;
	sendPlayerCount();
	clients = clients.filter(function(client){
		if (client.clientId != clientId){
			return client;
		}
	});
	drawables = drawables.filter(function(drawable){
		if (drawable.clientId != clientId){
			return drawable;
		}
	});
	findHighestScore();

	  
}




function checkCollision(drawable1, drawable2) {
	if (drawable1.parentId == drawable2.clientId || drawable2.parentId == drawable1.clientId || (drawable1.parentId == drawable2.parentId && drawable1.parentId != null)){
		return false;
	}
	// Rectangle to Rectangle Collision
	if (
		drawable1.topLeftX + drawable1.width > drawable2.topLeftX &&
		drawable1.topLeftX < drawable2.topLeftX + drawable2.width &&
		drawable1.topLeftY + drawable1.height > drawable2.topLeftY &&
		drawable1.topLeftY < drawable2.topLeftY + drawable2.height
	  ) {
		 // Oval to Oval Collision  
	  	 if (drawable1.shape == 'oval' && drawable2.shape == 'oval') {
	 	 	const dx = drawable2.topLeftX + drawable2.width / 2 - (drawable1.topLeftX + drawable1.width / 2);
	 	 	const dy = drawable2.topLeftY + drawable2.height / 2 - (drawable1.topLeftY + drawable1.height / 2);
	 	 	const radiiSumX = (drawable1.width + drawable2.width) / 2;
	 		const radiiSumY = (drawable1.height + drawable2.height) / 2;
  
	  		if (dx * dx / (radiiSumX * radiiSumX) + dy * dy / (radiiSumY * radiiSumY) < 1) {
				return true;
	  		}
		   // Rectangle to Oval Collision or Oval to Rectangle Collision
		 } else if (drawable1.shape == 'oval' || drawable2.shape == 'oval'){			
			var rectangle;
			var oval;
			if (drawable1.shape == 'oval'){
				oval = drawable1;
	  			rectangle = drawable2;  			
			} else {
				oval = drawable2;
	  			rectangle = drawable1;  
			}
			
	  		// Check if any edge of the rectangle intersects the oval
			const ovalCenterX = oval.topLeftX + oval.width / 2;
			const ovalCenterY = oval.topLeftY + oval.height / 2;
	  		const closestX = Math.max(rectangle.topLeftX, Math.min(ovalCenterX, rectangle.topLeftX + rectangle.width));
	  		const closestY = Math.max(rectangle.topLeftY, Math.min(ovalCenterY, rectangle.topLeftY + rectangle.height));
  
	  		const dx = closestX - ovalCenterX;
	  		const dy = closestY - ovalCenterY;
  
	  		if (dx * dx + dy * dy < (oval.width / 2) * (oval.width / 2)) {
				return true;
	  		}
		} else {
			return true;
		}
	}
	return false;
}



function KnockbackWithDmg(drawable1, drawable2){


	// knockback amount section

	let KnockbackAmount;

	if ((drawable1.direction == 'up' && drawable2.direction == 'down') || (drawable2.direction == 'up' && drawable1.direction == 'down') || 
		(drawable1.direction == 'right' && drawable2.direction == 'left') || (drawable2.direction == 'right' && drawable1.direction == 'left') || 
		(drawable1.direction == 'up+right' && drawable2.direction == 'down+left') || (drawable2.direction == 'up+right' && drawable1.direction == 'down+left') ||
		(drawable1.direction == 'up+left' && drawable2.direction == 'down+right') || (drawable2.direction == 'up+left' && drawable1.direction == 'down+right')){
		KnockbackAmount = (drawable1.speed + drawable2.speed)/2;
	} else if (drawable1.direction == null && drawable2.direction == null){
		KnockbackAmount = 1;
	} else if (drawable1.direction == null){
		KnockbackAmount = drawable2.speed/2;
	} else if (drawable2.direction == null){
		KnockbackAmount = drawable1.speed/2;
	} else {
		KnockbackAmount = Math.abs(drawable1.speed - drawable2.speed)/2;
	}

	if (KnockbackAmount < 1){
		KnockbackAmount = 1;
	}

	//KnockbackAmount *= 5;

	// damage section
	drawable1.currentHealth -= drawable2.bodyDamage;  // damage based on knockback and damage
	drawable2.currentHealth -= drawable1.bodyDamage;


	var time = Date.now();
	drawable1.lastDamageTime = time;
	drawable2.lastDamageTime = time;

	if (drawable2.parentId == null){
		drawable1.lastDamagerId = drawable2.clientId;
	} else {
		drawable1.lastDamagerId = drawable2.parentId;
	}

	if (drawable1.parentId == null){
		drawable2.lastDamagerId = drawable1.clientId;
	} else {
		drawable2.lastDamagerId = drawable1.parentId;
	}

	


	// knockback direction section

	// Calculate the center points of both drawables
	const drawable1CenterX = drawable1.topLeftX + drawable1.width / 2;
	const drawable1CenterY = drawable1.topLeftY + drawable1.height / 2;
	const drawable2CenterX = drawable2.topLeftX + drawable2.width / 2;
	const drawable2CenterY = drawable2.topLeftY + drawable2.height / 2;

	// Calculate the direction vector from drawable1's center to drawable2's center
	const directionX = drawable2CenterX - drawable1CenterX;
	const directionY = drawable2CenterY - drawable1CenterY;
	const distance = Math.sqrt(directionX * directionX + directionY * directionY);

	// Scale the direction vector by the KnockbackAmount
	const knockbackX = (directionX / distance) * KnockbackAmount;
	const knockbackY = (directionY / distance) * KnockbackAmount;

	// Apply the knockback to all corners of both drawables
	drawable1.topLeftX -= knockbackX;
	drawable1.topLeftY -= knockbackY;

	drawable2.topLeftX += knockbackX;
	drawable2.topLeftY += knockbackY;
	
}



function clamp(value, min, max) {
	return Math.min(Math.max(value, min), max);
}



function checkForRespawn(){
	clients.forEach(function(client){
		if (client.connection.readyState == 1){
			deadButConnected = deadButConnected.filter(function(drawable){
				if (drawable.clientId == client.clientId && drawable.isToBeRespawned == true){
					client.connection.removeAllListeners();
    		  		respawnPlayer(drawable, client.connection);	
					return false;  // remove if client sent message to respawn
				}
				return true;  // otherwise keep it
			});
		}
	});
}



function checkForDeath(){	
	drawables = drawables.filter(function(drawable){
		if (drawable.currentHealth <= 0){
			// experience distribution
			if (drawable.type == 'food' || drawable.type == 'player'){
				const pair = { lastDamagerId: drawable.lastDamagerId, experience: Math.round((drawable.experience + turnLevelToExperience(drawable.level)) / 2) };
				drawablesToBeRewarded.push(pair);
				// food replenish
				if (drawable.type == 'food'){
					setTimeout(createFood, 10000);
				} else
				// respawn players
				if (drawable.type == 'player'){
					deadButConnected.push(drawable);

					const message = JSON.stringify({type:'death',data:true});
					const nonCompressedData = message;
								
					const flag = "0"; // flag for noncompressed data
					const dataWithFlag = Buffer.concat([Buffer.from(nonCompressedData), Buffer.from(flag)]);
								
								
								
					clients.forEach(function(client){
						 if (client.connection.readyState == 1 && client.clientId == drawable.clientId){
							 client.connection.send(dataWithFlag);
						 }
					});
				}
			}
			return false;  // remove if drawable health <= 0
		}
		return true;  // otherwise keep it
	});	
}



function turnLevelToExperience(level) {
	let experience = 0;
	let baseDivisor = 1000; // You can adjust this as needed
  
	for (let i = 1; i < level; i++) {
	  experience += baseDivisor;
	  baseDivisor = Math.round(baseDivisor * 1.1); // Increase the divisor (you can adjust this factor)
	}
  
	return experience;
}



function updateLevel(drawable){
	let baseDivisor = Math.round(1000 * Math.pow(1.1, drawable.level - 1)); // Adjust baseDivisor based on starting level
	let levelupOccured = false;
	let sizeIncrease = 5;
	
	while (drawable.experience >= baseDivisor) {
	  drawable.experience -= baseDivisor;
	  baseDivisor = Math.round(baseDivisor * 1.1); // Increase the divisor (you can adjust this factor)
	  drawable.level++;
	  levelupOccured = true;
	}
	drawable.experience = Math.floor(drawable.experience);

	// Update drawable size if player levels up
	if (levelupOccured == true){
		drawable.topLeftX -= sizeIncrease/2;
		drawable.topLeftY -= sizeIncrease/2;
		drawable.width += sizeIncrease;
		drawable.height += sizeIncrease;
	}


	
	const message = JSON.stringify({type:'experience',data:{level:drawable.level,experience:drawable.experience,score:drawable.score}});
	const nonCompressedData = message;

	const flag = "0"; // flag for noncompressed data
	const dataWithFlag = Buffer.concat([Buffer.from(nonCompressedData), Buffer.from(flag)]);



	clients.forEach(function(client){
		 if (client.connection.readyState == 1 && client.clientId == drawable.clientId){
			 client.connection.send(dataWithFlag);
		 }
	});
}



function findHighestScore(){
	const playerDrawables = drawables.filter(function(drawable){
		if (drawable.type != 'player'){
			// experience distribution
			return false;  // remove if drawable isnt a player
		}
		return true;  // otherwise keep it
	});
	
	// Find the drawable with the highest experience.
	const highestScoreDrawable = playerDrawables.length > 0 ? playerDrawables.reduce((prev, current) => {
    	return current.score > prev.score ? current : prev;
    }) : null;
	

	// Check if the current highest experience has changed.
	if (highestScoreDrawable == null || highestScoreDrawable.score == 0){
		if (currentHighestScore != 0){
			currentHighestScore = 0;



			const message = JSON.stringify({type:'highScore',data:{highScore : currentHighestScore}});
			const nonCompressedData = message;

			const flag = "0"; // flag for noncompressed data
			const dataWithFlag = Buffer.concat([Buffer.from(nonCompressedData), Buffer.from(flag)]);


			clients.forEach(function(client){
			 	if (client.connection.readyState == 1){
				 	client.connection.send(dataWithFlag);
			 	}
			});
		}

	} else if (highestScoreDrawable.score != currentHighestScore){
		currentHighestScore = highestScoreDrawable.score;
		


		const message = JSON.stringify({type:'highScore',data:{highScore : currentHighestScore}});
		const nonCompressedData = message;

		const flag = "0"; // flag for noncompressed data
		const dataWithFlag = Buffer.concat([Buffer.from(nonCompressedData), Buffer.from(flag)]);


		clients.forEach(function(client){
		 	if (client.connection.readyState == 1){
			 	client.connection.send(dataWithFlag);
		 	}
		});

	}
	
}



async function sendPlayerCount(){
	thisServer.playerCount = currentPlayers;
	const playerCount = { playerCount: currentPlayers, senderURL: `http://${domain}:${port}` };
	const url = `http://${domain}:3000/playercount`;
    try {
        await fetch(url,{
			method:'POST',
			headers: {
				"Content-Type": "application/json",
				"Authorization": secretKey
			},   
			body: JSON.stringify(playerCount)       
		});
    } catch (error) {
        // Handle errors (e.g., server not found or offline)
        console.log(error.message);
    }
}



async function sendServerInfo(){
	const serverInfo = thisServer;
	const url = `http://localhost:3000/serverdata`;
    try {
        await fetch(url,{
			method:'POST',
			headers: {
				"Content-Type": "application/json",
				"Authorization": secretKey
			},   
			body: JSON.stringify(serverInfo),   
		});
    } catch (error) {
        // Handle errors (e.g., server not found or offline)
		// console.log(error.message);
    }
}






// ----------------------------------------- End Of Functions -----------------------------------------






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
		drawable.topLeftX = clamp(drawable.topLeftX, 0, canvas.width - drawable.width);
    	drawable.topLeftY = clamp(drawable.topLeftY, 0, canvas.height - drawable.height);

		// insert drawable into grid
		spatialGrid.insert(drawable);

	});



	// spawn projectiles in batches
	for (const drawable of projectilesToBeShot){
		createProjectile(drawable);
		drawable.lastProjectileTime = Date.now();
	}
	projectilesToBeShot.length = 0;

	


	// give experience while avoiding O(n^2)
	drawables.forEach(function (drawable) {
		drawablesToBeRewarded.forEach(function (pair) {
			if (pair.lastDamagerId === drawable.clientId && drawable.currentHealth > 0){
		  		drawable.experience += pair.experience;
				drawable.score += pair.experience;
		  		updateLevel(drawable);
				findHighestScore();
			}
		});
	  });
	drawablesToBeRewarded.length = 0;

	  
	// collision logic here
	for (let row = 0; row < rows; row++) {
		for (let col = 0; col < cols; col++) {
			const cellIndex = spatialGrid.getIndex(row, col);
			const cellDrawables = spatialGrid.cells[cellIndex];
	
			// Iterate through drawables within the current cell
			for (const drawable of cellDrawables) {
				// Check for collisions with other drawables in the same cell
				for (const otherDrawable of cellDrawables) {
					if (drawable !== otherDrawable && checkCollision(drawable, otherDrawable)) {
						KnockbackWithDmg(drawable, otherDrawable);
					}
				}
			}
		}
	}
	spatialGrid.clear();



	// check currentHealth for death
	checkForDeath();

	// check message from client for player respawn
	checkForRespawn();
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
	for (var i = 0; i < sortedDrawables.length; i++) {		 // reduce message size #1
		var drawable = sortedDrawables[i];
		var attributesArray = [
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
},40);







app.listen(port);