import {
    getClients,
    getCurrentHighestScore,
    getCurrentPlayers,
    getDrawables,
    setClients, setCurrentHighestScore, setCurrentPlayers,
    setDrawables,
    Drawable, getClientCounter, setClientCounter, pathGridDimensions
} from "./server.js";

function disconnect(clientId, clients, drawables, currentPlayers, currentHighestScore, domain, port, secretKey){
    currentPlayers--;
    sendPlayerCount(currentPlayers, domain, port, secretKey);
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
    currentHighestScore = findHighestScore(drawables, clients, currentHighestScore);
    return {clients , drawables, currentPlayers, currentHighestScore}


}


async function sendPlayerCount(currentPlayers, domain, port, secretKey){
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


function findHighestScore(drawables, clients, currentHighestScore){
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

    return currentHighestScore;

}



function addListeners(drawable, ws, domain, port, secretKey){


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



    ws.on('error', function () {
        let clients = getClients()
        let drawables = getDrawables()
        let currentPlayers = getCurrentPlayers()
        let currentHighestScore = getCurrentHighestScore();

        ({ clients, drawables, currentHighestScore, currentPlayers } = disconnect(drawable.clientId, clients, drawables, currentPlayers, currentHighestScore, domain, port, secretKey));

        setClients(clients)
        setDrawables(drawables)
        setCurrentPlayers(currentPlayers)
        setCurrentHighestScore(currentHighestScore)

    });

    ws.on('close', function () {
        let clients = getClients()
        let drawables = getDrawables()
        let currentPlayers = getCurrentPlayers()
        let currentHighestScore = getCurrentHighestScore();

        ({ clients, drawables, currentHighestScore, currentPlayers } = disconnect(drawable.clientId, clients, drawables, currentPlayers, currentHighestScore, domain, port, secretKey));

        setClients(clients)
        setDrawables(drawables)
        setCurrentPlayers(currentPlayers)
        setCurrentHighestScore(currentHighestScore)

    });

}

function getRandomPosition(cellSize, cellNum) {
    const offset = 1

    const minX = cellSize * offset;
    const maxX = cellSize * (cellNum - 1 - offset);
    const minY = cellSize * offset;
    const maxY = cellSize * (cellNum - 1 - offset);

    const centerX = minX + Math.random() * maxX;
    const centerY = minY + Math.random() * maxY;

    return { x: centerX, y: centerY };
}


function respawnPlayer(oldDrawable, ws, drawables, clients, currentHighestScore, domain, port, secretKey){
    const clientId = oldDrawable.clientId;

    const name = oldDrawable.name;
    let width = 150;
    let height = 150;
    const { x, y } = getRandomPosition(cellSize, cellNum)
    const topLeftX = x;
    const topLeftY = y;
    const color = '#' + (Math.random() * 0xFFFFFF << 0).toString(16).padStart(6, '0');
    let speed = 5;
    let direction = null;
    let maxHealth = 2000;
    let currentHealth = maxHealth;
    let healthRegenRate = 0.5 / 100 * maxHealth;  // change only the first part of the variable
    let healthRegenDelay = 10 * 1000;  // change only the first part of the variable (in seconds)
    let bodyDamage = 20;
    let lastDamageTime = null;
    let lastDamagerId = null;
    const shape = 'oval';
    const type = 'player';
    const parentId = null;
    let projectileVelocity = 20;
    let projectileHealth = 20;
    let projectileDamage = 50;
    let shootingAngle = null;
    let isShooting = false;
    let lastProjectileTime = null;
    let rateOfFire = 0.1 * 1000;  // change only the first part of the variable (in seconds)
    let level = 1;
    let score = (turnLevelToExperience(oldDrawable.level) + oldDrawable.experience) / 2;
    let experience = score;
    let isToBeRespawned = false;



    let drawable = new Drawable(clientId, name, width, height, topLeftX, topLeftY, color, speed, direction, maxHealth, currentHealth, healthRegenRate, healthRegenDelay, bodyDamage, lastDamageTime, lastDamagerId, shape, type, parentId, projectileVelocity, projectileHealth, projectileDamage, shootingAngle, isShooting, lastProjectileTime, rateOfFire, level, score, experience, isToBeRespawned);
    drawables.push(drawable);


    updateLevel(drawable, clients);
    currentHighestScore = findHighestScore(drawables, clients, currentHighestScore);
    addListeners(drawable, ws, domain, port, secretKey)

    return currentHighestScore

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


function updateLevel(drawable, clients){
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


function checkForDeath(drawables, drawablesToBeRewarded, deadButConnected, clients, cellSize, cellNum){
    drawables = drawables.filter(function(drawable){
        if (drawable.currentHealth <= 0){
            // experience distribution
            if (drawable.type == 'food' || drawable.type == 'player' || drawable.type == 'agent'){
                const pair = { lastDamagerId: drawable.lastDamagerId, experience: Math.round((drawable.experience + turnLevelToExperience(drawable.level)) / 2) };
                drawablesToBeRewarded.push(pair);
                // food replenish
                if (drawable.type == 'food'){
                    setTimeout(() => {
                        const drawables = getDrawables();
                        const clientCounter = getClientCounter();
                        const updatedClientCounter = createFood(clientCounter, drawables, cellSize, cellNum)
                        setClientCounter(updatedClientCounter)
                    }, 10000);
                } else if (drawable.type == 'agent'){
                    setTimeout(() => {
                        const drawables = getDrawables();
                        const clientCounter = getClientCounter();
                        const updatedClientCounter = createAgent(clientCounter, drawables, cellSize, cellNum)
                        setClientCounter(updatedClientCounter)
                    }, 10000);
                } else
                // put players to respawn queue
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

    return drawables
}


function createFood(clientCounter, drawables, cellSize, cellNum){
    const clientId = clientCounter;
    clientCounter++;

    const name = null;
    let width = 30;
    let height = 30;
    const { x, y } = getRandomPosition(cellSize, cellNum)
    const topLeftX = x;
    const topLeftY = y;
    let color;
    if (clientId % 2 == 0){
        color = '#AA0000'
    } else {
        color = '#00AAAA'
    }
    let speed = 0;
    let direction = null;
    let maxHealth = 20;
    let currentHealth = maxHealth;
    let healthRegenRate = 0.5 / 100 * maxHealth;  // change only the first part of the variable
    let healthRegenDelay = 10 * 1000;  // change only the first part of the variable (in seconds)
    let bodyDamage = 20;
    let lastDamageTime = null;
    let lastDamagerId = null;
    const shape = 'oval';
    // type
    const parentId = null;
    let projectileVelocity = null;
    let projectileHealth = null;
    let projectileDamage = null;
    let shootingAngle = null;
    let isShooting = null;
    let lastProjectileTime = null;
    let rateOfFire = null;
    let level = 1;
    let score = null;
    let experience = 1000;
    let isToBeRespawned = null;


    let drawable = new Drawable(clientId, name, width, height, topLeftX, topLeftY, color, speed, direction, maxHealth, currentHealth, healthRegenRate, healthRegenDelay, bodyDamage, lastDamageTime, lastDamagerId, shape, 'food', parentId, projectileVelocity, projectileHealth, projectileDamage, shootingAngle, isShooting, lastProjectileTime, rateOfFire, level, score, experience, isToBeRespawned);
    drawables.push(drawable);

    return clientCounter
}


function checkForRespawn(deadButConnected, drawables, clients, currentHighestScore, domain, port, secretKey, cellSize, cellNum){
    clients.forEach(function(client){
        if (client.connection.readyState == 1){
            deadButConnected = deadButConnected.filter(function(drawable){
                if (drawable.clientId == client.clientId && drawable.isToBeRespawned == true){
                    client.connection.removeAllListeners();
                    currentHighestScore = respawnPlayer(drawable, client.connection, drawables, clients, currentHighestScore, domain, port, secretKey, cellSize, cellNum)
                    return false;  // remove if client sent message to respawn
                }
                return true;  // otherwise keep it
            });
        }
    });

    return {deadButConnected, currentHighestScore}
}


function createProjectile(parent, clientCounter, drawables){


    const xaxis = parent.topLeftX + parent.width/2;
    const yaxis = parent.topLeftY + parent.height/2;


    const clientId = clientCounter;
    clientCounter++;

    const name = '';
    let width = parent.width/5;
    let height = parent.height/5;
    const topLeftX = xaxis - width/2 + Math.cos(parent.shootingAngle) * parent.width/2;
    const topLeftY = yaxis - height/2 + Math.sin(parent.shootingAngle) * parent.height/2;
    const color = parent.color;
    let speed = parent.projectileVelocity;
    let direction = parent.shootingAngle;
    let maxHealth = parent.projectileHealth;
    let currentHealth = maxHealth;
    let healthRegenRate = null;
    let healthRegenDelay = null;
    let bodyDamage = parent.projectileDamage;
    let lastDamageTime = null;
    let lastDamagerId = null;
    const shape = 'oval';
    const type = 'projectile';
    const parentId = parent.clientId;
    let projectileVelocity = null;
    let projectileHealth = null;
    let projectileDamage = null;
    let shootingAngle = null;
    let isShooting = null;
    let lastProjectileTime = null;
    let rateOfFire = null;
    let level = parent.level;
    let score = null;
    let experience = null;
    let isToBeRespawned = null;



    let drawable = new Drawable(clientId, name, width, height, topLeftX, topLeftY, color, speed, direction, maxHealth, currentHealth, healthRegenRate, healthRegenDelay, bodyDamage, lastDamageTime, lastDamagerId, shape, type, parentId, projectileVelocity, projectileHealth, projectileDamage, shootingAngle, isShooting, lastProjectileTime, rateOfFire, level, score, experience, isToBeRespawned);
    drawables.push(drawable);



    setTimeout(function() {
        drawable.currentHealth = 0;
    }, 4000);

    return clientCounter
}



async function sendServerInfo(thisServer, secretKey){
    const url = `http://localhost:3000/serverdata`;
    try {
        await fetch(url,{
            method:'POST',
            headers: {
                "Content-Type": "application/json",
                "Authorization": secretKey
            },
            body: JSON.stringify(thisServer),
        });
    } catch (error) {
        // console.log(error.message);
    }
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
            let rectangle;
            let oval;
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



function knockbackWithDmg(drawable1, drawable2){
    // knockback amount section

    let knockbackAmount;

    if ((drawable1.direction == 'up' && drawable2.direction == 'down') || (drawable2.direction == 'up' && drawable1.direction == 'down') ||
        (drawable1.direction == 'right' && drawable2.direction == 'left') || (drawable2.direction == 'right' && drawable1.direction == 'left') ||
        (drawable1.direction == 'up+right' && drawable2.direction == 'down+left') || (drawable2.direction == 'up+right' && drawable1.direction == 'down+left') ||
        (drawable1.direction == 'up+left' && drawable2.direction == 'down+right') || (drawable2.direction == 'up+left' && drawable1.direction == 'down+right')){
        knockbackAmount = (drawable1.speed + drawable2.speed)/2;
    } else if (drawable1.direction == null && drawable2.direction == null){
        knockbackAmount = 1;
    } else if (drawable1.direction == null){
        knockbackAmount = drawable2.speed/2;
    } else if (drawable2.direction == null){
        knockbackAmount = drawable1.speed/2;
    } else {
        knockbackAmount = Math.abs(drawable1.speed - drawable2.speed)/2;
    }

    if (knockbackAmount < 1){
        knockbackAmount = 1;
    }

    // damage section
    drawable1.currentHealth -= drawable2.bodyDamage;  // damage based on knockback and damage
    drawable2.currentHealth -= drawable1.bodyDamage;


    let time = Date.now();
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

    // Scale the direction vector by the knockbackAmount
    const knockbackX = (directionX / distance) * knockbackAmount;
    const knockbackY = (directionY / distance) * knockbackAmount;

    // Apply the knockback to all corners of both drawables
    drawable1.topLeftX -= knockbackX;
    drawable1.topLeftY -= knockbackY;

    drawable2.topLeftX += knockbackX;
    drawable2.topLeftY += knockbackY;

}


function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}


function range(xStart, yStart, xEnd, yEnd){
    const coordinates = [];

    for (let x = xStart; x <= xEnd; x++) {
        for (let y = yStart; y <= yEnd; y++) {
            coordinates.push([x, y]);
        }
    }

    return coordinates;
};

function outsideClamp(drawable, minX, maxX, minY, maxY) {

    // Check for specific corner cases

    const distanceFromTop = minY - drawable.topLeftY;  // Distance to the top boundary
    const distanceFromBottom = (drawable.topLeftY + drawable.height) - maxY;  // Distance to the bottom boundary
    const distanceFromLeft = minX - drawable.topLeftX;  // Distance to the left boundary
    const distanceFromRight = (drawable.topLeftX + drawable.width) - maxX;  // Distance to the right boundary


    if (distanceFromLeft == Math.max(distanceFromLeft,distanceFromRight,distanceFromTop,distanceFromBottom)) {
        drawable.topLeftX = minX - drawable.width;  // Push to the left boundary
    } else if (distanceFromRight == Math.max(distanceFromLeft,distanceFromRight,distanceFromTop,distanceFromBottom)) {
        drawable.topLeftX = maxX;  // Push to the right boundary
    } else if (distanceFromTop == Math.max(distanceFromLeft,distanceFromRight,distanceFromTop,distanceFromBottom)) {
        drawable.topLeftY = minY - drawable.height;  // Push above the top boundary
    } else if (distanceFromBottom == Math.max(distanceFromLeft,distanceFromRight,distanceFromTop,distanceFromBottom)) {
        drawable.topLeftY = maxY;  // Push below the bottom boundary
    }
}


function isInUnwalkableCell(drawable, unwalkableCells, gameBoundsDimensions, pathGridDimensions) {
    const cellWidth = gameBoundsDimensions.width / pathGridDimensions.cols;
    const cellHeight = gameBoundsDimensions.height / pathGridDimensions.rows;
    let cellInfo = {};

    // Find if drawable is in any unwalkable cell
    unwalkableCells.find(([gridXStart, gridYStart, gridXEnd, gridYEnd]) => {
        // Calculate the real top-left corner and size of the unwalkable cell
        const cellTopLeftX = gridXStart * cellWidth;
        const cellTopLeftY = gridYStart * cellHeight;
        const cellRealWidth = (gridXEnd - gridXStart + 1) * cellWidth; // Number of columns
        const cellRealHeight = (gridYEnd - gridYStart + 1) * cellHeight; // Number of rows

        // Check if drawable's position overlaps with the real position of the cell
        if (
            drawable.topLeftX + drawable.width > cellTopLeftX &&
            drawable.topLeftX < cellTopLeftX + cellRealWidth &&
            drawable.topLeftY + drawable.height > cellTopLeftY &&
            drawable.topLeftY < cellTopLeftY + cellRealHeight
        ) {
            outsideClamp(drawable, cellTopLeftX, cellTopLeftX + cellRealWidth, cellTopLeftY, cellTopLeftY + cellRealHeight)
        }
    });
}

function createAgent(clientCounter, drawables, cellSize, cellNum, evalMode){
    const clientId = clientCounter;
    clientCounter++;

    const name = '';
    let width = 35;
    let height = 35;
    const { x, y } = getRandomPosition(cellSize, cellNum)
    const topLeftX = x;
    const topLeftY = y;
    let color;
    if (clientId % 2 == 0){
        color = '#AA0000'
    } else {
        color = '#00AAAA'
    }
    let speed;
    if (evalMode) {
        speed = 15;
    } else {
        speed = 5;
    }
    let direction = null;
    let maxHealth = 2000;
    let currentHealth = maxHealth;
    let healthRegenRate = 0.5 / 100 * maxHealth;  // change only the first part of the variable
    let healthRegenDelay = 10 * 1000;  // change only the first part of the variable (in seconds)
    let bodyDamage = 20;
    let lastDamageTime = null;
    let lastDamagerId = null;
    const shape = 'oval';
    const type = 'agent';
    const parentId = null;
    let projectileVelocity = null;
    let projectileHealth = null;
    let projectileDamage = null;
    let shootingAngle = null;
    let isShooting = null;
    let lastProjectileTime = null;
    let rateOfFire = null;
    let level = 1;
    let score = null;
    let experience = 1000;
    let isToBeRespawned = null;


    let drawable = new Drawable(clientId, name, width, height, topLeftX, topLeftY, color, speed, direction, maxHealth, currentHealth, healthRegenRate, healthRegenDelay, bodyDamage, lastDamageTime, lastDamagerId, shape, type, parentId, projectileVelocity, projectileHealth, projectileDamage, shootingAngle, isShooting, lastProjectileTime, rateOfFire, level, score, experience, isToBeRespawned);
    drawables.push(drawable);

    return clientCounter
}


const getRandomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;


const generateRandomWalls = (numWalls) => {
    const walls = [];
    const maxExpansion = 5; // Maximum expansion for xEnd or yEnd

    for (let i = 0; i < numWalls; i++) {
        const xStart = getRandomInt(0, pathGridDimensions.cols - 1);
        const yStart = getRandomInt(0, pathGridDimensions.rows - 1);

        // Randomly decide whether to expand xEnd or yEnd
        if (Math.random() < 0.5) { // 50% chance to expand xEnd
            const xEnd = Math.min(xStart + getRandomInt(1, maxExpansion), pathGridDimensions.cols - 1);
            walls.push([xStart, yStart, xEnd, yStart]); // Keep yEnd the same
        } else { // Expand yEnd
            const yEnd = Math.min(yStart + getRandomInt(1, maxExpansion), pathGridDimensions.rows - 1);
            walls.push([xStart, yStart, xStart, yEnd]); // Keep xEnd the same
        }
    }

    return walls;
};

function sendDrawablesToPython(drawables, pythonProcess) {
    pythonProcess.stdin.write(JSON.stringify({
        type: 'drawables',
        data: drawables
    }) + '\n');  // Send data via stdin
}

function sendOneTimeDataToPython(gameBoundsDimensions, pathGridDimensions, unwalkableCellsExpanded, evalMode, pythonProcess) {
    // Convert the data to JSON and send it to Python
    pythonProcess.stdin.write(JSON.stringify({
        type: 'oneTimeData',
        data: {gameBoundsDimensions,pathGridDimensions, unwalkableCellsExpanded, evalMode}
    }) + '\n'); // Sending one-time data
}



export { disconnect, sendPlayerCount, findHighestScore, addListeners, respawnPlayer, turnLevelToExperience, updateLevel, checkForDeath, createFood, checkForRespawn, createProjectile, sendServerInfo, checkCollision, knockbackWithDmg, clamp, range, isInUnwalkableCell, createAgent, generateRandomWalls, getRandomInt, sendDrawablesToPython, sendOneTimeDataToPython, getRandomPosition }