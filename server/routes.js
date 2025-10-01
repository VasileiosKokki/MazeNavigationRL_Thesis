import { sendPlayerCount, addListeners, getRandomPosition } from "./functions.js";
import {
    Drawable,
    getClientCounter,
    getClients,
    getCurrentHighestScore,
    getCurrentPlayers,
    getDrawables,
    setClientCounter,
    setCurrentPlayers,
    maxPlayers,
    port,
    secretKey,
    gameBoundsDimensions,
    cellSize,
    cellNum,
    spatialGridDimensions,
    pathGridDimensions,
    getUnwalkableCells
} from "./server.js";

import express from 'express'
const router = express.Router(); // Create a new router instance
import expressWs from 'express-ws'
expressWs(router)

import wsRateLimitModule from 'ws-rate-limit';
import rateLimit from "express-rate-limit";
import path from "path";
wsRateLimitModule('0.05s', 5);



router.ws('/connected', function(ws, req){

    const clients = getClients();
    const drawables = getDrawables();
    const currentHighestScore = getCurrentHighestScore();
    let currentPlayers = getCurrentPlayers();
    let clientCounter = getClientCounter();
    const unwalkableCells = getUnwalkableCells();



    rateLimit(ws);
    if (currentPlayers >= maxPlayers) {
        ws.close(1000, 'Maximum player count reached');
    } else {
        const parts = req.headers.host.split(':');
        const domain = parts[0];


        const clientId = clientCounter;
        clients.push({clientId:clientId,connection:ws});
        clientCounter++;
        currentPlayers++;
        sendPlayerCount(currentPlayers, domain, port, secretKey);


        const name = req.url.split('?')[1];
        let width = 35;
        let height = 35;
        const { x, y } = getRandomPosition(cellSize, cellNum)
        const topLeftX = x;
        const topLeftY = y;
        const color = '#' + (Math.random() * 0xFFFFFF << 0).toString(16).padStart(6, '0');
        let speed = 10;
        let direction = null;
        let maxHealth = 3000;
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
        let projectileHealth = 40;
        let projectileDamage = 50;
        let shootingAngle = null;
        let isShooting = false;
        let lastProjectileTime = null;
        let rateOfFire = 0.1 * 1000;  // change only the first part of the variable (in seconds)
        let level = 1;
        let score = 0;
        let experience = 0;
        let isToBeRespawned = false;


        let drawable = new Drawable(clientId, name, width, height, topLeftX, topLeftY, color, speed, direction, maxHealth, currentHealth, healthRegenRate, healthRegenDelay, bodyDamage, lastDamageTime, lastDamagerId, shape, type, parentId, projectileVelocity, projectileHealth, projectileDamage, shootingAngle, isShooting, lastProjectileTime, rateOfFire, level, score, experience, isToBeRespawned);
        drawables.push(drawable);


        addListeners(drawable, ws, domain, port, secretKey)


        const message = JSON.stringify({type:'connected',data:{clientId:drawable.clientId,gameBoundsDimensions,spatialGridDimensions,pathGridDimensions,unwalkableCells}});
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

    setCurrentPlayers(currentPlayers)
    setClientCounter(clientCounter)


});

router.get("/styles.css", (req, res) => {
    res.sendFile(path.join(__dirname, '../public/client', 'styles.css'));
});


router.get('/*', (req, res) => {
    res.redirect(`http://${req.hostname}:3000`);
});


export { router };