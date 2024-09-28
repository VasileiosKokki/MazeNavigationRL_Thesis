const express = require('express');
const bodyParser = require('body-parser');
const app = express();
app.use(bodyParser.json());
const path = require('path');
const isReachable = require('is-reachable');
const rateLimit = require('express-rate-limit');





const secretKey = "fd867a587aa02407952a83e59675e99e4c8de5bb6640db609eb8e7fdfb358373";
var serverData = [];


const limiter = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 1000, // Number of requests allowed in the time window
    message: 'Too many requests from this IP, please try again later.',
});





setInterval(() => {   
    serverData.forEach(async (server) => {
        const status = (await isReachable(`localhost:${server.port}`));
        if (status){
            server.status = 'Online';
        } else {
            server.status = 'Offline'
            server.playerCount = 0;
        }
    });
    console.log(serverData);
}, 2000);







// ----------------------------------------- Start of Route Handlers -----------------------------------------




app.get("/serverdata", limiter, (req, res, next) => {
    res.send(serverData);
});





app.get("/redirect", limiter,  (req, res, next) => {
    const serverSelect = req.query.serverSelect;
    const targetServer = serverData.find((server) => server.name === serverSelect);
    // Check if the selected server exists in the mapping
    if (targetServer != undefined) { 
        if (targetServer.playerCount < targetServer.maxCount && targetServer.status == 'Online'){
            // Remove the found token from the array
            
            res.setHeader('Content-Type', 'text/html');
        
            // Send the merged content as an HTML response
            res.sendFile(path.join(__dirname, 'public', 'client.html'));
        } else if (!(targetServer.playerCount < targetServer.maxCount)){
            res.status(503).send("Server is Full");
        } else if (!(targetServer.status == 'Online')){
            res.status(503).send("Server is Offline");
        }
    } else {
        next();
    }
});







app.post("/playercount", (req, res) => {
    const playerCount = req.body.playerCount;
    const senderURL = req.body.senderURL;
    const receivedKey = req.headers["authorization"];

    if (receivedKey == secretKey){

        // Iterate through the serverData object to find a match based on the senderURL
        const foundServer = serverData.find((server) => {
            const url = "http://" + req.hostname + ":" + server.port;
            return url === senderURL;
        });

        if (foundServer) {
            // The senderURL corresponds to a server in serverData
            // You can update the player count for the found server here
            foundServer.playerCount = playerCount;
            console.log(`Received player count ${playerCount} from server ${foundServer.name}`);
        } else {
            // Handle the case where the senderURL does not match any known server
            console.error("Received data from an unknown server");
        }
        res.send("Ok");
    } else {
        res.send("Not Authorised");
    }
});




app.post("/serverdata", (req, res) => {
    const receivedServer = req.body;
    const receivedKey = req.headers["authorization"];

    if (receivedKey == secretKey){
        const existingServer = serverData.find((server) => server.port === receivedServer.port);
        if (!existingServer) {
            serverData.push(receivedServer);
            console.log("oof");
        }
        res.send("Ok");
    } else {
        res.send("Not Authorised");
    }
});





app.get("/favicon.ico", limiter, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'favicon.ico'));
});




app.get("/worker.js", limiter, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'worker.js'));
});




app.get("/*", limiter, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'clienthub.html'));
});



// ----------------------------------------- End of Route Handlers -----------------------------------------






app.listen(3000);