import express from 'express'
import path from 'path'
import rateLimit from 'express-rate-limit'
import { fileURLToPath } from 'url';

const router = express.Router(); // Create a new router instance

const secretKey = "fd867a587aa02407952a83e59675e99e4c8de5bb6640db609eb8e7fdfb358373";
let serverData = []; // Declare serverData here

const limiter = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 1000, // Number of requests allowed in the time window
    message: 'Too many requests from this IP, please try again later.',
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);



router.get("/serverdata", limiter, (req, res) => {
    res.send(serverData);
});

router.get("/redirect", limiter, (req, res, next) => {
    const serverSelect = req.query.serverSelect;
    const targetServer = serverData.find((server) => server.name === serverSelect);

    if (targetServer !== undefined) {
        if (targetServer.playerCount < targetServer.maxCount && targetServer.status === 'Online'){
            res.setHeader('Content-Type', 'text/html');
            res.sendFile(path.join(__dirname, '../public/client', 'client.html'));
        } else if (!(targetServer.playerCount < targetServer.maxCount)){
            res.status(503).send("Server is Full");
        } else if (!(targetServer.status === 'Online')){
            res.status(503).send("Server is Offline");
        }
    } else {
        next();
    }
});

router.post("/playercount", (req, res) => {
    const playerCount = req.body.playerCount;
    const senderURL = req.body.senderURL;
    const receivedKey = req.headers["authorization"];

    if (receivedKey === secretKey) {
        const foundServer = serverData.find((server) => {
            const url = "http://" + req.hostname + ":" + server.port;
            return url === senderURL;
        });

        if (foundServer) {
            foundServer.playerCount = playerCount;
            console.log(`Received player count ${playerCount} from server ${foundServer.name}`);
        } else {
            console.error("Received data from an unknown server");
        }
        res.send("Ok");
    } else {
        res.send("Not Authorised");
    }
});

router.post("/serverdata", (req, res) => {
    const receivedServer = req.body;
    const receivedKey = req.headers["authorization"];

    if (receivedKey === secretKey) {
        const existingServer = serverData.find((server) => server.port === receivedServer.port);
        if (!existingServer) {
            serverData.push(receivedServer);
            console.log("Server added:", receivedServer.name);
        }
        res.send("Ok");
    } else {
        res.send("Not Authorised");
    }
});

router.get("/favicon.ico", limiter, (req, res) => {
    res.sendFile(path.join(__dirname, '../public', 'favicon.ico'));
});

router.get("/worker.js", limiter, (req, res) => {
    res.sendFile(path.join(__dirname, '../public/worker', 'worker.js'));
});

router.get("/styles.css", limiter, (req, res) => {
    res.sendFile(path.join(__dirname, '../public/client', 'styles.css'));
});

router.get("/:folder/functions.js", limiter, (req, res) => {
    console.log(path.join(__dirname, '../public/', req.path));
    res.sendFile(path.join(__dirname, '../public/', req.path));
});



router.get("/*", limiter, (req, res) => {
    res.sendFile(path.join(__dirname, '../public/clienthub', 'clienthub.html'));
});


export { router, serverData }; // Export the router and serverData
