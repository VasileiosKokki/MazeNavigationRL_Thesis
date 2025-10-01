import express from 'express';
import bodyParser from 'body-parser'
import isReachable from 'is-reachable'
import { router, serverData } from './routes.js'

const app = express();
app.use(bodyParser.json());
app.use(router); // Use the routes

// Check server status at intervals
setInterval(() => {
    serverData.forEach(async (server) => {
        const status = await isReachable(`localhost:${server.port}`);
        server.status = status ? 'Online' : 'Offline';
        if (!status) server.playerCount = 0; // Reset player count if offline
    });
    console.log(serverData);
}, 2000);

// Start your server
app.listen(3000, () => {
    console.log("Server is running on http://localhost:3000");
});
