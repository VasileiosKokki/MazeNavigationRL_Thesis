import express from 'express';
import isReachable from 'is-reachable';
import path from 'path';
import { fileURLToPath } from 'url';
import { router, serverData } from './routes.js';

const app = express();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.json());

// Serve everything under public
app.use(express.static(path.join(__dirname, '../public')));

// API and custom routes after static files
app.use(router);

// Check server status at intervals
setInterval(async () => {
    await Promise.all(
        serverData.map(async (server) => {
            const status = await isReachable(`localhost:${server.port}`);
            server.status = status ? 'Online' : 'Offline';
            if (!status) {
                server.playerCount = 0;
            }
        })
    );
}, 2000);

// Start your server
app.listen(3000, () => {
    console.log('Server hub is running on http://localhost:3000');
});
