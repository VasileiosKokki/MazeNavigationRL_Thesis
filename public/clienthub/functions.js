async function fetchServerData(serverData){
    const url = `/serverdata`;
    const result = await fetch(url,{
        method:'GET',
        headers: {
            "Content-Type": "application/json"
        }
    });

    try {
        const receivedText = await result.text();
        if (result.ok) {
            serverData = JSON.parse(receivedText);
        } else {
            throw new Error(receivedText);
        }
    } catch (error) {
        alert(error.message);
    }

    return serverData
}


function updateServerOptions(serverData) {
    const serverSelectMenu = document.getElementById('serverSelect');
    let selectedServer = serverSelectMenu.value;
    // Clear existing options
    serverSelectMenu.innerHTML = '';

    // Iterate through serverData and add options
    serverData.forEach(function(server){
        const option = document.createElement('option');
        option.value = server.name;
        option.textContent = `Server ${server.name.charAt(server.name.length - 1)}`;
        serverSelectMenu.appendChild(option);

        if (option.value === selectedServer) {
            option.selected = true;
        }
    });
}


async function executeWithOrder(serverData){
    serverData = await fetchServerData(serverData);
    updateServerOptions(serverData);

    return serverData
}

export { fetchServerData, updateServerOptions, executeWithOrder };