function handleWebSocketErrorOrClose(event) {
    if (event != undefined){
        if (event.type === 'error') {
            console.error('WebSocket Error:', event);
        } else if (event.type === 'close') {
            console.error('WebSocket Connection Is Closed');
        }
    } else {
        console.error('Try Going Through ServerHub This Time')
    }
    document.getElementById('connection-element-2').style.opacity = '1';
}


function getBaseDivisorAtLevel(level) {
    let baseDivisor = 1000; // Initial base divisor
    for (let i = 1; i < level; i++) {
        baseDivisor = Math.round(baseDivisor * 1.1); // Increase the divisor (adjust this factor)
    }
    return baseDivisor;
}



function updateCanvasSize(scaleFactor, worker) {
    const canvasWidth = window.innerWidth / scaleFactor;
    const canvasHeight = window.innerHeight / scaleFactor;

    const sizeData = {canvasWidth,canvasHeight}
    worker.postMessage({ type:'resize', data:JSON.stringify(sizeData) });
}


function onMouseMove(event, lastMouseMoveTime, angle, throttleInterval, websocket, canvas, scaleFactor) {
    const currentTime = Date.now();
    if (currentTime - lastMouseMoveTime >= throttleInterval) {
        if ((websocket != undefined) && (websocket.readyState == 1)){   // need to limit to 40ms
            lastMouseMoveTime = currentTime;
            const xaxis = event.clientX - canvas.width * scaleFactor / 2;
            const yaxis = event.clientY - canvas.height * scaleFactor / 2;
            angle = Math.atan2(yaxis, xaxis);
            const message = {type:'userProjectile',shoot:'direction',data:angle};
            websocket.send(JSON.stringify(message));
        }
    }

    return { lastMouseMoveTime, angle }
}




// zoom in out
function handleMouseWheel(event, worker, zoomLevel) {
    const delta = Math.sign(event.deltaY);
    let zoomStep = 0.05;
    if (delta === -1) {
        zoomLevel += zoomStep;
        if (zoomLevel >= 1.55) {
            zoomLevel = 1.5;
            return zoomLevel;
        }
    } else if (delta === 1) {
        zoomLevel -= zoomStep;
        if (zoomLevel <= 0.7) {
            zoomLevel = 0.75;
            return zoomLevel;
        }
    }
    const zoomLevelData = {zoomLevel,zoomStep,delta}
    worker.postMessage({ type:'zoomlevel', data:JSON.stringify(zoomLevelData) });

    return zoomLevel
}



// swap tabs while key is pressed
function handleVisibilityChange(websocket, arrowUpPressed, arrowDownPressed, arrowRightPressed, arrowLeftPressed, isShooting, angle) {
    if ((websocket != undefined) && (websocket.readyState == 1)){
        if ((arrowUpPressed || arrowDownPressed || arrowRightPressed || arrowLeftPressed) || isShooting) {

            arrowUpPressed = false;
            arrowDownPressed = false;
            arrowRightPressed = false;
            arrowLeftPressed = false;
            isShooting = false;
            const message = {type:'userAction',data:'ArrowNotClicked'};
            websocket.send(JSON.stringify(message));
            const message2 = {type:'userProjectile',shoot:'false',data:angle};
            websocket.send(JSON.stringify(message2));


        }
    }

    return { arrowUpPressed, arrowDownPressed, arrowRightPressed, arrowLeftPressed, isShooting }
}


// common for both keydown and keyup event listener
function handleKeyAction(websocket, arrowUpPressed, arrowDownPressed, arrowRightPressed, arrowLeftPressed) {

    // 3rd level - 3 buttons
    if (arrowUpPressed && arrowRightPressed && arrowLeftPressed) {
        const message = {type: 'userAction', data: 'ArrowUp'};
        websocket.send(JSON.stringify(message));
    } else if (arrowDownPressed && arrowRightPressed && arrowLeftPressed)  {
        const message = {type: 'userAction', data: 'ArrowDown'};
        websocket.send(JSON.stringify(message));
    } else if (arrowRightPressed && arrowUpPressed && arrowDownPressed) {
        const message = {type: 'userAction', data: 'ArrowRight'};
        websocket.send(JSON.stringify(message));
    } else if (arrowLeftPressed && arrowUpPressed && arrowDownPressed) {
        const message = {type: 'userAction', data: 'ArrowLeft'};
        websocket.send(JSON.stringify(message));
    } else



    // 2nd level - 2 buttons
    if ((arrowUpPressed && arrowDownPressed) || (arrowRightPressed && arrowLeftPressed)) {
        const message = {type:'userAction',data:'ArrowNotClicked'};
        websocket.send(JSON.stringify(message));
    } else  if (arrowUpPressed && arrowRightPressed){
        const message = {type:'userAction',data:'ArrowUp+Right'};
        websocket.send(JSON.stringify(message));
    } else if (arrowUpPressed && arrowLeftPressed){
        const message = {type:'userAction',data:'ArrowUp+Left'};
        websocket.send(JSON.stringify(message));
    } else if (arrowDownPressed && arrowRightPressed){
        const message = {type:'userAction',data:'ArrowDown+Right'};
        websocket.send(JSON.stringify(message));
    } else if (arrowDownPressed && arrowLeftPressed){
        const message = {type:'userAction',data:'ArrowDown+Left'};
        websocket.send(JSON.stringify(message));
    } else



    // 1st level - 1 button
    if (arrowUpPressed) {
        const message = {type: 'userAction', data: 'ArrowUp'};
        websocket.send(JSON.stringify(message));
    } else if (arrowDownPressed)  {
        const message = {type: 'userAction', data: 'ArrowDown'};
        websocket.send(JSON.stringify(message));
    } else if (arrowRightPressed) {
        const message = {type: 'userAction', data: 'ArrowRight'};
        websocket.send(JSON.stringify(message));
    } else if (arrowLeftPressed) {
        const message = {type: 'userAction', data: 'ArrowLeft'};
        websocket.send(JSON.stringify(message));
    }

}

export { handleKeyAction, handleWebSocketErrorOrClose, handleVisibilityChange, handleMouseWheel, updateCanvasSize, getBaseDivisorAtLevel, onMouseMove }