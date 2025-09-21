import {
    changeZoom,
    drawCannon,
    drawHealthPart1,
    drawHealthPart2,
    drawName,
    mainDraw,
    makeColorLighter, recenterCamera, updateCamera
} from "../worker/functions.js";

let canvas;
let ctx;
let zoomLevel = 1;
let cameraReached = false;
let cameraSpeed = 1;
let cameraX;
let cameraY;

let properId;
let gameBoundsDimensions;
let spatialGridDimensions;
let pathGridDimensions;
let unwalkableCells;
let zoomStep;
let delta;
let targetCameraX;
let targetCameraY;




self.addEventListener('message', (event) => {
    const message = event.data;
    switch(message.type){
        case 'canvas':
            canvas = message.data;
            ctx = canvas.getContext('2d', {alpha: false});
            break;
        case 'model':
            const drawables = JSON.parse(message.data);
            Presentation(drawables);
            break;
        case 'connected':
            const connectedData = JSON.parse(message.data);
            properId = connectedData.properId
            gameBoundsDimensions = connectedData.gameBoundsDimensions
            spatialGridDimensions = connectedData.spatialGridDimensions
            pathGridDimensions = connectedData.pathGridDimensions
            unwalkableCells = connectedData.unwalkableCells
            cameraX = -gameBoundsDimensions.width/2;
            cameraY = -gameBoundsDimensions.height/2;
            break;
        case 'zoomlevel':
            const zoomLevelData = JSON.parse(message.data);
            zoomLevel = zoomLevelData.zoomLevel;
            zoomStep = zoomLevelData.zoomStep;
            delta = zoomLevelData.delta;
            ({cameraX, cameraY} = changeZoom(zoomLevel, zoomStep, delta, canvas, cameraX, cameraY))
            break;
        case 'resize':
            const sizeData = JSON.parse(message.data);
            canvas.width = sizeData.canvasWidth;
            canvas.height = sizeData.canvasHeight;
            ctx.fillStyle = "rgb(20, 20, 20)";
            break;
        default:
            break;
    }
});






function Presentation(drawables){

    const drawablesPlayers = [];
    const drawablesPlayersFood = [];
    const drawablesProjectiles = [];
    const unwalkableCellWidth = gameBoundsDimensions.width / pathGridDimensions.cols * zoomLevel
    const unwalkableCellHeight = gameBoundsDimensions.height / pathGridDimensions.rows * zoomLevel

    


    drawables.filter(drawable => {
        if (drawable.clientId == properId){  // check if it matches the player we control
            ({targetCameraX, targetCameraY} = recenterCamera(drawable, zoomLevel, canvas));
            ({cameraX, cameraY, cameraReached, cameraSpeed} = updateCamera(targetCameraX, targetCameraY, cameraX, cameraY, cameraReached, cameraSpeed));
        }
        if (drawable.type === 'player' || drawable.type === 'food' || drawable.type === 'agent'){
            drawablesPlayersFood.push(drawable);
            if (drawable.type === 'player' || drawable.type === 'agent') {
                drawablesPlayers.push(drawable);
            }
        } else if (drawable.type === 'projectile'){
            drawablesProjectiles.push(drawable);
        }
    });


    // 0th level - dark background
    ctx.fillRect(0, 0, canvas.width, canvas.height);	
    ctx.save();
    ctx.translate(cameraX, cameraY);
   
    

    // 1st level - green and grey background elements
    let collumnX = gameBoundsDimensions.width * zoomLevel / spatialGridDimensions.cols // width
    let collumnY = gameBoundsDimensions.height * zoomLevel
    let rowsX = gameBoundsDimensions.width * zoomLevel
    let rowsY = gameBoundsDimensions.height * zoomLevel / spatialGridDimensions.rows // height
    let x = 0;
    let y = 0;
    ctx.strokeStyle = "rgba(128, 128, 128, 0.1)";
    
    
    for (let i = 0; i < spatialGridDimensions.cols; i++){
        ctx.strokeRect(x,y,collumnX,collumnY);
        x += collumnX;
    }
    
    x = 0;
    for (let j = 0; j < spatialGridDimensions.rows; j++){
        ctx.strokeRect(x,y,rowsX,rowsY);
        y += rowsY;
    }						
    
    ctx.strokeStyle = "green";
    ctx.strokeRect(0,0,gameBoundsDimensions.width * zoomLevel,gameBoundsDimensions.height * zoomLevel);



    // 2nd level - walls
    ctx.fillStyle = "red";

    unwalkableCells.forEach(([gridXStart, gridYStart, gridXEnd, gridYEnd]) => {
        // Calculate the top-left corner of the entire rectangle
        const cellTopLeftX = gridXStart * unwalkableCellWidth;
        const cellTopLeftY = gridYStart * unwalkableCellHeight;

        // Calculate the width and height of the entire rectangle
        const rectangleWidth = (gridXEnd - gridXStart + 1) * unwalkableCellWidth;
        const rectangleHeight = (gridYEnd - gridYStart + 1) * unwalkableCellHeight;

        // Draw the rectangle covering all the cells in the range
        ctx.fillRect(cellTopLeftX, cellTopLeftY, rectangleWidth, rectangleHeight);
    });


    // 3rd level - projectiles
    let lastFillStyle;
    let lastStrokeStyle;
    drawablesProjectiles.forEach(function(drawable){
        

            if (drawable.color != lastFillStyle){
                ctx.fillStyle = drawable.color;
                lastFillStyle = drawable.color;
            }
            
        
            let slightlyLighterColor = makeColorLighter(drawable.color);
            if (slightlyLighterColor != lastStrokeStyle){
                ctx.strokeStyle = slightlyLighterColor;
                lastStrokeStyle = slightlyLighterColor;
            }
            
            
            ctx.beginPath();
            mainDraw(drawable, ctx, zoomLevel);
             				
    });


    
    // 4th level - cannon for players
    ctx.fillStyle = '#7B6C65';
    ctx.strokeStyle = '#B4BFC3';
    lastFillStyle = '#7B6C65';
    lastStrokeStyle = '#B4BFC3';
    drawablesPlayers.forEach(function(drawable){
            
            
            const centerX = (drawable.topLeftX + drawable.width / 2) * zoomLevel;
            const centerY = (drawable.topLeftY + drawable.height / 2) * zoomLevel;
            ctx.save();

            ctx.translate(centerX, centerY);
            ctx.rotate(drawable.shootingAngle);
            ctx.translate(-centerX, -centerY);
            ctx.beginPath();
            drawCannon(drawable, ctx, zoomLevel);

            ctx.restore();
            
            
        
    });
    
    
     
    // 5th level - players and food
    drawablesPlayersFood.forEach(function(drawable){

        
        
        
        
        if (drawable.color != lastFillStyle){
            ctx.fillStyle = drawable.color;
            lastFillStyle = drawable.color;
        }
        
       
        let slightlyLighterColor = makeColorLighter(drawable.color);
        if (slightlyLighterColor != lastStrokeStyle){
            ctx.strokeStyle = slightlyLighterColor;
            lastStrokeStyle = slightlyLighterColor;
        }

    
        
        ctx.beginPath();
        mainDraw(drawable, ctx, zoomLevel);




           				
    });


 
    // 6th level - health bar for players and food
    ctx.fillStyle = 'gray';
    ctx.lineWidth = 1;
    drawablesPlayersFood.forEach(function(drawable){
        if (drawable.currentHealth != drawable.maxHealth){
            drawHealthPart1(drawable, ctx, zoomLevel);
        }
    });
    ctx.fillStyle = 'green';
    drawablesPlayersFood.forEach(function(drawable){
        if (drawable.currentHealth != drawable.maxHealth){
            drawHealthPart2(drawable, ctx);
        }
    });


    
    // 7th level - name for players
    ctx.fillStyle = 'white'; // Set the text color
    drawablesPlayers.forEach(function(drawable){
        drawName(drawable, ctx, zoomLevel);
    });
    
    


    ctx.restore();
}
