var canvas;
var ctx;
let zoomLevel = 1;
var cameraReached = false;
var cameraSpeed;
var cameraX;
var cameraY;




self.addEventListener('message', (event) => {
    const message = event.data;
    switch(message.type){
        case 'canvas':
            canvas = message.data;
            ctx = canvas.getContext('2d', {alpha: false});
            break;
        case 'model':
            drawables = JSON.parse(message.data);
            Presentation(drawables);
            break;
        case 'connected':
            connectedData = JSON.parse(message.data);
            properId = connectedData.properId
            gameBoundsWidth = connectedData.gameBoundsWidth
            gameBoundsHeight = connectedData.gameBoundsHeight
            cameraX = -gameBoundsWidth/2;
            cameraY = -gameBoundsHeight/2;
            collumns = connectedData.collumns
            rows = connectedData.rows
            break;
        case 'zoomlevel':
            zoomLevelData = JSON.parse(message.data);
            zoomLevel = zoomLevelData.zoomLevel;
            zoomStep = zoomLevelData.zoomStep;
            delta = zoomLevelData.delta;
            changeZoom(zoomLevel,zoomStep,delta);
            break;
        case 'resize':
            sizeData = JSON.parse(message.data);
            canvas.width = sizeData.canvasWidth;
            canvas.height = sizeData.canvasHeight;
            ctx.fillStyle = "rgb(20, 20, 20)";
            break;
        default:
            break;
    }
});






function Presentation(drawables){

    drawablesPlayers = [];
    drawablesPlayersFood = [];
    drawablesProjectiles = [];
    
    


    drawables.filter(drawable => {
        if (drawable.clientId == properId){  // gia ton paikth pou elegxoume
            recenterCamera(drawable);
            updateCamera();         
        }
        if (drawable.type === 'player' || drawable.type === 'food'){
            drawablesPlayersFood.push(drawable);
            if (drawable.type === 'player') {
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
    var collumnX = gameBoundsWidth * zoomLevel / collumns // width
    var collumnY = gameBoundsHeight * zoomLevel
    var rowsX = gameBoundsWidth * zoomLevel
    var rowsY = gameBoundsHeight * zoomLevel / rows // height
    var x = 0;
    var y = 0;
    ctx.strokeStyle = "rgba(128, 128, 128, 0.1)";
    
    
    for (i = 0; i < collumns; i++){
        ctx.strokeRect(x,y,collumnX,collumnY);
        x += collumnX;
    }
    
    x = 0;
    for (j = 0; j < rows; j++){
        ctx.strokeRect(x,y,rowsX,rowsY);
        y += rowsY;
    }						
    
    ctx.strokeStyle = "green";
    ctx.strokeRect(0,0,gameBoundsWidth * zoomLevel,gameBoundsHeight * zoomLevel);



    // 2nd level - projectiles
    var lastFillStyle;
    var lastStrokeStyle;
    drawablesProjectiles.forEach(function(drawable){
        

            if (drawable.color != lastFillStyle){
                ctx.fillStyle = drawable.color;
                lastFillStyle = drawable.color;
            }
            
        
            var slightlyLighterColor = makeColorLighter(drawable.color);
            if (slightlyLighterColor != lastStrokeStyle){
                ctx.strokeStyle = slightlyLighterColor;
                lastStrokeStyle = slightlyLighterColor;
            }
            
            
            ctx.beginPath();
            mainDraw(drawable);
             				
    });


    
    // 3rd level - cannon for players
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
            //console.log(drawable.shootingAngle);
            ctx.beginPath();
            drawCannon(drawable);

            ctx.restore();
            
            
        
    });
    
    
     
    // 4th level - players and food
    drawablesPlayersFood.forEach(function(drawable){

        
        
        
        
        if (drawable.color != lastFillStyle){
            ctx.fillStyle = drawable.color;
            lastFillStyle = drawable.color;
        }
        
       
        var slightlyLighterColor = makeColorLighter(drawable.color);
        if (slightlyLighterColor != lastStrokeStyle){
            ctx.strokeStyle = slightlyLighterColor;
            lastStrokeStyle = slightlyLighterColor;
        }

    
        
        ctx.beginPath();
        mainDraw(drawable); 




           				
    });


 
    // 5th level - health bar for players and food
    ctx.fillStyle = 'gray';
    ctx.lineWidth = 1;
    drawablesPlayersFood.forEach(function(drawable){
        if (drawable.currentHealth != drawable.maxHealth){
            drawHealthPart1(drawable);
        }
    });
    ctx.fillStyle = 'green';
    drawablesPlayersFood.forEach(function(drawable){
        if (drawable.currentHealth != drawable.maxHealth){
            drawHealthPart2(drawable);
        }
    });


    
    // 6th level - name for players
    ctx.fillStyle = 'white'; // Set the text color
    drawablesPlayers.forEach(function(drawable){
        drawName(drawable);
    });
    
    


    ctx.restore();
}





function mainDraw(drawable){
    //const id = drawable.clientId;
    const x = drawable.topLeftX * zoomLevel;
    const y = drawable.topLeftY * zoomLevel;
    const width = drawable.width * zoomLevel;
    const height = drawable.height * zoomLevel;									

    const lineWidth = (Math.min(width, height) * 0.05);
    ctx.lineWidth = lineWidth;    

    if (drawable.shape == 'rectangle'){  

        const innerX = x + lineWidth / 2;
        const innerY = y + lineWidth / 2;
        const innerWidth = width - lineWidth;
        const innerHeight = height - lineWidth;

        ctx.rect(innerX, innerY, innerWidth, innerHeight);  

    } else {   //  anything other than rectangle causes more lag

        // Draw a triangle shape <- sadly deprecated because collision was impossible to do
        /*
        ctx.beginPath();
          ctx.moveTo(x, y+height);
          ctx.lineTo(x+width, y+height);
          ctx.lineTo(x+width/2, y);
          ctx.closePath();
          ctx.fill();
        */

        // Draw an oval shape
                        
        const centerX = x + width / 2;
        const centerY = y + height / 2;
        const radiusX = width / 2 - lineWidth / 2;
        const radiusY = height / 2 - lineWidth / 2;

        ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);
                                              

    }
    //if (drawable.type != 'player'){
        ctx.fill();
    //}
    //if (drawable.type != 'player'){
        ctx.stroke();
    //}						
}


function drawHealthPart1(drawable){
    const x = (drawable.topLeftX - drawable.width/2) * zoomLevel;
    const y = (drawable.topLeftY + drawable.height*1.1) * zoomLevel;
    const width = 2 * drawable.width * zoomLevel;
    const height = 5 * zoomLevel; // Height for displaying health bar




    drawable.healthBarX = x;
    drawable.healthBarY = y;
    drawable.healthBarWidth = width;
    drawable.healthBarHeight = height;
    ctx.fillRect(drawable.healthBarX, drawable.healthBarY, drawable.healthBarWidth, drawable.healthBarHeight);
}


function drawHealthPart2(drawable){
    const healthPercentage = drawable.currentHealth / drawable.maxHealth; 
    ctx.fillRect(drawable.healthBarX, drawable.healthBarY, drawable.healthBarWidth * healthPercentage, drawable.healthBarHeight);
}


function drawCannon(drawable){
    const x = (drawable.topLeftX + drawable.width * 9/10) * zoomLevel;
    const y = (drawable.topLeftY + drawable.height * 3/9) * zoomLevel;
    const width = drawable.width * zoomLevel * 2/5;
    const height = drawable.height * zoomLevel / 3;									

    const lineWidth = (Math.min(width*5/2, height*3) * 0.05);
    ctx.lineWidth = lineWidth;    

  

    const innerX = x + lineWidth / 2;
    const innerY = y + lineWidth / 2;
    const innerWidth = width - lineWidth;
    const innerHeight = height - lineWidth;

    ctx.rect(innerX, innerY, innerWidth, innerHeight);  

    
    //ctx.rotate(30 * Math.PI/360);
    //if (drawable.type != 'player'){
        ctx.fill();
    //}
    //if (drawable.type != 'player'){
        ctx.stroke();
    //}						
}


function drawName(drawable) {
    const fontSize = (drawable.width / 4) * zoomLevel + 5;
    ctx.font = fontSize + 'px Arial';
    const textWidth = ctx.measureText(drawable.name).width
    

    const x = drawable.topLeftX * zoomLevel + (drawable.width/2) * zoomLevel - (textWidth/2);
    const y = (drawable.topLeftY - drawable.height/10) * zoomLevel; // Adjust the vertical position
    

    // Draw the name above the drawable
    ctx.fillText(drawable.name, x, y);
    
    

}





function makeColorLighter(colorValue){
    // Remove the # symbol if it's present
    colorValue = colorValue.replace('#', '');

    // Convert the colorValue string to separate color components
    const redComponent = parseInt(colorValue.substr(0, 2), 16);
    const greenComponent = parseInt(colorValue.substr(2, 2), 16);
    const blueComponent = parseInt(colorValue.substr(4, 2), 16);

    // Subtract the amount from each color component
    const factor = 2; // Adjust this value to control the darkness
    const darkerRed = Math.min(255, Math.round(redComponent * factor));
    const darkerGreen = Math.min(255, Math.round(greenComponent * factor));
    const darkerBlue = Math.min(255, Math.round(blueComponent * factor));

    // Convert the adjusted color components back to hexadecimal and pad with zeros
    // Remove Here - Don't touch
    const lighterColorValue = `#${darkerRed.toString(16).padStart(2, '0')}${darkerGreen.toString(16).padStart(2, '0')}${darkerBlue.toString(16).padStart(2, '0')}`;

    return lighterColorValue;
}


// recenter function, xoris auto olo to camera logic prepei na ginei xeirokinhta, eg cameraY += 4.3 otan paththei to up arrow
function recenterCamera(drawable){
    const x = drawable.topLeftX * zoomLevel;
    const y = drawable.topLeftY * zoomLevel;
    const width = drawable.width * zoomLevel;
    const height = drawable.height * zoomLevel;

    const centerX = x + width / 2;
    const centerY = y + height / 2;

    const canvasCenterX = canvas.width / 2;
    const canvasCenterY = canvas.height / 2;

    targetCameraX = canvasCenterX - centerX;
    targetCameraY = canvasCenterY - centerY;
}


function updateCamera() {

    
    const dx = targetCameraX - cameraX;
    const dy = targetCameraY - cameraY;

    
    if (Math.abs(dx) < 100 && Math.abs(dy) < 100) {
        cameraReached = true;
    }

    if (cameraReached == false){
        cameraSpeed = 1;
    } else {
        if (cameraSpeed < 10){
            cameraSpeed += 1;
            console.log(cameraSpeed); 
        }
    }



    cameraX += dx * cameraSpeed / 10;
    cameraY += dy * cameraSpeed / 10;
}
    


function changeZoom(zoomLevel,zoomStep,delta){
    const centerX = canvas.width / 2; // Center of the canvas
    const centerY = canvas.height / 2;
    if (delta === -1) {			
   		cameraX = centerX - (centerX - cameraX) * (zoomLevel / (zoomLevel - zoomStep));
    	cameraY = centerY - (centerY - cameraY) * (zoomLevel / (zoomLevel - zoomStep));			
    } else if (delta === 1) {    		
   		cameraX = centerX - (centerX - cameraX) * (zoomLevel / (zoomLevel + zoomStep));
    	cameraY = centerY - (centerY - cameraY) * (zoomLevel / (zoomLevel + zoomStep));
	}
}