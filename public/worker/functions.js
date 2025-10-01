function mainDraw(drawable, ctx, zoomLevel){
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

        // Draw an oval shape

        const centerX = x + width / 2;
        const centerY = y + height / 2;
        const radiusX = width / 2 - lineWidth / 2;
        const radiusY = height / 2 - lineWidth / 2;

        ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);


    }
    ctx.fill();
    ctx.stroke();
}


function drawHealthPart1(drawable, ctx, zoomLevel){
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


function drawHealthPart2(drawable, ctx){
    const healthPercentage = drawable.currentHealth / drawable.maxHealth;
    ctx.fillRect(drawable.healthBarX, drawable.healthBarY, drawable.healthBarWidth * healthPercentage, drawable.healthBarHeight);
}


function drawCannon(drawable, ctx, zoomLevel){
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

    ctx.fill();

    ctx.stroke();
}



function drawName(drawable, ctx, zoomLevel) {
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
    const lighterColorValue = `#${darkerRed.toString(16).padStart(2, '0')}${darkerGreen.toString(16).padStart(2, '0')}${darkerBlue.toString(16).padStart(2, '0')}`;

    return lighterColorValue;
}



function recenterCamera(drawable, zoomLevel, canvas){
    const x = drawable.topLeftX * zoomLevel;
    const y = drawable.topLeftY * zoomLevel;
    const width = drawable.width * zoomLevel;
    const height = drawable.height * zoomLevel;

    const centerX = x + width / 2;
    const centerY = y + height / 2;

    const canvasCenterX = canvas.width / 2;
    const canvasCenterY = canvas.height / 2;

    const targetCameraX = canvasCenterX - centerX;
    const targetCameraY = canvasCenterY - centerY;

    return {targetCameraX, targetCameraY}
}


function updateCamera(targetCameraX, targetCameraY, cameraX, cameraY, cameraReached, cameraSpeed) {


    const dx = targetCameraX - cameraX;
    const dy = targetCameraY - cameraY;


    if (Math.abs(dx) < 100 && Math.abs(dy) < 100) {
        cameraReached = true;
    }

    if (cameraReached == false){
        cameraSpeed += 0.01;
    } else {
        if (cameraSpeed < 10){
            cameraSpeed += 1;
        }
        if (cameraSpeed > 10){
            cameraSpeed = 10;
        }
    }



    cameraX += dx * cameraSpeed / 10;
    cameraY += dy * cameraSpeed / 10;

    return {cameraX, cameraY, cameraReached, cameraSpeed}
}



function changeZoom(zoomLevel, zoomStep, delta, canvas, cameraX, cameraY){
    const centerX = canvas.width / 2; // Center of the canvas
    const centerY = canvas.height / 2;
    if (delta === -1) {
        cameraX = centerX - (centerX - cameraX) * (zoomLevel / (zoomLevel - zoomStep));
        cameraY = centerY - (centerY - cameraY) * (zoomLevel / (zoomLevel - zoomStep));
    } else if (delta === 1) {
        cameraX = centerX - (centerX - cameraX) * (zoomLevel / (zoomLevel + zoomStep));
        cameraY = centerY - (centerY - cameraY) * (zoomLevel / (zoomLevel + zoomStep));
    }
    return {cameraX, cameraY}
}



export { mainDraw, drawHealthPart1, drawHealthPart2, drawCannon, drawName, makeColorLighter, recenterCamera, updateCamera, changeZoom }