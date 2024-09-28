playersOnly.forEach(function(drawable){
    if (drawable.clientId == client.clientId && client.connection.readyState == 1){
        
        const message = JSON.stringify({type:'connected',data:{clientId:drawable.clientId,bounds:canvas,grid:{cols,rows}}});
        const compressedData = zlib.deflateSync(message);
        client.connection.send(compressedData);
         
        
    }
});





deadButConnected = deadButConnected.filter(function(drawable) {
    if (drawable.clientId == client.clientId && respawnFlag == true) {
      client.connection.removeAllListeners();
      respawnPlayer(drawable, client.connection);
      return false; // Don't keep this drawable in the new array
    }
    return true; // Keep other drawables in the new array
});






    const message = JSON.stringify({type:'unique',data:{clientId:drawable.clientId,bounds:canvas,grid:{cols,rows},level:drawable.level,experience:drawable.experience}});
	const nonCompressedData = message;

	const flag = "0"; // flag for noncompressed data
	const dataWithFlag = Buffer.concat([Buffer.from(nonCompressedData), Buffer.from(flag)]);



	clients.forEach(function(client){
		 if (client.connection.readyState == 1 && client.clientId == drawable.clientId){
			 client.connection.send(dataWithFlag);
		 }
	});


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