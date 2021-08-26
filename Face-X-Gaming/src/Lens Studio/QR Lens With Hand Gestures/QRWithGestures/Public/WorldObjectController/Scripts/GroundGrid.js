// GroundGrid.js
// Version: 0.0.4
// Event: Lens Initialized
// Description: Controls the visibility of the ground grid underneath the character

//@input float fadeUpSpeed
//@input float fadeDownSpeed

// Bool used to determine if the user is touching the object
var isTouching = false;

// A reference to the Mesh Visual Component on the ground grid object
var groundMeshVisual = script.getSceneObject().getFirstComponent("Component.MeshVisual");

function onLensTurnOn()
{
    isTouching = false;
}

// Function that runs every frame
function onFrameUpdated(eventData)
{
    // Update Alpha for the ground grid material
    var curColor = groundMeshVisual.mainMaterial.mainPass.mainColor;

    // Determines if the user is touching/moving the object around to lerp the alpha for the grid up and down
    if(isTouching)
    {
        var lerpedColor = vec4.lerp( curColor, new vec4(1, 1, 1, 1), script.fadeUpSpeed );
        groundMeshVisual.mainMaterial.mainPass.mainColor = lerpedColor;
    }
    else
    {
        var lerpedColor = vec4.lerp( curColor, new vec4(0, 0, 0, 1), script.fadeDownSpeed );
        groundMeshVisual.mainMaterial.mainPass.mainColor = lerpedColor;
    }

    // This controls hiding the grid if the user is recording on their device within Snapchat
    if(global.scene.isRecording())
    {
        groundMeshVisual.enabled = false;
    }
    else
    {
        groundMeshVisual.enabled = true;
    }
}

// Setup for events and callbacks
function onTouchStarted(eventData)
{
    isTouching = true;
}

function onTouchEnded(eventData)
{
    isTouching = false;
}

if(groundMeshVisual)
{    
    var turnOnEvent = script.createEvent("TurnOnEvent");
    turnOnEvent.bind(onLensTurnOn);

    var updateEvent = script.createEvent("UpdateEvent");
    updateEvent.bind(onFrameUpdated);

    var touchStartEvent = script.createEvent("TouchStartEvent");
    touchStartEvent.bind(onTouchStarted);

    var touchEndEvent = script.createEvent("TouchEndEvent");
    touchEndEvent.bind(onTouchEnded);       
}