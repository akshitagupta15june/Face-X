// -----JS CODE-----
// @input int timeoutSeconds
// @input Component.MaterialMeshVisual setTextureTarget
// @input Asset.Texture setTextureNewTexture


var delayedEvent = script.createEvent("DelayedCallbackEvent");
delayedEvent.bind(function(eventData)
{
    script.setTextureTarget.mainPass.baseTex = script.setTextureNewTexture
    print("delay is over");
    delayedEvent.reset(script.timeoutSeconds);
print("delay has started");
});

// Start with a 2 second delay
delayedEvent.reset(script.timeoutSeconds);
print("delay has started");