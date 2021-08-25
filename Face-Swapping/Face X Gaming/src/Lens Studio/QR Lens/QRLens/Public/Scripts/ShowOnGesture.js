// -----JS CODE-----

// @input SceneObject targetObject
// @input string showEvent
// @input string hideEvent

function onShow(time)
{
 global.controlTime = getTime();
 script.targetObject.enabled = true;
}
var showEvent = script.createEvent(script.showEvent);
showEvent.bind(onShow);


function onHide(time)
{
 global.controlTime = getTime();
 script.targetObject.enabled = false;
}
var hideEvent = script.createEvent(script.hideEvent);
hideEvent.bind(onHide);