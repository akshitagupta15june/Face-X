// ReparentHelper.js
// Version: 0.0.1
// Event: Lens Initialized
// Description: Re-parents this object to the given root screen transform while maintaining the same place on screen
//@input SceneObject handRoot

function init() {
    // Get this scene object
    var sceneObject = script.getSceneObject();

    // Get the current parent
    var currentParent = sceneObject.getParent();
    if (!currentParent) {
        print("ReparentHelper, ERROR: Please make sure the prefab is a child of any object under the orthographic camera!");
        return;
    }

    // Get screen transform of the current parent
    var parentScreenTransform = currentParent.getComponent("Component.ScreenTransform");
    if (!parentScreenTransform) {
        print("ReparentHelper, ERROR: Please make sure parent of the prefab has a Screen Transform component on it!");
        return;
    }
    
    var newParent = global.scene.createSceneObject("Parent");
    newParent.copyComponent(parentScreenTransform);
    newParent.setParent(script.handRoot);
    sceneObject.setParent(newParent);
    
    // This will create a new script and attach it to the parent object to make sure
    // that when the child gets destroyed using the Behavior script, this will destroy the
    // parent too. This will help with optimization by cleaning up extra objects.
    var newScript = newParent.createComponent("Component.ScriptComponent");
    newScript.createEvent("UpdateEvent").bind(function() {
        if (newParent.getChildrenCount() == 0) {
            newParent.destroy();
        }
    });
}

init();

