// MaterialDuplicator.js
// Version: 0.0.1
// Event: Lens Initialized
// Description: Duplicates material of the object it is attached to.

function init() {
    var comp = script.getSceneObject().getComponent("Component.MaterialMeshVisual");
    if (comp) {
        comp.mainMaterial = comp.mainMaterial.clone();
    }
}

init();