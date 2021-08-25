// TweenAlpha.js
// Version: 0.0.9
// Event: Any Event
// Description: Runs a tween on a Lens Studio object's alpha using TweenJS
// ----- USAGE -----
// Attach this script as a component after the Tween Manager script on either the same scene object or in a lower scene object in the Objects Panel.
//
// Assign a scene object that contains a non-Default Material to "Scene Object" on this script.
// -----------------

// @input SceneObject sceneObject
// @input string tweenName
// @input bool playAutomatically = true
// @input int loopType = 0 {"widget":"combobox", "values":[{"label":"None", "value":0}, {"label":"Loop", "value":1}, {"label":"Ping Pong", "value":2}, {"label":"Ping Pong Once", "value":3}]}
// @ui {"widget":"separator"}
// @input int movementType = 0 {"widget": "combobox", "values": [{"label": "From / To", "value": 0}, {"label": "To", "value": 1}, {"label":"From", "value": 2}, {"label":"Offset", "value": 3}]}
// @input float start = 0.0 {"widget":"slider", "min":0.0, "max":1.0, "step":0.01, "showIf":"movementType", "showIfValue":0}
// @input float end = 1.0 {"widget":"slider", "min":0.0, "max":1.0, "step":0.01, "showIf":"movementType", "showIfValue":0}
// @input float from = 0.0 {"widget":"slider", "min":0.0, "max":1.0, "step":0.01, "showIf":"movementType", "showIfValue":2, "label":"Start"}
// @input float to = 0.0 {"widget":"slider", "min":0.0, "max":1.0, "step":0.01, "showIf":"movementType", "showIfValue":1, "label":"End"}
// @input float offset = 0.0 {"widget":"slider", "min":-1.0, "max":1.0, "step":0.01, "showIf":"movementType", "showIfValue":3}
// @input bool additive {"showIf":"movementType", "showIfValue": 3}
// @ui {"widget":"label", "label":"(Use on Loop)", "showIf": "movementType", "showIfValue": 3}
// @input bool recursive = false
// @input float time = 1.0
// @input float delay = 0.0

// @ui {"widget":"separator"}
// @input string easingFunction = "Quadratic" {"widget":"combobox", "values":[{"label":"Linear", "value":"Linear"}, {"label":"Quadratic", "value":"Quadratic"}, {"label":"Cubic", "value":"Cubic"}, {"label":"Quartic", "value":"Quartic"}, {"label":"Quintic", "value":"Quintic"}, {"label":"Sinusoidal", "value":"Sinusoidal"}, {"label":"Exponential", "value":"Exponential"}, {"label":"Circular", "value":"Circular"}, {"label":"Elastic", "value":"Elastic"}, {"label":"Back", "value":"Back"}, {"label":"Bounce", "value":"Bounce"}]}
// @input string easingType = "Out" {"widget":"combobox", "values":[{"label":"In", "value":"In"}, {"label":"Out", "value":"Out"}, {"label":"In / Out", "value":"InOut"}]}

// If no scene object is specified, use object the script is attached to
if (!script.sceneObject) {
    script.sceneObject = script.getSceneObject();
}

// Setup the external API
script.api.tweenObject = script.getSceneObject();
script.api.tweenType = "alpha";
script.api.tweenName = script.tweenName;
script.api.time = script.time;
script.api.startTween = startTween;
script.api.resetObject = resetObject;
script.api.movementType = script.movementType;
script.api.loopType = script.loopType;
script.api.additive = script.additive;
script.api.tween = null;
script.api.tweenObjects = null;
script.api.setupTween = setupTween;
script.api.setupTweenBackwards = setupTweenBackwards;
script.api.sceneObject = script.sceneObject;
script.api.updateToStart = updateToStart;
script.api.updateToEnd = updateToEnd;
script.api.loopType = script.loopType;
script.api.start = null;
script.api.end = null;
script.api.setStart = setStart;
script.api.setEnd = setEnd;
script.api.manualStart = false;
script.api.manualEnd = false;
script.api.playAutomatically = script.playAutomatically;

if (global.tweenManager && global.tweenManager.addToRegistry) {
    global.tweenManager.addToRegistry(script);
}

// Manually set start value
function setStart(start) {
    script.api.manualStart = true;
    script.api.start = start;
}

// Manually set end value
function setEnd(end) {
    script.api.manualEnd = true;
    script.api.end = end;
}

// Update the tween to its start
function updateToStart() {
    for (var i = 0; i < script.api.tweenObjects.length; i++) {
        var tweenObject = script.api.tweenObjects[i];

        updateAlphaComponent(tweenObject.component, tweenObject.startValue);
    }
}

// Update the tween to its end
function updateToEnd() {
    for (var i = 0; i < script.api.tweenObjects.length; i++) {
        var tweenObject = script.api.tweenObjects[i];

        if (script.loopType == 3) {
            updateAlphaComponent(tweenObject.component, tweenObject.startValue);
        } else {
            updateAlphaComponent(tweenObject.component, tweenObject.endValue);
        }
    }
}

// Play it automatically if specified
if (script.playAutomatically) {
    // Start the tween
    startTween();
}

// Create the tween with passed in parameters
function startTween() {
    if (!global.tweenManager) {
        print("Tween Alpha: Tween Manager not initialized. Try moving the TweenManager script to the top of the Objects Panel or changing the event on this TweenType to \"Lens Turned On\".");
        return;
    }

    var tween = setupTween();

    if (tween) {
        if (script.api.tween.length > 0) {
            if (script.api.movementType == 3 && script.api.loopType == 1 && script.api.additive) {
                script.api.tween[script.api.tween.length - 1].onComplete(startTween);
            }

            // Start the tweens
            for (var i = 0; i < script.api.tween.length; i++) {
                script.api.tween[i].start();
            }
        }
    }
}

// Create the tween with passed in parameters
function setupTween() {
    script.api.tweenObjects = [];

    script.api.tween = [];

    var componentTypes = [
        "Component.MaterialMeshVisual",
        "Component.Text"
    ];

    for (var i = 0; i < componentTypes.length; i++) {
        setupAlphaComponentTweens(componentTypes[i], script.api.sceneObject);
    }

    if (script.api.tween.length == 0) {
        print("Tween Alpha: No compatible components found for SceneObject " + script.sceneObject.name);
        return;
    }

    return script.api.tween;
}

function setupAlphaComponentTweens(componentType, sceneObject) {
    var visualComponents = sceneObject.getComponents(componentType);

    for (var i = 0; i < visualComponents.length; i++) {
        var visualComponent = visualComponents[i];
        var startValue = null;
        var endValue = null;
        var tween = null;
        var tweenObject = null;

        if (visualComponent.getMaterialsCount() == 0) {
            continue;
        }

        if (!script.api.manualStart) {
            switch (script.api.movementType) {
                case 0:
                    script.api.start = {
                        a: script.start
                    };
                    break;
                case 2:
                    script.api.start = {
                        a: script.from
                    };
                    break;
                case 1:
                case 3:
                    script.api.start = {
                        a: getVisualComponentAlpha(visualComponent)
                    };
                    break;
            }
        }

        startValue = script.api.start;

        if (!script.api.manualEnd) {
            switch (script.api.movementType) {
                case 0:
                    script.api.end = {
                        a: script.end
                    };
                    break;
                case 2:
                    script.api.end = {
                        a: getVisualComponentAlpha(visualComponent)
                    };
                    break;
                case 1:
                    script.api.end = {
                        a: script.to
                    };
                    break;
                case 3:
                    script.api.end = {
                        a: startValue.a + script.offset
                    };
                    break;
            }
        }

        endValue = script.api.end;

        // Create the tween
        tween = new TWEEN.Tween(startValue)
            .to(endValue, script.api.time * 1000.0)
            .delay(script.delay * 1000.0)
            .easing(global.tweenManager.getTweenEasingType(script.easingFunction, script.easingType))
            .onUpdate(updateAlphaComponent(visualComponent));

        if (tween) {
            // Configure the type of looping based on the inputted parameters
            if (script.movementType == 3 && script.additive && script.loopType == 1) {
                global.tweenManager.setTweenLoopType(tween, 0);
            } else {
                global.tweenManager.setTweenLoopType(tween, script.api.loopType);
            }

            tweenObject = {
                tween: tween,
                startValue: {
                    a: startValue.a
                },
                endValue: {
                    a: endValue.a
                },
                component: visualComponent
            };

            script.api.tweenObjects.push(tweenObject);

            script.api.tween.push(tween);
        } else {
            print("Tween Alpha: Tween Manager not initialized. Try moving the TweenManager script to the top of the Objects Panel or changing the event on this TweenType to \"Lens Turned On\".");
        }
    }
    if (script.recursive) {
        for (var j = 0; j < sceneObject.getChildrenCount(); j++) {
            setupAlphaComponentTweens(componentType, sceneObject.getChild(j));
        }
    }
}

// Create the tween with swapped start and end parameters
function setupTweenBackwards() {
    var tempTweenObjectsArray = [];

    var tempTweenArray = [];

    var easingType = global.tweenManager.getSwitchedEasingType(script.easingType);

    for (var i = 0; i < script.api.tween.length; i++) {
        var newTweenObject = null;

        var tween = script.api.tweenObjects[i];

        var tweenStart = (script.loopType == 3) ? tween.startValue : tween.endValue;

        var tweenEnd = (script.loopType == 3) ? tween.endValue : tween.startValue;

        var tweenEasingType = global.tweenManager.getTweenEasingType(script.easingFunction, easingType);

        var newTween = new TWEEN.Tween(tweenStart)
            .to(tweenEnd, script.api.time * 1000.0)
            .delay(script.delay * 1000.0)
            .easing(tweenEasingType)
            .onUpdate(updateAlphaComponent(tween.component));

        if (newTween) {
            // Configure the type of looping based on the inputted parameters
            global.tweenManager.setTweenLoopType(newTween, script.api.loopType);

            newTweenObject = {
                tween: newTween,
                startValue: {
                    a: (script.loopType == 3) ? tween.startValue.a : tween.endValue.a
                },
                endValue: {
                    a: (script.loopType == 3) ? tween.endValue.a : tween.startValue.a
                },
                component: tween.component
            };

            // Save reference to tween
            tempTweenObjectsArray.push(newTweenObject);

            tempTweenArray.push(newTween);
        } else {
            print("Tween Alpha: Tween Manager not initialized. Try moving the TweenManager script to the top of the Objects Panel or changing the event on this TweenType to \"Lens Turned On\".");
            return;
        }
    }

    return tempTweenArray;
}

function getVisualComponentAlpha(visual) {
    var color;
    if (visual.getTypeName() == "Component.Text") {
        color = visual.textFill.color;
    } else {
        color = visual.getMaterial(0).getPass(0).baseColor;
    }

    if (color && color.a != undefined) {
        return color.a;
    } else {
        print("TweenAlpha: Visual Component on Object '" + visual.getSceneObject().name + "' does not have a material supported by TweenAlpha. Setting alpha to 1.0.");
        return 1.0;
    }
}

function updateText(visualComponent, value) {
    var fillColor = visualComponent.textFill.color;

    if (fillColor && fillColor.a != undefined) {
        fillColor.a = value.a;
        visualComponent.textFill.color = fillColor;
    }

    // Outline Color
    var outlineColor = visualComponent.outlineSettings.fill.color;

    if (outlineColor && outlineColor.a != undefined) {
        outlineColor.a = value.a;
        visualComponent.outlineSettings.fill.color = outlineColor;
    }

    // Drop Shadow Color
    var dropShadowColor = visualComponent.dropshadowSettings.fill.color;
    if (dropShadowColor && dropShadowColor.a != undefined) {
        dropShadowColor.a = value.a;

        visualComponent.dropshadowSettings.fill.color = dropShadowColor;
    }

    // Background Color
    var backgroundColor = visualComponent.backgroundSettings.fill.color;
    if (backgroundColor && backgroundColor.a != undefined) {
        backgroundColor.a = value.a;

        visualComponent.backgroundSettings.fill.color = backgroundColor;
    }
}

function updateVisual(visualComponent, value) {
    var currColor = visualComponent.getMaterial(0).getPass(0).baseColor;
    if (currColor && currColor.a != undefined) {
        currColor.a = value.a;

        visualComponent.getMaterial(0).getPass(0).baseColor = currColor;
    }
}

// Resets the object to its start
function resetObject() {
    if (script.api.tweenObjects == null) {
        return;
    }
    for (var i = 0; i < script.api.tweenObjects.length; i++) {
        var tweenObject = script.api.tweenObjects[i];
        updateAlphaComponent(tweenObject.component, tweenObject.startValue);
    }
}

function updateAlphaComponent(visualComponent, value) {
    if (value) {
        if (visualComponent.getTypeName() == "Component.Text") {
            updateText(visualComponent, value);
        } else {
            updateVisual(visualComponent, value);
        }
    } else {
        return function(value) {
            if (visualComponent.getTypeName() == "Component.Text") {
                updateText(visualComponent, value);
            } else {
                updateVisual(visualComponent, value);
            }
        };
    }
}
