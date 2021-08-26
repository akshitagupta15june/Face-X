// TweenColor.js
// Version: 0.0.9
// Event: Any Event
// Description: Runs a tween on a Lens Studio color using TweenJS
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

// @input int colorProperty = 0 {"widget":"combobox", "values":[{"label":"Default", "value":0}, {"label":"Outline (Text Only)", "value":1}, {"label":"Drop Shadow (Text Only)", "value":2}, {"label":"Background (Text Only)", "value":3}]}

// @input vec4 start = {1,1,1,1} {"widget":"color", "showIf":"movementType", "showIfValue":0}
// @input vec4 end = {1,1,1,1} {"widget":"color", "showIf":"movementType", "showIfValue":0}
// @input vec4 to = {1,1,1,1} {"widget":"color", "showIf":"movementType", "showIfValue":1, "label":"End"}
// @input vec4 from = {1,1,1,1} {"widget":"color", "showIf":"movementType", "showIfValue":2, "label":"Start"}
// @input vec4 offset = {1,1,1,1} {"showIf":"movementType", "showIfValue":3}

// @input bool additive {"showIf":"movementType", "showIfValue": 3}
// @ui {"widget":"label", "label":"(Use on Loop)", "showIf": "movementType", "showIfValue": 3}
// @input bool recursive = false
// @input bool ignoreAlpha = false
// @input float time = 1.0
// @input float delay = 0.0

// @ui {"widget":"separator"}
// @input string easingFunction = "Quadratic" {"widget":"combobox", "values":[{"label":"Linear", "value":"Linear"}, {"label":"Quadratic", "value":"Quadratic"}, {"label":"Cubic", "value":"Cubic"}, {"label":"Quartic", "value":"Quartic"}, {"label":"Quintic", "value":"Quintic"}, {"label":"Sinusoidal", "value":"Sinusoidal"}, {"label":"Exponential", "value":"Exponential"}, {"label":"Circular", "value":"Circular"}, {"label":"Elastic", "value":"Elastic"}, {"label":"Back", "value":"Back"}, {"label":"Bounce", "value":"Bounce"}]}
// @input string easingType = "Out" {"widget":"combobox", "values":[{"label":"In", "value":"In"}, {"label":"Out", "value":"Out"}, {"label":"In / Out", "value":"InOut"}]}

var TextPropertyNames = {
    0: "textFill",
    1: "outlineSettings",
    2: "dropshadowSettings",
    3: "backgroundSettings"
};

// If no scene object is specified, use object the script is attached to
if (!script.sceneObject) {
    script.sceneObject = script.getSceneObject();
}

// Setup the external API
script.api.tweenObject = script.getSceneObject();
script.api.tweenType = "color";
script.api.tweenName = script.tweenName;
script.api.movementType = script.movementType;
script.api.time = script.time;
script.api.startTween = startTween;
script.api.resetObject = resetObject;
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

        updateColorComponent(tweenObject.component, tweenObject.startValue);
    }
}

// Update the tween to its end
function updateToEnd() {
    for (var i = 0; i < script.api.tweenObjects.length; i++) {
        var tweenObject = script.api.tweenObjects[i];

        var copiedValue = {
            r: (script.loopType == 3) ? tweenObject.startValue.r : tweenObject.endValue.r,
            g: (script.loopType == 3) ? tweenObject.startValue.g : tweenObject.endValue.g,
            b: (script.loopType == 3) ? tweenObject.startValue.b : tweenObject.endValue.b,
            a: (script.loopType == 3) ? tweenObject.startValue.a : tweenObject.endValue.a
        };

        updateColorComponent(tweenObject.component, copiedValue);
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
        print("Tween Color: Tween Manager not initialized. Try moving the TweenManager script to the top of the Objects Panel or changing the event on this TweenType to \"Lens Turned On\".");
        return;
    }

    var tween = setupTween();

    if (tween) {
        if (script.api.tween.length > 0) {
            script.api.tween[script.api.tween.length - 1].onComplete((script.movementType == 3 && script.additive && script.loopType == 1) ? startTween : null);
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
        setupColorComponentTweens(componentTypes[i], script.api.sceneObject);
    }
    if (script.api.tween.length == 0) {
        print("Tween Color:  No compatible components found for SceneObject " + script.sceneObject.name);
    }
    return script.api.tween;
}

// Create Tweens for specific Visual Component (e.g. MaterialMeshVisual or Text)
function setupColorComponentTweens(componentType, sceneObject) {
    var visualComponents = sceneObject.getComponents(componentType);

    for (var i = 0; i < visualComponents.length; i++) {
        var visualComponent = visualComponents[i];

        if (visualComponent.getMaterialsCount() == 0) {
            continue;
        }

        var startValue = null;

        var endValue = null;

        var tween = null;

        var tweenObject = null;

        // Set start and end values for Tween based on movementType
        if (!script.api.manualStart) {
            switch (script.movementType) {
                case 0:
                    script.api.start = {
                        r: script.start.r,
                        g: script.start.g,
                        b: script.start.b,
                        a: script.start.a
                    };
                    break;
                case 2:
                    script.api.start = {
                        r: script.from.r,
                        g: script.from.g,
                        b: script.from.b,
                        a: script.from.a
                    };
                    break;
                case 1:
                case 3:
                    script.api.start = getVisualComponentStartColor(visualComponent);
                    break;
            }
        }

        startValue = script.api.start;

        if (!script.api.manualEnd) {
            switch (script.movementType) {
                case 0:
                    script.api.end = {
                        r: script.end.r,
                        g: script.end.g,
                        b: script.end.b,
                        a: (script.ignoreAlpha) ? startValue.a : script.end.a
                    };
                    break;
                case 1:
                    script.api.end = {
                        r: script.to.r,
                        g: script.to.g,
                        b: script.to.b,
                        a: (script.ignoreAlpha) ? startValue.a : script.to.a
                    };
                    break;
                case 2:
                    script.api.end = getVisualComponentEndColor(visualComponent, startValue);
                    break;
                case 3:
                    script.api.end = {
                        r: startValue.r + script.offset.w,
                        g: startValue.g + script.offset.x,
                        b: startValue.b + script.offset.y,
                        a: (script.ignoreAlpha) ? startValue.a : startValue.a + script.offset.z
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
            .onUpdate(updateColorComponent(visualComponent));

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
                    r: startValue.r,
                    g: startValue.g,
                    b: startValue.b,
                    a: startValue.a
                },
                endValue: {
                    r: endValue.r,
                    g: endValue.g,
                    b: endValue.b,
                    a: endValue.a
                },
                component: visualComponent
            };

            script.api.tweenObjects.push(tweenObject);

            script.api.tween.push(tween);
        } else {
            print("Tween Color: Tween Manager not initialized. Try moving the TweenManager script to the top of the Objects Panel or changing the event on this TweenType to \"Lens Turned On\".");
        }
    }

    if (script.recursive) {
        for (var j = 0; j < sceneObject.getChildrenCount(); j++) {
            setupColorComponentTweens(componentType, sceneObject.getChild(j));
        }
    }
}

function getVisualComponentEndColor(visualComponent, startValue) {
    if (visualComponent.getTypeName() == "Component.Text") {
        var textSetting = getTextSetting(visualComponent);
        return {
            r: textSetting.color.r,
            g: textSetting.color.g,
            b: textSetting.color.b,
            a: (script.ignoreAlpha) ? startValue.a : textSetting.color.a
        };
    } else {
        return {
            r: visualComponent.getMaterial(0).getPass(0).baseColor.r,
            g: visualComponent.getMaterial(0).getPass(0).baseColor.g,
            b: visualComponent.getMaterial(0).getPass(0).baseColor.b,
            a: (script.ignoreAlpha) ? startValue.a : visualComponent.getMaterial(0).getPass(0).baseColor.a
        };
    }
}

function getVisualComponentStartColor(visualComponent) {
    if (visualComponent.getTypeName() == "Component.Text") {
        var textSetting = getTextSetting(visualComponent);

        return {
            r: textSetting.color.r,
            g: textSetting.color.g,
            b: textSetting.color.b,
            a: textSetting.color.a
        };
    } else {
        return {
            r: visualComponent.getMaterial(0).getPass(0).baseColor.r,
            g: visualComponent.getMaterial(0).getPass(0).baseColor.g,
            b: visualComponent.getMaterial(0).getPass(0).baseColor.b,
            a: visualComponent.getMaterial(0).getPass(0).baseColor.a
        };
    }
}

// Create the tween with swapped start and end parameters
function setupTweenBackwards() {
    var tempTweenObjectsArray = [];

    var tempTweenArray = [];

    // Change easing type
    var easingType = global.tweenManager.getSwitchedEasingType(script.easingType);

    for (var i = 0; i < script.api.tween.length; i++) {
        var tween = script.api.tweenObjects[i];

        var newTween = new TWEEN.Tween((script.loopType == 3) ? tween.startValue : tween.endValue)
            .to((script.loopType == 3) ? tween.endValue : tween.startValue, script.api.time * 1000.0)
            .delay(script.delay * 1000.0)
            .easing(global.tweenManager.getTweenEasingType(script.easingFunction, easingType))
            .onUpdate(updateColorComponent(tween.component));

        var newTweenObject = null;

        if (newTween) {
            // Configure the type of looping based on the inputted parameters
            global.tweenManager.setTweenLoopType(newTween, script.api.loopType);

            newTweenObject = {
                tween: newTween,
                startValue: {
                    r: (script.loopType == 3) ? tween.startValue.r : tween.endValue.r,
                    g: (script.loopType == 3) ? tween.startValue.g : tween.endValue.g,
                    b: (script.loopType == 3) ? tween.startValue.b : tween.endValue.b,
                    a: (script.loopType == 3) ? tween.startValue.a : tween.endValue.a
                },
                endValue: {
                    r: (script.loopType == 3) ? tween.endValue.r : tween.startValue.r,
                    g: (script.loopType == 3) ? tween.endValue.g : tween.startValue.g,
                    b: (script.loopType == 3) ? tween.endValue.b : tween.startValue.b,
                    a: (script.loopType == 3) ? tween.endValue.a : tween.startValue.a
                },
                component: tween.component
            };

            // Save reference to tween
            tempTweenObjectsArray.push(newTweenObject);

            tempTweenArray.push(newTween);
        } else {
            print("Tween Color: Tween Manager not initialized. Try moving the TweenManager script to the top of the Objects Panel or changing the event on this TweenType to \"Lens Turned On\".");
            return;
        }
    }

    return tempTweenArray;
}

// Resets the object to its start
function resetObject() {
    if (script.api.tweenObjects == null) {
        return;
    }

    for (var i = 0; i < script.api.tweenObjects.length; i++) {
        var tweenObject = script.api.tweenObjects[i];

        updateColorComponent(tweenObject.component, tweenObject.startValue);
    }
}

function getTextSetting(visualComponent) {
    return script.colorProperty == 0 ? visualComponent[TextPropertyNames[script.colorProperty]] : visualComponent[TextPropertyNames[script.colorProperty]].fill;
}

function updateText(visualComponent, value) {
    var textSetting = getTextSetting(visualComponent);

    if (script.ignoreAlpha) {
        var currColor = textSetting.color;
        textSetting.color = new vec4(value.r, value.g, value.b, currColor.a);
    } else {
        textSetting.color = new vec4(value.r, value.g, value.b, value.a);
    }
}

function updateVisual(visualComponent, value) {
    if (script.ignoreAlpha) {
        var currColor = visualComponent.getMaterial(0).getPass(0).baseColor;
        visualComponent.getMaterial(0).getPass(0).baseColor = new vec4(value.r, value.g, value.b, currColor.a);
    } else {
        visualComponent.getMaterial(0).getPass(0).baseColor = new vec4(value.r, value.g, value.b, value.a);
    }
}

// Update single Visual Component (e.g. MaterialMeshVisual or Text)
function updateColorComponent(visualComponent, value) {
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
