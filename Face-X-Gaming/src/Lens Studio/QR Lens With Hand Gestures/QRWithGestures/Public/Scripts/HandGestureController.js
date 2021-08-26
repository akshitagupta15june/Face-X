// HandGestureController.js
// Version: 0.0.1
// Event: Lens Initialized
// Description: Sends given set of triggers on hand gesture detection

//@input Component.ObjectTracking tracker
//@input string[] openTriggers
//@input string[] closeTriggers
//@input string[] hornsTriggers
//@input string[] indexFingerTriggers
//@input string[] victoryTriggers
//@input bool debug


var labels = ["open", "close", "horns", "index_finger", "victory"];
var customMap = {
    open: script.openTriggers,
    close: script.closeTriggers,
    horns: script.hornsTriggers,
    index_finger: script.indexFingerTriggers,
    victory: script.victoryTriggers,
};


function generateTriggerResponse(evt) {
    return function() {
        if (script.debug) {
            print("Gesture detected: " + evt);
        }
        sendCustomTriggers(evt);
    };
}


function sendCustomTriggers(evt) {
    if (!customMap[evt]) {
        return;
    }

    for (var i in customMap[evt]) {
        global.behaviorSystem.sendCustomTrigger(customMap[evt][i]);
        if (script.debug) {
            print("Triggered: " + customMap[evt][i]);
        }
    }
}


function init() {
    for (var i = 0; i < labels.length; i++) {
        script.tracker.registerDescriptorStart(labels[i], generateTriggerResponse(labels[i]));
        if (script.debug) {
            print("Event created for: " + labels[i]);
        }
    }
}


init();