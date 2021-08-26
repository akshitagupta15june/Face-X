// -----JS CODE-----
// @input Component.Text label

var store = global.launchParams;
var testText = store.getString("test_text");

if(script.label && testText) { 
    script.label.text = testText;
} else {
    script.label.text = "Launch from the Android app to get instructions!";
}