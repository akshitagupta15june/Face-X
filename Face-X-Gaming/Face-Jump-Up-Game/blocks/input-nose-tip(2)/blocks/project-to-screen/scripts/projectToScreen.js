const console = require('Diagnostics');
const CameraInfo = require('CameraInfo');
const Patches = require('Patches');
const Reactive = require('Reactive');
const Scene = require('Scene');

(async function() {

    const cameraDistance = 0.53613; // default focal distance

    const inputPos = await Patches.outputs.getPoint('position'); // get object 3D position

    const screenSize = Reactive.pack2(
            CameraInfo.previewSize.width,
            CameraInfo.previewSize.height)
        .div(CameraInfo.previewScreenScale); // scaled screen size as vec2

    const aspect = Reactive.div(screenSize.x, screenSize.y); //screen aspect ratio

    const range = Reactive.pack2(aspect, -1).mul(0.25); // range multiplier

    const isFrontCamera = CameraInfo.captureDevicePosition.eq('FRONT'); // is front/back camera?

    const axisYSwap = Reactive.ifThenElse(isFrontCamera, 1, -1); // front camera y axis is wrong sign

    const pos = inputPos.mul(Reactive.pack3(1, axisYSwap, 1)) // correct wrong y axis

    const scaledPos = pos.mul(Reactive.div(cameraDistance, pos.z)).neg(); //calc scaled position

    const toRange = Reactive.fromRange(
        Reactive.pack2(scaledPos.x, scaledPos.y),
        range.neg(),
        range);

    const posXY = Reactive.toRange(
        toRange,
        Reactive.pack2(0, 0),
        screenSize);

    Patches.inputs.setPoint2D('position2D', posXY); // get object 3D position    

})();