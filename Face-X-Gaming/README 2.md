## Inspiration

People with certain physical disabilities often find themselves at an immediate disadvantage in gaming. There are some amazing people and organizations in the gaming accessibility world that have set out to make that statement less true. People like Bryce Johnson who created the [Xbox Adaptive Controller](https://www.xbox.com/en-US/accessories/controllers/xbox-adaptive-controller), or everyone from the Special Effect and Able Gamers charities. They use their time and money to create custom controllers that are fit to a specific user with their own unique situation.

Here's an example of those setups:
![Adaptive setup](https://compass-ssl.xbox.com/assets/d3/9d/d39d37c7-deb8-4088-bd2e-f2bb15631bc6.jpg?n=Assistive-Tech_Feature-0_Strengthened-Community_1040x585.jpg)

You can see the custom buttons on the pad and the headrest as well as the custom joysticks. These types of customized controllers using the XAC let the user make the controller work for them. These are absolutely amazing developments in the accessible gaming world, but we can do more.

Games that are fast paced or just challenging in general still leave an uneven playing field for people with disabilities. For example, I can tap a key or click my mouse drastically faster than the person in the example above can reach off the joystick to hit a button on a pad. I have a required range of motion of 2mm where he has a required range of over 12 inches.

I built Suave Keys to level the playing field, now made even better by Snap Keys! Combine voice input, facial expressions, and gestures to play games the way that works for you!

## What it does

SnapKeys + SuaveKeys lets you play games and use software with your voice, gestures, and expressions alongside the usual input of keyboard and mouse. 
It acts as a distributed system to allow users to make use of whatever resources they have to connect. Use your voice via any virtual assistant, smart speaker, or voice-enabled app. Then use the Suave Keys snap lens and the Snap Reader app to start using expressions and gestures too!

Here's what it looks like without Snap Keys: 
![pre-snap](https://i.imgur.com/4mKU6gZ.png)

The process is essentially:
- User signs into their smart speaker and client app
- User speaks to the smart speaker
- The request goes to Voicify to add context and routing
- Voicify sends the updated request to the SuaveKeys API
- The SuaveKeys API sends the processed input to the connected Client apps over websockets
- The Client app checks the input phrase against a selected keyboard profile
- The profile matches the phrase to a key or a macro
- The client app then sends the request over a serial data writer to an Arduino Leonardo
- The Arduino then sends USB keyboard commands back to the host computer
- The computer executes the action in game

Now here it is with Snap Keys:

![with-snap](https://i.imgur.com/S5XX5E2.png)

Snap Keys acts as an extension of Suave Keys. You launch the lens from the Android or iOS app, then launch the Snap Reader windows client. This client lets you choose an application to monitor such as your android or iPhone, then streams each frame of the application through the processor. Whenever a QR code is found, it will detect the inner-command of the QR code and send that command to Suave Keys. From there, Suave Keys takes over and sends the command down to the user's end-client which checks it against the current game profile and executes the key or macro of keys through the Arduino.

Here's a typical flow once everything is running:
- User raises eyebrows
- Snap reader detects the brow raise QR code
- Snap reader sends brow_raise command to Suave Keys for the authenticated user
- Suave Keys sends brow_raise to the end client via websocket
- End client sees that "brow_raise" matches with the space bar
- End client sends space bar key request to Arduino
- Arduino presses space bar
- Character jumps in game!

The app also allows the user to customize their profiles from their phone as well as their desktop client. So if you want to quickly create a new command or macro, you can register it right within the app.

Here's an example of a Fall Guys profile of commands - select a key, give a list of commands, and when you speak them, it works!
![keyboard](https://voicify-prod-files.s3.amazonaws.com/faebe924-8c96-45cd-971f-adadd6b125e2/e8df6980-0f14-4135-ba17-eb7ed9f6e2d2/keyboard.png)

You can also add macros to a profile:
![macros](https://voicify-prod-files.s3.amazonaws.com/faebe924-8c96-45cd-971f-adadd6b125e2/f8401a37-2602-4ea7-a3f8-0b2e46e52d18/keyboard2.png)

## How I built it

The Snap Keys apps were built using:
- Kotlin for Android
- Swift for iOS
- C#, .NET, and UWP for the Snap Reader
- The lense was built using lens studio combined with custom scripts and assets for the QR codes generated online

While the SuaveKeys API and Authentication layers already existed, we were able to build the client apps to act as a whole new input type. 

The most important piece was the Snap Reader Windows app. I made use of the `GraphicsCaptureSession` library in Windows along side a Direct 3D encoder to take each frame from the stream, process it in memory to a bitmap, then run the bitmap through a ZXing barcode scanner set to scan for QR codes.

Here's the method that is invoked on each frame being processed from the screen stream:
```csharp
private async void OnFrameArrived(Direct3D11CaptureFramePool sender, object args)
{
    _currentFrame = sender.TryGetNextFrame();

    BarcodeReader reader = new BarcodeReader();

    reader.AutoRotate = true;
    reader.Options.TryHarder = true;
    reader.Options.PureBarcode = false;
    reader.Options.PossibleFormats = new List<BarcodeFormat>();
    reader.Options.PossibleFormats.Add(BarcodeFormat.QR_CODE);

    var bitmap = await SoftwareBitmap.CreateCopyFromSurfaceAsync(_currentFrame.Surface).AsTask();
    var result = reader.Decode(bitmap);
    if (!string.IsNullOrEmpty(result?.Text) && (result.Text.StartsWith("suavekeys|expression") || result.Text.StartsWith("suavekeys|gesture")))
    {
        Debug.WriteLine("WOOHOO WE FOUND A CODE");
        if(!_isSending)
        {
            _isSending = true;
            var command = result.Text.Split('|')[2];
            await _suaveKeysService.SendCommandAsync(command);
            _isSending = false;
        }
    }
    _frameEvent.Set();
}
```

I added the `_isSending` lock so that we weren't constantly feeding HTTP requests to SuaveKeys' API on every single frame since on a decent machine, the Snap Reader app can process about 60 fps. That means if you were raising an eye brow to jump in game, holding your brow up for 1 second, it would send 60 jump requests to the game. This acted as a safe throttle while still allowing for hold actions.

## Challenges I ran into

The biggest challenge was testing while also talking to my chat on stream! Since the whole thing was built live on my twitch channel, I'm always talking to chat about my thought process, what I'm doing, telling jokes, and answering questions. I also talk with a lot of facial expressions naturally, so talking to chat triggered tons of extra mouth, smile, and brow events. Hoenstly it ended up being pretty funny though.

Other than that, it just took me an hour or so to really figure out how to use lens studio to its potential. I've done a bit of Unity and Unreal work in the past, so it wasn't too bad.

## Accomplishments that I'm proud of

The biggest accomplishment was being able to see it in use! I was able to play games like Call of Duty, Dark Souls, and Fall Guys using my face and gestures! It is far more performant for primary actions than just voice and feels like there is some real potential to use this type of technology or direction to give people more options for how to interact with games and software that works for them.


## What I learned

I learned a lot about multi-modality on the input side of conversational AI and commands and was able to use snapchat to push that to new limits. I also learned tons about how to use lens studio and create some really cool, funny, and innovative lenses that people will hopefully love ♥

## What's next for Snap Keys

Tons of stuff! For Snap Keys:
- More gestures and expressions
- Making the lens look a lot better
- Tweaking performance

Then within just Suave Keys:
- Making the UI a WHOLE lot cleaner and easier to use
- Enabling more platforms to help more people use it
- Distributing hardware creation to let people actually use it
- Adding more device support for the XAC
- Building shareable game profiles

I'm working on it twice a week on stream, so we are always making tons of progress :)

## Conclusion

I think Suave and Snap Keys has the chance to enable so many more people to play games that they never could before using whatever they have available to them!
