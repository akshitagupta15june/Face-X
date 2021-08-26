using AsyncAwaitBestPractices.MVVM;
using ServiceResult;
using SharpDX.Direct3D11;
using SuaveKeys.SnapReader.Uwp.Services;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using Windows.Devices.PointOfService;
using Windows.Graphics;
using Windows.Graphics.Capture;
using Windows.Graphics.DirectX;
using Windows.Graphics.DirectX.Direct3D11;
using Windows.Graphics.Imaging;
using Windows.Media.Core;
using Windows.Media.MediaProperties;
using Windows.Media.Transcoding;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.UI.Xaml;
using ZXing;
using ZXing.QrCode;

namespace SuaveKeys.SnapReader.Uwp.ViewModels
{
    public class MainPageViewModel : INotifyPropertyChanged
    {
        private readonly ISuaveKeysService _suaveKeysService;
        public event PropertyChangedEventHandler PropertyChanged;
        public ICommand SuaveKeysSignInCommand { get; set; }
        public ICommand ToggleStartCommand { get; set; }
        public bool IsListening { get; set; }
        public Visibility SuaveKeysSignInVisibility { get; set; }
        public string CommandLog { get; set; }
        public string Error { get; set; }
        public MainPageViewModel()
        {
            _suaveKeysService = App.Current?.Container?.Resolve<ISuaveKeysService>();
            CommandLog = "";
            SuaveKeysSignInCommand = new AsyncCommand(async () =>
            {
                var signInResult = await _suaveKeysService.StartSignInAsync();
                SuaveKeysSignInVisibility = signInResult?.ResultType == ResultType.Ok ? Visibility.Visible : Visibility.Collapsed;
            });
            ToggleStartCommand = new AsyncCommand(async () =>
            {
                if (!GraphicsCaptureSession.IsSupported())
                {
                    Error = "This device does not support the capture requirements";
                    return;
                }
                if (!_isRecording)
                {
                    await SetupEncoding();

                }
                else
                {
                    Stop();
                    Cleanup();
                }
            });
        }

        private IDirect3DDevice _device;
        private bool _isRecording;
        private bool _closed;
        private bool _isSending;
        private Device _sharpDxD3dDevice;
        private GraphicsCaptureItem _captureItem;
        private Texture2D _composeTexture;
        private RenderTargetView _composeRenderTargetView;
        private MediaEncodingProfile _encodingProfile;
        private VideoStreamDescriptor _videoDescriptor;
        private MediaStreamSource _mediaStreamSource;
        private MediaTranscoder _transcoder;
        private Multithread _multithread;
        private ManualResetEvent _frameEvent;
        private ManualResetEvent _closedEvent;
        private ManualResetEvent[] _events;
        private Direct3D11CaptureFramePool _framePool;
        private GraphicsCaptureSession _session;
        private Direct3D11CaptureFrame _currentFrame;

        private async Task SetupEncoding()
        {
            if (!GraphicsCaptureSession.IsSupported())
            {
                // Show message to user that screen capture is unsupported
                return;
            }

            // Create the D3D device and SharpDX device
            if (_device == null)
            {
                _device = Direct3D11Helpers.CreateD3DDevice();
            }
            if (_sharpDxD3dDevice == null)
            {
                _sharpDxD3dDevice = Direct3D11Helpers.CreateSharpDXDevice(_device);
            }



            try
            {
                // Let the user pick an item to capture
                var picker = new GraphicsCapturePicker();
                _captureItem = await picker.PickSingleItemAsync();
                if (_captureItem == null)
                {
                    return;
                }

                // Initialize a blank texture and render target view for copying frames, using the same size as the capture item
                _composeTexture = Direct3D11Helpers.InitializeComposeTexture(_sharpDxD3dDevice, _captureItem.Size);
                _composeRenderTargetView = new SharpDX.Direct3D11.RenderTargetView(_sharpDxD3dDevice, _composeTexture);

                // This example encodes video using the item's actual size.
                var width = (uint)_captureItem.Size.Width;
                var height = (uint)_captureItem.Size.Height;

                // Make sure the dimensions are are even. Required by some encoders.
                width = (width % 2 == 0) ? width : width + 1;
                height = (height % 2 == 0) ? height : height + 1;


                var temp = MediaEncodingProfile.CreateMp4(VideoEncodingQuality.HD1080p);
                var bitrate = temp.Video.Bitrate;
                uint framerate = 30;

                _encodingProfile = new MediaEncodingProfile();
                _encodingProfile.Container.Subtype = "MPEG4";
                _encodingProfile.Video.Subtype = "H264";
                _encodingProfile.Video.Width = width;
                _encodingProfile.Video.Height = height;
                _encodingProfile.Video.Bitrate = bitrate;
                _encodingProfile.Video.FrameRate.Numerator = framerate;
                _encodingProfile.Video.FrameRate.Denominator = 1;
                _encodingProfile.Video.PixelAspectRatio.Numerator = 1;
                _encodingProfile.Video.PixelAspectRatio.Denominator = 1;

                var videoProperties = VideoEncodingProperties.CreateUncompressed(MediaEncodingSubtypes.Bgra8, width, height);
                _videoDescriptor = new VideoStreamDescriptor(videoProperties);

                // Create our MediaStreamSource
                _mediaStreamSource = new MediaStreamSource(_videoDescriptor);
                _mediaStreamSource.BufferTime = TimeSpan.FromSeconds(0);
                _mediaStreamSource.Starting += OnMediaStreamSourceStarting;
                _mediaStreamSource.SampleRequested += OnMediaStreamSourceSampleRequested;

                // Create our transcoder
                _transcoder = new MediaTranscoder();
                _transcoder.HardwareAccelerationEnabled = true;

                using (var stream = new InMemoryRandomAccessStream())
                    await EncodeAsync(stream);

            }
            catch (Exception ex)
            {
                return;
            }
        }
        private async Task EncodeAsync(IRandomAccessStream stream)
        {
            if (!_isRecording)
            {
                _isRecording = true;

                StartCapture();

                var transcode = await _transcoder.PrepareMediaStreamSourceTranscodeAsync(_mediaStreamSource, stream, _encodingProfile);

                await transcode.TranscodeAsync();
            }
        }
        private async void OnMediaStreamSourceSampleRequested(MediaStreamSource sender, MediaStreamSourceSampleRequestedEventArgs args)
        {
            if (_isRecording && !_closed)
            {
                try
                {
                    using (var frame = WaitForNewFrame())
                    {
                        if (frame == null)
                        {
                            args.Request.Sample = null;
                            //Dispose();
                            return;
                        }

                        var timeStamp = frame.SystemRelativeTime;

                        var sample = MediaStreamSample.CreateFromDirect3D11Surface(frame.Surface, timeStamp);

                        args.Request.Sample = sample;
                    }
                }
                catch (Exception e)
                {
                    Debug.WriteLine(e.Message);
                    Debug.WriteLine(e.StackTrace);
                    Debug.WriteLine(e);
                    args.Request.Sample = null;
                    Stop();
                    Cleanup();
                }
            }
            else
            {
                args.Request.Sample = null;
                Stop();
                Cleanup();
            }
        }
        private void OnMediaStreamSourceStarting(MediaStreamSource sender, MediaStreamSourceStartingEventArgs args)
        {
            using (var frame = WaitForNewFrame())
            {
                args.Request.SetActualStartPosition(frame.SystemRelativeTime);
            }
        }
        public void StartCapture()
        {

            _multithread = _sharpDxD3dDevice.QueryInterface<SharpDX.Direct3D11.Multithread>();
            _multithread.SetMultithreadProtected(true);
            _frameEvent = new ManualResetEvent(false);
            _closedEvent = new ManualResetEvent(false);
            _events = new[] { _closedEvent, _frameEvent };

            _captureItem.Closed += OnClosed;
            _framePool = Direct3D11CaptureFramePool.CreateFreeThreaded(
                _device,
                DirectXPixelFormat.B8G8R8A8UIntNormalized,
                1,
                _captureItem.Size);
            _framePool.FrameArrived += OnFrameArrived;
            _session = _framePool.CreateCaptureSession(_captureItem);
            _session.StartCapture();
        }
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
        private void OnClosed(GraphicsCaptureItem sender, object args)
        {
            _closedEvent.Set();
        }
        public SurfaceWithInfo WaitForNewFrame()
        {
            // Let's get a fresh one.
            _currentFrame?.Dispose();
            _frameEvent.Reset();

            var signaledEvent = _events[WaitHandle.WaitAny(_events)];
            if (signaledEvent == _closedEvent)
            {
                Cleanup();
                return null;
            }

            var result = new SurfaceWithInfo();
            result.SystemRelativeTime = _currentFrame.SystemRelativeTime;
            using (var multithreadLock = new MultithreadLock(_multithread))
            using (var sourceTexture = Direct3D11Helpers.CreateSharpDXTexture2D(_currentFrame.Surface))
            {

                _sharpDxD3dDevice.ImmediateContext.ClearRenderTargetView(_composeRenderTargetView, new SharpDX.Mathematics.Interop.RawColor4(0, 0, 0, 1));

                var width = Math.Clamp(_currentFrame.ContentSize.Width, 0, _currentFrame.Surface.Description.Width);
                var height = Math.Clamp(_currentFrame.ContentSize.Height, 0, _currentFrame.Surface.Description.Height);
                var region = new SharpDX.Direct3D11.ResourceRegion(0, 0, 0, width, height, 1);
                _sharpDxD3dDevice.ImmediateContext.CopySubresourceRegion(sourceTexture, 0, region, _composeTexture, 0);

                var description = sourceTexture.Description;
                description.Usage = SharpDX.Direct3D11.ResourceUsage.Default;
                description.BindFlags = SharpDX.Direct3D11.BindFlags.ShaderResource | SharpDX.Direct3D11.BindFlags.RenderTarget;
                description.CpuAccessFlags = SharpDX.Direct3D11.CpuAccessFlags.None;
                description.OptionFlags = SharpDX.Direct3D11.ResourceOptionFlags.None;

                using (var copyTexture = new SharpDX.Direct3D11.Texture2D(_sharpDxD3dDevice, description))
                {
                    _sharpDxD3dDevice.ImmediateContext.CopyResource(_composeTexture, copyTexture);
                    result.Surface = Direct3D11Helpers.CreateDirect3DSurfaceFromSharpDXTexture(copyTexture);
                }
            }

            return result;
        }
        private void Stop()
        {
            _closedEvent.Set();
        }
        private void Cleanup()
        {
            _framePool?.Dispose();
            _session?.Dispose();
            if (_captureItem != null)
            {
                _captureItem.Closed -= OnClosed;
            }
            _captureItem = null;
            _device = null;
            _sharpDxD3dDevice = null;
            _composeTexture?.Dispose();
            _composeTexture = null;
            _composeRenderTargetView?.Dispose();
            _composeRenderTargetView = null;
            _currentFrame?.Dispose();
        }
    }


    public class MultithreadLock : IDisposable
    {
        public MultithreadLock(SharpDX.Direct3D11.Multithread multithread)
        {
            _multithread = multithread;
            _multithread?.Enter();
        }

        public void Dispose()
        {
            _multithread?.Leave();
            _multithread = null;
        }

        private SharpDX.Direct3D11.Multithread _multithread;
    }
    public sealed class SurfaceWithInfo : IDisposable
    {
        public IDirect3DSurface Surface { get; internal set; }
        public TimeSpan SystemRelativeTime { get; internal set; }

        public void Dispose()
        {
            Surface?.Dispose();
            Surface = null;
        }
    }
    [ComImport]
    [Guid("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    [ComVisible(true)]
    interface IDirect3DDxgiInterfaceAccess
    {
        IntPtr GetInterface([In] ref Guid iid);
    };

    public static class Direct3D11Helpers
    {
        internal static Guid IInspectable = new Guid("AF86E2E0-B12D-4c6a-9C5A-D7AA65101E90");
        internal static Guid ID3D11Resource = new Guid("dc8e63f3-d12b-4952-b47b-5e45026a862d");
        internal static Guid IDXGIAdapter3 = new Guid("645967A4-1392-4310-A798-8053CE3E93FD");
        internal static Guid ID3D11Device = new Guid("db6f6ddb-ac77-4e88-8253-819df9bbf140");
        internal static Guid ID3D11Texture2D = new Guid("6f15aaf2-d208-4e89-9ab4-489535d34f9c");

        [DllImport(
            "d3d11.dll",
            EntryPoint = "CreateDirect3D11DeviceFromDXGIDevice",
            SetLastError = true,
            CharSet = CharSet.Unicode,
            ExactSpelling = true,
            CallingConvention = CallingConvention.StdCall
            )]
        internal static extern UInt32 CreateDirect3D11DeviceFromDXGIDevice(IntPtr dxgiDevice, out IntPtr graphicsDevice);

        [DllImport(
            "d3d11.dll",
            EntryPoint = "CreateDirect3D11SurfaceFromDXGISurface",
            SetLastError = true,
            CharSet = CharSet.Unicode,
            ExactSpelling = true,
            CallingConvention = CallingConvention.StdCall
            )]
        internal static extern UInt32 CreateDirect3D11SurfaceFromDXGISurface(IntPtr dxgiSurface, out IntPtr graphicsSurface);

        public static IDirect3DDevice CreateD3DDevice()
        {
            return CreateD3DDevice(false);
        }

        public static IDirect3DDevice CreateD3DDevice(bool useWARP)
        {
            var d3dDevice = new SharpDX.Direct3D11.Device(
                useWARP ? SharpDX.Direct3D.DriverType.Software : SharpDX.Direct3D.DriverType.Hardware,
                SharpDX.Direct3D11.DeviceCreationFlags.BgraSupport);
            IDirect3DDevice device = null;

            // Acquire the DXGI interface for the Direct3D device.
            using (var dxgiDevice = d3dDevice.QueryInterface<SharpDX.DXGI.Device3>())
            {
                // Wrap the native device using a WinRT interop object.
                uint hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.NativePointer, out IntPtr pUnknown);

                if (hr == 0)
                {
                    device = Marshal.GetObjectForIUnknown(pUnknown) as IDirect3DDevice;
                    Marshal.Release(pUnknown);
                }
            }

            return device;
        }


        internal static IDirect3DSurface CreateDirect3DSurfaceFromSharpDXTexture(SharpDX.Direct3D11.Texture2D texture)
        {
            IDirect3DSurface surface = null;

            // Acquire the DXGI interface for the Direct3D surface.
            using (var dxgiSurface = texture.QueryInterface<SharpDX.DXGI.Surface>())
            {
                // Wrap the native device using a WinRT interop object.
                uint hr = CreateDirect3D11SurfaceFromDXGISurface(dxgiSurface.NativePointer, out IntPtr pUnknown);

                if (hr == 0)
                {
                    surface = Marshal.GetObjectForIUnknown(pUnknown) as IDirect3DSurface;
                    Marshal.Release(pUnknown);
                }
            }

            return surface;
        }



        internal static SharpDX.Direct3D11.Device CreateSharpDXDevice(IDirect3DDevice device)
        {
            var access = (IDirect3DDxgiInterfaceAccess)device;
            var d3dPointer = access.GetInterface(ID3D11Device);
            var d3dDevice = new SharpDX.Direct3D11.Device(d3dPointer);
            return d3dDevice;
        }

        internal static SharpDX.Direct3D11.Texture2D CreateSharpDXTexture2D(IDirect3DSurface surface)
        {
            var access = (IDirect3DDxgiInterfaceAccess)surface;
            var d3dPointer = access.GetInterface(ID3D11Texture2D);
            var d3dSurface = new SharpDX.Direct3D11.Texture2D(d3dPointer);
            return d3dSurface;
        }


        public static SharpDX.Direct3D11.Texture2D InitializeComposeTexture(
            SharpDX.Direct3D11.Device sharpDxD3dDevice,
            SizeInt32 size)
        {
            var description = new SharpDX.Direct3D11.Texture2DDescription
            {
                Width = size.Width,
                Height = size.Height,
                MipLevels = 1,
                ArraySize = 1,
                Format = SharpDX.DXGI.Format.B8G8R8A8_UNorm,
                SampleDescription = new SharpDX.DXGI.SampleDescription()
                {
                    Count = 1,
                    Quality = 0
                },
                Usage = SharpDX.Direct3D11.ResourceUsage.Default,
                BindFlags = SharpDX.Direct3D11.BindFlags.ShaderResource | SharpDX.Direct3D11.BindFlags.RenderTarget,
                CpuAccessFlags = SharpDX.Direct3D11.CpuAccessFlags.None,
                OptionFlags = SharpDX.Direct3D11.ResourceOptionFlags.None
            };
            var composeTexture = new SharpDX.Direct3D11.Texture2D(sharpDxD3dDevice, description);


            using (var renderTargetView = new SharpDX.Direct3D11.RenderTargetView(sharpDxD3dDevice, composeTexture))
            {
                sharpDxD3dDevice.ImmediateContext.ClearRenderTargetView(renderTargetView, new SharpDX.Mathematics.Interop.RawColor4(0, 0, 0, 1));
            }

            return composeTexture;
        }
    }
}
