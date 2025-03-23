import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';
import 'dart:async';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  MyApp({required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: CameraScreen(cameras: cameras),
    );
  }
}

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  CameraScreen({required this.cameras});

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  Interpreter? _interpreter;
  List<Map<String, dynamic>> _recognitions = [];
  bool isProcessing = false;
  List<String> _labels = [];
  int _selectedCameraIndex = 0;

  @override
  void initState() {
    super.initState();
    initCamera(_selectedCameraIndex);
    loadModel();
    loadLabels();
  }

  Future<void> loadLabels() async {
    final labelsData = await DefaultAssetBundle.of(
      context,
    ).loadString('assets/labels.txt');
    _labels = labelsData.split('\n');
  }

  Future<void> initCamera(int cameraIndex) async {
    if (_controller != null) {
      await _controller!.dispose();
    }

    if (widget.cameras.isNotEmpty && cameraIndex < widget.cameras.length) {
      _controller = CameraController(
        widget.cameras[cameraIndex],
        ResolutionPreset.medium,
      );

      try {
        await _controller!.initialize();
        if (!mounted) return;
        setState(() {});
        _controller!.startImageStream((CameraImage image) {
          if (!isProcessing) {
            isProcessing = true;
            runModelOnFrame(image).then((_) => isProcessing = false);
          }
        });
      } catch (e) {
        print('Error initializing camera: $e');
        // If there's an error with this camera, try to use the next one
        if (cameraIndex + 1 < widget.cameras.length) {
          initCamera(cameraIndex + 1);
        }
      }
    }
  }

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset(
      "assets/mobilenet_v1_1.0_224.tflite",
    );
  }

  Future<void> runModelOnFrame(CameraImage image) async {
    if (_interpreter == null) return;

    // Convert CameraImage to Image
    img.Image imgData = convertCameraImage(image);

    // Resize image to 224x224
    img.Image resized = img.copyResize(imgData, width: 224, height: 224);

    // Normalize pixel values
    List<List<List<List<double>>>> input = preprocessImage(resized);

    // Model output
    List<List<double>> output = List.generate(1, (_) => List.filled(1001, 0.0));

    _interpreter!.run(input, output);

    // Get highest confidence labels
    List<Map<String, dynamic>> results = parseOutput(output[0]);

    setState(() {
      _recognitions = results;
    });
  }

  img.Image convertCameraImage(CameraImage image) {
    // Convert YUV420 format to RGB
    final int width = image.width;
    final int height = image.height;

    final img.Image imgData = img.Image(width, height);

    final yPlane = image.planes[0].bytes;
    final uPlane = image.planes[1].bytes;
    final vPlane = image.planes[2].bytes;

    final yRowStride = image.planes[0].bytesPerRow;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel!;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int yIndex = y * yRowStride + x;

        // uvRow and uvColumn are calculated based on downsampled size
        final int uvRow = y ~/ 2;
        final int uvColumn = x ~/ 2;

        // The u and v values are interleaved in the UV planes, with a pixel stride
        final int uvIndex = uvRow * uvRowStride + uvColumn * uvPixelStride;

        final int yValue = yPlane[yIndex];
        final int uValue = uPlane[uvIndex];
        final int vValue = vPlane[uvIndex];

        // YUV to RGB conversion
        int r = (yValue + 1.402 * (vValue - 128)).round().clamp(0, 255);
        int g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128))
            .round()
            .clamp(0, 255);
        int b = (yValue + 1.772 * (uValue - 128)).round().clamp(0, 255);

        imgData.setPixel(x, y, img.getColor(r, g, b));
      }
    }
    return imgData;
  }

  List<List<List<List<double>>>> preprocessImage(img.Image image) {
    List<List<List<List<double>>>> input = List.generate(
      1,
      (_) => List.generate(
        224,
        (_) => List.generate(224, (_) => List.filled(3, 0.0)),
      ),
    );

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = image.getPixel(x, y);
        input[0][y][x][0] = img.getRed(pixel) / 255.0;
        input[0][y][x][1] = img.getGreen(pixel) / 255.0;
        input[0][y][x][2] = img.getBlue(pixel) / 255.0;
      }
    }
    return input;
  }

  List<Map<String, dynamic>> parseOutput(List<double> output) {
    List<Map<String, dynamic>> results = [];
    for (int i = 0; i < output.length; i++) {
      if (output[i] > 0.2) {
        // Confidence threshold
        String label = i < _labels.length ? _labels[i] : "Unknown";
        results.add({"label": label, "confidence": output[i]});
      }
    }
    results.sort((a, b) => b["confidence"].compareTo(a["confidence"]));
    return results.take(5).toList();
  }

  void _switchCamera() {
    setState(() {
      _selectedCameraIndex = (_selectedCameraIndex + 1) % widget.cameras.length;
    });
    initCamera(_selectedCameraIndex);
  }

  @override
  Widget build(BuildContext context) {
    final isTablet = MediaQuery.of(context).size.shortestSide >= 600;
    final orientation = MediaQuery.of(context).orientation;

    return Scaffold(
      backgroundColor: Colors.black,
      body:
          _controller == null || !_controller!.value.isInitialized
              ? Center(child: CircularProgressIndicator())
              : OrientationBuilder(
                builder: (context, orientation) {
                  return isTablet && orientation == Orientation.landscape
                      ? _buildTabletLandscapeLayout()
                      : _buildPhoneLayout();
                },
              ),
    );
  }

  Widget _buildPhoneLayout() {
    return Stack(
      children: [
        Positioned.fill(child: CameraPreview(_controller!)),
        Positioned(top: 40, right: 20, child: _buildCameraSwitchButton()),
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: _buildRecognitionsPanel(),
        ),
      ],
    );
  }

  Widget _buildTabletLandscapeLayout() {
    return Row(
      children: [
        // Camera preview takes 2/3 of the screen in landscape
        Expanded(
          flex: 2,
          child: Stack(
            children: [
              Positioned.fill(child: CameraPreview(_controller!)),
              Positioned(top: 40, right: 20, child: _buildCameraSwitchButton()),
            ],
          ),
        ),
        // Recognition panel takes 1/3 of the screen
        Expanded(
          flex: 1,
          child: Container(
            color: Colors.black,
            child: Padding(
              padding: EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        "Object Recognition",
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        "Camera ${_selectedCameraIndex + 1}/${widget.cameras.length}",
                        style: TextStyle(color: Colors.white60, fontSize: 14),
                      ),
                    ],
                  ),
                  SizedBox(height: 20),
                  Expanded(
                    child:
                        _recognitions.isEmpty
                            ? Center(
                              child: Text(
                                "No objects detected",
                                style: TextStyle(
                                  color: Colors.white70,
                                  fontSize: 18,
                                ),
                              ),
                            )
                            : ListView.builder(
                              itemCount: _recognitions.length,
                              itemBuilder: (context, index) {
                                final rec = _recognitions[index];
                                return _buildTabletRecognitionItem(
                                  rec["label"],
                                  (rec["confidence"] * 100).toInt(),
                                );
                              },
                            ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildRecognitionsPanel() {
    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.8),
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            "Recognitions",
            style: TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          SizedBox(height: 8),
          if (_recognitions.isNotEmpty)
            for (var rec in _recognitions)
              _buildRecognition(
                rec["label"],
                (rec["confidence"] * 100).toInt(),
              ),
        ],
      ),
    );
  }

  Widget _buildTabletRecognitionItem(String label, int confidence) {
    return Container(
      margin: EdgeInsets.symmetric(vertical: 8),
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black45,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey.shade800),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.w500,
            ),
          ),
          SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                flex: 3,
                child: LinearProgressIndicator(
                  value: confidence / 100,
                  backgroundColor: Colors.grey.shade800,
                  color: _getConfidenceColor(confidence),
                  minHeight: 10,
                  borderRadius: BorderRadius.circular(5),
                ),
              ),
              SizedBox(width: 12),
              Text(
                "$confidence%",
                style: TextStyle(
                  color: _getConfidenceColor(confidence),
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Color _getConfidenceColor(int confidence) {
    if (confidence > 85) return Colors.green;
    if (confidence > 60) return Colors.lightGreen;
    if (confidence > 40) return Colors.amber;
    return Colors.orange;
  }

  Widget _buildRecognition(String label, int confidence) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: TextStyle(color: Colors.white, fontSize: 16),
            ),
          ),
          Expanded(
            flex: 2,
            child: LinearProgressIndicator(
              value: confidence / 100,
              backgroundColor: Colors.grey,
              color: _getConfidenceColor(confidence),
            ),
          ),
          SizedBox(width: 8),
          Text(
            "$confidence%",
            style: TextStyle(color: Colors.white, fontSize: 16),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraSwitchButton() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(50),
      ),
      child: IconButton(
        icon: Icon(Icons.switch_camera, color: Colors.white, size: 28),
        onPressed: widget.cameras.length > 1 ? _switchCamera : null,
        tooltip: 'Switch Camera',
        padding: EdgeInsets.all(12),
      ),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    _interpreter?.close();
    super.dispose();
  }
}
