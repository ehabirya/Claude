// flutter_integration.dart
// Digital Twin API Integration for Flutter

import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

class DigitalTwinAPI {
  final String apiEndpoint;
  final String apiKey;

  DigitalTwinAPI({
    required this.apiEndpoint,
    required this.apiKey,
  });

  /// Convert image file to base64 string
  Future<String> imageToBase64(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    return base64Encode(bytes);
  }

  /// Analyze single photo quality
  Future<Map<String, dynamic>> analyzePhotoQuality(File photo) async {
    try {
      final photoBase64 = await imageToBase64(photo);

      final response = await http.post(
        Uri.parse(apiEndpoint),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $apiKey',
        },
        body: jsonEncode({
          'input': {
            'action': 'analyze_photo',
            'photo': photoBase64,
          }
        }),
      );

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        return result;
      } else {
        throw Exception('Failed to analyze photo: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error analyzing photo: $e');
    }
  }

  /// Generate digital twin from 3 photos
  Future<DigitalTwinResult> generateDigitalTwin({
    required File frontPhoto,
    required File sidePhoto,
    required File backPhoto,
    required BodyMeasurements measurements,
  }) async {
    try {
      // Convert photos to base64
      final photo1Base64 = await imageToBase64(frontPhoto);
      final photo2Base64 = await imageToBase64(sidePhoto);
      final photo3Base64 = await imageToBase64(backPhoto);

      // Make API request
      final response = await http.post(
        Uri.parse(apiEndpoint),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $apiKey',
        },
        body: jsonEncode({
          'input': {
            'action': 'generate_twin',
            'photos': [photo1Base64, photo2Base64, photo3Base64],
            'measurements': measurements.toJson(),
          }
        }),
      );

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);

        if (result['success'] == true) {
          return DigitalTwinResult.fromJson(result);
        } else {
          throw Exception(result['error'] ?? 'Unknown error occurred');
        }
      } else {
        throw Exception('API request failed: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error generating digital twin: $e');
    }
  }

  /// Save OBJ file to device
  Future<String> saveMeshToFile(String meshObjBase64, String filePath) async {
    try {
      final meshData = base64Decode(meshObjBase64);
      final file = File(filePath);
      await file.writeAsBytes(meshData);
      return filePath;
    } catch (e) {
      throw Exception('Error saving mesh file: $e');
    }
  }
}

/// Body measurements model
class BodyMeasurements {
  final int height; // cm
  final int weight; // kg
  final int shoulderWidth; // cm
  final int chestCircumference; // cm
  final int waistCircumference; // cm
  final int hipCircumference; // cm

  BodyMeasurements({
    required this.height,
    required this.weight,
    required this.shoulderWidth,
    required this.chestCircumference,
    required this.waistCircumference,
    required this.hipCircumference,
  });

  Map<String, dynamic> toJson() {
    return {
      'height': height,
      'weight': weight,
      'shoulderWidth': shoulderWidth,
      'chestCircumference': chestCircumference,
      'waistCircumference': waistCircumference,
      'hipCircumference': hipCircumference,
    };
  }
}

/// Digital twin result model
class DigitalTwinResult {
  final bool success;
  final String meshObjBase64;
  final String meshObj;
  final MeshStatistics statistics;

  DigitalTwinResult({
    required this.success,
    required this.meshObjBase64,
    required this.meshObj,
    required this.statistics,
  });

  factory DigitalTwinResult.fromJson(Map<String, dynamic> json) {
    return DigitalTwinResult(
      success: json['success'] ?? false,
      meshObjBase64: json['mesh_obj_base64'] ?? '',
      meshObj: json['mesh_obj'] ?? '',
      statistics: MeshStatistics.fromJson(json['statistics'] ?? {}),
    );
  }
}

/// Mesh statistics model
class MeshStatistics {
  final int numVertices;
  final int numFaces;
  final int numMatches;

  MeshStatistics({
    required this.numVertices,
    required this.numFaces,
    required this.numMatches,
  });

  factory MeshStatistics.fromJson(Map<String, dynamic> json) {
    return MeshStatistics(
      numVertices: json['num_vertices'] ?? 0,
      numFaces: json['num_faces'] ?? 0,
      numMatches: json['num_matches'] ?? 0,
    );
  }
}

// ============================================
// USAGE EXAMPLE IN FLUTTER APP
// ============================================

/*
// Add these dependencies to pubspec.yaml:
// dependencies:
//   http: ^1.1.0
//   image_picker: ^1.0.4
//   path_provider: ^2.1.1

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

class DigitalTwinScreen extends StatefulWidget {
  @override
  _DigitalTwinScreenState createState() => _DigitalTwinScreenState();
}

class _DigitalTwinScreenState extends State<DigitalTwinScreen> {
  final DigitalTwinAPI api = DigitalTwinAPI(
    apiEndpoint: 'https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync',
    apiKey: 'YOUR_API_KEY',
  );

  File? frontPhoto;
  File? sidePhoto;
  File? backPhoto;

  bool isGenerating = false;
  String? resultMessage;

  final ImagePicker _picker = ImagePicker();

  // Body measurements
  int height = 170;
  int weight = 70;
  int shoulderWidth = 40;
  int chest = 90;
  int waist = 80;
  int hips = 95;

  Future<void> pickPhoto(String position) async {
    final XFile? image = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 85,
    );

    if (image != null) {
      setState(() {
        if (position == 'front') frontPhoto = File(image.path);
        if (position == 'side') sidePhoto = File(image.path);
        if (position == 'back') backPhoto = File(image.path);
      });

      // Optional: Analyze photo quality immediately
      try {
        final quality = await api.analyzePhotoQuality(File(image.path));
        if (!quality['is_good']) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Photo quality issues: ${quality['issues'].join(', ')}'),
              backgroundColor: Colors.red,
            ),
          );
        }
      } catch (e) {
        print('Error analyzing photo: $e');
      }
    }
  }

  Future<void> generateDigitalTwin() async {
    if (frontPhoto == null || sidePhoto == null || backPhoto == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please upload all 3 photos')),
      );
      return;
    }

    setState(() {
      isGenerating = true;
      resultMessage = null;
    });

    try {
      final measurements = BodyMeasurements(
        height: height,
        weight: weight,
        shoulderWidth: shoulderWidth,
        chestCircumference: chest,
        waistCircumference: waist,
        hipCircumference: hips,
      );

      final result = await api.generateDigitalTwin(
        frontPhoto: frontPhoto!,
        sidePhoto: sidePhoto!,
        backPhoto: backPhoto!,
        measurements: measurements,
      );

      setState(() {
        resultMessage = 'Digital twin created successfully!\n'
            'Vertices: ${result.statistics.numVertices}\n'
            'Faces: ${result.statistics.numFaces}\n'
            'Matches: ${result.statistics.numMatches}';
      });

      // Save mesh to file
      final directory = await getApplicationDocumentsDirectory();
      final savedPath = await api.saveMeshToFile(
        result.meshObjBase64,
        '${directory.path}/digital_twin.obj',
      );

      print('Mesh saved to: $savedPath');

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Mesh saved to: $savedPath'),
          backgroundColor: Colors.green,
        ),
      );
    } catch (e) {
      setState(() {
        resultMessage = 'Error: $e';
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: $e'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      setState(() {
        isGenerating = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Digital Twin Generator'),
        backgroundColor: Color(0xFF667eea),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Instructions
            Container(
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(10),
                border: Border(left: BorderSide(color: Color(0xFF667eea), width: 4)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '📸 Photo Instructions',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  SizedBox(height: 8),
                  Text('• Take 3 photos: Front, Side (90°), and Back (180°)'),
                  Text('• Stand 2-3 meters from camera'),
                  Text('• Use good lighting'),
                  Text('• Wear fitted clothing'),
                ],
              ),
            ),
            SizedBox(height: 20),

            // Photo upload buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildPhotoCard('Front', frontPhoto, () => pickPhoto('front')),
                _buildPhotoCard('Side', sidePhoto, () => pickPhoto('side')),
                _buildPhotoCard('Back', backPhoto, () => pickPhoto('back')),
              ],
            ),
            SizedBox(height: 20),

            // Measurements
            Text(
              '📏 Body Measurements',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 10),
            _buildSlider('Height (cm)', height, 140, 210, (val) => setState(() => height = val.round())),
            _buildSlider('Weight (kg)', weight, 40, 150, (val) => setState(() => weight = val.round())),
            _buildSlider('Shoulder (cm)', shoulderWidth, 30, 60, (val) => setState(() => shoulderWidth = val.round())),
            _buildSlider('Chest (cm)', chest, 70, 140, (val) => setState(() => chest = val.round())),
            _buildSlider('Waist (cm)', waist, 60, 130, (val) => setState(() => waist = val.round())),
            _buildSlider('Hips (cm)', hips, 70, 140, (val) => setState(() => hips = val.round())),
            
            SizedBox(height: 20),

            // Generate button
            ElevatedButton(
              onPressed: isGenerating ? null : generateDigitalTwin,
              child: isGenerating
                  ? Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            color: Colors.white,
                            strokeWidth: 2,
                          ),
                        ),
                        SizedBox(width: 10),
                        Text('Generating...'),
                      ],
                    )
                  : Text('🎯 Generate Digital Twin'),
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                backgroundColor: Color(0xFF667eea),
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
            ),

            // Result
            if (resultMessage != null)
              Container(
                margin: EdgeInsets.only(top: 20),
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: resultMessage!.contains('Error') 
                      ? Colors.red[50] 
                      : Colors.green[50],
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                    color: resultMessage!.contains('Error') 
                        ? Colors.red 
                        : Colors.green,
                  ),
                ),
                child: Text(
                  resultMessage!,
                  style: TextStyle(
                    color: resultMessage!.contains('Error') 
                        ? Colors.red[900] 
                        : Colors.green[900],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildPhotoCard(String label, File? photo, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 100,
        height: 130,
        decoration: BoxDecoration(
          border: Border.all(
            color: photo != null ? Colors.green : Colors.grey[400]!,
            width: 2,
          ),
          borderRadius: BorderRadius.circular(10),
        ),
        child: photo != null
            ? ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(photo, fit: BoxFit.cover),
              )
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.camera_alt, size: 40, color: Colors.grey[600]),
                  SizedBox(height: 8),
                  Text(
                    label,
                    style: TextStyle(fontSize: 12, color: Colors.grey[700]),
                  ),
                ],
              ),
      ),
    );
  }

  Widget _buildSlider(String label, int value, int min, int max, Function(double) onChanged) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('$label: $value', style: TextStyle(fontWeight: FontWeight.w500)),
        Slider(
          value: value.toDouble(),
          min: min.toDouble(),
          max: max.toDouble(),
          activeColor: Color(0xFF667eea),
          onChanged: onChanged,
        ),
      ],
    );
  }
}
*/
