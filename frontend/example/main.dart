// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo_example/presentation/screens/loading_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Load Glass',
      debugShowCheckedModeBanner: false,
      home: const SingleImageExample(), // Directly launch YOLO test screen for desktop
      routes: {
      },
    );
  }
}

// Single image inference example
class SingleImageExample extends StatefulWidget {
  const SingleImageExample({super.key});

  @override
  State<SingleImageExample> createState() => _SingleImageExampleState();
}

class _SingleImageExampleState extends State<SingleImageExample> {
  YOLO? _yolo;

  @override
  void initState() {
    super.initState();
    _initializeYOLO();
  }

  Future<void> _initializeYOLO() async {
    _yolo = YOLO(modelPath: 'yolo11n', task: YOLOTask.detect);
    await _yolo!.loadModel();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('YOLO Desktop Test')),
      body: Center(
        child: YOLOView(
          modelPath: 'yolo11n',
          task: YOLOTask.detect,
          onResult: (results) {
            for (final r in results) {
              debugPrint('Result: ${r.className} (${r.confidence})');
            }
          },
        ),
      ),
    );
  }
}
