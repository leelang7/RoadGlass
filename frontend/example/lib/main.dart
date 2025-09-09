import 'package:flutter/material.dart';
import 'presentation/screens/camera_inference_screen.dart';
import 'presentation/screens/main_screen.dart';
import 'presentation/screens/loading_screen.dart';

void main() {
  runApp(const App());
}

class App extends StatelessWidget {
  const App({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Road Glass',
      theme: ThemeData(useMaterial3: true),
      debugShowCheckedModeBanner: false,
      initialRoute: '/',
      routes: {
        '/': (_) => const LoadingScreen(),
        '/main': (_) => const MainScreen(),
        '/camera': (_) => const CameraInferenceScreen(),
      },
    );
  }
}