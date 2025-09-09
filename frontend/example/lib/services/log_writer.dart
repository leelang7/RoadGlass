import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

class LogWriter {
  IOSink? _sink;
  String? _path;

  Future<void> openJsonl(String fileName) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/$fileName');
      await file.parent.create(recursive: true);
      _sink = file.openWrite(mode: FileMode.append);
      _path = file.path;
      debugPrint('[LOG] Opened $_path');
    } catch (e) {
      debugPrint('[LOG][ERR] openJsonl failed: $e');
      rethrow;
    }
  }

  void write(Map<String, dynamic> record) {
    final s = _sink;
    if (s == null) {
      debugPrint('[LOG][WARN] write called but sink==null (path=$_path)');
      return;
    }
    try {
      s.writeln(jsonEncode(record));
      s.flush();
    } catch (e) {
      debugPrint('[LOG][ERR] write failed: $e');
    }
  }

  Future<void> close() async {
    try {
      await _sink?.flush();
      await _sink?.close();
      debugPrint('[LOG] Closed $_path');
    } catch (e) {
      debugPrint('[LOG][ERR] close failed: $e');
    } finally {
      _sink = null;
    }
  }
}