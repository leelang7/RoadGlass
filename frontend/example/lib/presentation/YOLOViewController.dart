import 'dart:async';
import 'package:flutter/services.dart';

class YOLOViewController {
  final int viewId;
  late final EventChannel _eventChannel;
  StreamSubscription? _subscription;

  YOLOViewController(this.viewId) {
    _eventChannel = EventChannel('yolo_event_channel_$viewId');
  }

  /// YOLO 네이티브에서 오는 결과 구독
  void listenResults(void Function(dynamic result) onData) {
    _subscription = _eventChannel.receiveBroadcastStream().listen(
      onData,
      onError: (err) {
        print("❌ YOLO event error: $err");
      },
      cancelOnError: false,
    );
  }

  /// 구독 해제
  void dispose() {
    _subscription?.cancel();
  }
}