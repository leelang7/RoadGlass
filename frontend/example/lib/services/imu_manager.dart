import 'dart:async';
import 'package:sensors_plus/sensors_plus.dart';

class ImuSample {
  ImuSample({
    required this.tUs,
    required this.ax, required this.ay, required this.az,
    required this.gx, required this.gy, required this.gz,
    required this.lax, required this.lay, required this.laz,
  });

  final int tUs;               // microseconds since epoch
  final double ax, ay, az;     // accelerometer incl. gravity (m/s^2)
  final double gx, gy, gz;     // gyroscope (rad/s)
  final double lax, lay, laz;  // linear acceleration (m/s^2)
}

class ImuManager {
  StreamSubscription<AccelerometerEvent>? _accSub;
  StreamSubscription<UserAccelerometerEvent>? _linAccSub;
  StreamSubscription<GyroscopeEvent>? _gyroSub;

  double? _ax, _ay, _az;
  double? _gx, _gy, _gz;
  double? _lax, _lay, _laz;
  int _lastTsUs = 0;
  bool _started = false;

  void start() {
    if (_started) return;
    _started = true;

    _accSub = accelerometerEvents.listen((e) {
      _ax = e.x.toDouble(); _ay = e.y.toDouble(); _az = e.z.toDouble();
      _lastTsUs = DateTime.now().microsecondsSinceEpoch;
    });

    // linear acceleration (gravity removed)
    _linAccSub = userAccelerometerEvents.listen((e) {
      _lax = e.x.toDouble(); _lay = e.y.toDouble(); _laz = e.z.toDouble();
      _lastTsUs = DateTime.now().microsecondsSinceEpoch;
    });

    _gyroSub = gyroscopeEvents.listen((e) {
      _gx = e.x.toDouble(); _gy = e.y.toDouble(); _gz = e.z.toDouble();
      _lastTsUs = DateTime.now().microsecondsSinceEpoch;
    });
  }

  void stop() {
    _accSub?.cancel(); _accSub = null;
    _linAccSub?.cancel(); _linAccSub = null;
    _gyroSub?.cancel(); _gyroSub = null;
    _started = false;
  }

  /// 최근 샘플을 반환(아직 없다면 null)
  ImuSample? closest(int _targetUs) {
    if (_ax == null && _lax == null && _gx == null) return null;
    return ImuSample(
      tUs: _lastTsUs == 0 ? DateTime.now().microsecondsSinceEpoch : _lastTsUs,
      ax: _ax ?? 0.0, ay: _ay ?? 0.0, az: _az ?? 0.0,
      gx: _gx ?? 0.0, gy: _gy ?? 0.0, gz: _gz ?? 0.0,
      lax: _lax ?? 0.0, lay: _lay ?? 0.0, laz: _laz ?? 0.0,
    );
  }
}