import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:async';
import 'dart:convert'; // add at top
import 'dart:ui' as ui;
import 'package:http/http.dart' as http;
import 'package:geolocator/geolocator.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/scheduler.dart' show Ticker;
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import '../../services/imu_manager.dart';
import '../../services/log_writer.dart'; 

// (프로젝트에 이미 있다면 유지) 모델 타입 커스텀 enum을 쓰지 않고 detect 고정으로 갑니다.
// import '../../models/model_type.dart';  // ❌ 불필요

class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() => _CameraInferenceScreenState();
}

class _CameraInferenceScreenState extends State<CameraInferenceScreen> with SingleTickerProviderStateMixin {
  // Backend base URL (unified)
  static const String kBaseUrl = 'http://ec2-54-67-48-0.us-west-1.compute.amazonaws.com';
  static const String kInferPath = '/lane_wear_infer';
  static const String kHealthPath = '/health';
  final _yoloController = YOLOViewController();
  final _imu = ImuManager();
  final _log = LogWriter();      // 선택
  final GlobalKey _yoloKey = GlobalKey();
  bool _logging = false;         // 선택
  double _uiFps = 0.0;
  double _engineFps = 0.0;
  int _lastDetLogUs = 0; // 마지막 onResult 로그 시각

  // ===== Wear (마모도) 임계값 / 설정 =====
  double _wearThreshold = 0.60; // 0~1 사이 점수, 이 이상이면 '마모 심함'으로 간주
  // 각 구성요소 가중치 (합이 1.0 근처면 이해 쉬움)
  double _wContinuity = 0.45;   // 점/세그먼트 간 간격 기반 연속성 결손
  double _wRoughness  = 0.35;   // 각도 변화 분산 기반 가장자리 거칠기
  double _wThinness   = 0.20;   // bbox/폴리곤 치수 기반 '얇아짐' 정도

  String? _modelPath;
  bool _isModelLoading = true;
  String _loadingMessage = 'Loading model...';
  bool _yoloViewInitialized = false;

  // FPS 계산 및 표시
  int _frameCount = 0;
  DateTime _lastFpsUpdate = DateTime.now();
  double _currentFps = 0.0;
  DateTime? _lastFrameTime;   // fallback: previous frame time
  double _emaFps = 0.0;       // fallback: smoothed FPS
  // UI-frame-based FPS fallback (when engine metrics are unavailable)
  late final Ticker _uiTicker;
  int _uiFrameCount = 0;
  DateTime _lastUiFpsTs = DateTime.now();
  DateTime? _lastEnqueueAt;
  static const Duration _minEnqueueInterval = Duration(seconds: 2);
  static const int _maxQueueItems = 1000; // PNG count upper bound
  static const double _maxQueueMB = 500.0; // total size upper bound
  
  // FPS 히스토리 (최근 10개 프레임)
  final List<double> _fpsHistory = [];
  static const int _maxHistorySize = 10;

  // 세그멘테이션 결과를 오버레이에 그리기 위해 저장
  List<YOLOResult> _lastSegResults = [];
  Map<String, dynamic>? _lastImu;
  Timer? _imuTicker;
  Timer? _syncTimer; // offline 큐 동기화 타이머

  // YOLOView 인스턴스 캐시
  Widget? _yoloView;

  // One-time debug dump flag (replaces the invalid local static var)
  bool _dumpedFirstResult = false;

  

  // === Wear score helpers ===
  // 다각선 길이 (정규화 좌표 기준)
  double _polylineLength(List pts) {
    double len = 0.0;
    for (int i = 1; i < pts.length; i++) {
      final p0 = pts[i - 1];
      final p1 = pts[i];
      final x0 = (p0 is List) ? (p0[0] as num).toDouble() : (p0['x'] as num).toDouble();
      final y0 = (p0 is List) ? (p0[1] as num).toDouble() : (p0['y'] as num).toDouble();
      final x1 = (p1 is List) ? (p1[0] as num).toDouble() : (p1['x'] as num).toDouble();
      final y1 = (p1 is List) ? (p1[1] as num).toDouble() : (p1['y'] as num).toDouble();
      final dx = x1 - x0, dy = y1 - y0;
      len += math.sqrt(dx * dx + dy * dy);
    }
    return len;
  }

  // 간격 기반 연속성 결손(0=좋음, 커질수록 안좋음). threshold 이상 벌어진 간격의 비율
  double _gapContinuityScore(List pts, {double gapThresh = 0.02}) {
    if (pts.length < 2) return 1.0; // 점이 너무 적으면 연속성 나쁨으로 간주
    double gaps = 0.0;
    double total = 0.0;
    for (int i = 1; i < pts.length; i++) {
      final p0 = pts[i - 1];
      final p1 = pts[i];
      final x0 = (p0 is List) ? (p0[0] as num).toDouble() : (p0['x'] as num).toDouble();
      final y0 = (p0 is List) ? (p0[1] as num).toDouble() : (p0['y'] as num).toDouble();
      final x1 = (p1 is List) ? (p1[0] as num).toDouble() : (p1['x'] as num).toDouble();
      final y1 = (p1 is List) ? (p1[1] as num).toDouble() : (p1['y'] as num).toDouble();
      final d = math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
      total += d;
      if (d > gapThresh) gaps += (d - gapThresh);
    }
    if (total <= 1e-6) return 1.0;
    double s = (gaps / total).clamp(0.0, 1.0);
    return s;
  }

  // 각도 변화 분산(0=부드러움, 커질수록 거칠다=마모) → 0..1로 정규화
  double _angleRoughnessScore(List pts) {
    if (pts.length < 3) return 1.0;
    final angles = <double>[];
    for (int i = 2; i < pts.length; i++) {
      double x0, y0, x1, y1, x2, y2;
      final p0 = pts[i - 2];
      final p1 = pts[i - 1];
      final p2 = pts[i];
      x0 = (p0 is List) ? (p0[0] as num).toDouble() : (p0['x'] as num).toDouble();
      y0 = (p0 is List) ? (p0[1] as num).toDouble() : (p0['y'] as num).toDouble();
      x1 = (p1 is List) ? (p1[0] as num).toDouble() : (p1['x'] as num).toDouble();
      y1 = (p1 is List) ? (p1[1] as num).toDouble() : (p1['y'] as num).toDouble();
      x2 = (p2 is List) ? (p2[0] as num).toDouble() : (p2['x'] as num).toDouble();
      y2 = (p2 is List) ? (p2[1] as num).toDouble() : (p2['y'] as num).toDouble();
      final v1x = x1 - x0, v1y = y1 - y0;
      final v2x = x2 - x1, v2y = y2 - y1;
      final dot = v1x * v2x + v1y * v2y;
      final n1 = math.sqrt(v1x * v1x + v1y * v1y);
      final n2 = math.sqrt(v2x * v2x + v2y * v2y);
      if (n1 < 1e-6 || n2 < 1e-6) continue;
      double cosA = (dot / (n1 * n2)).clamp(-1.0, 1.0);
      final a = math.acos(cosA); // 0..pi
      angles.add(a);
    }
    if (angles.isEmpty) return 1.0;
    final mean = angles.reduce((a, b) => a + b) / angles.length;
    double variance = 0.0;
    for (final ang in angles) { variance += (ang - mean) * (ang - mean); }
    variance /= angles.length; // 라디안^2
    // 대략적인 정규화: pi^2 를 상한으로 보고 0..1로 클램프
    final num normNum = (variance / (math.pi * math.pi)).clamp(0.0, 1.0);
    return normNum.toDouble();
  }

  // bbox/폴리곤으로 '얇아짐'(thinness) 정도 추정 (0=정상, 1=많이 얇아짐)
  double _thinnessScore(dynamic bbox, List? poly) {
    double ref = 0.02; // 기대 최소 두께(정규화). 프로젝트에 맞춰 조정
    double th;
    if (bbox is Map && bbox.containsKey('w') && bbox.containsKey('h')) {
      final w = (bbox['w'] as num).toDouble();
      final h = (bbox['h'] as num).toDouble();
      th = (w < h ? w : h);
    } else if (bbox is List && bbox.length >= 4) {
      final w = ((bbox[2] as num) - (bbox[0] as num)).abs().toDouble();
      final h = ((bbox[3] as num) - (bbox[1] as num)).abs().toDouble();
      th = (w < h ? w : h);
    } else if (poly != null && poly.length >= 2) {
      // 근사: 폴리라인 길이에 비해 bbox 면적이 작으면 얇다고 판단
      // 간단화를 위해 폴리라인 segment 평균 간격을 두께 근사치로 사용
      double sum = 0.0; int cnt = 0;
      for (int i = 1; i < poly.length; i++) {
        final p0 = poly[i - 1];
        final p1 = poly[i];
        final dx = ((p1 is List ? p1[0] : p1['x']) as num).toDouble() - ((p0 is List ? p0[0] : p0['x']) as num).toDouble();
        final dy = ((p1 is List ? p1[1] : p1['y']) as num).toDouble() - ((p0 is List ? p0[1] : p0['y']) as num).toDouble();
        sum += math.sqrt(dx * dx + dy * dy);
        cnt++;
      }
      th = (cnt > 0) ? (sum / cnt) : ref;
    } else {
      th = ref; // 정보 없으면 보수적으로
    }
    // 얇을수록 점수↑, ref 대비 비율로 정규화
    final s = (1.0 - (th / ref)).clamp(0.0, 1.0);
    return s;
  }

  // 단일 결과에 대한 wear score (0=정상~1=심함)
  double _wearScoreForResult(YOLOResult r) {
    final name = (r.className ?? '').toLowerCase();
    final dyn = r as dynamic;
    final poly = (dyn.polygon ?? dyn.points);
    final bbox = (dyn.bbox ?? dyn.rect ?? dyn.box);

    // 폴리곤/포인트가 있으면 연속성/거칠기 우선
    double continuity = 0.5, rough = 0.5, thin = 0.5;
    if (poly is List && poly.length >= 2) {
      continuity = _gapContinuityScore(poly);    // 0..1 (클수록 안좋음)
      rough = _angleRoughnessScore(poly);        // 0..1 (클수록 안좋음)
    }
    thin = _thinnessScore(bbox, poly);           // 0..1 (클수록 얇음)

    // 유형별 가중치 미세 조정 (필요시 클래스별 튜닝)
    double wC = _wContinuity, wR = _wRoughness, wT = _wThinness;
    if (name.contains('crosswalk') || name.contains('stop_line')) {
      // 횡단보도/정지선은 두께와 연속성이 더 중요
      wC = 0.50; wR = 0.20; wT = 0.30;
    }

    final score = (wC * continuity + wR * rough + wT * thin).clamp(0.0, 1.0);
    return score;
  }

  // 프레임 전체(다수 객체)에 대한 wear score: 가장 심한 객체 기준(max)
  double _calcWear(List<YOLOResult> results) {
    double worst = 0.0;
    for (final r in results) {
      final s = _wearScoreForResult(r);
      if (s > worst) worst = s;
    }
    return worst;
  }

  @override
  void initState() {
    super.initState();
    debugPrint("🚀 CameraInferenceScreen initState called");
    debugPrint("🚀 YOLOViewController created: $_yoloController");

    // Request location permission at startup
    _requestLocationPermission();

    _imu.start();
    // 🔄 IMU overlay updater (independent of YOLO onResult)
    _imuTicker = Timer.periodic(const Duration(milliseconds: 150), (_) {
      final nowUs = DateTime.now().microsecondsSinceEpoch;
      final s = _imu.closest(nowUs);
      if (!mounted) return;
      setState(() {
        _lastImu = s == null
            ? null
            : {
                "t_us": s.tUs,
                "acc": {"x": s.ax, "y": s.ay, "z": s.az},
                "gyro": {"x": s.gx, "y": s.gy, "z": s.gz},
                "lin_acc": {"x": s.lax, "y": s.lay, "z": s.laz},
              };
        // Call IMU check/send after updating _lastImu
        _checkImuAndSend();
      });
      if (_logging && _lastImu != null) {
        final nowUs2 = DateTime.now().microsecondsSinceEpoch;
        if (nowUs2 - _lastDetLogUs > 500000) { // 0.5초 경과
          final rec = {
            "t_us": nowUs2,
            "fps": (_currentFps > 0.1 ? _currentFps : _uiFps),
            "fps_engine": _currentFps,
            "fps_ui": _uiFps,
            "fps_ema": _emaFps,
            "imu": _lastImu,
            "results": [],
            "event": "periodic_imu"
          };
          _log.write(rec);
          _lastDetLogUs = nowUs2;
        }
      }
    });
    // FPS 상태 초기화
    _frameCount = 0;
    _lastFpsUpdate = DateTime.now();
    _currentFps = 0.0;
    _lastFrameTime = null;
    _emaFps = 0.0;
    // UI ticker to approximate preview FPS when engine metrics are missing
    _uiTicker = createTicker((_) {
      _uiFrameCount++;
      final now = DateTime.now();
      final elapsedMs = now.difference(_lastUiFpsTs).inMilliseconds;
      if (elapsedMs >= 1000) {
        final fps = (_uiFrameCount * 1000) / elapsedMs;
        _uiFrameCount = 0;
        _lastUiFpsTs = now;
        if (_currentFps <= 0.1) {
          // Only use UI FPS as fallback when engine/result FPS is missing
          if (mounted) {
            setState(() { _uiFps = fps; });
          } else {
            _uiFps = fps;
          }
        } else {
          _uiFps = fps; // keep updated for display/debug
        }
      }
    });
    _uiTicker.start();
    // Offline queue sync timer (주기적 재시도) with small random jitter
    final delay = Duration(seconds: math.Random().nextInt(5));
    Future.delayed(delay, () {
      _syncTimer = Timer.periodic(const Duration(seconds: 30), (_) {
        _trySyncPending();
      });
      _trySyncPending(); // first attempt after jitter
    });
    // 선택: 로깅 켜고 싶으면
    _logging = true;
    _log.openJsonl('run_log.jsonl');
    final tUs0 = DateTime.now().microsecondsSinceEpoch;
    _log.write({
      "t_us": tUs0,
      "event": "app_start",
      "device": Platform.isAndroid ? "android" : (Platform.isIOS ? "ios" : "other"),
      "note": "startup marker"
    });
    _lastDetLogUs = tUs0;
    _loadModel().then((_) {
      // Only create YOLOView after model is loaded
      if (mounted && _modelPath != null && _yoloView == null) {
        setState(() {
          _yoloView = YOLOView(
            controller: _yoloController,
            modelPath: _modelPath!,
            task: YOLOTask.segment, // or detect depending on model
            onResult: (results) {
              final nowUs = DateTime.now().microsecondsSinceEpoch;
              final imu = _imu.closest(nowUs);

              _log.write({
                "t_us": nowUs,
                "event": "on_result_raw",
                "result_count": results.length,
                "imu": imu == null
                    ? null
                    : {
                        "t_us": imu.tUs,
                        "acc": {"x": imu.ax, "y": imu.ay, "z": imu.az},
                        "gyro": {"x": imu.gx, "y": imu.gy, "z": imu.gz},
                        "lin_acc": {"x": imu.lax, "y": imu.lay, "z": imu.laz},
                      },
              });

              debugPrint("🎯 YOLOView onResult fired (raw) with ${results.length} results");
              _onDetectionResults(results);
            },
            onPerformanceMetrics: (metrics) {
              final val = (metrics.fps.isFinite && metrics.fps > 0) ? metrics.fps : 0.0;
              setState(() {
                _engineFps = val;
                _currentFps = (val > 0.1) ? val : (_emaFps > 0.1 ? _emaFps : _currentFps);
              });
              if (_logging) {
                _log.write({
                  "t_us": DateTime.now().microsecondsSinceEpoch,
                  "event": "perf",
                  "source": "engine",
                  "engine_fps": val
                });
              }
            },
          );
        });
      }
    });
  }

  @override
  void dispose() {
    _imu.stop();
    if (_logging) _log.close();
    _imuTicker?.cancel();
    _imuTicker = null;
    _uiTicker.stop();
    _syncTimer?.cancel();
    _syncTimer = null;
    super.dispose();
  }

  // 필요한 TFLite 파일명만 지정해서 사용하세요.
  // 예시: assets/models/base_model_float16.tflite
  String get _modelFileName => 'base_model_float16.tflite';

  // === Offline upload queue (file-based) ===
  Future<Directory> _pendingDir() async {
    final dir = await getApplicationDocumentsDirectory();
    final d = Directory('${dir.path}/pending_uploads');
    if (!await d.exists()) {
      await d.create(recursive: true);
    }
    return d;
  }

  Future<void> _trimQueueIfNeeded() async {
    try {
      final dir = await _pendingDir();
      final files = await dir
          .list()
          .where((e) => e is File && (e.path.endsWith('.png') || e.path.endsWith('.json')))
          .cast<File>()
          .toList();
      if (files.isEmpty) return;
      files.sort((a, b) => a.lastModifiedSync().compareTo(b.lastModifiedSync())); // oldest first
      final pngCount = files.where((f) => f.path.endsWith('.png')).length;
      final totalMB = files.fold<int>(0, (acc, f) => acc + f.lengthSync()) / (1024 * 1024);
      if (pngCount <= _maxQueueItems && totalMB <= _maxQueueMB) return;

      int overItems = (pngCount - _maxQueueItems).clamp(0, pngCount);
      int deleteBudget = overItems;
      if (totalMB > _maxQueueMB) {
        deleteBudget += 50; // heuristic: extra cleanup when size too big
      }
      int deleted = 0;
      for (final f in files) {
        if (deleted >= deleteBudget) break;
        try { await f.delete(); deleted++; } catch (_) {}
      }
      debugPrint('🧹 Queue trimmed: deleted $deleted files (items=$pngCount, totalMB=${totalMB.toStringAsFixed(1)})');
    } catch (e) {
      debugPrint('🧹 Queue trim failed: $e');
    }
  }

  Future<File> _policyFile() async {
    final dir = await _pendingDir();
    return File('${dir.path}/_policy.json');
  }

  Future<bool> _loadWifiOnlyPolicy() async {
    try {
      final f = await _policyFile();
      if (await f.exists()) {
        final j = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
        return (j['wifi_only'] ?? true) == true;
      }
    } catch (_) {}
    return true;
  }

  Future<bool> _isOnline() async {
    try {
      final resp = await http.get(Uri.parse('$kBaseUrl$kHealthPath')).timeout(const Duration(seconds: 2));
      return resp.statusCode == 200;
    } catch (_) { return false; }
  }

  Future<void> _enqueueUpload({required Uint8List pngBytes, required Position pos, required String deviceId, required String trigger}) async {
    final dir = await _pendingDir();
    final ts = DateTime.now();
    final safeTs = ts.toIso8601String().replaceAll(':', '-');
    final base = 'item_${deviceId}_$safeTs';
    final imgFile = File('${dir.path}/$base.png');
    final metaFile = File('${dir.path}/$base.json');

    final meta = {
      "t_us": ts.microsecondsSinceEpoch,
      "device_id": deviceId,
      "gps": {
        "lat": pos.latitude,
        "lon": pos.longitude,
        "accuracy": pos.accuracy,
        "speed_kmh": ((pos.speed.isNaN || pos.speed.isInfinite) ? 0.0 : pos.speed * 3.6),
      },
      "trigger": trigger,
    };

    await imgFile.writeAsBytes(pngBytes, flush: true);
    await metaFile.writeAsString(jsonEncode(meta), flush: true);
    await _trimQueueIfNeeded();
    debugPrint('📦 Enqueued offline upload: ${imgFile.path}');
  }

  Future<bool> _trySyncPending() async {
    try {
      final wifiOnly = await _loadWifiOnlyPolicy();
      final online = await _isOnline();
      if (wifiOnly && !online) {
        debugPrint('📶 Wi‑Fi only policy active and offline → skip sync');
        return false;
      }
      final dir = await _pendingDir();
      final entries = await dir.list().toList();
      final pngs = entries.whereType<File>().where((f) => f.path.endsWith('.png')).toList();
      int success = 0;
      for (final png in pngs) {
        final metaPath = png.path.replaceAll('.png', '.json');
        final metaFile = File(metaPath);
        if (!await metaFile.exists()) continue;
        Map<String, dynamic> meta;
        try {
          meta = jsonDecode(await metaFile.readAsString()) as Map<String, dynamic>;
        } catch (e) {
          debugPrint('⚠️ Meta parse failed for ${metaFile.path}: $e');
          continue;
        }
        final bytes = await png.readAsBytes();

        if (bytes.length > 20 * 1024 * 1024) {
          debugPrint('🚫 Skip upload (>20MB): ${png.path}');
          continue;
        }
        final request = http.MultipartRequest('POST', Uri.parse('$kBaseUrl$kInferPath'));
        final fname = png.uri.pathSegments.last;
        request.files.add(http.MultipartFile.fromBytes('file', bytes, filename: fname));

        final gps = (meta['gps'] as Map?) ?? {};
        request.fields['gps_lat'] = (gps['lat'] ?? '').toString();
        request.fields['gps_lon'] = (gps['lon'] ?? '').toString();
        request.fields['timestamp'] = DateTime.now().toIso8601String();
        request.fields['device_id'] = (meta['device_id'] ?? 'unknown').toString();

        try {
          final resp = await request.send().timeout(const Duration(seconds: 12));
          String body = '';
          try { body = await resp.stream.bytesToString(); } catch (_) {}
          if (resp.statusCode == 200) {
            success++;
            try {
              if (body.isNotEmpty) {
                final j = jsonDecode(body) as Map<String, dynamic>;
                final overlayUrl = j['overlay_url'] as String?;
                if (overlayUrl != null && overlayUrl.isNotEmpty) {
                  final ovResp = await http.get(Uri.parse(overlayUrl)).timeout(const Duration(seconds: 3));
                  if (ovResp.statusCode == 200 && ovResp.bodyBytes.isNotEmpty) {
                    final ovPath = png.path.replaceAll('.png', '_overlay.jpg');
                    await File(ovPath).writeAsBytes(ovResp.bodyBytes, flush: true);
                  }
                }
              }
            } catch (_) {}
            await png.delete();
            await metaFile.delete();
          } else if (resp.statusCode == 413) {
            debugPrint('🚫 413 Payload Too Large');
          } else if (resp.statusCode == 415) {
            debugPrint('🚫 415 Unsupported Media Type');
          } else if (resp.statusCode == 400) {
            debugPrint('🚫 400 Bad Request: $body');
          } else {
            debugPrint('❌ Sync failed (${resp.statusCode}): $body');
          }
        } catch (e) {
          debugPrint('📶 Offline: sync attempt failed for ${png.path}: $e');
        }
      }
      debugPrint('🔁 Sync complete: $success item(s) uploaded');
      await _trimQueueIfNeeded();
      return success > 0;
    } catch (e) {
      debugPrint('⚠️ Sync error: $e');
      return false;
    }
  }

  Future<void> _checkImuAndSend() async {
    if (_lastImu == null) return;

    // 먼저 GPS 위치(속도 포함) 가져오기
    Position? pos;
    try {
      pos = await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high);
    } catch (e) {
      debugPrint("⚠️ GPS position fetch failed: $e");
      return;
    }

    // 속도 체크 (단위: m/s → km/h)
    double speedKmh = (pos.speed.isNaN || pos.speed.isInfinite) ? 0.0 : pos.speed * 3.6;
    if (speedKmh < 15.0 || speedKmh > 50.0) {
      debugPrint("⏩ Skip send: speed $speedKmh km/h not in [15, 50]");
      return;
    }

    // IMU zAcc 체크
    final double zAcc = (_lastImu?["acc"]["z"] as num).toDouble();
    if (zAcc.abs() < 3.0) {
      debugPrint("⏩ Skip send: zAcc $zAcc < 3.0");
      return; // 방지턱 제외
    }

    try {
      // 1. 카메라 캡쳐 (null safety + lower pixel ratio)
      final ctx = _yoloKey.currentContext;
      if (ctx == null) {
        debugPrint('⚠️ yoloKey context is null');
        return;
      }
      final boundary = ctx.findRenderObject() as RenderRepaintBoundary?;
      if (boundary == null) {
        debugPrint('⚠️ RepaintBoundary not found');
        return;
      }

      ui.Image? image;
      Uint8List pngBytes;
      int capW = 0, capH = 0;
      try {
        image = await boundary.toImage(pixelRatio: 1.5);
        capW = image.width;
        capH = image.height;
        final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
        if (byteData == null) {
          debugPrint('⚠️ Failed to get image bytes');
          return;
        }
        pngBytes = byteData.buffer.asUint8List();
      } finally {
        try { image?.dispose(); } catch (_) {}
      }

      // Downscale to max 1280px (longest side)
      Uint8List resizedBytes = pngBytes;
      try {
        final codec = await ui.instantiateImageCodec(
          pngBytes,
          targetWidth: (capW >= capH && capW > 0) ? 1280 : null,
          targetHeight: (capH > capW && capH > 0) ? 1280 : null,
        );
        final fi = await codec.getNextFrame();
        final rb = await fi.image.toByteData(format: ui.ImageByteFormat.png);
        if (rb != null) resizedBytes = rb.buffer.asUint8List();
      } catch (_) {}

      // Upload throttle (min interval)
      final now = DateTime.now();
      if (_lastEnqueueAt != null && now.difference(_lastEnqueueAt!) < _minEnqueueInterval) {
        debugPrint('⏸️ Cooldown: skip enqueue');
        return;
      }

      // 3. Device ID 가져오기
      final deviceInfo = DeviceInfoPlugin();
      String deviceId = "unknown";
      if (Platform.isAndroid) {
        final androidInfo = await deviceInfo.androidInfo;
        deviceId = androidInfo.id;
      } else if (Platform.isIOS) {
        final iosInfo = await deviceInfo.iosInfo;
        deviceId = iosInfo.identifierForVendor ?? "ios-unknown";
      }

      // 5. Offline-first: 큐에 저장 후 백그라운드 동기화 시도
      await _enqueueUpload(
        pngBytes: resizedBytes,
        pos: pos,
        deviceId: deviceId,
        trigger: 'z_acc_threshold',
      );
      _lastEnqueueAt = now;
      await _trySyncPending();
    } catch (e) {
      debugPrint("⚠️ Capture/Upload failed: $e");
    }
  }

  Future<void> _loadModel() async {
    setState(() {
      _isModelLoading = true;
      _loadingMessage = 'Loading model...';
    });

    try {
      final ByteData data = await rootBundle.load('assets/models/$_modelFileName');

      final Directory appDir = await getApplicationDocumentsDirectory();
      final Directory modelDir = Directory('${appDir.path}/assets/models');
      if (!await modelDir.exists()) {
        await modelDir.create(recursive: true);
      }

      final File file = File('${modelDir.path}/$_modelFileName');
      if (!await file.exists()) {
        await file.writeAsBytes(data.buffer.asUint8List());
      }

      if (!mounted) return;
      setState(() {
        _modelPath = file.path;
        _isModelLoading = false;
        _loadingMessage = '';
      });

      // 기본 임계치 세팅(조금 더 현실적인 값으로 조정)
      _yoloController.setThresholds(
        confidenceThreshold: 0.05, // 조금 더 현실적인 값
        iouThreshold: 0.4,         // 일반적인 기본값
        numItemsThreshold: 1,
      );
      
      debugPrint("✅ YOLO model loaded successfully: $_modelPath");
      debugPrint("✅ Thresholds set: conf=0.05, iou=0.4, numItems=1");
      // YOLOView 생성은 initState에서 처리

      // YOLOView 초기화 확인 (매우 긴 대기 시간)
      Future.delayed(const Duration(milliseconds: 5000), () {
        if (mounted) {
          debugPrint("🔄 Checking YOLOView status after 5 seconds...");
          debugPrint("🔄 Current FPS: $_currentFps, EMA FPS: $_emaFps");
          debugPrint("🔄 YOLOView should be running now");
          debugPrint("🔄 YOLOController: $_yoloController");
          debugPrint("🔄 Model path: $_modelPath");
          
          setState(() {
            _yoloViewInitialized = true;
          });
        }
      });
      // 5초 후에도 결과가 없으면 경고
      Future.delayed(const Duration(seconds: 5), () {
        if (mounted) {
          debugPrint("⚠️ 5초 후에도 YOLO 결과가 없습니다. 모델이나 설정을 확인해주세요.");
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isModelLoading = false;
        _loadingMessage = 'Failed to load model';
      });
      debugPrint('Model load error: $e');
      // 최소 화면만 요구하셨으므로 다이얼로그는 생략
    }
  }

  // === JSON-safe serializers for YOLO outputs ===
  Map<String, double>? _encodeBbox(dynamic bbox) {
    if (bbox == null) return null;
    try {
      if (bbox is Map) {
        if (bbox.containsKey('x') && bbox.containsKey('y') &&
            bbox.containsKey('w') && bbox.containsKey('h')) {
          final x = (bbox['x'] as num).toDouble();
          final y = (bbox['y'] as num).toDouble();
          final w = (bbox['w'] as num).toDouble();
          final h = (bbox['h'] as num).toDouble();
          if ([x,y,w,h].any((v) => v.isNaN || v.isInfinite)) return null;
          return {"x": x, "y": y, "w": w, "h": h};
        }
        if (bbox.containsKey('x1') && bbox.containsKey('y1') &&
            bbox.containsKey('x2') && bbox.containsKey('y2')) {
          final x1 = (bbox['x1'] as num).toDouble();
          final y1 = (bbox['y1'] as num).toDouble();
          final x2 = (bbox['x2'] as num).toDouble();
          final y2 = (bbox['y2'] as num).toDouble();
          if ([x1,y1,x2,y2].any((v) => v.isNaN || v.isInfinite)) return null;
          return {"x": x1, "y": y1, "w": (x2 - x1).abs(), "h": (y2 - y1).abs()};
        }
      }
      if (bbox is List && bbox.length >= 4) {
        final x1 = (bbox[0] as num).toDouble();
        final y1 = (bbox[1] as num).toDouble();
        final x2 = (bbox[2] as num).toDouble();
        final y2 = (bbox[3] as num).toDouble();
        if ([x1,y1,x2,y2].any((v) => v.isNaN || v.isInfinite)) return null;
        return {"x": x1, "y": y1, "w": (x2 - x1).abs(), "h": (y2 - y1).abs()};
      }
    } catch (_) {}
    return null;
  }

  List<List<double>>? _encodePolygon(dynamic poly) {
    if (poly == null) return null;
    try {
      if (poly is List) {
        final out = <List<double>>[];
        for (final p in poly) {
          double x, y;
          if (p is List && p.length >= 2) {
            x = (p[0] as num).toDouble();
            y = (p[1] as num).toDouble();
          } else if (p is Map) {
            x = (p['x'] as num).toDouble();
            y = (p['y'] as num).toDouble();
          } else {
            continue;
          }
          if (x.isNaN || y.isNaN || x.isInfinite || y.isInfinite) continue;
          out.add([x, y]);
        }
        if (out.isEmpty) return null;
        return out;
      }
    } catch (_) {}
    return null;
  }

  Map<String, dynamic> _encodeDetection(YOLOResult r) {
    try {
      return {
        "class": r.className ?? "unknown",
        "score": r.confidence ?? 0.0,
      };
    } catch (e) {
      debugPrint("❌ Detection encoding failed: $e");
      return {
        "class": r.className ?? "unknown",
        "score": r.confidence ?? 0.0,
        "error": "encoding_failed"
      };
    }
  }
  void _onDetectionResults(List<YOLOResult> results) {
    debugPrint("🔍 _onDetectionResults called with ${results.length} results");
    debugPrint("🔍 Results: $results");
    
    // --- 실시간 FPS 계산 ---
    final nowTs = DateTime.now();
    if (_lastFrameTime != null) {
      final dtMs = nowTs.difference(_lastFrameTime!).inMilliseconds;
      if (dtMs > 0) {
        final instFps = 1000.0 / dtMs;
        // FPS 히스토리에 추가
        _fpsHistory.add(instFps);
        if (_fpsHistory.length > _maxHistorySize) {
          _fpsHistory.removeAt(0);
        }
        // Exponential moving average to stabilize
        _emaFps = (_emaFps == 0.0) ? instFps : (_emaFps * 0.7 + instFps * 0.3);
      }
    }
    _lastFrameTime = nowTs;

    // === FPS 업데이트 (1초마다) ===
    _frameCount++;
    final now = DateTime.now();
    final elapsedMs = now.difference(_lastFpsUpdate).inMilliseconds;
    if (elapsedMs >= 1000) {
      final fps = (_frameCount * 1000) / elapsedMs;
      _frameCount = 0;
      _lastFpsUpdate = now;
      
      // FPS 히스토리에서 평균 계산 (더 안정적인 FPS)
      double avgFps = fps;
      if (_fpsHistory.isNotEmpty) {
        avgFps = _fpsHistory.reduce((a, b) => a + b) / _fpsHistory.length;
      }
      
      debugPrint("🔄 FPS Update: frameFps=$fps, avgFps=$avgFps, _currentFps=$_currentFps, _emaFps=$_emaFps");
      
      // Engine FPS가 없을 때만 fallback FPS 사용
      if (_currentFps <= 0.1) {
        if (mounted) {
          setState(() {
            _currentFps = _emaFps > 0.1 ? _emaFps : avgFps;
          });
        } else {
          _currentFps = _emaFps > 0.1 ? _emaFps : avgFps;
        }
        debugPrint("🔄 Using fallback FPS: $_currentFps");
      }
    }

    // 1) 프레임 타임스탬프
    final tUs = DateTime.now().microsecondsSinceEpoch;

    // 2) IMU에서 가장 가까운 샘플
    final imu = _imu.closest(tUs);

    // === YOLO detection 직렬화 ===
    final detections = <Map<String, dynamic>>[];
    debugPrint("🔍 Processing ${results.length} detection results");

    // --- Debug dump of raw YOLOResult objects ---
    for (int i = 0; i < results.length; i++) {
      final r = results[i];
      try {
        final rawJson = jsonEncode(r as dynamic);
        debugPrint("🟢 Raw result dump $i: $rawJson");
      } catch (e) {
        debugPrint("⚠️ Failed to jsonEncode YOLOResult $i: $e");
        debugPrint("⚠️ Fallback toString: ${r.toString()}");
      }
    }

    for (int i = 0; i < results.length; i++) {
      final r = results[i];
      try {
        debugPrint("🔍 Result $i: class=${r.className}, confidence=${r.confidence}");
        final enc = _encodeDetection(r);
        detections.add(enc);
        debugPrint("✅ Encoded result $i: $enc");
      } catch (e) {
        debugPrint("❌ Failed to encode result $i: $e");
        // 직렬화 실패하는 항목은 스킵
      }
    }
    
    debugPrint("🔍 Total detections encoded: ${detections.length}");

    // === JSON 레코드 ===
    final record = {
      "t_us": tUs,
      "event": "on_result",
      "fps": _currentFps > 0.1 ? _currentFps : _uiFps,
      "fps_engine": _currentFps,
      "fps_ui": _uiFps,
      "fps_ema": _emaFps,
      "fps_history": _fpsHistory.length > 0 ? _fpsHistory.take(5).toList() : null, // 최근 5개 FPS 값
      "imu": imu == null
          ? null
          : {
              "t_us": imu.tUs,
              "acc": {"x": imu.ax, "y": imu.ay, "z": imu.az},
              "gyro": {"x": imu.gx, "y": imu.gy, "z": imu.gz},
              "lin_acc": {"x": imu.lax, "y": imu.lay, "z": imu.laz},
            },
      // 👇 sanity fields
      "result_count": results.length,
      "classes": results.map((e) => e.className).whereType<String>().take(8).toList(),
      // 실제 직렬화된 결과
      "results": detections,
    };

    // 4) 파일로 저장
    if (_logging) {
      // detections가 비어있으면 최소 정보라도 남기기
      if (detections.isEmpty && results.isNotEmpty) {
        debugPrint("⚠️ Detections empty but results not empty, creating minimal records");
        for (final r in results) {
          detections.add({
            "class": r.className,
            "score": r.confidence,
          });
        }
        record["results"] = detections;
      }
      
      // results가 비어있어도 로그 남기기
      if (results.isEmpty) {
        debugPrint("⚠️ No detection results, but logging anyway");
        record["results"] = [];
      }
      
      debugPrint("🔍 Writing record: ${record.keys.join(', ')}");
      debugPrint("🔍 Results count in record: ${(record["results"] as List).length}");
      
      // 전체 record 저장 (results 포함)
      _log.write(record);
      _lastDetLogUs = tUs;
    }

    // 6) 화면 갱신 (기존대로)
    setState(() {
      _lastSegResults = results;
      _lastImu = record["imu"] as Map<String, dynamic>?;
    });
  }

  @override
  Widget build(BuildContext context) {
    debugPrint("🔄 Building CameraInferenceScreen - modelPath: $_modelPath, isLoading: $_isModelLoading");
    debugPrint("🔄 YOLOView will be built with controller: $_yoloController");

    // YOLOView 초기화 상태 확인
    if (_modelPath != null && !_isModelLoading) {
      debugPrint("🔄 YOLOView should be initialized now");
      debugPrint("🔄 YOLOView initialized: $_yoloViewInitialized");
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: OrientationBuilder(
        builder: (context, orientation) {
          Widget content;
          if (_modelPath != null && !_isModelLoading) {
            content = Stack(
              children: [
                // 카메라 프리뷰 전체 화면
                const Positioned.fill(child: SizedBox()),
                // YOLOView 인스턴스 사용 (초기화된 경우)
                Positioned.fill(
                  child: RepaintBoundary(
                    key: _yoloKey,
                    child: _yoloView ?? const SizedBox(),
                  ),
                ),
                //  세그 폴리곤/마스크를 그리는 오버레이
                Positioned.fill(
                  child: IgnorePointer(
                    child: CustomPaint(
                      painter: LaneOverlayPainter(results: _lastSegResults),
                    ),
                  ),
                ),
                // FPS 표시 (우상단)
                Positioned(
                  top: MediaQuery.of(context).padding.top + 12,
                  right: 16,
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
                    decoration: BoxDecoration(
                      color: Colors.black87,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.white24, width: 1),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Icon(Icons.speed, color: Colors.greenAccent, size: 16),
                            const SizedBox(width: 4),
                            Text(
                              "FPS: ${(_currentFps > 0.1 ? _currentFps : _uiFps).toStringAsFixed(1)}",
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                fontFamily: 'monospace',
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Container(
                              width: 8,
                              height: 8,
                              decoration: BoxDecoration(
                                color: _currentFps > 0.1 ? Colors.greenAccent : Colors.orange,
                                shape: BoxShape.circle,
                              ),
                            ),
                            const SizedBox(width: 4),
                            Text(
                              "Engine: ${_currentFps > 0.1 ? 'ON' : 'OFF'}",
                              style: TextStyle(
                                color: _currentFps > 0.1 ? Colors.greenAccent : Colors.orange,
                                fontSize: 12,
                                fontFamily: 'monospace',
                              ),
                            ),
                          ],
                        ),
                        if (_emaFps > 0.1 && _currentFps <= 0.1)
                          Text(
                            "EMA: ${_emaFps.toStringAsFixed(1)}",
                            style: const TextStyle(
                              color: Colors.blueAccent,
                              fontSize: 12,
                              fontFamily: 'monospace',
                            ),
                          ),
                        if (_fpsHistory.isNotEmpty) ...[
                          const SizedBox(height: 2),
                          Text(
                            "Min: ${_fpsHistory.reduce((a, b) => a < b ? a : b).toStringAsFixed(1)}",
                            style: const TextStyle(
                              color: Colors.redAccent,
                              fontSize: 10,
                              fontFamily: 'monospace',
                            ),
                          ),
                          Text(
                            "Max: ${_fpsHistory.reduce((a, b) => a > b ? a : b).toStringAsFixed(1)}",
                            style: const TextStyle(
                              color: Colors.greenAccent,
                              fontSize: 10,
                              fontFamily: 'monospace',
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
                // IMU 정보 (좌상단 - FPS 패널 아래로 이동)
                Positioned(
                  top: MediaQuery.of(context).padding.top + 120, // FPS 패널 아래로 이동
                  left: 16,
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        if (_lastImu == null)
                          const Text(
                            "IMU: -",
                            style: TextStyle(color: Colors.white, fontSize: 14),
                          )
                        else ...[
                          Text(
                            "ACC  x:${(_lastImu?["acc"]["x"] as num).toStringAsFixed(2)}  "
                            "y:${(_lastImu?["acc"]["y"] as num).toStringAsFixed(2)}  "
                            "z:${(_lastImu?["acc"]["z"] as num).toStringAsFixed(2)}",
                            style: const TextStyle(color: Colors.white, fontSize: 12, fontFamily: 'monospace'),
                          ),
                          Text(
                            "GYRO x:${(_lastImu?["gyro"]["x"] as num).toStringAsFixed(2)}  "
                            "y:${(_lastImu?["gyro"]["y"] as num).toStringAsFixed(2)}  "
                            "z:${(_lastImu?["gyro"]["z"] as num).toStringAsFixed(2)}",
                            style: const TextStyle(color: Colors.white, fontSize: 12, fontFamily: 'monospace'),
                          ),
                          Text(
                            "LIN  x:${(_lastImu?["lin_acc"]["x"] as num).toStringAsFixed(2)}  "
                            "y:${(_lastImu?["lin_acc"]["y"] as num).toStringAsFixed(2)}  "
                            "z:${(_lastImu?["lin_acc"]["z"] as num).toStringAsFixed(2)}",
                            style: const TextStyle(color: Colors.white, fontSize: 12, fontFamily: 'monospace'),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ],
            );
          } else {
            content = const Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 12),
                  Text('Loading model...', style: TextStyle(color: Colors.white70)),
                ],
              ),
            );
          }
          // If in landscape, rotate the content so it renders correctly
          if (orientation == Orientation.landscape) {
            return RotatedBox(
              quarterTurns: 1,
              child: content,
            );
          } else {
            return content;
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          final dir = await getApplicationDocumentsDirectory();
          final file = File('${dir.path}/run_log.jsonl');
          if (await file.exists()) {
            await file.delete();
            debugPrint("🗑️ run_log.jsonl deleted");
            if (context.mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('run_log.jsonl deleted')),
              );
            }
          } else {
            debugPrint("⚠️ run_log.jsonl not found");
            if (context.mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('run_log.jsonl not found')),
              );
            }
          }
        },
        backgroundColor: Colors.red,
        child: const Icon(Icons.delete),
      ),
    );
  }
}

class LaneOverlayPainter extends CustomPainter {
  LaneOverlayPainter({required this.results});
  final List<YOLOResult> results;

  Color _colorForResult(YOLOResult r) {
    // 1) Exact class name mapping first (authoritative)
    final n = (r.className ?? '').toLowerCase();
    switch (n) {
      case 'crosswalk':
        return Colors.greenAccent;
      case 'stop_line':
        return Colors.redAccent;
      case 'traffic_lane_blue_dotted':
      case 'traffic_lane_blue_solid':
        return const Color(0xFF00B2FF);
      case 'traffic_lane_white_dotted':
      case 'traffic_lane_white_solid':
        return Colors.white;
      case 'traffic_lane_yellow_dotted':
      case 'traffic_lane_yellow_solid':
        return const Color(0xFFFFD400);
    }

    // 2) Fallback by class index if provided
    int? idx;
    try { idx = (r as dynamic).classIndex as int?; } catch (_) {}
    try { idx ??= (r as dynamic).classId as int?; } catch (_) {}
    if (idx != null) {
      switch (idx) {
        case 0: return Colors.greenAccent; // crosswalk
        case 1: return Colors.redAccent;   // stop_line
        case 2:
        case 3:
          return const Color(0xFF00B2FF);
        case 4:
        case 5:
          return Colors.white;
        case 6:
        case 7:
          return const Color(0xFFFFD400);
      }
    }

    // 3) Last fallback by keyword
    if (n.contains('yellow')) return const Color(0xFFFFD400);
    if (n.contains('blue')) return const Color(0xFF00B2FF);
    if (n.contains('white')) return Colors.white;
    if (n.contains('stop_line')) return Colors.redAccent;
    if (n.contains('crosswalk')) return Colors.greenAccent;
    return Colors.white;
  }

  double _strokeFor(String? className) {
    final s = (className ?? '').toLowerCase();
    if (s.contains('stop_line')) return 8;
    if (s.contains('crosswalk')) return 7;
    return 6; // default for lanes
  }

  @override
  void paint(Canvas canvas, Size size) {
    for (final r in results) {
      final name = r.className;
      final color = _colorForResult(r);
      final strokeW = _strokeFor(name);

      final outline = Paint()
        ..color = Colors.black.withOpacity(0.75)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeW + 3.5
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;

      final paint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeW
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;

      final dyn = (r as dynamic);
      bool drawn = false;

      // 1) Prefer polygon/points if present
      try {
        final List? poly = dyn.polygon ?? dyn.points;
        if (poly is List && poly.isNotEmpty) {
          final path = Path();
          for (int i = 0; i < poly.length; i++) {
            final p = poly[i];
            double x, y;
            if (p is List && p.length >= 2) {
              x = (p[0] as num).toDouble();
              y = (p[1] as num).toDouble();
            } else if (p is Map) {
              x = (p['x'] as num).toDouble();
              y = (p['y'] as num).toDouble();
            } else {
              continue;
            }
            final dx = x * size.width;
            final dy = y * size.height;
            if (i == 0) {
              path.moveTo(dx, dy);
            } else {
              path.lineTo(dx, dy);
            }
          }
          final nn = (name ?? '').toLowerCase();
          final isDotted = nn == 'traffic_lane_blue_dotted' ||
                           nn == 'traffic_lane_white_dotted' ||
                           nn == 'traffic_lane_yellow_dotted' ||
                           nn.contains('dotted');
          final toDraw = isDotted ? _dashPath(path) : path;
          canvas.drawPath(toDraw, outline);
          canvas.drawPath(toDraw, paint);
          drawn = true;
        }
      } catch (_) {}

      if (drawn) continue;

      // 2) Mask overlay if available
      try {
        final mask = dyn.mask;
        final mw = (dyn.maskWidth as int?);
        final mh = (dyn.maskHeight as int?);
        if (mask != null && mw != null && mh != null && mask is Uint8List) {
          // Convert binary/prob mask into ImageShader
          // Since decodeImageFromPixels is async, this block must be adapted for sync paint.
          // In practice, mask overlays should be pre-decoded to Image and passed in, but for
          // demonstration, we use instantiateImageCodec for RGBA8888 mask.
          // WARNING: decodeImageFromPixels is async and can't be used directly here!
          // So we fallback to drawing a color overlay using alpha mask if available.
          // If mask is a binary mask (0/1 or 0/255), we can draw pixels manually.
          final w = mw;
          final h = mh;
          final maskBytes = mask;
          // Try to draw as alpha mask (1 byte per pixel)
          if (maskBytes.length == w * h) {
            final imgBytes = Uint8List(w * h * 4);
            for (int i = 0; i < w * h; i++) {
              final alpha = maskBytes[i];
              imgBytes[i * 4 + 0] = color.red;
              imgBytes[i * 4 + 1] = color.green;
              imgBytes[i * 4 + 2] = color.blue;
              imgBytes[i * 4 + 3] = (alpha * 0.3).toInt().clamp(0, 255); // semi-transparent
            }
            // ignore: deprecated_member_use
            final paintImage = Paint()
              ..filterQuality = FilterQuality.low
              ..isAntiAlias = false;
            // decodeImageFromPixels is async, so we cannot call it here synchronously.
            // Instead, fallback: just paint a translucent rectangle.
            canvas.drawRect(
              Rect.fromLTWH(0, 0, size.width, size.height),
              Paint()
                ..color = color.withOpacity(0.15)
                ..style = PaintingStyle.fill,
            );
            drawn = true;
          } else {
            // Fallback: just paint a translucent rectangle overlay if mask bytes are not expected shape
            canvas.drawRect(
              Rect.fromLTWH(0, 0, size.width, size.height),
              Paint()
                ..color = color.withOpacity(0.15)
                ..style = PaintingStyle.fill,
            );
            drawn = true;
          }
        }
      } catch (e) {
        debugPrint("⚠️ Mask paint failed: $e");
      }

      if (drawn) continue;

      // 3) Fallback to bbox
      try {
        final bbox = dyn.bbox ?? dyn.rect ?? dyn.box;
        if (bbox != null) {
          double x1, y1, x2, y2;
          if (bbox is Map && bbox.containsKey('x') && bbox.containsKey('y') && bbox.containsKey('w') && bbox.containsKey('h')) {
            x1 = (bbox['x'] as num).toDouble();
            y1 = (bbox['y'] as num).toDouble();
            x2 = x1 + (bbox['w'] as num).toDouble();
            y2 = y1 + (bbox['h'] as num).toDouble();
          } else if (bbox is Map && bbox.containsKey('x1')) {
            x1 = (bbox['x1'] as num).toDouble();
            y1 = (bbox['y1'] as num).toDouble();
            x2 = (bbox['x2'] as num).toDouble();
            y2 = (bbox['y2'] as num).toDouble();
          } else if (bbox is List && bbox.length >= 4) {
            x1 = (bbox[0] as num).toDouble();
            y1 = (bbox[1] as num).toDouble();
            x2 = (bbox[2] as num).toDouble();
            y2 = (bbox[3] as num).toDouble();
          } else {
            continue;
          }
          final rect = Rect.fromLTRB(x1 * size.width, y1 * size.height, x2 * size.width, y2 * size.height);
          canvas.drawRect(rect, outline..strokeWidth = strokeW + 2);
          canvas.drawRect(rect, paint..strokeWidth = strokeW);
        }
      } catch (_) {}
    }
  }

  Path _dashPath(Path source, {double dashWidth = 16, double gapWidth = 10}) {
    final Path dest = Path();
    for (final metric in source.computeMetrics()) {
      double distance = 0.0;
      while (distance < metric.length) {
        final double next = (distance + dashWidth).clamp(0, metric.length);
        dest.addPath(metric.extractPath(distance, next), Offset.zero);
        distance = next + gapWidth;
      }
    }
    return dest;
  }

  @override
  bool shouldRepaint(covariant LaneOverlayPainter old) => old.results != results;
}
  // Location permission request at startup
  Future<void> _requestLocationPermission() async {
    debugPrint("📍 Checking location permission...");
    LocationPermission permission = await Geolocator.checkPermission();
    debugPrint("📍 Initial location permission status: $permission");
    if (permission == LocationPermission.denied) {
      debugPrint("📍 Location permission denied, requesting...");
      permission = await Geolocator.requestPermission();
      debugPrint("📍 Requested location permission, new status: $permission");
    }
    if (permission == LocationPermission.deniedForever) {
      debugPrint("📍 Location permission denied forever. Opening app settings...");
      await Geolocator.openAppSettings();
      // Optionally, you can show a dialog/snackbar here
    }
    if (permission == LocationPermission.whileInUse || permission == LocationPermission.always) {
      debugPrint("📍 Location permission granted: $permission");
    } else {
      debugPrint("📍 Location permission NOT granted: $permission");
    }
  }