import 'dart:io';
import 'dart:convert';
import 'package:exif/exif.dart';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:photo_manager/photo_manager.dart';
import 'package:ultralytics_yolo_example/presentation/screens/location_picker_bottom_sheet.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart' as ll;
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show Clipboard, ClipboardData, HapticFeedback;

/// 기기 갤러리에서 사진을 선택하여 업로드하는 화면
/// 기존 파일명을 유지하되, 화면 클래스명은 UploadGalleryScreen 으로 구성
class UploadGalleryScreen extends StatefulWidget {
  const UploadGalleryScreen({super.key});

  @override
  State<UploadGalleryScreen> createState() => _UploadGalleryScreenState();
}

class _UploadGalleryScreenState extends State<UploadGalleryScreen> {
  final _picker = ImagePicker();
  File? _imageFile;

  bool _isUploading = false;
  double _progress = 0.0;
  String? _serverMsg;

  String? _deviceId; // 단말 자동 식별자
  // 미리보기용 메타(선택된 사진에서 읽어온 값)
  String? _pvCapture;
  String? _pvMake;
  String? _pvModel;
  double? _pvLat;
  double? _pvLon;

  // ──(UX) 진행률/속도/ETA 표시용
  DateTime? _uploadStartAt;
  int _lastSentBytes = 0;
  double _speedKbps = 0; // 최근 계산 속도
  String? _etaText;      // 남은 시간 추정

  // ──(UX) 에러/재시도
  String? _lastErrorMsg; // 사람이 읽기 쉬운 에러 메시지

  // ──(UX) 버튼 탭 애니메이션
  double _btnScale = 1.0;

  String? _firstPrintable(Map<String, IfdTag> tags, List<String> keys) {
    for (final k in keys) {
      final v = tags[k];
      if (v != null) {
        final s = v.printable.trim();
        if (s.isNotEmpty && s.toLowerCase() != 'null') return s;
      }
    }
    return null;
  }

  Future<Map<String, dynamic>> _readExifMap(String path) async {
    try {
      final bytes = await File(path).readAsBytes();
      final tags = await readExifFromBytes(bytes);

      // 날짜(GPS DateStamp/TimeStamp 보조), 촬영일자 우선순위
      String? capture = _firstPrintable(tags, [
        'EXIF DateTimeOriginal',
        'Image DateTime',
        'EXIF DateTimeDigitized',
      ]);
      // 제조사/모델 대체 키
      final make  = _firstPrintable(tags, ['Image Make', 'Make']);
      final model = _firstPrintable(tags, ['Image Model', 'Model']);

      // 다양한 GPS 키 케이스 지원
      final latStr = _firstPrintable(tags, [
        'GPS GPSLatitude',
        'GPSLatitude',
        'GPS Latitude',
      ]);
      final latRef = _firstPrintable(tags, [
        'GPS GPSLatitudeRef',
        'GPSLatitudeRef',
        'GPS Latitude Ref',
      ]);
      final lonStr = _firstPrintable(tags, [
        'GPS GPSLongitude',
        'GPSLongitude',
        'GPS Longitude',
      ]);
      final lonRef = _firstPrintable(tags, [
        'GPS GPSLongitudeRef',
        'GPSLongitudeRef',
        'GPS Longitude Ref',
      ]);

      double? lat;
      double? lon;
      try {
        lat = _exifToDecimal(latStr, latRef);
        lon = _exifToDecimal(lonStr, lonRef);
      } catch (_) {}

      // 축약 + 원본도 함께 전달
      final raw = <String, dynamic>{};
      for (final k in tags.keys) {
        final v = tags[k];
        if (v != null) raw[k] = v.printable;
      }

      return {
        'captureDate': capture,
        'make': make,
        'model': model,
        'lat': lat,
        'lon': lon,
        'raw': raw,
      };
    } catch (e) {
      return {'error': e.toString()};
    }
  }

  double? _exifToDecimal(String? printable, String? ref) {
    if (printable == null) return null;
    // printable 예: [37, 33, 24.12] 또는 37, 33, 24/1
    final cleaned = printable
        .replaceAll('[', '')
        .replaceAll(']', '')
        .replaceAll('deg', '')
        .replaceAll('°', '')
        .replaceAll('\'', '')
        .replaceAll('"', '')
        .trim();
    final parts = cleaned.split(RegExp(r'[ ,]+')).where((e) => e.isNotEmpty).toList();
    if (parts.isEmpty) return null;

    double parseFrac(String s) {
      if (s.contains('/')) {
        final sp = s.split('/');
        final a = double.tryParse(sp[0]) ?? 0.0;
        final b = double.tryParse(sp.length > 1 ? sp[1] : '1') ?? 1.0;
        return b == 0 ? 0.0 : a / b;
      }
      return double.tryParse(s) ?? 0.0;
    }

    final deg = parseFrac(parts[0]);
    final min = parts.length > 1 ? parseFrac(parts[1]) : 0.0;
    final sec = parts.length > 2 ? parseFrac(parts[2]) : 0.0;

    double dec = deg + (min / 60.0) + (sec / 3600.0);
    final r = (ref ?? '').toUpperCase();
    if (r == 'S' || r == 'W') dec = -dec;
    return dec;
  }

  String _toIso8601FromExif(String? exifDate) {
    // exifDate 예: '2024:09:21 12:34:56' 또는 null
    if (exifDate == null || exifDate.trim().isEmpty) {
      return DateTime.now().toIso8601String();
    }
    try {
      // 시도 1: EXIF 포맷 'yyyy:MM:dd HH:mm:ss'
      final dt = DateFormat('yyyy:MM:dd HH:mm:ss').parseUtc(exifDate.replaceAll('-', ':'));
      return dt.toIso8601String();
    } catch (_) {
      try {
        // 시도 2: 일반 포맷 'yyyy-MM-dd HH:mm:ss'
        final dt = DateFormat('yyyy-MM-dd HH:mm:ss').parseUtc(exifDate);
        return dt.toIso8601String();
      } catch (_) {
        // 마지막: now
        return DateTime.now().toIso8601String();
      }
    }
  }

  // --- MediaStore(갤러리 DB)에서 좌표 보강 시도 ---
  Future<({double? lat, double? lon, DateTime? takenAt})> _findFromMediaDb(String filePath) async {
    try {
      // 권한 확인/요청
      final ps = await PhotoManager.requestPermissionExtend();
      if (!ps.isAuth) {
        return (lat: null, lon: null, takenAt: null);
      }

      final fileName = filePath.split('/').last.toLowerCase();

      // 최근 이미지 앨범 우선 탐색
      final paths = await PhotoManager.getAssetPathList(
        type: RequestType.image,
        filterOption: FilterOptionGroup(
          orders: [const OrderOption(type: OrderOptionType.createDate, asc: false)],
        ),
      );

      for (final p in paths) {
        // 최대 200개만 훑어서 매칭 (퍼포먼스/안정성 타협)
        final assets = await p.getAssetListPaged(page: 0, size: 200);
        for (final a in assets) {
          final title = (a.title ?? '').toLowerCase();
          if (title == fileName) {
            // 정확 매칭
            final lat = (a.latitude == 0.0 && a.longitude == 0.0) ? null : a.latitude;
            final lon = (a.latitude == 0.0 && a.longitude == 0.0) ? null : a.longitude;
            // 촬영/생성 시각
            final taken = a.createDateTime;
            return (lat: lat, lon: lon, takenAt: taken);
          }
          // 느슨 매칭: 확장자 제거 후 비교
          final base = fileName.split('.').first;
          final titleBase = title.split('.').first;
          if (titleBase == base) {
            final lat = (a.latitude == 0.0 && a.longitude == 0.0) ? null : a.latitude;
            final lon = (a.latitude == 0.0 && a.longitude == 0.0) ? null : a.longitude;
            final taken = a.createDateTime;
            return (lat: lat, lon: lon, takenAt: taken);
          }
        }
      }
    } catch (_) {}
    return (lat: null, lon: null, takenAt: null);
  }


  @override
  void dispose() {
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    final di = DeviceInfoPlugin();
    () async {
      try {
        if (Platform.isAndroid) {
          final a = await di.androidInfo;
          setState(() => _deviceId = a.id); // Android 13+: use a.id (non-resettable Android ID)
        } else if (Platform.isIOS) {
          final i = await di.iosInfo;
          setState(() => _deviceId = i.identifierForVendor);
        } else {
          setState(() => _deviceId = 'unknown-device');
        }
      } catch (_) {
        setState(() => _deviceId = 'unknown-device');
      }
    }();
  }

  Future<void> _pickFromGallery() async {
    final x = await _picker.pickImage(
      source: ImageSource.gallery,
    );
    if (x == null) return;
    setState(() { _serverMsg = null; _lastErrorMsg = null; });
    setState(() {
      _imageFile = File(x.path); // 옵션 제거로 원본에 최대한 가깝게 유지(메타 보존)
    });
    try {
      final meta = await _readExifMap(_imageFile!.path);
      setState(() {
        _pvCapture = (meta['captureDate'] ?? '') as String?;
        _pvMake = (meta['make'] ?? '') as String?;
        _pvModel = (meta['model'] ?? '') as String?;
        final latAny = meta['lat'];
        final lonAny = meta['lon'];
        _pvLat = (latAny is num) ? latAny.toDouble() : null;
        _pvLon = (lonAny is num) ? lonAny.toDouble() : null;
      });
    } catch (_) {
      setState(() { _pvCapture = null; _pvMake = null; _pvModel = null; _pvLat = null; _pvLon = null; });
    }
  }

  Future<void> _upload() async {
    if (_imageFile == null) {
      _snack('먼저 갤러리에서 사진을 선택하세요.');
      return;
    }

    HapticFeedback.lightImpact();
    setState(() { _btnScale = 0.96; });
    Future.delayed(const Duration(milliseconds: 120), () {
      if (mounted) setState(() { _btnScale = 1.0; });
    });

    setState(() {
      _isUploading = true;
      _progress = 0.0;
      _serverMsg = null;
      _lastErrorMsg = null;
      _uploadStartAt = DateTime.now();
      _lastSentBytes = 0;
      _speedKbps = 0;
      _etaText = null;
    });

    try {
      final dio = Dio(BaseOptions(
        baseUrl: 'http://ec2-54-67-48-0.us-west-1.compute.amazonaws.com', // TODO: 서버 주소로 교체
        connectTimeout: const Duration(seconds: 10),
        receiveTimeout: const Duration(minutes: 2),
        sendTimeout: const Duration(minutes: 2),
      ));

      // Read EXIF metadata from the selected image
      final exifMap = await _readExifMap(_imageFile!.path);
      final metaJson = jsonEncode(exifMap); // full meta as JSON
      final captureDate = (exifMap['captureDate'] ?? '') as String? ?? '';
      final lat = exifMap['lat'];
      final lon = exifMap['lon'];
      final make = (exifMap['make'] ?? '') as String? ?? '';
      final model = (exifMap['model'] ?? '') as String? ?? '';

      // 위치 결정: EXIF → MediaDB → 수동
      double? useLat = (lat is num) ? (lat as num).toDouble() : null;
      double? useLon = (lon is num) ? (lon as num).toDouble() : null;
      DateTime? takenAt;

      if (useLat == null || useLon == null) {
        // MediaStore에서 보강
        final media = await _findFromMediaDb(_imageFile!.path);
        if (useLat == null && media.lat != null) useLat = media.lat;
        if (useLon == null && media.lon != null) useLon = media.lon;
        takenAt = media.takenAt ?? takenAt;
      }

      if (useLat == null || useLon == null) {
        // 수동 입력: 공용 위치 바텀시트 호출 (입력/지도 탭)
        final picked = await showLocationPickerBottomSheet(
          context,
          initialLat: useLat,
          initialLon: useLon,
          initialAddressHint: '',
        );
        if (picked != null) {
          useLat = picked.lat;
          useLon = picked.lon;
        }
      }

      if (useLat == null || useLon == null) {
        _snack('위치 정보가 필요합니다. 위도/경도를 지정해 주세요.');
        setState(() => _isUploading = false);
        return;
      }

      // 타임스탬프 결정: EXIF → MediaStore → now
      String isoTimestamp = _toIso8601FromExif(captureDate);
      if (captureDate.toString().isEmpty && takenAt != null) {
        isoTimestamp = takenAt.toUtc().toIso8601String();
      }

      final form = FormData.fromMap({
        'file': await MultipartFile.fromFile(
          _imageFile!.path,
          filename: _imageFile!.path.split('/').last,
        ),
        // 백엔드에서 요구하는 필드 (Form(...))
        'gps_lat': useLat,
        'gps_lon': useLon,
        'timestamp': isoTimestamp,
        'device_id': _deviceId ?? 'unknown-device',
        // 선택 메타(서버에선 무시해도 됨)
        'meta_make': make,
        'meta_model': model,
        'meta_json': metaJson,
        'meta_location_source': (lat != null && lon != null)
            ? 'exif'
            : (takenAt != null ? 'media_db' : 'manual'),
      });

      final resp = await dio.post(
        '/lane_wear_infer?conf=0.25&iou=0.50&max_size=1280',
        data: form,
        onSendProgress: (sent, total) {
          if (total > 0) {
            final now = DateTime.now();
            final dt = _uploadStartAt != null ? now.difference(_uploadStartAt!).inMilliseconds : 0;
            final progress = sent / total;
            // kbps 계산 (간단 평균)
            final kbTotal = total / 1024.0;
            final kbSent = sent / 1024.0;
            final sec = dt > 0 ? dt / 1000.0 : 1.0;
            final speed = sec > 0 ? kbSent / sec : 0.0; // KB/s
            // ETA
            final kbLeft = (kbTotal - kbSent).clamp(0.0, double.infinity);
            final etaSec = speed > 0 ? (kbLeft / speed) : double.nan;
            String? etaText;
            if (etaSec.isFinite) {
              final m = etaSec ~/ 60;
              final s = (etaSec % 60).round();
              etaText = m > 0 ? '${m}m ${s}s 남음' : '${s}s 남음';
            }
            setState(() {
              _progress = progress;
              _speedKbps = speed;
              _etaText = etaText;
            });
          }
        },
      );

      setState(() {
        _serverMsg = '서버 응답: ${resp.statusCode} ${resp.statusMessage ?? ''}';
      });
      _snack('업로드 완료');
    } on DioException catch (e) {
      String friendly;
      final code = e.response?.statusCode;
      final msg = e.message ?? '';
      if (code == 404) {
        friendly = '요청 경로가 맞는지 확인해 주세요 (404). 서버의 /lane_wear_infer 경로와 베이스 URL을 점검하세요.';
      } else if (code == 422) {
        friendly = '요청 파라미터 형식이 맞는지 확인해 주세요 (422). 필수 Form 필드(gps_lat/gps_lon/timestamp/device_id)가 있는지 점검하세요.';
      } else if (msg.contains('Failed host lookup') || msg.contains('failed host lookup')) {
        friendly = '서버 주소를 확인할 수 없습니다. 네트워크 연결/도메인/포트를 확인해 주세요.';
      } else if (code == 500) {
        friendly = '서버 내부 오류(500)가 발생했습니다. 잠시 후 다시 시도해 주세요.';
      } else {
        friendly = '업로드 실패: ${code ?? ''} ${msg}'.trim();
      }
      setState(() {
        _serverMsg = friendly;
        _lastErrorMsg = friendly;
      });
      _snack(friendly);
    } catch (e) {
      setState(() {
        _serverMsg = '오류: $e';
        _lastErrorMsg = '오류: $e';
      });
      _snack('오류: $e');
    } finally {
      setState(() => _isUploading = false);
    }
  }

  void _snack(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  Widget build(BuildContext context) {
    final canPreview = _imageFile != null;

    return Scaffold(
      appBar: AppBar(title: const Text('갤러리 업로드')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text('사진 선택', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            OutlinedButton.icon(
              icon: const Icon(Icons.photo_library_outlined),
              label: Text(_imageFile == null
                  ? '갤러리에서 사진 선택'
                  : '선택됨: ${_imageFile!.path.split("/").last}'),
              onPressed: _isUploading ? null : _pickFromGallery,
            ),
            const SizedBox(height: 12),

            if (!canPreview) ...[
              const SizedBox(height: 32),
              Center(
                child: Column(
                  children: const [
                    Icon(Icons.photo_library_outlined, size: 72),
                    SizedBox(height: 12),
                    Text('갤러리에서 사진을 선택하세요', textAlign: TextAlign.center),
                  ],
                ),
              ),
              const SizedBox(height: 24),
            ],

            if (canPreview) ...[
              AspectRatio(
                aspectRatio: 16 / 9,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.file(_imageFile!, fit: BoxFit.cover),
                ),
              ),
              const SizedBox(height: 16),
            ],

            if (canPreview) ...[
              Text('사진 정보', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.info_outline),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              [
                                if (_pvCapture != null && _pvCapture!.isNotEmpty) '촬영일시: $_pvCapture',
                                if (_pvMake != null && _pvMake!.isNotEmpty) '제조사: $_pvMake',
                                if (_pvModel != null && _pvModel!.isNotEmpty) '모델: $_pvModel',
                                if (_pvLat != null && _pvLon != null) '위치: ${_pvLat!.toStringAsFixed(6)}, ${_pvLon!.toStringAsFixed(6)}',
                                if (_deviceId != null) 'Device ID: $_deviceId',
                              ].where((e) => e.isNotEmpty).join('\n'),
                              style: Theme.of(context).textTheme.bodyMedium,
                            ),
                          ),
                        ],
                      ),
                      if (_pvLat != null && _pvLon != null) ...[
                        const SizedBox(height: 12),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: GestureDetector(
                            onTap: () async {
                              final picked = await showLocationPickerBottomSheet(
                                context,
                                initialLat: _pvLat,
                                initialLon: _pvLon,
                                initialAddressHint: '',
                              );
                              if (picked != null) {
                                setState(() { _pvLat = picked.lat; _pvLon = picked.lon; });
                              }
                            },
                            child: SizedBox(
                              height: 160,
                              child: FlutterMap(
                                options: MapOptions(
                                  initialCenter: ll.LatLng(_pvLat!, _pvLon!),
                                  initialZoom: 14,
                                  interactionOptions: const InteractionOptions(
                                    flags: InteractiveFlag.none, // 프리뷰이므로 비활성화
                                  ),
                                ),
                                children: [
                                  TileLayer(
                                    urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                                    userAgentPackageName: 'com.roadglass.app',
                                  ),
                                  MarkerLayer(markers: [
                                    Marker(
                                      point: ll.LatLng(_pvLat!, _pvLon!),
                                      width: 36,
                                      height: 36,
                                      child: const Icon(Icons.location_on, size: 36),
                                    )
                                  ]),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],

            if (_isUploading) ...[
              LinearProgressIndicator(value: _progress == 0 ? null : _progress),
              const SizedBox(height: 8),
              Center(
                child: Text(
                  () {
                    final pct = (_progress * 100).clamp(0, 100).toStringAsFixed(0);
                    final spd = _speedKbps > 0 ? '${_speedKbps.toStringAsFixed(1)} KB/s' : '';
                    final eta = _etaText ?? '';
                    final parts = [
                      '$pct%',
                      if (spd.isNotEmpty) spd,
                      if (eta.isNotEmpty) eta,
                    ];
                    return parts.join(' · ');
                  }(),
                ),
              ),
              const SizedBox(height: 8),
            ],

            if (canPreview) ...[
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  if (_pvCapture?.isNotEmpty == true) Chip(label: Text(_pvCapture!)),
                  if (_pvMake?.isNotEmpty == true) Chip(label: Text(_pvMake!)),
                  if (_pvModel?.isNotEmpty == true) Chip(label: Text(_pvModel!)),
                  if (_pvLat != null && _pvLon != null)
                    ActionChip(
                      label: Text('${_pvLat!.toStringAsFixed(5)}, ${_pvLon!.toStringAsFixed(5)}'),
                      onPressed: () {
                        Clipboard.setData(ClipboardData(text: '${_pvLat},${_pvLon}'));
                        _snack('좌표를 클립보드에 복사했습니다');
                      },
                    ),
                ],
              ),
              const SizedBox(height: 12),
            ],

            AnimatedScale(
              scale: _btnScale,
              duration: const Duration(milliseconds: 120),
              child: FilledButton.icon(
                onPressed: _isUploading ? null : _upload,
                icon: const Icon(Icons.cloud_upload_outlined),
                label: const Text('업로드'),
              ),
            ),
            const SizedBox(height: 8),
            if (_serverMsg != null)
              Text(_serverMsg!, style: Theme.of(context).textTheme.bodySmall),
            if (_lastErrorMsg != null && !_isUploading) ...[
              const SizedBox(height: 8),
              OutlinedButton.icon(
                onPressed: _upload,
                icon: const Icon(Icons.refresh),
                label: const Text('다시 시도'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}