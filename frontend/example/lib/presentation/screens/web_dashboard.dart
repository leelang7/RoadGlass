// lib/presentation/screens/web_dashboard.dart
// ignore_for_file: avoid_web_libraries_in_flutter

import 'dart:async';
import 'dart:convert';
import 'dart:ui' as ui;            // 공용 (모바일/데스크탑)
import 'dart:ui_web' as ui_web;    // 웹 전용 platformViewRegistry

// 웹에서만 쓰는 iframe 임베드
import 'dart:html' as html;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart' as gmap;
import 'package:http/http.dart' as http;
import 'package:flutter_web_plugins/flutter_web_plugins.dart' as web;


/// ===============================
/// 환경설정
/// ===============================

const String kApiBase =
    'http://ec2-54-67-48-0.us-west-1.compute.amazonaws.com:80'; // API 주소
const gmap.LatLng kSeoul = gmap.LatLng(37.5665, 126.9780);

const double kLeftWidth = 320;
const double kRightWidth = 380;
const double kGutter = 16;

/// ===============================
/// 공통 유틸
/// ===============================
String _fmtDate(DateTime? dt) {
  if (dt == null) return '-';
  final y = dt.year.toString().padLeft(4, '0');
  final m = dt.month.toString().padLeft(2, '0');
  final d = dt.day.toString().padLeft(2, '0');
  final hh = dt.hour.toString().padLeft(2, '0');
  final mm = dt.minute.toString().padLeft(2, '0');
  return '$y-$m-$d $hh:$mm';
}

Color _colorForWear(double s) {
  if (s >= 70) return Colors.red.shade600;
  if (s >= 40) return Colors.orange.shade600;
  return Colors.teal.shade600;
}

Widget _chip(String k, String? v, {IconData? icon}) {
  return Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
    decoration: BoxDecoration(
      color: Colors.grey.shade100,
      borderRadius: BorderRadius.circular(999),
    ),
    child: Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (icon != null) ...[
          Icon(icon, size: 14, color: Colors.black54),
          const SizedBox(width: 6),
        ],
        Text('$k ${v ?? '-'}', style: const TextStyle(fontSize: 12)),
      ],
    ),
  );
}

void _launchWeb(String url) {
  if (kIsWeb) {
    html.window.open(url, '_blank');
  } else {
    debugPrint('Open in browser: $url');
  }
}

Map<String, String> _roadviewUrls(double lat, double lon) => {
      'kakao': 'https://map.kakao.com/link/roadview/$lat,$lon',
      'naver':
          'https://map.naver.com/v5/entry/panorama?lat=$lat&lng=$lon&fov=120',
      'google':
          'https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=$lat,$lon',
    };

/// ===============================
/// API 모델/클라이언트
/// ===============================
class StatsSummary {
  final int windowH;
  final int detections24h;
  final int activeDevices24h;
  final int alertsCritical24h;
  final int alertsWarning24h;
  final double? alertsTrend;
  final int latestOk;
  final int latestWarning;
  final int latestCritical;
  final int maintenanceCandidates;

  StatsSummary({
    required this.windowH,
    required this.detections24h,
    required this.activeDevices24h,
    required this.alertsCritical24h,
    required this.alertsWarning24h,
    required this.alertsTrend,
    required this.latestOk,
    required this.latestWarning,
    required this.latestCritical,
    required this.maintenanceCandidates,
  });

  factory StatsSummary.fromJson(Map<String, dynamic> j) => StatsSummary(
        windowH: (j['window_h'] ?? 24) as int,
        detections24h: (j['detections_24h'] ?? 0) as int,
        activeDevices24h: (j['active_devices_24h'] ?? 0) as int,
        alertsCritical24h: (j['alerts_24h']?['critical'] ?? 0) as int,
        alertsWarning24h: (j['alerts_24h']?['warning'] ?? 0) as int,
        alertsTrend: (j['alerts_24h']?['trend_vs_prev'] as num?)?.toDouble(),
        latestOk: (j['latest_device_state']?['ok'] ?? 0) as int,
        latestWarning: (j['latest_device_state']?['warning'] ?? 0) as int,
        latestCritical: (j['latest_device_state']?['critical'] ?? 0) as int,
        maintenanceCandidates: (j['maintenance_candidates'] ?? 0) as int,
      );
}

class RecentItem {
  final int id;
  final double wear;
  final double? lat, lon;
  final DateTime? ts;
  final String? deviceId;
  final String? overlayUrl;
  final String? origUrl;

  RecentItem({
    required this.id,
    required this.wear,
    this.lat,
    this.lon,
    this.ts,
    this.deviceId,
    this.overlayUrl,
    this.origUrl,
  });

  factory RecentItem.fromJson(Map<String, dynamic> j) => RecentItem(
        id: (j['id'] ?? j['db_id']) as int,
        wear: (j['overall']?['wear_score'] as num?)?.toDouble() ??
            (j['wear'] as num?)?.toDouble() ??
            0.0,
        lat: (j['gps_lat'] as num?)?.toDouble(),
        lon: (j['gps_lon'] as num?)?.toDouble(),
        ts: j['timestamp'] != null ? DateTime.tryParse(j['timestamp']) : null,
        deviceId: j['device_id'] as String?,
        overlayUrl: j['overlay_url'] as String?,
        origUrl: j['orig_url'] as String?,
      );
}

class LaneWearResult {
  final int? id;
  final String model;
  final int width, height;
  final double runtimeMs;
  final Map<String, dynamic> overall;
  final Map<String, dynamic> perClass;
  final double? gpsLat, gpsLon;
  final DateTime? timestamp;
  final String? deviceId;
  final String? overlayUrl;
  final String? origUrl;

  LaneWearResult({
    required this.id,
    required this.model,
    required this.width,
    required this.height,
    required this.runtimeMs,
    required this.overall,
    required this.perClass,
    this.gpsLat,
    this.gpsLon,
    this.timestamp,
    this.deviceId,
    this.overlayUrl,
    this.origUrl,
  });

  double get wear => (overall['wear_score'] as num?)?.toDouble() ?? 0.0;

  factory LaneWearResult.fromJson(Map<String, dynamic> j) => LaneWearResult(
        id: (j['db_id'] ?? j['id']) as int?,
        model: (j['model'] ?? '') as String,
        width: (j['image_size']?['width'] ?? j['width'] ?? 0) as int,
        height: (j['image_size']?['height'] ?? j['height'] ?? 0) as int,
        runtimeMs: (j['runtime_ms'] as num?)?.toDouble() ?? 0.0,
        overall: (j['overall'] ?? const {}) as Map<String, dynamic>,
        perClass: (j['per_class'] ?? const {}) as Map<String, dynamic>,
        gpsLat: (j['gps_lat'] as num?)?.toDouble(),
        gpsLon: (j['gps_lon'] as num?)?.toDouble(),
        timestamp:
            j['timestamp'] != null ? DateTime.tryParse(j['timestamp']) : null,
        deviceId: j['device_id'] as String?,
        overlayUrl: j['overlay_url'] as String?,
        origUrl: j['orig_url'] as String?,
      );
}

class CandidateScore {
  final String deviceId;
  final double priority;
  final double wLast;
  final double trend;
  final int crit3;
  final int n;
  final DateTime? lastTs;

  CandidateScore({
    required this.deviceId,
    required this.priority,
    required this.wLast,
    required this.trend,
    required this.crit3,
    required this.n,
    this.lastTs,
  });

  factory CandidateScore.fromJson(Map<String, dynamic> j) => CandidateScore(
        deviceId: (j['device_id'] ?? '') as String,
        priority: (j['priority'] as num?)?.toDouble() ?? 0.0,
        wLast: (j['w_last'] as num?)?.toDouble() ?? 0.0,
        trend: (j['trend'] as num?)?.toDouble() ?? 0.0,
        crit3: (j['crit3'] ?? 0) as int,
        n: (j['n'] ?? 0) as int,
        lastTs: j['last_ts'] != null ? DateTime.tryParse(j['last_ts']) : null,
      );
}

class RankForId {
  final int? rank;
  final int total;
  final CandidateScore? row;
  final List<CandidateScore> top10;

  RankForId({
    required this.rank,
    required this.total,
    required this.row,
    required this.top10,
  });

  factory RankForId.fromJson(Map<String, dynamic> j) => RankForId(
        rank: j['rank'] as int?,
        total: (j['total'] ?? 0) as int,
        row: j['row'] != null
            ? CandidateScore.fromJson(j['row'] as Map<String, dynamic>)
            : null,
        top10: (j['top10'] as List<dynamic>? ?? const [])
            .map((e) => CandidateScore.fromJson(e as Map<String, dynamic>))
            .toList(),
      );
}

class ApiClient {
  final String base;
  const ApiClient(this.base);

  Future<StatsSummary> fetchStats({int windowH = 24}) async {
    final uri = Uri.parse('$base/stats/summary?window_h=$windowH');
    final r = await http.get(uri);
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: ${r.body}');
    }
    return StatsSummary.fromJson(jsonDecode(r.body));
  }

  Future<List<RecentItem>> recent({int limit = 30}) async {
    final uri = Uri.parse('$base/lane_wear/recent?limit=$limit');
    final r = await http.get(uri);
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: ${r.body}');
    }
    final List arr = jsonDecode(r.body) as List;
    return arr
        .map((e) => RecentItem.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<LaneWearResult> latest() async {
    final uri = Uri.parse('$base/lane_wear/latest');
    final r = await http.get(uri);
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: ${r.body}');
    }
    return LaneWearResult.fromJson(jsonDecode(r.body));
  }

  Future<LaneWearResult> byId(int id) async {
    final uris = <Uri>[
      Uri.parse('$base/lane_wear/by_id/$id'),
      Uri.parse('$base/lane_wear/$id'), // 구 백엔드 호환
    ];
    for (final u in uris) {
      final r = await http.get(u);
      if (r.statusCode == 200) {
        return LaneWearResult.fromJson(jsonDecode(r.body));
      }
    }
    throw Exception('not found: id=$id');
  }

  Future<List<CandidateScore>> rank({
    int windowH = 168,
    int limit = 10,
    int offset = 0,
  }) async {
    final uri = Uri.parse(
        '$base/candidates/rank?window_h=$windowH&limit=$limit&offset=$offset');
    final r = await http.get(uri);
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: ${r.body}');
    }
    final List arr = jsonDecode(r.body) as List;
    return arr
        .map((e) => CandidateScore.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<RankForId> rankForId(int id, {int windowH = 168}) async {
    final uri =
        Uri.parse('$base/candidates/rank_for_id/$id?window_h=$windowH');
    final r = await http.get(uri);
    if (r.statusCode != 200) {
      throw Exception('HTTP ${r.statusCode}: ${r.body}');
    }
    return RankForId.fromJson(jsonDecode(r.body));
  }
}

/// ===============================
/// 대시보드
/// ===============================
class WebDashboard extends StatefulWidget {
  const WebDashboard({super.key});
  @override
  State<WebDashboard> createState() => _WebDashboardState();
}

class _WebDashboardState extends State<WebDashboard> {
  final _api = const ApiClient(kApiBase);

  gmap.GoogleMapController? _mapCtrl;
  bool _darkMap = false;

  final _darkStyleJson = '''
[
  {"elementType":"geometry","stylers":[{"color":"#1f2937"}]},
  {"elementType":"labels.text.fill","stylers":[{"color":"#9aa0a6"}]},
  {"elementType":"labels.text.stroke","stylers":[{"color":"#1f2937"}]},
  {"featureType":"road","elementType":"geometry","stylers":[{"color":"#374151"}]},
  {"featureType":"road","elementType":"labels.text.fill","stylers":[{"color":"#cbd5e1"}]},
  {"featureType":"water","elementType":"geometry","stylers":[{"color":"#0ea5e9"}]}
]
''';

  // 데이터 상태
  late Future<StatsSummary> _statsF;
  List<RecentItem> _recent = [];
  Set<gmap.Marker> _markers = {};
  RecentItem? _selected;
  int? _selectedId;

  // 선택 상세 Future
  Future<LaneWearResult>? _detailF;

  // 지도가 아직 준비전이면 카메라 이동을 보류 후 실행
  gmap.CameraUpdate? _pendingCamera;

  @override
  void initState() {
    super.initState();
    _statsF = _api.fetchStats(windowH: 24);
    _loadRecent();
  }

  Future<void> _loadRecent() async {
    try {
      final xs = await _api.recent(limit: 30);
      setState(() {
        _recent = xs;
        _markers = {
          for (final it in xs.where((e) => e.lat != null && e.lon != null))
            gmap.Marker(
              markerId: gmap.MarkerId('r_${it.id}'),
              position: gmap.LatLng(it.lat!, it.lon!),
              infoWindow: gmap.InfoWindow(
                title: 'ID ${it.id}',
                snippet:
                    'wear ${it.wear.toStringAsFixed(1)} • ${_fmtDate(it.ts)}',
                onTap: () => _selectItem(it, focus: false),
              ),
              onTap: () => _selectItem(it),
            )
        };
        if (_selected == null && xs.isNotEmpty) {
          _selectItem(xs.first, focus: false);
        }
      });
    } catch (_) {
      // 무시하고 빈 상태 유지
    }
  }

  void _onMapCreated(gmap.GoogleMapController c) {
    _mapCtrl = c;
    if (_darkMap) _mapCtrl?.setMapStyle(_darkStyleJson);
    if (_pendingCamera != null) {
      _mapCtrl?.animateCamera(_pendingCamera!);
      _pendingCamera = null;
    }
  }

  void _toggleMapTheme() {
    setState(() => _darkMap = !_darkMap);
    _mapCtrl?.setMapStyle(_darkMap ? _darkStyleJson : null);
  }

  void _zoom(double delta) {
    _mapCtrl?.animateCamera(gmap.CameraUpdate.zoomBy(delta));
  }

  Future<void> _flyTo(double lat, double lon, {double zoom = 16}) async {
    final cam = gmap.CameraUpdate.newLatLngZoom(gmap.LatLng(lat, lon), zoom);
    if (_mapCtrl == null) {
      _pendingCamera = cam;
    } else {
      await _mapCtrl?.animateCamera(cam);
    }
  }

  void _selectItem(RecentItem it, {bool focus = true}) async {
    setState(() {
      _selected = it;
      _selectedId = it.id;
      _detailF = _api.byId(it.id);
    });
    if (focus && it.lat != null && it.lon != null) {
      _flyTo(it.lat!, it.lon!);
    }
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final isWide = size.width >= 1280;
    final isMedium = size.width >= 960;

    return Scaffold(
      backgroundColor: const Color(0xFFF4F6F8),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(kGutter),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const _HeaderBar(),
              const SizedBox(height: 12),
              _StatsRow(
                future: _statsF,
                onReload: () {
                  setState(() => _statsF = _api.fetchStats(windowH: 24));
                },
              ),
              const SizedBox(height: 12),

              // ===== 메인 레이아웃 =====
              Expanded(
                child: isWide
                    ? Row(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          SizedBox(
                            width: kLeftWidth,
                            child: _LeftRecentPanel(
                              items: _recent,
                              selectedId: _selectedId,
                              onTap: (it) => _selectItem(it),
                            ),
                          ),
                          const SizedBox(width: kGutter),
                          Expanded(
                            child: _MapCard(
                              markers: _markers,
                              onCreated: _onMapCreated,
                              zoom: _zoom,
                              toggleTheme: _toggleMapTheme,
                            ),
                          ),
                          const SizedBox(width: kGutter),
                          SizedBox(
                            width: kRightWidth,
                            child: _RightDetailPanel(
                              api: _api,
                              detailFuture: _detailF,
                              selected: _selected,
                              onReload: () {
                                if (_selectedId != null) {
                                  setState(() {
                                    _detailF = _api.byId(_selectedId!);
                                  });
                                } else {
                                  setState(() {
                                    _detailF = _api.latest();
                                  });
                                }
                              },
                            ),
                          )
                        ],
                      )
                    : (isMedium
                        ? Column(
                            children: [
                              Expanded(
                                child: Row(
                                  children: [
                                    SizedBox(
                                      width: kLeftWidth,
                                      child: _LeftRecentPanel(
                                        items: _recent,
                                        selectedId: _selectedId,
                                        onTap: (it) => _selectItem(it),
                                      ),
                                    ),
                                    const SizedBox(width: kGutter),
                                    Expanded(
                                      child: _MapCard(
                                        markers: _markers,
                                        onCreated: _onMapCreated,
                                        zoom: _zoom,
                                        toggleTheme: _toggleMapTheme,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(height: kGutter),
                              SizedBox(
                                height: 420,
                                child: _RightDetailPanel(
                                  api: _api,
                                  detailFuture: _detailF,
                                  selected: _selected,
                                  onReload: () {
                                    if (_selectedId != null) {
                                      setState(() {
                                        _detailF = _api.byId(_selectedId!);
                                      });
                                    } else {
                                      setState(() {
                                        _detailF = _api.latest();
                                      });
                                    }
                                  },
                                ),
                              ),
                            ],
                          )
                        : ListView(
                            children: [
                              _MapCard(
                                markers: _markers,
                                onCreated: _onMapCreated,
                              ),
                              const SizedBox(height: kGutter),
                              _LeftRecentPanel(
                                items: _recent,
                                selectedId: _selectedId,
                                onTap: (it) => _selectItem(it),
                              ),
                              const SizedBox(height: kGutter),
                              _RightDetailPanel(
                                api: _api,
                                detailFuture: _detailF,
                                selected: _selected,
                                onReload: () {
                                  if (_selectedId != null) {
                                    setState(() {
                                      _detailF = _api.byId(_selectedId!);
                                    });
                                  } else {
                                    setState(() {
                                      _detailF = _api.latest();
                                    });
                                  }
                                },
                              ),
                            ],
                          )),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// ===============================
/// 상단 헤더
/// ===============================
class _HeaderBar extends StatelessWidget {
  const _HeaderBar();
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        const Text('RoadGlass Dashboard',
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800)),
        const Spacer(),
        SizedBox(
          width: 280,
          child: TextField(
            decoration: InputDecoration(
              prefixIcon: const Icon(Icons.search),
              hintText: '지역/키워드',
              isDense: true,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              filled: true,
              fillColor: Colors.white,
            ),
          ),
        ),
      ],
    );
  }
}

/// ===============================
/// 상단 통계
/// ===============================
class _StatsRow extends StatelessWidget {
  final Future<StatsSummary> future;
  final VoidCallback onReload;
  const _StatsRow({required this.future, required this.onReload});

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<StatsSummary>(
      future: future,
      builder: (context, snap) {
        if (snap.connectionState == ConnectionState.waiting) {
          return Row(
            children: const [
              Expanded(child: _SkeletonStatCard()),
              SizedBox(width: 12),
              Expanded(child: _SkeletonStatCard()),
              SizedBox(width: 12),
              Expanded(child: _SkeletonStatCard()),
              SizedBox(width: 12),
              Expanded(child: _SkeletonStatCard()),
            ],
          );
        }
        if (snap.hasError) {
          return Row(
            children: [
              Expanded(
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Row(
                      children: [
                        const Icon(Icons.error, color: Colors.red),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text('통계 불러오기 실패: ${snap.error}',
                              overflow: TextOverflow.ellipsis),
                        ),
                        const SizedBox(width: 8),
                        FilledButton(
                            onPressed: onReload, child: const Text('다시 시도')),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          );
        }

        final s = snap.data!;
        String trendText;
        Color trendColor;
        if (s.alertsTrend == null) {
          trendText = '—';
          trendColor = Colors.blueGrey;
        } else if (s.alertsTrend! >= 0) {
          trendText = '+${(s.alertsTrend! * 100).toStringAsFixed(0)}%';
          trendColor = Colors.red;
        } else {
          trendText = '${(s.alertsTrend! * 100).toStringAsFixed(0)}%';
          trendColor = Colors.teal;
        }

        final cards = [
          _StatCard(
            title: '활성 위치 (${s.windowH}h)',
            value: s.detections24h.toString(),
            trend: '${s.activeDevices24h} devices',
          ),
          _StatCard(
            title: '심각 알림 (최신 기준)',
            value: s.latestCritical.toString(),
            trend: trendText,
            trendColor: trendColor,
          ),
          _StatCard(
            title: '보수 후보 위치',
            value: s.maintenanceCandidates.toString(),
            trend: '경고 ${s.latestWarning}',
          ),
          _StatCard(
            title: '정상 위치',
            value: s.latestOk.toString(),
            trend: 'OK',
          ),
        ];

        return LayoutBuilder(builder: (context, c) {
          final isNarrow = c.maxWidth < 900;
          return Wrap(
            spacing: 12,
            runSpacing: 12,
            children: cards
                .map((w) => SizedBox(
                      width: isNarrow
                          ? (c.maxWidth - 12) / 2
                          : (c.maxWidth - 36) / 4,
                      child: w,
                    ))
                .toList(),
          );
        });
      },
    );
  }
}

class _SkeletonStatCard extends StatelessWidget {
  const _SkeletonStatCard();
  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: SizedBox(height: 74, child: Container(color: Colors.grey.shade100)),
    );
  }
}

class _StatCard extends StatelessWidget {
  final String title;
  final String value;
  final String trend;
  final Color? trendColor;
  const _StatCard({
    required this.title,
    required this.value,
    required this.trend,
    this.trendColor,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Row(
          children: [
            Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                color: Colors.blue.shade50,
              ),
              child: const Icon(Icons.analytics, color: Colors.blue),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title,
                      style:
                          const TextStyle(fontSize: 13, color: Colors.black54)),
                  const SizedBox(height: 2),
                  Text(value,
                      style: const TextStyle(
                          fontSize: 20, fontWeight: FontWeight.w800)),
                ],
              ),
            ),
            Text(
              trend,
              style: TextStyle(
                fontWeight: FontWeight.w600,
                color: trendColor ??
                    (trend.startsWith('+')
                        ? Colors.green
                        : (trend == 'OK' ? Colors.blueGrey : Colors.red)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// ===============================
/// 왼쪽: 최근 결과 목록
/// ===============================
class _LeftRecentPanel extends StatelessWidget {
  final List<RecentItem> items;
  final int? selectedId;
  final void Function(RecentItem) onTap;

  const _LeftRecentPanel({
    required this.items,
    required this.selectedId,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final sorted = [...items]
      ..sort((a, b) => (b.ts ?? DateTime(0)).compareTo(a.ts ?? DateTime(0)));

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: sorted.isEmpty
          ? const Center(
              child: Padding(
                padding: EdgeInsets.all(24),
                child: Text('데이터가 없습니다'),
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(12),
              itemCount: sorted.length,
              itemBuilder: (_, i) {
                final x = sorted[i];
                final sel = x.id == selectedId;

                // 타일 (날짜 2줄 레이아웃 – 이전 답변의 가독성 개선 버전 그대로 사용)
                final tile = Container(
                  margin: const EdgeInsets.symmetric(vertical: 4),
                  decoration: BoxDecoration(
                    color: sel ? Colors.blue.shade50 : Colors.white,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                      color: sel ? Colors.blue.shade100 : Colors.grey.shade200,
                    ),
                  ),
                  child: ListTile(
                    dense: true,
                    isThreeLine: true,
                    contentPadding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    leading: Icon(Icons.place, color: _colorForWear(x.wear)),
                    title: Text(
                      'ID ${x.id} • wear ${x.wear.toStringAsFixed(1)}',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Row(
                          children: [
                            const Icon(Icons.access_time,
                                size: 14, color: Colors.black45),
                            const SizedBox(width: 4),
                            Text(
                              _fmtDate(x.ts),
                              style: const TextStyle(
                                  fontSize: 12, color: Colors.black87),
                              softWrap: false,
                              overflow: TextOverflow.visible,
                            ),
                          ],
                        ),
                        if (x.lat != null && x.lon != null) ...[
                          const SizedBox(height: 2),
                          Row(
                            children: [
                              const Icon(Icons.gps_fixed,
                                  size: 14, color: Colors.black45),
                              const SizedBox(width: 4),
                              Flexible(
                                child: Text(
                                  '${x.lat!.toStringAsFixed(4)}, ${x.lon!.toStringAsFixed(4)}',
                                  style: const TextStyle(
                                      fontSize: 12, color: Colors.black54),
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ],
                    ),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => onTap(x),
                  ),
                );

                return tile;
              },
            ),
    );
  }
}


/// ===============================
/// 오른쪽: 선택 상세 + 분석 보기 + 로드뷰 + 보수 후보
/// ===============================
class _RightDetailPanel extends StatelessWidget {
  final ApiClient api;
  final Future<LaneWearResult>? detailFuture;
  final RecentItem? selected;
  final VoidCallback onReload;

  const _RightDetailPanel({
    required this.api,
    required this.detailFuture,
    required this.selected,
    required this.onReload,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: detailFuture == null
            ? const Center(child: Text('항목을 선택하세요'))
            : FutureBuilder<LaneWearResult>(
                future: detailFuture,
                builder: (context, snap) {
                  if (snap.connectionState == ConnectionState.waiting) {
                    return const Center(
                        child: CircularProgressIndicator(strokeWidth: 2));
                  }
                  if (snap.hasError) {
                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('추론 상세',
                            style: TextStyle(fontWeight: FontWeight.w800)),
                        const SizedBox(height: 8),
                        Text('불러오기 실패: ${snap.error}',
                            style: const TextStyle(color: Colors.red)),
                        const SizedBox(height: 8),
                        FilledButton(
                            onPressed: onReload, child: const Text('다시 시도')),
                      ],
                    );
                  }

                  final x = snap.data!;
                  final pcs = x.perClass.entries.map((e) {
                    final m = e.value as Map<String, dynamic>;
                    final s = (m['wear_score'] as num?)?.toDouble() ?? 0.0;
                    final name =
                        (m['class_name'] ?? e.key.toString()).toString();
                    final thick = (m['thickness_px'] as num?)?.toDouble();
                    return {'name': name, 'score': s, 'thick': thick};
                  }).toList()
                    ..sort((a, b) => (b['score'] as double)
                        .compareTo(a['score'] as double));

                  return LayoutBuilder(
                    builder: (context, _) => SingleChildScrollView(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              const Text('추론 상세',
                                  style:
                                      TextStyle(fontWeight: FontWeight.w800)),
                              const Spacer(),
                              FilledButton.icon(
                                onPressed: x.id == null
                                    ? null
                                    : () => _openAnalysisViewer(
                                        context, x.id!),
                                icon: const Icon(Icons.image_outlined, size: 18),
                                label: const Text('분석 보기'),
                              ),
                            ],
                          ),
                          const SizedBox(height: 10),
                          _WearGauge(score: x.wear),
                          const SizedBox(height: 12),
                          Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: [
                              _chip('Vis',
                                  (x.overall['visibility'] as num?)
                                      ?.toStringAsFixed(1),
                                  icon: Icons.visibility),
                              _chip('Edge',
                                  (x.overall['edge_contrast'] as num?)
                                      ?.toStringAsFixed(1),
                                  icon: Icons.auto_graph),
                              _chip('Thick',
                                  (x.overall['thickness_px'] as num?)
                                      ?.toStringAsFixed(2),
                                  icon: Icons.straight),
                              _chip('CC',
                                  (x.overall['cc_count'] as num?)?.toString(),
                                  icon: Icons.view_module),
                              _chip(
                                  'Main',
                                  (x.overall['main_component_ratio'] as num?)
                                      ?.toStringAsFixed(2),
                                  icon: Icons.bubble_chart),
                              _chip('Area',
                                  (x.overall['area_px'] as num?)?.toString(),
                                  icon: Icons.aspect_ratio),
                            ],
                          ),
                          const SizedBox(height: 10),
                          const Text('클래스별',
                              style: TextStyle(fontWeight: FontWeight.w800)),
                          ...pcs.take(6).map((m) => ListTile(
                                dense: true,
                                contentPadding: EdgeInsets.zero,
                                leading: Icon(Icons.horizontal_rule,
                                    color: _colorForWear(
                                        (m['score'] as double))),
                                title: Text(m['name'] as String),
                                subtitle: Text(
                                    'wear ${(m['score'] as double).toStringAsFixed(1)}'
                                    '${m['thick'] != null ? ' • thick ${(m['thick'] as double).toStringAsFixed(2)}' : ''}'),
                              )),
                          const SizedBox(height: 8),
                          Wrap(
                            spacing: 6,
                            runSpacing: 6,
                            children: [
                              _chip('ID', '${x.id ?? '-'}',
                                  icon: Icons.tag),
                              _chip('Time', _fmtDate(x.timestamp),
                                  icon: Icons.access_time),
                              _chip('Size', '${x.width}×${x.height}',
                                  icon: Icons.photo_size_select_large),
                              _chip('RT', '${x.runtimeMs.toStringAsFixed(1)} ms',
                                  icon: Icons.speed),
                              if (x.deviceId != null)
                                _chip('Device', x.deviceId,
                                    icon: Icons.devices),
                              if (x.gpsLat != null && x.gpsLon != null)
                                _chip(
                                    'GPS',
                                    '${x.gpsLat!.toStringAsFixed(4)}, ${x.gpsLon!.toStringAsFixed(4)}',
                                    icon: Icons.gps_fixed),
                            ],
                          ),

                          // ---------- 로드뷰 ----------
                          if (x.gpsLat != null && x.gpsLon != null) ...[
                            const SizedBox(height: 14),
                            const Divider(),
                            const Text('로드뷰 (베타)',
                                style:
                                    TextStyle(fontWeight: FontWeight.w800)),
                            const SizedBox(height: 8),
                            Row(
                              children: [
                                OutlinedButton.icon(
                                  onPressed: () => _launchWeb(_roadviewUrls(
                                      x.gpsLat!, x.gpsLon!)['kakao']!),
                                  icon: const Icon(
                                      Icons.directions_car_filled_outlined,
                                      size: 18),
                                  label: const Text('카카오 (새 탭)'),
                                ),
                                const SizedBox(width: 8),
                                OutlinedButton(
                                  onPressed: () => _launchWeb(_roadviewUrls(
                                      x.gpsLat!, x.gpsLon!)['naver']!),
                                  child: const Text('네이버'),
                                ),
                                const SizedBox(width: 8),
                                OutlinedButton(
                                  onPressed: () => _launchWeb(_roadviewUrls(
                                      x.gpsLat!, x.gpsLon!)['google']!),
                                  child: const Text('구글'),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            if (kIsWeb)
                              _WebIFrame(
                                url: _roadviewUrls(
                                    x.gpsLat!, x.gpsLon!)['kakao']!,
                                viewType:
                                    'rv-${x.id ?? '${x.gpsLat}-${x.gpsLon}'}',
                                height: 240,
                              ),
                          ],

                          const SizedBox(height: 14),
                          const Divider(),
                          const SizedBox(height: 6),

                          // === 보수 후보 순위 (현재 디바이스 + Top10) ===
                          if (x.id != null)
                            FutureBuilder<RankForId>(
                              future: api.rankForId(x.id!),
                              builder: (context, rs) {
                                if (rs.connectionState ==
                                    ConnectionState.waiting) {
                                  return const Center(
                                      child: Padding(
                                    padding:
                                        EdgeInsets.symmetric(vertical: 8),
                                    child: CircularProgressIndicator(
                                        strokeWidth: 2),
                                  ));
                                }
                                if (rs.hasError || rs.data == null) {
                                  return const Text('보수 후보 순위를 불러올 수 없습니다.',
                                      style: TextStyle(color: Colors.red));
                                }
                                final r = rs.data!;
                                return Column(
                                  crossAxisAlignment:
                                      CrossAxisAlignment.start,
                                  children: [
                                    Row(
                                      children: [
                                        const Text('보수 후보 순위',
                                            style: TextStyle(
                                                fontWeight:
                                                    FontWeight.w800)),
                                        const Spacer(),
                                        Text(
                                          r.rank != null
                                              ? '현재 디바이스 순위  #${r.rank}/${r.total}'
                                              : '순위 정보 없음',
                                          style: const TextStyle(
                                              fontSize: 12,
                                              color: Colors.black54),
                                        ),
                                      ],
                                    ),
                                    const SizedBox(height: 6),
                                    if (r.row != null)
                                      _rankRowTile(r.row!, highlight: true),
                                    const SizedBox(height: 6),
                                    const Text('상위 후보 (Top 10)',
                                        style: TextStyle(
                                            fontWeight: FontWeight.w700,
                                            fontSize: 13)),
                                    const SizedBox(height: 4),
                                    ...r.top10.map((e) => _rankRowTile(e)),
                                  ],
                                );
                              },
                            ),
                        ],
                      ),
                    ),
                  );
                },
              ),
      ),
    );
  }

  Widget _rankRowTile(CandidateScore e, {bool highlight = false}) {
    return ListTile(
      dense: true,
      contentPadding: EdgeInsets.zero,
      tileColor: highlight ? Colors.orange.shade50 : null,
      leading:
          Icon(Icons.build_circle_rounded, color: _colorForWear(e.wLast)),
      title: Text(
          'device ${e.deviceId}  •  wear ${e.wLast.toStringAsFixed(1)}'),
      subtitle: Wrap(
        spacing: 6,
        runSpacing: 6,
        children: [
          _chip('prio', e.priority.toStringAsFixed(3)),
          _chip('trend',
              '${e.trend >= 0 ? '+' : ''}${e.trend.toStringAsFixed(1)}'),
          _chip('crit3', '${e.crit3}/3'),
          _chip('n', '${e.n}'),
          if (e.lastTs != null) _chip('last', _fmtDate(e.lastTs)),
        ],
      ),
    );
  }
}

/// 분석 이미지 뷰어(원본/오버레이) — 확대/이동 지원
Future<void> _openAnalysisViewer(BuildContext context, int id) async {
  final urlsOrig = [
    '$kApiBase/lane_wear/image/$id/orig',
    '$kApiBase/lane_wear/image/$id?type=orig',
  ];
  final urlsOv = [
    '$kApiBase/lane_wear/image/$id/overlay',
    '$kApiBase/lane_wear/image/$id?type=overlay',
  ];

  showDialog(
    context: context,
    builder: (_) {
      final size = MediaQuery.of(context).size;
      final isWide = size.width >= 900;
      final imgH = size.height * 0.70;
      final maxW = size.width * 0.98;
      final maxH = size.height * 0.92;

      final imgOrig = _FailoverImage(urls: urlsOrig);
      final imgOver = _FailoverImage(urls: urlsOv);

      return Dialog(
        insetPadding: const EdgeInsets.all(16),
        child: ConstrainedBox(
          constraints: BoxConstraints(maxWidth: maxW, maxHeight: maxH),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text('분석 이미지 • ID $id',
                        style: const TextStyle(
                            fontSize: 16, fontWeight: FontWeight.w800)),
                    const Spacer(),
                    IconButton(
                      icon: const Icon(Icons.close),
                      onPressed: () => Navigator.of(context).pop(),
                    )
                  ],
                ),
                const SizedBox(height: 8),
                Expanded(
                  child: isWide
                      ? Row(
                          children: [
                            Expanded(
                                child: _ImageCard(
                                    title: '원본',
                                    child: imgOrig,
                                    height: imgH)),
                            const SizedBox(width: 12),
                            Expanded(
                                child: _ImageCard(
                                    title: '오버레이',
                                    child: imgOver,
                                    height: imgH)),
                          ],
                        )
                      : ListView(
                          children: [
                            _ImageCard(
                                title: '원본', child: imgOrig, height: imgH),
                            const SizedBox(height: 12),
                            _ImageCard(
                                title: '오버레이',
                                child: imgOver,
                                height: imgH),
                          ],
                        ),
                ),
              ],
            ),
          ),
        ),
      );
    },
  );
}

class _ImageCard extends StatelessWidget {
  final String title;
  final Widget child;
  final double? height;

  const _ImageCard({required this.title, required this.child, this.height});

  @override
  Widget build(BuildContext context) {
    final h = height ?? MediaQuery.of(context).size.height * 0.7;

    return Card(
      elevation: 2,
      clipBehavior: Clip.antiAlias,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            color: Colors.grey.shade50,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Text(title,
                style:
                    const TextStyle(fontWeight: FontWeight.w700, fontSize: 13)),
          ),
          SizedBox(
            height: h,
            child: InteractiveViewer(
              panEnabled: true,
              minScale: 1.0,
              maxScale: 5.0,
              child: Center(child: child),
            ),
          ),
        ],
      ),
    );
  }
}

/// 여러 URL 시도하는 이미지
class _FailoverImage extends StatefulWidget {
  final List<String> urls;
  const _FailoverImage({required this.urls});
  @override
  State<_FailoverImage> createState() => _FailoverImageState();
}

class _FailoverImageState extends State<_FailoverImage> {
  int _idx = 0;
  @override
  Widget build(BuildContext context) {
    if (_idx >= widget.urls.length) {
      return const Text('이미지를 불러올 수 없습니다.',
          style: TextStyle(color: Colors.red));
    }
    final url = widget.urls[_idx];
    return Image.network(
      url,
      fit: BoxFit.contain,
      errorBuilder: (_, __, ___) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted) setState(() => _idx++);
        });
        return const SizedBox(
            height: 40,
            child:
                Center(child: CircularProgressIndicator(strokeWidth: 2)));
      },
    );
  }
}

/// ===============================
/// 지도 카드
/// ===============================
class _MapCard extends StatelessWidget {
  final void Function(gmap.GoogleMapController c)? onCreated;
  final Set<gmap.Marker>? markers;
  final void Function(double)? zoom;
  final VoidCallback? toggleTheme;

  const _MapCard({
    super.key,
    this.onCreated,
    this.markers,
    this.zoom,
    this.toggleTheme,
  });

  @override
  Widget build(BuildContext context) {
    final map = gmap.GoogleMap(
      onMapCreated: onCreated,
      initialCameraPosition:
          const gmap.CameraPosition(target: kSeoul, zoom: 12),
      markers: markers ?? const <gmap.Marker>{},
      zoomControlsEnabled: false,
      myLocationButtonEnabled: false,
      mapToolbarEnabled: false,
    );

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      clipBehavior: Clip.antiAlias,
      child: LayoutBuilder(
        builder: (_, c) {
          return Stack(
            children: [
              SizedBox(width: c.maxWidth, height: c.maxHeight, child: map),
              Positioned(
                right: 12,
                top: 12,
                child: Column(
                  children: [
                    _RoundIconButton(icon: Icons.add, onTap: () => zoom?.call(1)),
                    const SizedBox(height: 8),
                    _RoundIconButton(icon: Icons.remove, onTap: () => zoom?.call(-1)),
                    const SizedBox(height: 8),
                    _RoundIconButton(icon: Icons.layers, onTap: toggleTheme),
                  ],
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _RoundIconButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback? onTap;
  const _RoundIconButton({required this.icon, this.onTap});
  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      shape: const CircleBorder(),
      elevation: 3,
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Icon(icon, size: 18, color: Colors.black87),
        ),
      ),
    );
  }
}

class _WearGauge extends StatelessWidget {
  final double score;
  const _WearGauge({required this.score});
  @override
  Widget build(BuildContext context) {
    final color = _colorForWear(score);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('${score.toStringAsFixed(1)} / 100',
            style: const TextStyle(fontWeight: FontWeight.w800)),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: LinearProgressIndicator(
            value: (score.clamp(0, 100)) / 100.0,
            minHeight: 10,
            color: color,
            backgroundColor: color.withOpacity(0.15),
          ),
        ),
      ],
    );
  }
}

/// ===============================
/// 웹 iframe 임베드 (전버전 호환)
/// ===============================
class _WebIFrame extends StatelessWidget {
  final String url;
  final String viewType;  // 예: 'roadview-<id>'
  final double height;
  const _WebIFrame({
    required this.url,
    required this.viewType,
    this.height = 260,
  });

  @override
  Widget build(BuildContext context) {
    if (!kIsWeb) return const SizedBox.shrink();

    // 한 번만 등록 (이미 등록돼 있으면 예외 무시)
    try {
      ui_web.platformViewRegistry.registerViewFactory(viewType, (int viewId) {
        final el = html.IFrameElement()
          ..src = url
          ..style.border = '0'
          ..style.width = '100%'
          ..style.height = '100%'
          ..allow = 'fullscreen';
        return el;
      });
    } catch (_) {
      // 동일 viewType 재등록 시 발생 → 무시
    }

    return SizedBox(
      height: height,
      child: HtmlElementView(
        viewType: viewType,
        key: ValueKey('$viewType::$url'),
      ),
    );
  }
}
