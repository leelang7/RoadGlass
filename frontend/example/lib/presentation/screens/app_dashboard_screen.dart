import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

class AppDashboardScreen extends StatefulWidget {
  const AppDashboardScreen({super.key});

  @override
  State<AppDashboardScreen> createState() => _AppDashboardScreenState();
}

class _AppDashboardScreenState extends State<AppDashboardScreen> {
  static const _endpoint = 'http://ec2-54-67-48-0.us-west-1.compute.amazonaws.com/lane_wear_infer';

  int _pendingCount = 0;
  double _pendingMB = 0;
  DateTime? _lastSyncAt;
  bool _online = false;
  bool _syncing = false;

  // 추가 상태 (정책/로그/성능/버전)
  bool _wifiOnly = true; // 네트워크 정책: Wi‑Fi 전용 업로드
  List<_EventRow> _recent = const []; // 최근 이벤트 5건 (pending 기준)
  double _storagePercent = 0; // 전체 대비 비율(알 수 없으면 0)
  Map<String, dynamic>? _perf; // {engineFps, uiFps, emaFps}
  String? _appVersion; // 하단 표기 (없으면 N/A)
  String? _modelVersion; // 하단 표기 (없으면 N/A)

  // 간단 행 데이터 모델
  static const _warnMB = 200.0; // 경고 배지 임계값
  static const _policyFile = '_policy.json';
  static const _perfFile = 'perf_stats.json';
  static const _versionFile = 'version.txt';
  static const _modelFile = 'model.txt';

  @override
  void initState() {
    super.initState();
    _refreshAll();
  }

  Future<Directory> _pendingDir() async {
    final dir = await getApplicationDocumentsDirectory();
    final d = Directory('${dir.path}/pending_uploads');
    if (!await d.exists()) {
      await d.create(recursive: true);
    }
    return d;
  }

  Future<File> _policyPath() async {
    final dir = await _pendingDir();
    return File('${dir.path}/$_policyFile');
  }

  Future<void> _loadPolicy() async {
    try {
      final f = await _policyPath();
      if (await f.exists()) {
        final j = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
        _wifiOnly = (j['wifi_only'] ?? true) == true;
      }
    } catch (_) { /* ignore */ }
  }

  Future<void> _savePolicy() async {
    try {
      final f = await _policyPath();
      await f.writeAsString(jsonEncode({'wifi_only': _wifiOnly}), flush: true);
    } catch (_) { /* ignore */ }
  }

  Future<void> _loadPerfAndMeta() async {
    try {
      final dir = await _pendingDir();
      final pf = File('${dir.path}/$_perfFile');
      if (await pf.exists()) {
        final j = jsonDecode(await pf.readAsString()) as Map<String, dynamic>;
        _perf = j;
      }
      final vf = File('${dir.path}/$_versionFile');
      if (await vf.exists()) {
        _appVersion = (await vf.readAsString()).trim();
      }
      final mf = File('${dir.path}/$_modelFile');
      if (await mf.exists()) {
        _modelVersion = (await mf.readAsString()).trim();
      }
    } catch (_) { /* ignore */ }
  }

  Future<void> _refreshAll() async {
    await _loadPolicy();
    await Future.wait([
      _scanQueue(),
      _readLastSync(),
      _pingOnline(),
      _loadPerfAndMeta(),
      _scanRecentEvents(limit: 5),
    ]);
    if (mounted) setState(() {});
  }

  Future<void> _scanQueue() async {
    try {
      final dir = await _pendingDir();
      if (!await dir.exists()) {
        setState(() { _pendingCount = 0; _pendingMB = 0; });
        return;
      }
      final entries = await dir.list().toList();
      final files = entries.whereType<File>().toList();
      final pngs = files.where((f) => f.path.endsWith('.png')).toList();
      int bytes = 0;
      for (final f in files) {
        try { bytes += await f.length(); } catch (_) {}
      }
      setState(() {
        _pendingCount = pngs.length; // 1건 = 1 png + 1 json 쌍
        _pendingMB = bytes / (1024 * 1024);
      });
    } catch (e) {
      debugPrint('scanQueue error: $e');
    }
  }

  Future<void> _scanRecentEvents({int limit = 5}) async {
    try {
      final dir = await _pendingDir();
      final entries = await dir.list().toList();
      final jsons = entries.whereType<File>().where((f) => f.path.endsWith('.json')).toList();
      final rows = <_EventRow>[];
      for (final jf in jsons) {
        try {
          final meta = jsonDecode(await jf.readAsString()) as Map<String, dynamic>;
          final tUs = (meta['t_us'] as num?)?.toInt();
          final gps = (meta['gps'] as Map?) ?? const {};
          final sp = (gps['speed_kmh'] as num?)?.toDouble();
          final trig = (meta['trigger'] ?? '').toString();
          final ts = tUs != null ? DateTime.fromMicrosecondsSinceEpoch(tUs) : null;
          rows.add(_EventRow(
            time: ts ?? DateTime.fromMillisecondsSinceEpoch(jf.statSync().modified.millisecondsSinceEpoch),
            speedKmh: sp ?? 0,
            zAcc: null, // 메타에 없으면 빈값
            status: '대기',
            trigger: trig,
          ));
        } catch (_) { /* ignore bad meta */ }
      }
      rows.sort((a,b) => b.time.compareTo(a.time));
      _recent = rows.take(limit).toList();
    } catch (e) {
      debugPrint('scanRecentEvents error: $e');
      _recent = const [];
    }
  }

  double _storagePctForDisplay() {
    // 전체 저장소 용량 알 수 없으면 0 반환 (UI에서는 표기 생략)
    return _storagePercent;
  }

  Future<void> _readLastSync() async {
    try {
      final dir = await _pendingDir();
      final marker = File('${dir.path}/_last_sync.txt');
      if (await marker.exists()) {
        final s = await marker.readAsString();
        final dt = DateTime.tryParse(s.trim());
        if (dt != null) {
          setState(() { _lastSyncAt = dt.toLocal(); });
        }
      }
    } catch (e) {
      debugPrint('readLastSync error: $e');
    }
  }

  Future<void> _writeLastSyncNow() async {
    try {
      final dir = await _pendingDir();
      final marker = File('${dir.path}/_last_sync.txt');
      await marker.writeAsString(DateTime.now().toUtc().toIso8601String(), flush: true);
      setState(() { _lastSyncAt = DateTime.now(); });
    } catch (e) {
      debugPrint('writeLastSync error: $e');
    }
  }

  Future<void> _pingOnline() async {
    try {
      final uri = Uri.parse(_endpoint);
      final resp = await http.get(uri).timeout(const Duration(seconds: 3));
      setState(() { _online = resp.statusCode < 500; });
    } catch (_) {
      setState(() { _online = false; });
    }
  }

  Future<void> _syncNow() async {
    if (_syncing) return;
    if (_wifiOnly == true && !_online) {
      // 온라인 체크가 OFFLINE이면 동기화 차단 (Wi‑Fi만 허용 정책 가정)
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('오프라인 상태입니다. Wi‑Fi 연결 후 다시 시도하세요.'), duration: Duration(seconds: 2)),
      );
      return;
    }
    setState(() { _syncing = true; });
    try {
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
          debugPrint('meta parse fail: $e');
          continue;
        }
        final bytes = await png.readAsBytes();
        final req = http.MultipartRequest('POST', Uri.parse(_endpoint));
        final fname = png.uri.pathSegments.last;
        req.files.add(http.MultipartFile.fromBytes('file', bytes, filename: fname));
        final gps = (meta['gps'] as Map?) ?? {};
        req.fields['gps_lat'] = (gps['lat'] ?? '').toString();
        req.fields['gps_lon'] = (gps['lon'] ?? '').toString();
        req.fields['timestamp'] = DateTime.now().toIso8601String();
        req.fields['device_id'] = (meta['device_id'] ?? 'unknown').toString();
        try {
          final resp = await req.send();
          if (resp.statusCode == 200) {
            success++;
            await png.delete();
            await metaFile.delete();
          }
        } catch (_) {/* ignore and retry next time */}
      }
      if (success > 0) {
        await _writeLastSyncNow();
      }
    } catch (e) {
      debugPrint('syncNow error: $e');
    } finally {
      await _scanQueue();
      await _pingOnline();
      if (mounted) setState(() { _syncing = false; });
    }
  }

  String _fmtTime(DateTime? dt) {
    if (dt == null) return '기록 없음';
    final d = dt.toLocal();
    return '${d.year}-${d.month.toString().padLeft(2,'0')}-${d.day.toString().padLeft(2,'0')} ${d.hour.toString().padLeft(2,'0')}:${d.minute.toString().padLeft(2,'0')}';
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(title: const Text('상태 패널')),
      body: RefreshIndicator(
        onRefresh: _refreshAll,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _InfoCard(
              title: '연결 상태',
              value: _online ? 'ONLINE' : 'OFFLINE',
              caption: '마지막 동기화: ${_fmtTime(_lastSyncAt)}',
              accent: _online ? Colors.greenAccent : Colors.orangeAccent,
              actions: [
                IconButton(
                  icon: const Icon(Icons.refresh),
                  tooltip: '상태 새로고침',
                  onPressed: _refreshAll,
                )
              ],
            ),
            const SizedBox(height: 12),
            _InfoCard(
              title: '오프라인 큐',
              value: '$_pendingCount건',
              caption: '${_pendingMB.toStringAsFixed(2)} MB 대기 중',
              accent: cs.primary,
              trailing: ElevatedButton.icon(
                onPressed: (_pendingCount == 0 || _syncing) ? null : _syncNow,
                icon: _syncing ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.sync),
                label: Text(_syncing ? '동기화 중…' : '지금 동기화'),
              ),
            ),
            const SizedBox(height: 12),
            // 최근 이벤트 5건
            _RecentEventsCard(events: _recent),
            const SizedBox(height: 12),
            // 저장 공간 사용량
            _StorageUsageCard(mb: _pendingMB, warn: _pendingMB >= _warnMB, percent: _storagePctForDisplay()),
            const SizedBox(height: 12),
            // 네트워크 정책 토글
            _PolicyCard(
              wifiOnly: _wifiOnly,
              onChanged: (v) async {
                setState(() { _wifiOnly = v; });
                await _savePolicy();
              },
            ),
            const SizedBox(height: 12),
            // 성능 간단 표시 + 버전 정보 (하단)
            _PerfAndVersionCard(
              perf: _perf,
              appVersion: _appVersion,
              modelVersion: _modelVersion,
            ),
          ],
        ),
      ),
    );
  }
}

class _InfoCard extends StatelessWidget {
  const _InfoCard({
    required this.title,
    required this.value,
    required this.caption,
    required this.accent,
    this.trailing,
    this.actions,
  });
  final String title;
  final String value;
  final String caption;
  final Color accent;
  final Widget? trailing;
  final List<Widget>? actions;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Material(
      elevation: 2,
      borderRadius: BorderRadius.circular(16),
      color: cs.surface,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 10,
              height: 10,
              margin: const EdgeInsets.only(top: 6, right: 12),
              decoration: BoxDecoration(color: accent, shape: BoxShape.circle),
            ),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(title, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                      ),
                      if (actions != null) Row(children: actions!),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Text(value, style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800, letterSpacing: -0.2)),
                  const SizedBox(height: 4),
                  Text(caption, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Theme.of(context).hintColor)),
                ],
              ),
            ),
            if (trailing != null) ...[
              const SizedBox(width: 12),
              trailing!,
            ]
          ],
        ),
      ),
    );
  }
}

class _RecentEventsCard extends StatelessWidget {
  const _RecentEventsCard({required this.events});
  final List<_EventRow> events;
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Material(
      elevation: 2,
      borderRadius: BorderRadius.circular(16),
      color: cs.surface,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('최근 이벤트', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            if (events.isEmpty)
              Text('표시할 이벤트가 없습니다.', style: Theme.of(context).textTheme.bodySmall)
            else
              ...events.take(5).map((e) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Row(
                  children: [
                    Expanded(child: Text(_fmt(e.time), style: Theme.of(context).textTheme.bodyMedium)),
                    SizedBox(width: 8, child: Text(e.status, style: Theme.of(context).textTheme.labelSmall)),
                    const SizedBox(width: 12),
                    Text('${e.speedKmh.toStringAsFixed(1)} km/h', style: Theme.of(context).textTheme.bodySmall),
                    const SizedBox(width: 12),
                    if (e.zAcc != null) Text('zAcc ${e.zAcc!.toStringAsFixed(1)}', style: Theme.of(context).textTheme.bodySmall),
                  ],
                ),
              )),
          ],
        ),
      ),
    );
  }

  String _fmt(DateTime dt) {
    return '${dt.year}-${dt.month.toString().padLeft(2,'0')}-${dt.day.toString().padLeft(2,'0')} ${dt.hour.toString().padLeft(2,'0')}:${dt.minute.toString().padLeft(2,'0')}';
  }
}

class _StorageUsageCard extends StatelessWidget {
  const _StorageUsageCard({required this.mb, required this.warn, required this.percent});
  final double mb; final bool warn; final double percent;
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Material(
      elevation: 2,
      borderRadius: BorderRadius.circular(16),
      color: cs.surface,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(children: [
                    Text('저장 공간', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                    if (warn) ...[
                      const SizedBox(width: 8),
                      Icon(Icons.warning_amber_rounded, color: Colors.orangeAccent, size: 18),
                      const SizedBox(width: 4),
                      Text('용량이 커지고 있어요', style: Theme.of(context).textTheme.labelSmall?.copyWith(color: Colors.orangeAccent)),
                    ],
                  ]),
                  const SizedBox(height: 6),
                  Text('${mb.toStringAsFixed(2)} MB 대기 중', style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800, letterSpacing: -0.2)),
                  const SizedBox(height: 4),
                  if (percent > 0)
                    Text('전체 저장소의 ${(percent*100).toStringAsFixed(1)}%', style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Theme.of(context).hintColor)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PolicyCard extends StatelessWidget {
  const _PolicyCard({required this.wifiOnly, required this.onChanged});
  final bool wifiOnly; final ValueChanged<bool> onChanged;
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Material(
      elevation: 2,
      borderRadius: BorderRadius.circular(16),
      color: cs.surface,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Expanded(child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('네트워크 정책', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                const SizedBox(height: 6),
                Text('Wi‑Fi 전용 업로드', style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Theme.of(context).hintColor)),
              ],
            )),
            Switch(value: wifiOnly, onChanged: onChanged),
          ],
        ),
      ),
    );
  }
}

class _PerfAndVersionCard extends StatelessWidget {
  const _PerfAndVersionCard({required this.perf, required this.appVersion, required this.modelVersion});
  final Map<String, dynamic>? perf; final String? appVersion; final String? modelVersion;
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    String _fmtNum(num? v) => v == null ? '—' : v.toStringAsFixed(1);
    return Material(
      elevation: 2,
      borderRadius: BorderRadius.circular(16),
      color: cs.surface,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('성능 & 버전', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            Row(children: [
              Expanded(child: _kv('Engine FPS', _fmtNum(perf?['engineFps'] as num?))),
              Expanded(child: _kv('UI FPS', _fmtNum(perf?['uiFps'] as num?))),
              Expanded(child: _kv('EMA FPS', _fmtNum(perf?['emaFps'] as num?))),
            ]),
            const SizedBox(height: 12),
            Text('앱 버전: ${appVersion ?? 'N/A'}  ·  모델: ${modelVersion ?? 'N/A'}', style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Theme.of(context).hintColor)),
          ],
        ),
      ),
    );
  }

  Widget _kv(String k, String v) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(k, style: const TextStyle(fontSize: 12)),
      const SizedBox(height: 2),
      Text(v, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
    ]);
  }
}

class _EventRow {
  const _EventRow({required this.time, required this.speedKmh, required this.zAcc, required this.status, required this.trigger});
  final DateTime time; final double speedKmh; final double? zAcc; final String status; final String trigger;
}
