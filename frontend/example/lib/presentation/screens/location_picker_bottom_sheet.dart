import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart' as ll;
import 'package:geocoding/geocoding.dart' as geocoding;

/// 위치 선택 바텀시트 (입력 탭 + 지도 탭)
/// - 좌표 직접 입력 / 주소 검색
/// - 지도에서 핀 찍기/드래그로 선택
Future<({double lat, double lon})?> showLocationPickerBottomSheet(
  BuildContext context, {
  double? initialLat,
  double? initialLon,
  String? initialAddressHint,
}) {
  return showModalBottomSheet<({double lat, double lon})>(
    context: context,
    isScrollControlled: true,
    useSafeArea: true,
    builder: (ctx) => _LocationPickerBottomSheet(
      initialLat: initialLat,
      initialLon: initialLon,
      initialAddressHint: initialAddressHint,
    ),
  );
}

class _LocationPickerBottomSheet extends StatefulWidget {
  const _LocationPickerBottomSheet({
    this.initialLat,
    this.initialLon,
    this.initialAddressHint,
  });
  final double? initialLat;
  final double? initialLon;
  final String? initialAddressHint;

  @override
  State<_LocationPickerBottomSheet> createState() => _LocationPickerBottomSheetState();
}

class _LocationPickerBottomSheetState extends State<_LocationPickerBottomSheet>
    with SingleTickerProviderStateMixin {
  late final TextEditingController _latCtrl;
  late final TextEditingController _lonCtrl;
  late final TextEditingController _addrCtrl;

  bool _isGeocoding = false;

  // 지도 상태
  late ll.LatLng _center;
  ll.LatLng? _picked;
  double _zoom = 13; // 현재 줌 레벨 추적
  final _mapController = MapController();

  @override
  void initState() {
    super.initState();
    _latCtrl = TextEditingController(text: widget.initialLat?.toStringAsFixed(7) ?? '');
    _lonCtrl = TextEditingController(text: widget.initialLon?.toStringAsFixed(7) ?? '');
    _addrCtrl = TextEditingController(text: widget.initialAddressHint ?? '');

    // 초기 중심 (초기값 없으면 서울시청 근처)
    _center = ll.LatLng(widget.initialLat ?? 37.5665, widget.initialLon ?? 126.9780);
    if (widget.initialLat != null && widget.initialLon != null) {
      _picked = ll.LatLng(widget.initialLat!, widget.initialLon!);
    }
    _zoom = 13;
  }

  @override
  void dispose() {
    _latCtrl.dispose();
    _lonCtrl.dispose();
    _addrCtrl.dispose();
    super.dispose();
  }

  Future<void> _searchAddress() async {
    final q = _addrCtrl.text.trim();
    if (q.isEmpty) return;
    setState(() => _isGeocoding = true);
    try {
      final list = await geocoding.locationFromAddress(q);
      if (list.isNotEmpty) {
        final p = list.first;
        _latCtrl.text = p.latitude.toStringAsFixed(7);
        _lonCtrl.text = p.longitude.toStringAsFixed(7);
        // 지도 탭에도 반영
        final ll.LatLng lp = ll.LatLng(p.latitude, p.longitude);
        setState(() {
          _picked = lp;
          _center = lp;
        });
        _mapController.move(lp, _zoom);
      } else {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('해당 주소로 좌표를 찾지 못했습니다.')),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('주소 검색 실패: $e')),
      );
    } finally {
      if (mounted) setState(() => _isGeocoding = false);
    }
  }

  void _syncFromFieldsToMap() {
    final lat = double.tryParse(_latCtrl.text.trim());
    final lon = double.tryParse(_lonCtrl.text.trim());
    if (lat != null && lon != null) {
      final p = ll.LatLng(lat, lon);
      setState(() {
        _picked = p;
        _center = p;
      });
      _mapController.move(p, _zoom);
    }
  }

  void _confirm() {
    final lat = double.tryParse(_latCtrl.text.trim());
    final lon = double.tryParse(_lonCtrl.text.trim());
    if (lat == null || lon == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('위도/경도를 올바르게 입력하세요.')),
      );
      return;
    }
    Navigator.of(context).pop((lat: lat, lon: lon));
  }

  void _zoomBy(double dz) {
    final newZoom = (_zoom + dz).clamp(3.0, 19.0);
    setState(() => _zoom = newZoom);
    _mapController.move(_center, _zoom);
  }

  void _recenterToPicked() {
    final target = _picked ?? _center;
    _mapController.move(target, _zoom);
  }

  @override
  Widget build(BuildContext context) {
    final pad = EdgeInsets.only(
      bottom: MediaQuery.of(context).viewInsets.bottom + 16,
      left: 16,
      right: 16,
      top: 8,
    );

    return DefaultTabController(
      length: 2,
      child: Padding(
        padding: pad,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    '위치 지정',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                IconButton(
                  onPressed: () => Navigator.pop(context, null),
                  icon: const Icon(Icons.close),
                )
              ],
            ),
            const SizedBox(height: 8),
            Container(
              height: 44,
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceVariant.withOpacity(0.6),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const TabBar(
                tabs: [
                  Tab(text: '입력'),
                  Tab(text: '지도'),
                ],
              ),
            ),
            const SizedBox(height: 12),
            SizedBox(
              height: 360, // 바텀시트 내 적당한 지도 높이
              child: TabBarView(
                children: [
                  // 입력 탭
                  Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: TextField(
                              controller: _latCtrl,
                              decoration: const InputDecoration(
                                labelText: '위도 (lat)',
                                border: OutlineInputBorder(),
                              ),
                              keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: true),
                              onChanged: (_) => _syncFromFieldsToMap(),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: TextField(
                              controller: _lonCtrl,
                              decoration: const InputDecoration(
                                labelText: '경도 (lon)',
                                border: OutlineInputBorder(),
                              ),
                              keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: true),
                              onChanged: (_) => _syncFromFieldsToMap(),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: TextField(
                              controller: _addrCtrl,
                              decoration: const InputDecoration(
                                labelText: '주소로 찾기 (선택)',
                                hintText: '예) 서울특별시 중구 을지로 100',
                                border: OutlineInputBorder(),
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          SizedBox(
                            height: 48,
                            child: FilledButton(
                              onPressed: _isGeocoding ? null : _searchAddress,
                              child: _isGeocoding
                                  ? const SizedBox(
                                      width: 18,
                                      height: 18,
                                      child: CircularProgressIndicator(strokeWidth: 2),
                                    )
                                  : const Text('검색'),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),

                  // 지도 탭
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Stack(
                      children: [
                        FlutterMap(
                          mapController: _mapController,
                          options: MapOptions(
                            initialCenter: _center,
                            initialZoom: _zoom,
                            onTap: (tapPos, latlng) {
                              setState(() {
                                _picked = latlng;
                                _latCtrl.text = latlng.latitude.toStringAsFixed(7);
                                _lonCtrl.text = latlng.longitude.toStringAsFixed(7);
                              });
                            },
                            onLongPress: (tapPos, latlng) {
                              setState(() {
                                _picked = latlng;
                                _latCtrl.text = latlng.latitude.toStringAsFixed(7);
                                _lonCtrl.text = latlng.longitude.toStringAsFixed(7);
                              });
                            },
                            onMapEvent: (evt) {
                              // 카메라 상태 동기화
                              final cam = evt.camera;
                              setState(() {
                                _center = cam.center;
                                _zoom = cam.zoom;
                              });
                            },
                          ),
                          children: [
                            TileLayer(
                              urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                              userAgentPackageName: 'com.roadglass.app',
                            ),
                            if (_picked != null)
                              MarkerLayer(
                                markers: [
                                  Marker(
                                    point: _picked!,
                                    width: 40,
                                    height: 40,
                                    child: const Icon(Icons.location_on, size: 40),
                                  ),
                                ],
                              ),
                          ],
                        ),

                        // 상단 검색바 오버레이
                        Positioned(
                          left: 12,
                          right: 12,
                          top: 12,
                          child: Material(
                            elevation: 2,
                            borderRadius: BorderRadius.circular(10),
                            clipBehavior: Clip.antiAlias,
                            child: Row(
                              children: [
                                Expanded(
                                  child: TextField(
                                    controller: _addrCtrl,
                                    decoration: const InputDecoration(
                                      hintText: '주소/장소명으로 검색',
                                      contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 14),
                                      border: InputBorder.none,
                                    ),
                                    textInputAction: TextInputAction.search,
                                    onSubmitted: (_) => _searchAddress(),
                                  ),
                                ),
                                SizedBox(
                                  height: 48,
                                  child: TextButton.icon(
                                    onPressed: _isGeocoding ? null : _searchAddress,
                                    icon: _isGeocoding
                                        ? const SizedBox(
                                            width: 16,
                                            height: 16,
                                            child: CircularProgressIndicator(strokeWidth: 2),
                                          )
                                        : const Icon(Icons.search),
                                    label: const Text('검색'),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),

                        // 우측 하단 컨트롤(줌/리센터)
                        Positioned(
                          right: 12,
                          bottom: 12,
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              FloatingActionButton.small(
                                heroTag: 'zoom_in',
                                onPressed: () => _zoomBy(1.0),
                                child: const Icon(Icons.add),
                              ),
                              const SizedBox(height: 8),
                              FloatingActionButton.small(
                                heroTag: 'zoom_out',
                                onPressed: () => _zoomBy(-1.0),
                                child: const Icon(Icons.remove),
                              ),
                              const SizedBox(height: 8),
                              FloatingActionButton.small(
                                heroTag: 'recenter',
                                onPressed: _recenterToPicked,
                                child: const Icon(Icons.my_location),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () => Navigator.pop(context, null),
                    child: const Text('취소'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _confirm,
                    icon: const Icon(Icons.check),
                    label: const Text('확인'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}