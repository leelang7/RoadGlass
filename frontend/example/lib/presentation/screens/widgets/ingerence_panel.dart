import 'dart:math';
import 'package:flutter/material.dart';

class InferenceResult {
  final String model;
  final Size imageSize;
  final double runtimeMs;
  final Map<String, num> overall; // wear_score 등
  final List<InferenceClass> perClass;

  InferenceResult({
    required this.model,
    required this.imageSize,
    required this.runtimeMs,
    required this.overall,
    required this.perClass,
  });

  factory InferenceResult.fromJson(Map<String, dynamic> j) {
    final sz = j['image_size'] ?? {};
    final pc = (j['per_class'] as Map<String, dynamic>? ?? {});
    final classes = <InferenceClass>[];
    pc.forEach((k, v) => classes.add(InferenceClass.fromJson(k, v)));
    classes.sort((a, b) => (b.wearScore).compareTo(a.wearScore));
    return InferenceResult(
      model: j['model']?.toString() ?? 'unknown',
      imageSize: Size(
        (sz['width'] as num?)?.toDouble() ?? 0,
        (sz['height'] as num?)?.toDouble() ?? 0,
      ),
      runtimeMs: (j['runtime_ms'] as num?)?.toDouble() ?? 0,
      overall: Map<String, num>.from(j['overall'] ?? {}),
      perClass: classes,
    );
  }

  double get wearScore => (overall['wear_score'] as num?)?.toDouble() ?? 0;
  double get visibility => ((overall['visibility'] as num?)?.toDouble() ?? 0) * 100;
}

class InferenceClass {
  final int id;
  final String name;
  final double wearScore;
  final double visibility;
  final double thickness;
  final int ccCount;

  InferenceClass({
    required this.id,
    required this.name,
    required this.wearScore,
    required this.visibility,
    required this.thickness,
    required this.ccCount,
  });

  factory InferenceClass.fromJson(String id, Map<String, dynamic> j) {
    return InferenceClass(
      id: int.tryParse(id) ?? -1,
      name: j['class_name']?.toString() ?? id,
      wearScore: (j['wear_score'] as num?)?.toDouble() ?? 0,
      visibility: ((j['visibility'] as num?)?.toDouble() ?? 0) * 100,
      thickness: (j['thickness_px'] as num?)?.toDouble() ?? 0,
      ccCount: (j['cc_count'] as num?)?.toInt() ?? 0,
    );
  }
}

class InferencePanel extends StatelessWidget {
  final InferenceResult data;
  final ImageProvider? image; // 원본 이미지 보여줄 때 전달(선택)

  const InferencePanel({super.key, required this.data, this.image});

  @override
  Widget build(BuildContext context) {
    final status = _statusOf(data.wearScore);
    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Header
            Row(
              children: [
                Text('검사 결과', style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w800)),
                const Spacer(),
                _Tag('model: ${data.model}'),
                const SizedBox(width: 6),
                _Tag('${data.imageSize.width.toInt()}×${data.imageSize.height.toInt()}'),
                const SizedBox(width: 6),
                _Tag('${data.runtimeMs.toStringAsFixed(1)} ms'),
              ],
            ),
            const SizedBox(height: 12),
            // Wear score gauge + image
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // 게이지
                Expanded(
                  child: _WearGauge(score: data.wearScore, label: status.label, color: status.color),
                ),
                if (image != null) ...[
                  const SizedBox(width: 12),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(10),
                    child: Image(
                      image: image!,
                      width: 120, height: 90, fit: BoxFit.cover,
                    ),
                  ),
                ]
              ],
            ),
            const SizedBox(height: 12),
            // Overall KPI grid
            _OverallGrid(data: data),
            const SizedBox(height: 12),
            // Per-class list
            Text('클래스별 상세', style: Theme.of(context).textTheme.titleMedium!.copyWith(fontWeight: FontWeight.w800)),
            const SizedBox(height: 6),
            _PerClassList(classes: data.perClass),
          ],
        ),
      ),
    );
  }

  _Status _statusOf(double score) {
    // wear_score: 높을수록 마모 심함 가정 (0~100)
    if (score >= 70) return _Status('심각', Colors.red.shade500);
    if (score >= 40) return _Status('주의', Colors.orange.shade600);
    return _Status('양호', Colors.teal.shade600);
  }
}

class _WearGauge extends StatelessWidget {
  final double score; // 0~100
  final String label;
  final Color color;
  const _WearGauge({required this.score, required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.grey.shade50,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('Wear score', style: TextStyle(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            LayoutBuilder(builder: (context, c) {
              final v = score.clamp(0, 100) / 100;
              return Stack(
                alignment: Alignment.centerLeft,
                children: [
                  Container(
                    height: 16,
                    decoration: BoxDecoration(
                      color: Colors.grey.shade200,
                      borderRadius: BorderRadius.circular(999),
                    ),
                  ),
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 350),
                    height: 16,
                    width: c.maxWidth * v,
                    decoration: BoxDecoration(
                      color: color,
                      borderRadius: BorderRadius.circular(999),
                    ),
                  ),
                  Positioned.fill(
                    child: Center(
                      child: Text('${score.toStringAsFixed(1)} / 100  •  $label',
                        style: const TextStyle(fontWeight: FontWeight.w700)),
                    ),
                  ),
                ],
              );
            }),
          ],
        ),
      ),
    );
  }
}

class _OverallGrid extends StatelessWidget {
  final InferenceResult data;
  const _OverallGrid({required this.data});

  @override
  Widget build(BuildContext context) {
    final items = <_KpiItem>[
      _KpiItem('Visibility', '${data.visibility.toStringAsFixed(1)} %'),
      _KpiItem('Edge contrast', data.overall['edge_contrast']?.toStringAsFixed(1) ?? '-'),
      _KpiItem('Thickness(px)', data.overall['thickness_px']?.toStringAsFixed(2) ?? '-'),
      _KpiItem('CC count', (data.overall['cc_count'] ?? '-').toString()),
      _KpiItem('Main comp.', (data.overall['main_component_ratio'] as num?)?.toStringAsFixed(2) ?? '-'),
      _KpiItem('Area(px²)', (data.overall['area_px'] ?? '-').toString()),
    ];
    return GridView.builder(
      shrinkWrap: true,
      itemCount: items.length,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3, mainAxisExtent: 56, crossAxisSpacing: 8, mainAxisSpacing: 8),
      itemBuilder: (_, i) => _KpiTile(item: items[i]),
    );
  }
}

class _KpiItem { final String title; final String value; _KpiItem(this.title, this.value); }

class _KpiTile extends StatelessWidget {
  final _KpiItem item;
  const _KpiTile({required this.item});
  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        child: Row(
          children: [
            Expanded(child: Text(item.title, style: const TextStyle(color: Colors.black54))),
            Text(item.value, style: const TextStyle(fontWeight: FontWeight.w800)),
          ],
        ),
      ),
    );
  }
}

class _PerClassList extends StatelessWidget {
  final List<InferenceClass> classes;
  const _PerClassList({required this.classes});

  @override
  Widget build(BuildContext context) {
    if (classes.isEmpty) {
      return Container(
        padding: const EdgeInsets.symmetric(vertical: 16),
        alignment: Alignment.center,
        child: const Text('클래스 데이터 없음'),
      );
    }
    final maxScore = classes.map((e) => e.wearScore).fold<double>(0, max);

    return Column(
      children: classes.map((c) {
        final ratio = maxScore > 0 ? c.wearScore / maxScore : 0;
        final color = c.wearScore >= 70 ? Colors.red
          : (c.wearScore >= 40 ? Colors.orange : Colors.teal);

        return Container(
          margin: const EdgeInsets.only(bottom: 8),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(10),
            border: Border(left: BorderSide(color: color, width: 4)),
            color: Colors.grey.shade50,
          ),
          child: ListTile(
            dense: true,
            title: Text(c.name, style: const TextStyle(fontWeight: FontWeight.w700)),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 4),
                LayoutBuilder(builder: (_, cons) => Container(
                  width: cons.maxWidth, height: 8,
                  decoration: BoxDecoration(
                    color: Colors.grey.shade200, borderRadius: BorderRadius.circular(999),
                  ),
                  child: Align(
                    alignment: Alignment.centerLeft,
                    child: Container(
                      width: cons.maxWidth * ratio,
                      decoration: BoxDecoration(
                        color: color, borderRadius: BorderRadius.circular(999),
                      ),
                    ),
                  ),
                )),
                const SizedBox(height: 6),
                Text('wear ${c.wearScore.toStringAsFixed(1)} · vis ${c.visibility.toStringAsFixed(1)}% · thick ${c.thickness.toStringAsFixed(2)} · cc ${c.ccCount}'),
              ],
            ),
          ),
        );
      }).toList(),
    );
  }
}

class _Tag extends StatelessWidget {
  final String text;
  const _Tag(this.text);
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.grey.shade200, borderRadius: BorderRadius.circular(999),
      ),
      child: Text(text, style: const TextStyle(fontSize: 12)),
    );
  }
}

class _Status { final String label; final Color color; _Status(this.label, this.color); }
