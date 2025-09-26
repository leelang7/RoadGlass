import 'dart:convert';
import 'package:http/http.dart' as http;

class LaneWearResult {
  final int id;
  final DateTime? createdAt;
  final String? imageName;
  final String model;
  final int width, height;
  final double runtimeMs;
  final Map<String, dynamic> overall;
  final Map<String, dynamic> perClass;

  LaneWearResult({
    required this.id,
    required this.createdAt,
    required this.imageName,
    required this.model,
    required this.width,
    required this.height,
    required this.runtimeMs,
    required this.overall,
    required this.perClass,
  });

  factory LaneWearResult.fromJson(Map<String, dynamic> j) {
    final sz = (j['image_size'] ?? {}) as Map<String, dynamic>;
    return LaneWearResult(
      id: j['id'] ?? -1,
      createdAt: j['created_at'] != null ? DateTime.tryParse(j['created_at']) : null,
      imageName: j['image_name'] as String?,
      model: j['model']?.toString() ?? 'unknown',
      width: (sz['width'] ?? 0) as int,
      height: (sz['height'] ?? 0) as int,
      runtimeMs: ((j['runtime_ms'] ?? 0) as num).toDouble(),
      overall: Map<String, dynamic>.from(j['overall'] ?? const {}),
      perClass: Map<String, dynamic>.from(j['per_class'] ?? const {}),
    );
  }

  double get wearScore => ((overall['wear_score'] ?? 0) as num).toDouble();
  double get visibilityPct => (((overall['visibility'] ?? 0) as num).toDouble() * 100);
}

class LaneWearApi {
  final String base; // 예: http://<EC2_IP>:8000  (Nginx 프록시면 '' or '/api')
  const LaneWearApi(this.base);

  Future<LaneWearResult> latest({String? imageName}) async {
    final uri = Uri.parse('$base/lane_wear/latest${imageName != null ? '?image_name=$imageName' : ''}');
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('HTTP ${res.statusCode}: ${res.body}');
    }
    return LaneWearResult.fromJson(jsonDecode(res.body));
  }

  Future<List<LaneWearResult>> recent({int limit = 10}) async {
    final uri = Uri.parse('$base/lane_wear/recent?limit=$limit');
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('HTTP ${res.statusCode}: ${res.body}');
    }
    final List list = jsonDecode(res.body);
    return list.map((e) => LaneWearResult.fromJson(e)).toList();
  }

  Future<LaneWearResult> byId(int id) async {
    final uri = Uri.parse('$base/lane_wear/$id');
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('HTTP ${res.statusCode}: ${res.body}');
    }
    return LaneWearResult.fromJson(jsonDecode(res.body));
  }
}
