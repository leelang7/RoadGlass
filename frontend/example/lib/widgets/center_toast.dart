import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

enum ToastType { info, success, error }

class CenterToast {
  static OverlayEntry? _current;

  static void show(
    BuildContext context, {
    required String message,
    ToastType type = ToastType.info,
    Duration duration = const Duration(milliseconds: 1300),
  }) {
    // 기존 토스트가 있으면 제거
    _current?.remove();
    _current = null;

    final overlay = Overlay.of(context);
    if (overlay == null) return;

    final Color bg = switch (type) {
      ToastType.success => const Color(0xE600C853), // 초록
      ToastType.error   => const Color(0xE6D32F2F), // 빨강
      ToastType.info    => const Color(0xE6121212), // 검정
    };

    // 가벼운 햅틱(선택)
    HapticFeedback.selectionClick();

    late OverlayEntry entry;
    entry = OverlayEntry(
      builder: (ctx) {
        return Stack(
          children: [
            // 터치 통과용
            const Positioned.fill(child: IgnorePointer()),
            // 중앙 토스트
            Positioned.fill(
              child: Center(
                child: _ToastBubble(
                  bg: bg,
                  message: message,
                ),
              ),
            ),
          ],
        );
      },
    );

    overlay.insert(entry);
    _current = entry;

    // 자동 제거
    Timer(duration, () {
      if (_current == entry) {
        _current?.remove();
        _current = null;
      }
    });
  }
}

class _ToastBubble extends StatefulWidget {
  const _ToastBubble({required this.bg, required this.message});
  final Color bg;
  final String message;

  @override
  State<_ToastBubble> createState() => _ToastBubbleState();
}

class _ToastBubbleState extends State<_ToastBubble>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ac =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 160))
        ..forward();
  @override
  void dispose() { _ac.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _ac.drive(Tween<double>(begin: 0.0, end: 1.0)),
      child: ScaleTransition(
        scale: _ac.drive(Tween<double>(begin: 0.98, end: 1.0)),
        child: Material(
          color: Colors.transparent,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            margin: const EdgeInsets.symmetric(horizontal: 32),
            decoration: BoxDecoration(
              color: widget.bg,
              borderRadius: BorderRadius.circular(12),
              boxShadow: const [BoxShadow(blurRadius: 14, color: Colors.black54)],
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.info_outline, color: Colors.white, size: 18),
                const SizedBox(width: 8),
                Flexible(
                  child: Text(
                    widget.message,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 14.5,
                      height: 1.2,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}