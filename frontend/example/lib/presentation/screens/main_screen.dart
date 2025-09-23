import 'package:flutter/material.dart';
import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/gestures.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Neo Bento Grid + Immersive Glass Tiles + Kinetic Typography
class MainScreen extends StatelessWidget {
  const MainScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CustomScrollView(
        physics: const BouncingScrollPhysics(),
        slivers: [
          // Kinetic typography app bar
          SliverAppBar(
            pinned: true,
            floating: false,
            snap: false,
            expandedHeight: 56,
            toolbarHeight: 56,
            backgroundColor: Colors.white.withOpacity(0.08),
            elevation: 0,
            title: const _KineticTitle(text: 'RoadGlass', compact: true),
            centerTitle: false,
            actions: [
              IconButton(
                icon: const Icon(Icons.settings),
                onPressed: () => Navigator.pushNamed(context, '/settings'),
                tooltip: 'Settings',
              ),
            ],
            flexibleSpace: ClipRect(
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                child: const SizedBox.expand(),
              ),
            ),
          ),

          // Bento Grid section
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
              child: _BentoGrid(
                spacing: 12,
                runSpacing: 12,
                children: [
                  _BentoItem(spanX: 2, spanY: 2, child: _GlassTile(label: '실시간 감지', icon: Icons.camera_alt_rounded, route: '/guide', accent: const Color(0xFF60A5FA))),
                  _BentoItem(spanX: 2, spanY: 2, child: _GlassTile(label: '대시보드', icon: Icons.dashboard_rounded, route: '/dashboard', accent: const Color(0xFFF59E0B))),
                  _BentoItem(spanX: 2, spanY: 2, child: _GlassTile(label: '이미지 업로드', icon: Icons.file_upload_rounded, route: '/upload_gallery', accent: const Color(0xFFA78BFA))),
                  _BentoItem(spanX: 2, spanY: 2, child: _GlassTile(label: '히스토리', icon: Icons.history_rounded, route: '/history', accent: const Color(0xFF34D399))),
                  _BentoItem(spanX: 2, spanY: 2, child: _GlassTile(label: '알림 센터', icon: Icons.notifications_active_rounded, route: '/notifications', accent: const Color(0xFFFB7185))),
                ],
              ),
            ),
          ),
        ],
      ),
      // global grain overlay for subtle texture
      extendBodyBehindAppBar: false,
    );
  }
}

/// -------- KINETIC TYPOGRAPHY --------
class _KineticTitle extends StatefulWidget {
  const _KineticTitle({required this.text, this.compact = false});
  final String text;
  final bool compact;
  @override
  State<_KineticTitle> createState() => _KineticTitleState();
}

class _KineticTitleState extends State<_KineticTitle> with SingleTickerProviderStateMixin {
  late final AnimationController _c;
  late final Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _c = AnimationController(vsync: this, duration: const Duration(seconds: 3))..repeat(reverse: true);
    _anim = CurvedAnimation(parent: _c, curve: Curves.easeInOut);
  }

  @override
  void dispose() { _c.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final base = widget.compact
        ? Theme.of(context).textTheme.headlineSmall?.copyWith(
            fontWeight: FontWeight.w800,
            letterSpacing: -0.2,
            color: Colors.white,
          )
        : Theme.of(context).textTheme.headlineMedium?.copyWith(
            fontWeight: FontWeight.w900,
            letterSpacing: -0.4,
            color: Colors.white,
          );
    return AnimatedBuilder(
      animation: _anim,
      builder: (context, _) {
        final wave = (math.sin(_anim.value * math.pi * 2) + 1) / 2; // 0..1
        return ShaderMask(
          shaderCallback: (rect) {
            return LinearGradient(
              colors: [
                Colors.white,
                Colors.white.withOpacity(0.85 + 0.05 * wave),
                Colors.white,
              ],
              stops: const [0.0, 0.5, 1.0],
            ).createShader(rect);
          },
          blendMode: BlendMode.srcATop,
          child: Text(
            widget.text,
            style: base?.copyWith(letterSpacing: (widget.compact ? -0.2 : -0.4) + 0.4 * wave),
          ),
        );
      },
    );
  }
}

/// -------- BENTO GRID LAYOUT (no extra deps) --------
class _BentoGrid extends StatelessWidget {
  const _BentoGrid({required this.children, this.spacing = 12, this.runSpacing = 12});
  final List<_BentoItem> children;
  final double spacing; final double runSpacing;

  int _columnsForWidth(double w) {
    if (w >= 1000) return 6; // desktop/tablet large
    if (w >= 700) return 4;  // tablet
    return 2;                // phone
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, c) {
        final cols = _columnsForWidth(c.maxWidth);
        final cellW = (c.maxWidth - spacing * (cols - 1)) / cols;
        final cellH = cellW * 0.72; // 균등 카드 비율 (시각적 안정감)

        // Simple flow layout using Wrap
        return Wrap(
          spacing: spacing,
          runSpacing: runSpacing,
          children: children.map((it) {
            final w = cellW * it.spanX + spacing * (it.spanX - 1);
            final h = cellH * it.spanY + runSpacing * (it.spanY - 1);
            return SizedBox(width: w, height: h, child: it.child);
          }).toList(),
        );
      },
    );
  }
}

class _BentoItem {
  const _BentoItem({required this.child, this.spanX = 1, this.spanY = 1});
  final Widget child; final int spanX; final int spanY;
}

/// -------- GLASS TILE (3D + blur + grain + kinetic hover) --------
class _GlassTile extends StatefulWidget {
  const _GlassTile({required this.label, required this.icon, required this.route, required this.accent});
  final String label; final IconData icon; final String route; final Color accent;
  @override
  State<_GlassTile> createState() => _GlassTileState();
}

class _GlassTileState extends State<_GlassTile> with SingleTickerProviderStateMixin {
  Offset _tilt = Offset.zero;
  bool _pressed = false;
  late final AnimationController _pressC = AnimationController(vsync: this, duration: const Duration(milliseconds: 140));

  void _onPointer(PointerEvent e, Size size) {
    final local = e.localPosition;
    final dx = (local.dx / size.width - 0.5) * 2;  // -1..1
    final dy = (local.dy / size.height - 0.5) * 2; // -1..1
    setState(() => _tilt = Offset(dx, dy));
  }

  void _resetTilt() => setState(() => _tilt = Offset.zero);

  @override
  void dispose() { _pressC.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return LayoutBuilder(builder: (context, c) {
      final size = Size(c.maxWidth, c.maxHeight);
      final angleX = -_tilt.dy * 0.035; // radians
      final angleY = _tilt.dx * 0.035;

      final isPointerHeavy = kIsWeb || const {
        TargetPlatform.windows,
        TargetPlatform.linux,
        TargetPlatform.macOS,
      }.contains(Theme.of(context).platform);

      return MouseRegion(
        onHover: isPointerHeavy ? (e) => _onPointer(e, size) : null,
        onExit: isPointerHeavy ? (_) => _resetTilt() : null,
        child: Listener(
          onPointerMove: isPointerHeavy ? (e) => _onPointer(e, size) : null,
          onPointerUp: isPointerHeavy ? (_) => _resetTilt() : null,
          child: GestureDetector(
            onTapDown: (_) { HapticFeedback.lightImpact(); setState(() => _pressed = true); _pressC.forward(from: 0); },
            onTapUp: (_) { setState(() => _pressed = false); Navigator.pushNamed(context, widget.route); },
            onTapCancel: () { setState(() => _pressed = false); },
            child: AnimatedScale(
              scale: _pressed ? 0.98 : 1.0,
              duration: const Duration(milliseconds: 120),
              child: Transform(
                alignment: Alignment.center,
                transform: Matrix4.identity()
                  ..setEntry(3, 2, 0.001)
                  ..rotateX(angleX)
                  ..rotateY(angleY),
                child: Stack(
                  children: [
                    // Soft 3D shadow
                    Positioned.fill(
                      child: DecoratedBox(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(20),
                          boxShadow: [
                            BoxShadow(color: Colors.black.withOpacity(0.22), blurRadius: 16, spreadRadius: -6, offset: const Offset(0, 16)),
                            BoxShadow(color: widget.accent.withOpacity(0.15), blurRadius: 14, spreadRadius: -8),
                          ],
                        ),
                      ),
                    ),

                    // Glass layer
                    ClipRRect(
                      borderRadius: BorderRadius.circular(20),
                      child: Stack(
                        children: [
                          // Backdrop blur
                          BackdropFilter(
                            filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                            child: Container(),
                          ),
                          // Inner gradient & border
                          Container(
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  Colors.white.withOpacity(0.16),
                                  Colors.white.withOpacity(0.06),
                                ],
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                              ),
                              border: Border.all(color: Colors.white.withOpacity(0.22)),
                              borderRadius: BorderRadius.circular(20),
                            ),
                          ),
                          // Grain painter (procedural, no assets)
                          const _GrainOverlay(opacity: 0.025),

                          // Content
                          _TileContent(label: widget.label, icon: widget.icon, accent: widget.accent, cs: cs),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      );
    });
  }
}

class _TileContent extends StatelessWidget {
  const _TileContent({required this.label, required this.icon, required this.accent, required this.cs});
  final String label; final IconData icon; final Color accent; final ColorScheme cs;
  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, c) {
        final h = c.maxHeight;
        final iconSize = h.isFinite ? h * 0.28 : 56.0;
        final clampedIcon = iconSize.clamp(40.0, 56.0);
        final gap = (h.isFinite ? h * 0.06 : 8.0).clamp(4.0, 12.0);
        final pad = EdgeInsets.all(h.isFinite && h < 140 ? 16 : 20);

        return Padding(
          padding: pad,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: clampedIcon,
                height: clampedIcon,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(16),
                  gradient: LinearGradient(
                    colors: [accent, accent.withOpacity(0.6)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  border: Border.all(color: Colors.white.withOpacity(0.22)),
                ),
                child: Icon(icon, color: Colors.white),
              ),
              SizedBox(height: gap),
              Text(
                label,
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w800,
                  color: cs.onSurface,
                  letterSpacing: -0.1,
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        );
      },
    );
  }
}

/// Procedural grain overlay painter
class _GrainOverlay extends StatelessWidget {
  const _GrainOverlay({this.opacity = 0.05});
  final double opacity;
  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _GrainPainter(opacity: opacity),
      size: Size.infinite,
    );
  }
}

class _GrainPainter extends CustomPainter {
  _GrainPainter({required this.opacity});
  final double opacity;
  final math.Random _rng = math.Random(42);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.black.withOpacity(opacity);
    // draw sparse noise
    final count = (size.width * size.height / 600).clamp(200, 1200).toInt();
    for (int i = 0; i < count; i++) {
      final dx = _rng.nextDouble() * size.width;
      final dy = _rng.nextDouble() * size.height;
      canvas.drawRect(Rect.fromLTWH(dx, dy, 1, 1), paint);
    }
  }

  @override
  bool shouldRepaint(covariant _GrainPainter oldDelegate) => false;
}