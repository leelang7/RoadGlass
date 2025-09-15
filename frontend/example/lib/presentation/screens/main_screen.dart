import 'package:flutter/material.dart';
import 'dart:ui';

class MainScreen extends StatelessWidget {
  const MainScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      resizeToAvoidBottomInset: false,
      body: Stack(
        children: [
          // 1) Soft gradient background
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  Color(0xFFEFF6FF), // very light blue
                  Color(0xFFDCEBFF), // soft pastel blue
                ],
              ),
            ),
          ),

          // 2) Decorative blurred bubbles for a modern vibe
          Positioned(
            top: -40,
            left: -30,
            child: _BlurBubble(
              diameter: 180,
              color: const Color(0xFF8AB6FF).withOpacity(0.25),
            ),
          ),
          Positioned(
            bottom: -50,
            right: -40,
            child: _BlurBubble(
              diameter: 220,
              color: const Color(0xFF5CA8FF).withOpacity(0.20),
            ),
          ),

          // 3) Center frosted-glass card with logo + title + subtitle
          SafeArea(
            child: Center(
              child: _GlassCard(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Logo a little lower from very top
                    const SizedBox(height: 8),
                    Hero(
                      tag: 'app-logo',
                      child: CircleAvatar(
                        radius: 56,
                        backgroundColor: Colors.white,
                        backgroundImage: const AssetImage(
                          'assets/icon/MapPinRoad_maxfill_blue_1024.png',
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),
                    Hero(
                      tag: 'app-title',
                      child: Text(
                        'RoadGlass',
                        style: const TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.w900,
                          color: Color(0xFF1565C0),
                          decoration: TextDecoration.none,
                          letterSpacing: -0.2,
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '도로 상태를 더 똑똑하게 기록해요',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: const Color(0xFF334155),
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'IMU 기반 이벤트 순간만 캡쳐해\n백엔드로 전송합니다.',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: const Color(0xFF64748B),
                        height: 1.4,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 8),
                  ],
                ),
              ),
            ),
          ),

          // 4) Big bottom CTA (capsule) — leads to Guide screen
          SafeArea(
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 520),
                  child: _PrimaryCta(
                    label: '촬영 시작',
                    icon: Icons.camera_alt_rounded,
                    onPressed: () => Navigator.pushNamed(context, '/guide'),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Frosted glass container with subtle border + shadow
class _GlassCard extends StatelessWidget {
  const _GlassCard({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 16, sigmaY: 16),
        child: Container(
          padding: const EdgeInsets.fromLTRB(24, 18, 24, 22),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.55),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: Colors.white.withOpacity(0.8),
              width: 1,
            ),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF3B82F6).withOpacity(0.10),
                blurRadius: 28,
                spreadRadius: 4,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: child,
        ),
      ),
    );
  }
}

/// Big pill-shaped primary button with slight shadow
class _PrimaryCta extends StatelessWidget {
  const _PrimaryCta({
    required this.label,
    required this.icon,
    required this.onPressed,
  });

  final String label;
  final IconData icon;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Material(
      elevation: 8,
      shadowColor: const Color(0xFF2563EB).withOpacity(0.25),
      borderRadius: BorderRadius.circular(40),
      child: Ink(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(40),
          gradient: const LinearGradient(
            colors: [Color(0xFF3B82F6), Color(0xFF2563EB)], // vibrant blue
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: InkWell(
          borderRadius: BorderRadius.circular(40),
          onTap: onPressed,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 18),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon, color: Colors.white),
                const SizedBox(width: 10),
                Text(
                  label,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                    letterSpacing: 0.2,
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

/// Soft blurred colored circle used as background accent
class _BlurBubble extends StatelessWidget {
  const _BlurBubble({required this.diameter, required this.color});

  final double diameter;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return ClipOval(
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
        child: Container(
          width: diameter,
          height: diameter,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: color,
          ),
        ),
      ),
    );
  }
}