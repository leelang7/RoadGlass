import 'package:flutter/material.dart';
import 'dart:ui';

class GuideScreen extends StatefulWidget {
  const GuideScreen({Key? key}) : super(key: key);

  @override
  _GuideScreenState createState() => _GuideScreenState();
}

class _GuideScreenState extends State<GuideScreen> with SingleTickerProviderStateMixin {
  final Color _primary = const Color(0xFF1565C0); // match main primary
  final Color _navy = const Color(0xFF0D1B2A);

  late AnimationController _controller;
  late Animation<double> _iconAnimation;
  late Animation<double> _titleAnimation;
  late Animation<double> _subtitleAnimation;
  bool _showScrollHint = true;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    );

    _iconAnimation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _controller, curve: const Interval(0.0, 0.5, curve: Curves.easeOut)),
    );

    _titleAnimation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _controller, curve: const Interval(0.4, 0.8, curve: Curves.easeOut)),
    );

    _subtitleAnimation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _controller, curve: const Interval(0.7, 1.0, curve: Curves.easeOut)),
    );

    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Widget _buildGlassmorphicButton(BuildContext context) {
    return GestureDetector(
      onTap: () {
        Navigator.pushNamed(context, '/camera');
      },
      child: ClipRRect(
        borderRadius: BorderRadius.circular(40),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(vertical: 18),
            decoration: BoxDecoration(
              color: _primary.withOpacity(0.12),
              borderRadius: BorderRadius.circular(40),
              border: Border.all(color: _primary.withOpacity(0.35), width: 1.2),
              boxShadow: [
                BoxShadow(
                  color: _primary.withOpacity(0.18),
                  offset: const Offset(0, 4),
                  blurRadius: 10,
                  spreadRadius: 1,
                ),
              ],
            ),
            alignment: Alignment.center,
            child: Text(
              '시작하기',
              style: TextStyle(
                color: _primary,
                fontWeight: FontWeight.bold,
                fontSize: 22,
                letterSpacing: 1.1,
              ),
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Full-bleed background gradient (covers entire device width)
          Positioned.fill(
            child: Container(
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  colors: [Color(0xFFEAF6FF), Color(0xFFB3E5FC)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
              ),
            ),
          ),

          // Main content with horizontal padding (does NOT affect overlays)
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 36),
              child: Stack(
                children: [
                  Center(
                    child: NotificationListener<ScrollNotification>(
                      onNotification: (notification) {
                        if (_showScrollHint && notification.metrics.pixels > 8) {
                          setState(() {
                            _showScrollHint = false;
                          });
                        }
                        return false;
                      },
                      child: SingleChildScrollView(
                        physics: const BouncingScrollPhysics(),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const SizedBox(height: 56),
                            FadeTransition(
                              opacity: _iconAnimation,
                              child: Hero(
                                tag: 'app-logo',
                                child: Container(
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    gradient: LinearGradient(
                                      colors: [_primary.withOpacity(0.85), const Color(0xFF64B5F6).withOpacity(0.65)],
                                      begin: Alignment.topLeft,
                                      end: Alignment.bottomRight,
                                    ),
                                    boxShadow: [
                                      BoxShadow(
                                        color: _primary.withOpacity(0.25),
                                        blurRadius: 20,
                                        offset: const Offset(0, 8),
                                      ),
                                    ],
                                  ),
                                  padding: const EdgeInsets.all(28),
                                  child: Icon(
                                    Icons.photo_camera_rounded,
                                    size: 96,
                                    color: Colors.white.withOpacity(0.9),
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(height: 48),
                            FadeTransition(
                              opacity: _titleAnimation,
                              child: Hero(
                                tag: 'app-title',
                                child: Text(
                                  'RoadGlass',
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                    fontSize: 48,
                                    fontWeight: FontWeight.w900,
                                    color: _primary,
                                    letterSpacing: 1.5,
                                    shadows: [
                                      Shadow(
                                        color: _primary.withOpacity(0.18),
                                        offset: const Offset(0, 3),
                                        blurRadius: 6,
                                      ),
                                    ],
                                    decoration: TextDecoration.none,
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(height: 16),
                            FadeTransition(
                              opacity: _subtitleAnimation,
                              child: Text(
                                '당신의 안전한 운전을 위한 가이드',
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.w500,
                                  color: _navy.withOpacity(0.85),
                                  letterSpacing: 0.8,
                                  height: 1.4,
                                ),
                              ),
                            ),
                            const SizedBox(height: 32),
                            FadeTransition(
                              opacity: _subtitleAnimation,
                              child: Column(
                                children: [
                                  _buildGuideStep(Icons.stay_current_landscape, "스마트폰을 차량 거치대에\n단단히 고정하세요."),
                                  const SizedBox(height: 16),
                                  _buildGuideStep(Icons.cleaning_services, "렌즈를 깨끗하게 유지해\n인식률을 높이세요."),
                                  const SizedBox(height: 16),
                                  _buildGuideStep(Icons.traffic, "표지판, 차선 등 도로 요소가\n잘 보이게 촬영하세요."),
                                  const SizedBox(height: 16),
                                  _buildGuideStep(Icons.block, "운전 중에는 조작하지 말고,\n정차 후 이용하세요."),
                                  const SizedBox(height: 16),
                                  _buildGuideStep(Icons.lock, "촬영 데이터는\n안전하게 보호됩니다"),
                                ],
                              ),
                            ),
                            const SizedBox(height: 48),
                            _buildGlassmorphicButton(context),
                            const SizedBox(height: 48),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Scroll hint overlay (full width, unaffected by inner padding)
          if (_showScrollHint) ...[
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              height: 80,
              child: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [
                      Colors.transparent,
                      Colors.black.withOpacity(0.15),
                      Colors.black.withOpacity(0.25),
                    ],
                  ),
                ),
              ),
            ),
            Positioned(
              left: 0,
              right: 0,
              bottom: 32,
              child: _AnimatedScrollHintArrow(),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildGuideStep(IconData icon, String text) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.9),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.blueGrey.withOpacity(0.1)),
        boxShadow: [
          BoxShadow(
            color: Colors.blueGrey.withOpacity(0.15),
            blurRadius: 12,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Row(
        children: [
          Icon(icon, color: _primary, size: 28),
          const SizedBox(width: 16),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                color: _navy,
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _AnimatedScrollHintArrow extends StatefulWidget {
  @override
  State<_AnimatedScrollHintArrow> createState() => _AnimatedScrollHintArrowState();
}

class _AnimatedScrollHintArrowState extends State<_AnimatedScrollHintArrow> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _offsetAnimation;
  late Animation<double> _opacityAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1100),
      vsync: this,
    )..repeat(reverse: true);
    _offsetAnimation = Tween<double>(begin: 0, end: 18).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
    _opacityAnimation = Tween<double>(begin: 1, end: 0.4).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final Color _navy = const Color(0xFF0D1B2A);
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Opacity(
          opacity: _opacityAnimation.value,
          child: Transform.translate(
            offset: Offset(0, _offsetAnimation.value),
            child: Icon(
              Icons.keyboard_arrow_down,
              size: 48,
              color: _navy.withOpacity(0.75),
            ),
          ),
        );
      },
    );
  }
}