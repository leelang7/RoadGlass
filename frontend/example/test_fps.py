#!/usr/bin/env python3
"""
FPS 테스트 및 분석 스크립트
실시간으로 FPS 로그를 모니터링하고 분석합니다.
"""

import json
import time
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class FPSMonitor:
    def __init__(self, log_file="run_log.jsonl"):
        self.log_file = log_file
        self.fps_data = []
        self.timestamps = []
        self.last_position = 0
        
    def monitor_realtime(self, duration=60):
        """실시간 FPS 모니터링"""
        print(f"🔍 실시간 FPS 모니터링 시작 (최대 {duration}초)")
        print("=" * 50)
        
        start_time = time.time()
        last_fps_update = 0
        
        while time.time() - start_time < duration:
            if os.path.exists(self.log_file):
                try:
                    with open(self.log_file, 'r', encoding='utf-8') as f:
                        f.seek(self.last_position)
                        new_lines = f.readlines()
                        self.last_position = f.tell()
                        
                        for line in new_lines:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    if data.get("event") == "on_result" and "fps" in data:
                                        fps = data["fps"]
                                        if isinstance(fps, (int, float)) and fps > 0:
                                            self.fps_data.append(fps)
                                            self.timestamps.append(data.get("t_us", 0))
                                            
                                            # 실시간 FPS 출력 (1초마다)
                                            current_time = time.time()
                                            if current_time - last_fps_update >= 1.0:
                                                avg_fps = np.mean(self.fps_data[-10:]) if len(self.fps_data) >= 10 else np.mean(self.fps_data)
                                                print(f"⏱️  {datetime.now().strftime('%H:%M:%S')} - 현재 FPS: {fps:.1f} (평균: {avg_fps:.1f})")
                                                last_fps_update = current_time
                                                
                                except json.JSONDecodeError:
                                    continue
                                    
            time.sleep(0.1)  # 100ms마다 체크
            
        print(f"\n✅ 모니터링 완료! 총 {len(self.fps_data)}개 FPS 데이터 수집")
        return self.fps_data
    
    def analyze_data(self):
        """수집된 데이터 분석"""
        if not self.fps_data:
            print("❌ 분석할 FPS 데이터가 없습니다.")
            return
            
        print("\n📊 FPS 분석 결과")
        print("=" * 50)
        
        fps_array = np.array(self.fps_data)
        
        print(f"총 프레임 수: {len(fps_array)}")
        print(f"평균 FPS: {np.mean(fps_array):.2f}")
        print(f"중간값 FPS: {np.median(fps_array):.2f}")
        print(f"최소 FPS: {np.min(fps_array):.2f}")
        print(f"최대 FPS: {np.max(fps_array):.2f}")
        print(f"표준편차: {np.std(fps_array):.2f}")
        
        # FPS 분포
        fps_30_plus = np.sum(fps_array >= 30)
        fps_60_plus = np.sum(fps_array >= 60)
        fps_120_plus = np.sum(fps_array >= 120)
        
        print(f"\nFPS 분포:")
        print(f"  30+ FPS: {fps_30_plus}/{len(fps_array)} ({fps_30_plus/len(fps_array)*100:.1f}%)")
        print(f"  60+ FPS: {fps_60_plus}/{len(fps_array)} ({fps_60_plus/len(fps_array)*100:.1f}%)")
        print(f"  120+ FPS: {fps_120_plus}/{len(fps_array)} ({fps_120_plus/len(fps_array)*100:.1f}%)")
        
        # 성능 등급
        avg_fps = np.mean(fps_array)
        if avg_fps >= 60:
            grade = "🟢 우수"
        elif avg_fps >= 30:
            grade = "🟡 보통"
        else:
            grade = "🔴 개선 필요"
            
        print(f"\n성능 등급: {grade} (평균 {avg_fps:.1f} FPS)")
        
    def plot_fps(self, save_file="fps_realtime.png"):
        """FPS 그래프 생성"""
        if not self.fps_data:
            print("❌ 그래프를 그릴 데이터가 없습니다.")
            return
            
        plt.figure(figsize=(12, 6))
        
        # 시간축 생성 (초 단위)
        if self.timestamps:
            start_time = min(self.timestamps)
            time_seconds = [(t - start_time) / 1_000_000 for t in self.timestamps]
        else:
            time_seconds = list(range(len(self.fps_data)))
            
        plt.plot(time_seconds, self.fps_data, 'b-', linewidth=1, alpha=0.7)
        plt.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='30 FPS')
        plt.axhline(y=60, color='g', linestyle='--', alpha=0.5, label='60 FPS')
        plt.axhline(y=120, color='purple', linestyle='--', alpha=0.5, label='120 FPS')
        
        plt.title('실시간 FPS 모니터링')
        plt.xlabel('시간 (초)')
        plt.ylabel('FPS')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"📈 그래프가 '{save_file}'로 저장되었습니다.")
        
    def run_test(self, duration=60):
        """전체 테스트 실행"""
        print("🚀 FPS 테스트 시작")
        print(f"로그 파일: {self.log_file}")
        print(f"테스트 시간: {duration}초")
        
        # 실시간 모니터링
        self.monitor_realtime(duration)
        
        # 데이터 분석
        self.analyze_data()
        
        # 그래프 생성
        self.plot_fps()
        
        print("\n✅ FPS 테스트 완료!")

def main():
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    else:
        duration = 60
        
    monitor = FPSMonitor()
    monitor.run_test(duration)

if __name__ == "__main__":
    main()

