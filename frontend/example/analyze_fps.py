import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def analyze_fps(log_file):
    fps_values = []
    timestamps = []
    imu_fps_values = []
    imu_timestamps = []
    
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # on_result 이벤트 (실제 추론 결과)
            if rec.get("event") == "on_result":
                fps = rec.get("fps", 0)
                if isinstance(fps, (int, float)) and fps > 0:
                    fps_values.append(fps)
                    timestamps.append(rec.get("t_us", 0))
            
            # periodic_imu 이벤트 (IMU 기반 FPS)
            elif rec.get("event") == "periodic_imu":
                fps = rec.get("fps", 0)
                if isinstance(fps, (int, float)) and fps > 0:
                    imu_fps_values.append(fps)
                    imu_timestamps.append(rec.get("t_us", 0))

    if not fps_values and not imu_fps_values:
        print("⚠️ FPS 데이터가 없습니다.")
        return

    print("=" * 50)
    print("📊 FPS 분석 결과")
    print("=" * 50)
    
    # on_result FPS 분석
    if fps_values:
        avg_fps = sum(fps_values) / len(fps_values)
        min_fps = min(fps_values)
        max_fps = max(fps_values)
        std_fps = np.std(fps_values)
        
        print(f"\n🎯 추론 결과 FPS (on_result):")
        print(f"   총 프레임: {len(fps_values)}")
        print(f"   평균 FPS: {avg_fps:.2f}")
        print(f"   최소 FPS: {min_fps:.2f}")
        print(f"   최대 FPS: {max_fps:.2f}")
        print(f"   표준편차: {std_fps:.2f}")
        
        # FPS 분포 분석
        fps_30_plus = sum(1 for f in fps_values if f >= 30)
        fps_60_plus = sum(1 for f in fps_values if f >= 60)
        print(f"   30+ FPS: {fps_30_plus}/{len(fps_values)} ({fps_30_plus/len(fps_values)*100:.1f}%)")
        print(f"   60+ FPS: {fps_60_plus}/{len(fps_values)} ({fps_60_plus/len(fps_values)*100:.1f}%)")
    
    # IMU FPS 분석
    if imu_fps_values:
        avg_imu_fps = sum(imu_fps_values) / len(imu_fps_values)
        min_imu_fps = min(imu_fps_values)
        max_imu_fps = max(imu_fps_values)
        std_imu_fps = np.std(imu_fps_values)
        
        print(f"\n📱 IMU 기반 FPS (periodic_imu):")
        print(f"   총 샘플: {len(imu_fps_values)}")
        print(f"   평균 FPS: {avg_imu_fps:.2f}")
        print(f"   최소 FPS: {min_imu_fps:.2f}")
        print(f"   최대 FPS: {max_imu_fps:.2f}")
        print(f"   표준편차: {std_imu_fps:.2f}")
    
    # 그래프 생성
    if fps_values or imu_fps_values:
        plt.figure(figsize=(12, 8))
        
        if fps_values:
            # 시간축을 초 단위로 변환
            start_time = min(timestamps) if timestamps else 0
            time_seconds = [(t - start_time) / 1_000_000 for t in timestamps]
            plt.subplot(2, 1, 1)
            plt.plot(time_seconds, fps_values, 'b-', linewidth=1, alpha=0.7)
            plt.title('추론 결과 FPS (on_result)')
            plt.xlabel('시간 (초)')
            plt.ylabel('FPS')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='30 FPS')
            plt.axhline(y=60, color='g', linestyle='--', alpha=0.5, label='60 FPS')
            plt.legend()
        
        if imu_fps_values:
            start_time = min(imu_timestamps) if imu_timestamps else 0
            time_seconds = [(t - start_time) / 1_000_000 for t in imu_timestamps]
            plt.subplot(2, 1, 2)
            plt.plot(time_seconds, imu_fps_values, 'g-', linewidth=1, alpha=0.7)
            plt.title('IMU 기반 FPS (periodic_imu)')
            plt.xlabel('시간 (초)')
            plt.ylabel('FPS')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='30 FPS')
            plt.axhline(y=60, color='g', linestyle='--', alpha=0.5, label='60 FPS')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('fps_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n📈 그래프가 'fps_analysis.png'로 저장되었습니다.")
        
        # 히스토그램
        plt.figure(figsize=(10, 6))
        if fps_values:
            plt.hist(fps_values, bins=30, alpha=0.7, label='추론 결과 FPS', color='blue')
        if imu_fps_values:
            plt.hist(imu_fps_values, bins=30, alpha=0.7, label='IMU 기반 FPS', color='green')
        plt.xlabel('FPS')
        plt.ylabel('빈도')
        plt.title('FPS 분포 히스토그램')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('fps_histogram.png', dpi=150, bbox_inches='tight')
        print(f"📊 히스토그램이 'fps_histogram.png'로 저장되었습니다.")

# 실행 예시
if __name__ == "__main__":
    analyze_fps("run_log.jsonl")