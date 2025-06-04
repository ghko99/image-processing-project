"""
카메라 기반 손가락 인식 피아노 시스템
Author: [Your Name]
Date: 2025-06-04

이 프로젝트는 MediaPipe를 기반으로 한 손가락 인식을 통해 가상 피아노를 연주할 수 있는 시스템입니다.
기존 오픈소스 대비 다음과 같은 개선사항을 포함합니다:
1. 적응적 임계값을 통한 터치 감지 정확도 향상
2. 손가락별 개별 감도 조정 기능
3. 음표 지속시간 제어 및 화음 연주 지원
4. 실시간 연주 녹음 및 재생 기능
"""

import cv2
import numpy as np
import pygame
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from piano_detector import PianoDetector
from audio_manager import AudioManager
from ui_manager import UIManager
from recorder import RecordingManager

@dataclass
class Config:
    """시스템 설정 관리 클래스"""
    # 카메라 설정
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    CAMERA_FPS: int = 30
    
    # 피아노 키 설정
    NUM_KEYS: int = 14  # 2옥타브 (C부터 시작)
    KEY_WIDTH: int = 80
    KEY_HEIGHT: int = 200
    
    # 터치 감지 설정
    TOUCH_THRESHOLD: float = 0.02  # 기본 터치 임계값
    MIN_TOUCH_DURATION: float = 0.1  # 최소 터치 지속시간 (초)
    
    # UI 설정
    WINDOW_NAME: str = "Virtual Piano - Finger Recognition"
    
    @classmethod
    def load_from_file(cls, filename: str = "config.json") -> 'Config':
        """설정 파일에서 로드"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                return cls(**data)
        except FileNotFoundError:
            return cls()
    
    def save_to_file(self, filename: str = "config.json"):
        """설정을 파일로 저장"""
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

class VirtualPiano:
    """
    메인 가상 피아노 클래스
    카메라 입력을 받아 손가락 위치를 인식하고 피아노 연주를 처리합니다.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        
        # 구성 요소 초기화
        self.detector = PianoDetector(config)
        self.audio_manager = AudioManager(config)
        self.ui_manager = UIManager(config)
        self.recorder = RecordingManager(config)
        
        # 카메라 초기화
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        # 성능 모니터링
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("Virtual Piano System initialized successfully!")
        print(f"Camera resolution: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
        print(f"Number of keys: {config.NUM_KEYS}")
    
    def run(self):
        """메인 실행 루프"""
        self.running = True
        print("Starting Virtual Piano... Press 'q' to quit")
        
        try:
            while self.running:
                success, frame = self.cap.read()
                if not success:
                    print("Failed to read from camera")
                    break
                
                # 프레임 처리
                processed_frame = self.process_frame(frame)
                
                # UI 업데이트 및 표시
                display_frame = self.ui_manager.render_frame(
                    processed_frame, 
                    self.detector.get_current_state(),
                    self.current_fps
                )
                
                cv2.imshow(self.config.WINDOW_NAME, display_frame)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
                
                # FPS 계산
                self.calculate_fps()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임 처리 및 손가락 인식
        
        Args:
            frame: 입력 카메라 프레임
            
        Returns:
            처리된 프레임
        """
        # 프레임 전처리
        frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 효과)
        
        # 손가락 위치 감지
        finger_positions = self.detector.detect_fingers(frame)
        
        # 피아노 키 터치 감지 및 오디오 재생
        touched_keys = self.detector.detect_key_touches(finger_positions)
        
        # 터치된 키에 대해 음성 재생
        for key_idx in touched_keys:
            self.audio_manager.play_note(key_idx)
        
        # 녹음 중이면 터치 정보 기록
        if self.recorder.is_recording():
            self.recorder.add_touch_event(touched_keys, time.time())
        
        return frame
    
    def handle_keyboard_input(self, key: int) -> bool:
        """
        키보드 입력 처리
        
        Args:
            key: 입력된 키 코드
            
        Returns:
            계속 실행할지 여부
        """
        if key == ord('q'):
            return False
        elif key == ord('r'):
            # 녹음 시작/중지
            if self.recorder.is_recording():
                self.recorder.stop_recording()
                print("Recording stopped")
            else:
                self.recorder.start_recording()
                print("Recording started")
        elif key == ord('p'):
            # 녹음 재생
            self.recorder.play_recording()
            print("Playing recording")
        elif key == ord('s'):
            # 설정 저장
            self.config.save_to_file()
            print("Configuration saved")
        elif key == ord('c'):
            # 설정 조정 모드
            self.ui_manager.toggle_calibration_mode()
        elif key == ord('h'):
            # 도움말 표시
            self.show_help()
        
        return True
    
    def calculate_fps(self):
        """FPS 계산 및 업데이트"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # 30프레임마다 FPS 업데이트
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
            self.fps_counter = 0
    
    def show_help(self):
        """도움말 출력"""
        help_text = """
        === Virtual Piano Controls ===
        q: Quit
        r: Start/Stop recording
        p: Play recorded performance
        s: Save configuration
        c: Toggle calibration mode
        h: Show this help
        
        === Performance Tips ===
        - Keep hands at comfortable distance from camera
        - Ensure good lighting for better detection
        - Use calibration mode to adjust sensitivity
        """
        print(help_text)
    
    def cleanup(self):
        """리소스 정리"""
        print("Cleaning up resources...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # 오디오 시스템 정리
        self.audio_manager.cleanup()
        
        print("Cleanup completed")

def main():
    """메인 함수"""
    print("=== Camera-based Finger Recognition Piano ===")
    print("Loading configuration...")
    
    # 설정 로드
    config = Config.load_from_file()
    
    # 가상 피아노 시스템 생성 및 실행
    piano = VirtualPiano(config)
    piano.run()

if __name__ == "__main__":
    main()