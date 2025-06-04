"""
UI 관리 모듈
사용자 인터페이스 및 시각화를 담당합니다.

개선사항:
1. 실시간 성능 모니터링 UI
2. 터치 감지 상태 시각화
3. 조정 가능한 UI 요소들
4. 사용자 친화적인 정보 표시
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

class UIManager:
    """
    사용자 인터페이스 관리 클래스
    프레임에 피아노 키, 상태 정보, 성능 지표 등을 그려줍니다.
    """
    
    def __init__(self, config):
        self.config = config
        
        # UI 설정
        self.show_debug_info = True
        self.show_performance_info = True
        self.calibration_mode = False
        
        # 색상 정의
        self.colors = {
            'white_key': (240, 240, 240),
            'black_key': (30, 30, 30),
            'touched_white': (100, 255, 100),
            'touched_black': (50, 200, 50),
            'finger_point': (0, 0, 255),
            'text': (255, 255, 255),
            'background': (40, 40, 40),
            'success': (0, 255, 0),
            'warning': (0, 255, 255),
            'error': (0, 0, 255)
        }
        
        # 폰트 설정
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        
        # UI 영역 정의
        self.info_panel_width = 300
        self.setup_ui_regions()
        
        print("UIManager initialized")
    
    def setup_ui_regions(self):
        """UI 영역 설정"""
        # 정보 패널 영역 (우측)
        self.info_panel = {
            'x': self.config.CAMERA_WIDTH - self.info_panel_width,
            'y': 0,
            'width': self.info_panel_width,
            'height': 400
        }
        
        # 피아노 키 영역 (하단)
        self.piano_area = {
            'x': 0,
            'y': self.config.CAMERA_HEIGHT - 250,
            'width': self.config.CAMERA_WIDTH - self.info_panel_width,
            'height': 250
        }
        
        # 메인 카메라 영역
        self.camera_area = {
            'x': 0,
            'y': 0,
            'width': self.config.CAMERA_WIDTH - self.info_panel_width,
            'height': self.config.CAMERA_HEIGHT - 250
        }
    
    def render_frame(self, frame: np.ndarray, detector_state: Dict, 
                    fps: float, touched_keys: List[int] = None) -> np.ndarray:
        """
        완전한 UI가 포함된 프레임 렌더링
        
        Args:
            frame: 기본 카메라 프레임
            detector_state: 감지기 상태 정보
            fps: 현재 FPS
            touched_keys: 터치된 키 리스트
            
        Returns:
            렌더링된 프레임
        """
        if touched_keys is None:
            touched_keys = []
        
        # 프레임 크기 조정 (필요시)
        display_frame = cv2.resize(frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
        
        # 배경 확장 (정보 패널용)
        extended_frame = np.zeros((self.config.CAMERA_HEIGHT, 
                                 self.config.CAMERA_WIDTH + self.info_panel_width, 3), 
                                dtype=np.uint8)
        extended_frame[:, :self.config.CAMERA_WIDTH] = display_frame
        
        # 정보 패널 배경
        cv2.rectangle(extended_frame, 
                     (self.config.CAMERA_WIDTH, 0),
                     (self.config.CAMERA_WIDTH + self.info_panel_width, self.config.CAMERA_HEIGHT),
                     self.colors['background'], -1)
        
        # 피아노 키 그리기
        self.draw_piano_interface(extended_frame, touched_keys)
        
        # 손가락 포인트 그리기
        self.draw_finger_points(extended_frame, detector_state)
        
        # 정보 패널 그리기
        self.draw_info_panel(extended_frame, detector_state, fps)
        
        # 상태 메시지 그리기
        self.draw_status_messages(extended_frame, detector_state)
        
        # 도움말 그리기 (필요시)
        if self.calibration_mode:
            self.draw_calibration_help(extended_frame)
        
        return extended_frame
    
    def draw_piano_interface(self, frame: np.ndarray, touched_keys: List[int]):
        """피아노 인터페이스 그리기"""
        # 피아노 영역 배경
        piano_bg = self.piano_area
        cv2.rectangle(frame, 
                     (piano_bg['x'], piano_bg['y']),
                     (piano_bg['x'] + piano_bg['width'], piano_bg['y'] + piano_bg['height']),
                     (20, 20, 20), -1)
        
        # 키 설정
        key_width = 60
        key_height = 180
        start_x = piano_bg['x'] + 20
        start_y = piano_bg['y'] + 30
        
        # 음표 이름
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C5', 'C#5']
        
        # 흰 키 먼저 그리기
        white_key_index = 0
        for i in range(self.config.NUM_KEYS):
            if '#' not in note_names[i]:  # 흰 키
                x = start_x + white_key_index * (key_width + 2)
                y = start_y
                
                # 터치 상태에 따른 색상
                color = self.colors['touched_white'] if i in touched_keys else self.colors['white_key']
                
                # 키 그리기
                cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), color, -1)
                cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), (0, 0, 0), 2)
                
                # 음표 이름
                text_color = (0, 0, 0)
                cv2.putText(frame, note_names[i], 
                           (x + key_width//2 - 8, y + key_height - 10),
                           self.font, 0.5, text_color, 1)
                
                white_key_index += 1
        
        # 검은 키 그리기
        black_key_positions = [0.5, 1.5, 3.5, 4.5, 5.5, 7.5, 8.5]  # 흰 키 기준 상대 위치
        black_key_index = 0
        
        for i in range(self.config.NUM_KEYS):
            if '#' in note_names[i]:  # 검은 키
                if black_key_index < len(black_key_positions):
                    x = start_x + int(black_key_positions[black_key_index] * (key_width + 2)) - key_width//4
                    y = start_y
                    
                    # 터치 상태에 따른 색상
                    color = self.colors['touched_black'] if i in touched_keys else self.colors['black_key']
                    
                    # 키 그리기 (검은 키는 더 작음)
                    black_width = key_width // 2
                    black_height = key_height * 2 // 3
                    
                    cv2.rectangle(frame, (x, y), (x + black_width, y + black_height), color, -1)
                    cv2.rectangle(frame, (x, y), (x + black_width, y + black_height), (100, 100, 100), 2)
                    
                    # 음표 이름
                    text_color = (255, 255, 255)
                    cv2.putText(frame, note_names[i], 
                               (x + 5, y + black_height - 10),
                               self.font, 0.4, text_color, 1)
                    
                    black_key_index += 1
    
    def draw_finger_points(self, frame: np.ndarray, detector_state: Dict):
        """손가락 포인트 시각화"""
        # 이 부분은 detector에서 finger_states를 받아올 수 있도록 수정 필요
        # 현재는 기본 시각화만 구현
        pass
    
    def draw_info_panel(self, frame: np.ndarray, detector_state: Dict, fps: float):
        """정보 패널 그리기"""
        panel_x = self.config.CAMERA_WIDTH + 10
        y_offset = 30
        line_height = 25
        
        # 제목
        cv2.putText(frame, "Virtual Piano Status", 
                   (panel_x, y_offset), 
                   self.font, 0.7, self.colors['text'], 2)
        y_offset += line_height * 2
        
        # FPS 정보
        fps_color = self.colors['success'] if fps > 25 else self.colors['warning']
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (panel_x, y_offset), 
                   self.font, self.font_scale, fps_color, self.font_thickness)
        y_offset += line_height
        
        # 캘리브레이션 상태
        if detector_state.get('is_calibrated', False):
            cv2.putText(frame, "Status: Ready", 
                       (panel_x, y_offset), 
                       self.font, self.font_scale, self.colors['success'], self.font_thickness)
        else:
            progress = detector_state.get('calibration_progress', 0.0)
            cv2.putText(frame, f"Calibrating: {progress*100:.0f}%", 
                       (panel_x, y_offset), 
                       self.font, self.font_scale, self.colors['warning'], self.font_thickness)
        y_offset += line_height
        
        # 활성 키 수
        active_keys = detector_state.get('active_keys', 0)
        cv2.putText(frame, f"Active Keys: {active_keys}", 
                   (panel_x, y_offset), 
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        y_offset += line_height
        
        # 감지 품질
        quality = detector_state.get('detection_quality', 'Unknown')
        quality_color = self.colors['success'] if quality == 'Good' else self.colors['warning']
        cv2.putText(frame, f"Quality: {quality}", 
                   (panel_x, y_offset), 
                   self.font, self.font_scale, quality_color, self.font_thickness)
        y_offset += line_height * 2
        
        # 컨트롤 도움말
        self.draw_control_help(frame, panel_x, y_offset)
    
    def draw_control_help(self, frame: np.ndarray, x: int, y: int):
        """컨트롤 도움말 그리기"""
        controls = [
            "Controls:",
            "Q: Quit",
            "R: Record",
            "P: Play recording",
            "C: Calibrate",
            "S: Save config",
            "H: Help"
        ]
        
        for i, control in enumerate(controls):
            color = self.colors['text'] if i == 0 else (200, 200, 200)
            weight = 2 if i == 0 else 1
            cv2.putText(frame, control, 
                       (x, y + i * 20), 
                       self.font, 0.5, color, weight)
    
    def draw_status_messages(self, frame: np.ndarray, detector_state: Dict):
        """상태 메시지 그리기"""
        if not detector_state.get('is_calibrated', False):
            msg = "Keep hands steady for calibration..."
            text_size = cv2.getTextSize(msg, self.font, 0.7, 2)[0]
            x = (self.config.CAMERA_WIDTH - text_size[0]) // 2
            y = 50
            
            # 배경 사각형
            cv2.rectangle(frame, (x-10, y-25), (x+text_size[0]+10, y+10), (0, 0, 0), -1)
            cv2.putText(frame, msg, (x, y), self.font, 0.7, self.colors['warning'], 2)
    
    def draw_calibration_help(self, frame: np.ndarray):
        """캘리브레이션 도움말 그리기"""
        help_text = [
            "Calibration Mode",
            "1. Keep hands at normal playing position",
            "2. Stay still for 2-3 seconds",
            "3. System will auto-calibrate touch sensitivity"
        ]
        
        y_start = self.config.CAMERA_HEIGHT - 150
        for i, text in enumerate(help_text):
            color = self.colors['warning'] if i == 0 else self.colors['text']
            weight = 2 if i == 0 else 1
            cv2.putText(frame, text, (10, y_start + i * 25), 
                       self.font, 0.6, color, weight)
    
    def toggle_calibration_mode(self):
        """캘리브레이션 모드 토글"""
        self.calibration_mode = not self.calibration_mode
        status = "enabled" if self.calibration_mode else "disabled"
        print(f"Calibration mode {status}")
    
    def toggle_debug_info(self):
        """디버그 정보 표시 토글"""
        self.show_debug_info = not self.show_debug_info
    
    def set_ui_theme(self, theme: str):
        """UI 테마 변경"""
        if theme == "dark":
            self.colors['background'] = (40, 40, 40)
            self.colors['text'] = (255, 255, 255)
        elif theme == "light":
            self.colors['background'] = (240, 240, 240)
            self.colors['text'] = (0, 0, 0)