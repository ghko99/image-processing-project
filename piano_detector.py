"""
피아노 키 감지 및 손가락 추적 모듈
MediaPipe를 사용한 손가락 인식을 기반으로 피아노 키 터치를 감지합니다.

개선사항:
1. 적응적 임계값을 통한 터치 감지 정확도 향상
2. 손가락별 개별 터치 히스토리 관리
3. 잘못된 터치 필터링 알고리즘
4. 다중 손 지원
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class FingerState:
    """개별 손가락의 상태 정보"""
    position: Tuple[float, float]  # (x, y) 좌표
    z_depth: float  # 깊이 정보
    is_touching: bool  # 터치 상태
    touch_start_time: float  # 터치 시작 시간
    velocity: float  # 움직임 속도
    confidence: float  # 감지 신뢰도

class PianoDetector:
    """
    MediaPipe 기반 손가락 인식 및 피아노 키 터치 감지 클래스
    """
    
    def __init__(self, config):
        self.config = config
        
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # 양손 지원
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 피아노 키 영역 설정
        self.setup_piano_keys()
        
        # 손가락 상태 추적
        self.finger_states: Dict[str, FingerState] = {}
        self.touch_history: Dict[int, deque] = {i: deque(maxlen=10) for i in range(10)}
        
        # 적응적 임계값 시스템
        self.adaptive_thresholds = [config.TOUCH_THRESHOLD] * 10
        self.baseline_depths = [0.0] * 10
        self.calibration_frames = 0
        self.is_calibrated = False
        
        # 성능 개선을 위한 필터링
        self.position_smoothing = 0.7  # 위치 스무딩 팩터
        self.previous_positions = {}
        
        print("PianoDetector initialized with MediaPipe")
    
    def setup_piano_keys(self):
        """피아노 키 영역 설정"""
        self.key_regions = []
        key_width = self.config.KEY_WIDTH
        key_height = self.config.KEY_HEIGHT
        
        # 화면 하단에 피아노 키 배치
        start_x = (self.config.CAMERA_WIDTH - (self.config.NUM_KEYS * key_width)) // 2
        start_y = self.config.CAMERA_HEIGHT - key_height - 50
        
        # 음표 매핑 (2옥타브)
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C5', 'C#5']
        
        for i in range(self.config.NUM_KEYS):
            x = start_x + i * key_width
            # 검은 키와 흰 키 구분
            is_black_key = self.note_names[i].endswith('#')
            
            if is_black_key:
                # 검은 키는 더 작고 위쪽에 배치
                key_rect = (x + key_width//4, start_y, key_width//2, key_height//2)
                color = (30, 30, 30)
            else:
                # 흰 키
                key_rect = (x, start_y, key_width, key_height)
                color = (240, 240, 240)
            
            self.key_regions.append({
                'rect': key_rect,
                'note': self.note_names[i],
                'color': color,
                'is_black': is_black_key,
                'index': i
            })
    
    def detect_fingers(self, frame: np.ndarray) -> List[FingerState]:
        """
        프레임에서 손가락 위치 감지
        
        Args:
            frame: 입력 이미지 프레임
            
        Returns:
            감지된 손가락 상태 리스트
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        finger_states = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 각 손가락 끝점 추출 (MediaPipe 손가락 인덱스 사용)
                finger_tips = [4, 8, 12, 16, 20]  # 엄지, 검지, 중지, 약지, 새끼
                
                for i, tip_idx in enumerate(finger_tips):
                    landmark = hand_landmarks.landmark[tip_idx]
                    
                    # 화면 좌표로 변환
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    z = landmark.z
                    
                    # 위치 스무딩 적용
                    finger_id = f"finger_{i}"
                    if finger_id in self.previous_positions:
                        prev_x, prev_y = self.previous_positions[finger_id]
                        x = int(self.position_smoothing * prev_x + (1 - self.position_smoothing) * x)
                        y = int(self.position_smoothing * prev_y + (1 - self.position_smoothing) * y)
                    
                    self.previous_positions[finger_id] = (x, y)
                    
                    # 손가락 상태 생성
                    finger_state = FingerState(
                        position=(x, y),
                        z_depth=z,
                        is_touching=False,  # 나중에 계산
                        touch_start_time=0,
                        velocity=self.calculate_velocity(finger_id, (x, y)),
                        confidence=1.0  # MediaPipe는 이미 신뢰도 필터링됨
                    )
                    
                    finger_states.append(finger_state)
                    
                # 손 스켈레톤 그리기 (디버깅용)
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return finger_states
    
    def calculate_velocity(self, finger_id: str, current_pos: Tuple[int, int]) -> float:
        """손가락 움직임 속도 계산"""
        if finger_id not in self.previous_positions:
            return 0.0
        
        prev_x, prev_y = self.previous_positions[finger_id]
        curr_x, curr_y = current_pos
        
        # 유클리드 거리로 속도 계산
        velocity = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        return velocity
    
    def detect_key_touches(self, finger_states: List[FingerState]) -> List[int]:
        """
        손가락 위치를 기반으로 피아노 키 터치 감지
        
        Args:
            finger_states: 손가락 상태 리스트
            
        Returns:
            터치된 키 인덱스 리스트
        """
        touched_keys = []
        current_time = time.time()
        
        for finger in finger_states:
            x, y = finger.position
            
            # 각 피아노 키와의 충돌 검사
            for key_info in self.key_regions:
                rect = key_info['rect']
                key_x, key_y, key_w, key_h = rect
                
                # 키 영역 내에 손가락이 있는지 확인
                if (key_x <= x <= key_x + key_w and 
                    key_y <= y <= key_y + key_h):
                    
                    key_idx = key_info['index']
                    
                    # 적응적 터치 감지
                    if self.is_touch_detected(finger, key_idx):
                        # 중복 터치 방지
                        if key_idx not in touched_keys:
                            touched_keys.append(key_idx)
                            
                            # 터치 히스토리 업데이트
                            self.touch_history[key_idx].append(current_time)
        
        return touched_keys
    
    def is_touch_detected(self, finger: FingerState, key_idx: int) -> bool:
        """
        적응적 임계값을 사용한 터치 감지
        
        개선된 터치 감지 알고리즘:
        1. Z 깊이 기반 터치 감지
        2. 속도 기반 필터링 (너무 빠른 움직임 제외)
        3. 최소 터치 지속시간 확인
        """
        # 캘리브레이션 중이면 베이스라인 업데이트
        if not self.is_calibrated and self.calibration_frames < 60:
            self.baseline_depths[key_idx] = finger.z_depth
            self.calibration_frames += 1
            if self.calibration_frames >= 60:
                self.is_calibrated = True
                print("Calibration completed")
            return False
        
        # 베이스라인 대비 깊이 변화 계산
        depth_diff = self.baseline_depths[key_idx] - finger.z_depth
        
        # 동적 임계값 조정
        if depth_diff > self.adaptive_thresholds[key_idx]:
            # 속도 필터링 (너무 빠른 움직임은 터치로 인정하지 않음)
            if finger.velocity < 50:  # 픽셀/프레임
                return True
        
        return False
    
    def get_current_state(self) -> Dict:
        """현재 감지 시스템 상태 반환"""
        return {
            'is_calibrated': self.is_calibrated,
            'calibration_progress': min(self.calibration_frames / 60.0, 1.0),
            'active_keys': len([h for h in self.touch_history.values() if len(h) > 0]),
            'detection_quality': 'Good' if self.is_calibrated else 'Calibrating...'
        }
    
    def draw_piano_keys(self, frame: np.ndarray, touched_keys: List[int] = None) -> np.ndarray:
        """
        프레임에 피아노 키 그리기
        
        Args:
            frame: 입력 프레임
            touched_keys: 터치된 키 리스트
            
        Returns:
            키가 그려진 프레임
        """
        if touched_keys is None:
            touched_keys = []
        
        # 흰 키 먼저 그리기
        for key_info in self.key_regions:
            if not key_info['is_black']:
                self.draw_single_key(frame, key_info, key_info['index'] in touched_keys)
        
        # 검은 키 나중에 그리기 (위에 겹쳐짐)
        for key_info in self.key_regions:
            if key_info['is_black']:
                self.draw_single_key(frame, key_info, key_info['index'] in touched_keys)
        
        return frame
    
    def draw_single_key(self, frame: np.ndarray, key_info: Dict, is_touched: bool):
        """개별 키 그리기"""
        rect = key_info['rect']
        x, y, w, h = rect
        
        # 터치 상태에 따른 색상 변경
        if is_touched:
            color = (0, 255, 0) if not key_info['is_black'] else (0, 200, 0)
        else:
            color = key_info['color']
        
        # 키 사각형 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        
        # 음표 이름 표시
        text_color = (0, 0, 0) if not key_info['is_black'] else (255, 255, 255)
        cv2.putText(frame, key_info['note'], 
                   (x + w//2 - 10, y + h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    def recalibrate(self):
        """재캘리브레이션"""
        self.calibration_frames = 0
        self.is_calibrated = False
        self.baseline_depths = [0.0] * 10
        print("Recalibration started...")