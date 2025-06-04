"""
연주 녹음 및 재생 모듈
사용자의 피아노 연주를 녹음하고 재생할 수 있습니다.

개선사항:
1. 타이밍 정확도가 높은 녹음 시스템
2. JSON 형식으로 연주 데이터 저장
3. 재생 시 원본과 동일한 타이밍 재현
4. 여러 녹음 파일 관리 기능
"""

import json
import time
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os

@dataclass
class TouchEvent:
    """터치 이벤트 정보"""
    timestamp: float
    keys: List[int]
    duration: float = 0.0

@dataclass
class Recording:
    """녹음 정보"""
    name: str
    created_at: str
    duration: float
    events: List[TouchEvent]
    metadata: Dict

class RecordingManager:
    """
    연주 녹음 및 재생 관리 클래스
    """
    
    def __init__(self, config):
        self.config = config
        
        # 녹음 상태
        self.is_recording_active = False
        self.current_recording: Optional[Recording] = None
        self.recording_start_time = 0.0
        self.touch_events: List[TouchEvent] = []
        
        # 재생 상태
        self.is_playing = False
        self.playback_thread: Optional[threading.Thread] = None
        
        # 저장 디렉토리
        self.recordings_dir = "recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # 로드된 녹음들
        self.saved_recordings: Dict[str, Recording] = {}
        self.load_all_recordings()
        
        print(f"RecordingManager initialized")
        print(f"Recordings directory: {self.recordings_dir}")
    
    def start_recording(self, name: Optional[str] = None) -> bool:
        """
        녹음 시작
        
        Args:
            name: 녹음 이름 (None이면 자동 생성)
            
        Returns:
            녹음 시작 성공 여부
        """
        if self.is_recording_active:
            print("Recording already in progress")
            return False
        
        if name is None:
            name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.is_recording_active = True
        self.recording_start_time = time.time()
        self.touch_events = []
        
        # 새 녹음 객체 생성
        self.current_recording = Recording(
            name=name,
            created_at=datetime.now().isoformat(),
            duration=0.0,
            events=[],
            metadata={
                "config": {
                    "num_keys": self.config.NUM_KEYS,
                    "touch_threshold": self.config.TOUCH_THRESHOLD
                }
            }
        )
        
        print(f"Recording started: {name}")
        return True
    
    def stop_recording(self) -> Optional[Recording]:
        """
        녹음 중지 및 저장
        
        Returns:
            완성된 녹음 객체
        """
        if not self.is_recording_active:
            print("No recording in progress")
            return None
        
        self.is_recording_active = False
        recording_duration = time.time() - self.recording_start_time
        
        if self.current_recording:
            self.current_recording.duration = recording_duration
            self.current_recording.events = self.touch_events.copy()
            
            # 터치 이벤트들의 지속시간 계산
            self.calculate_event_durations()
            
            # 파일로 저장
            self.save_recording(self.current_recording)
            
            print(f"Recording stopped: {self.current_recording.name}")
            print(f"Duration: {recording_duration:.2f}s, Events: {len(self.touch_events)}")
            
            return self.current_recording
        
        return None
    
    def add_touch_event(self, keys: List[int], timestamp: float):
        """
        터치 이벤트 추가
        
        Args:
            keys: 터치된 키 리스트
            timestamp: 타임스탬프
        """
        if not self.is_recording_active or not keys:
            return
        
        # 녹음 시작 시간 기준으로 상대 시간 계산
        relative_time = timestamp - self.recording_start_time
        
        # 새 터치 이벤트 생성
        event = TouchEvent(
            timestamp=relative_time,
            keys=keys.copy()
        )
        
        self.touch_events.append(event)
    
    def calculate_event_durations(self):
        """터치 이벤트들의 지속시간 계산"""
        if len(self.touch_events) <= 1:
            return
        
        # 연속된 이벤트 간의 시간 차이로 지속시간 추정
        for i in range(len(self.touch_events) - 1):
            current_event = self.touch_events[i]
            next_event = self.touch_events[i + 1]
            
            # 같은 키가 다음 이벤트에도 있으면 지속시간 연장
            time_diff = next_event.timestamp - current_event.timestamp
            current_event.duration = min(time_diff, 2.0)  # 최대 2초
        
        # 마지막 이벤트는 기본 지속시간
        if self.touch_events:
            self.touch_events[-1].duration = 0.5
    
    def save_recording(self, recording: Recording):
        """
        녹음을 파일로 저장
        
        Args:
            recording: 저장할 녹음 객체
        """
        filename = f"{recording.name}.json"
        filepath = os.path.join(self.recordings_dir, filename)
        
        try:
            # TouchEvent 객체들을 딕셔너리로 변환
            recording_dict = asdict(recording)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(recording_dict, f, indent=2, ensure_ascii=False)
            
            self.saved_recordings[recording.name] = recording
            print(f"Recording saved: {filepath}")
            
        except Exception as e:
            print(f"Error saving recording: {e}")
    
    def load_recording(self, filename: str) -> Optional[Recording]:
        """
        파일에서 녹음 로드
        
        Args:
            filename: 녹음 파일명
            
        Returns:
            로드된 녹음 객체
        """
        filepath = os.path.join(self.recordings_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # TouchEvent 객체들 복원
            events = [TouchEvent(**event) for event in data['events']]
            
            recording = Recording(
                name=data['name'],
                created_at=data['created_at'],
                duration=data['duration'],
                events=events,
                metadata=data.get('metadata', {})
            )
            
            return recording
            
        except Exception as e:
            print(f"Error loading recording {filename}: {e}")
            return None
    
    def load_all_recordings(self):
        """모든 저장된 녹음 로드"""
        if not os.path.exists(self.recordings_dir):
            return
        
        for filename in os.listdir(self.recordings_dir):
            if filename.endswith('.json'):
                recording = self.load_recording(filename)
                if recording:
                    self.saved_recordings[recording.name] = recording
        
        print(f"Loaded {len(self.saved_recordings)} recordings")
    
    def play_recording(self, recording_name: Optional[str] = None) -> bool:
        """
        녹음 재생
        
        Args:
            recording_name: 재생할 녹음 이름 (None이면 가장 최근 녹음)
            
        Returns:
            재생 시작 성공 여부
        """
        if self.is_playing:
            print("Already playing a recording")
            return False
        
        # 재생할 녹음 선택
        target_recording = None
        if recording_name and recording_name in self.saved_recordings:
            target_recording = self.saved_recordings[recording_name]
        elif self.current_recording:
            target_recording = self.current_recording
        elif self.saved_recordings:
            # 가장 최근 녹음 선택
            latest_name = max(self.saved_recordings.keys(), 
                            key=lambda x: self.saved_recordings[x].created_at)
            target_recording = self.saved_recordings[latest_name]
        
        if not target_recording or not target_recording.events:
            print("No recording to play")
            return False
        
        # 별도 스레드에서 재생
        self.playback_thread = threading.Thread(
            target=self._playback_worker, 
            args=(target_recording,), 
            daemon=True
        )
        self.playback_thread.start()
        
        return True
    
    def _playback_worker(self, recording: Recording):
        """
        재생 작업자 스레드
        
        Args:
            recording: 재생할 녹음
        """
        self.is_playing = True
        print(f"Playing recording: {recording.name}")
        print(f"Duration: {recording.duration:.2f}s, Events: {len(recording.events)}")
        
        try:
            # 재생 시작 시간
            playback_start = time.time()
            
            # 이벤트별 재생
            for event in recording.events:
                if not self.is_playing:  # 중도 중단 확인
                    break
                
                # 이벤트 시간까지 대기
                target_time = playback_start + event.timestamp
                current_time = time.time()
                
                if target_time > current_time:
                    time.sleep(target_time - current_time)
                
                # 오디오 매니저에서 음성 재생 (연결 필요)
                self._play_event_audio(event)
                
                print(f"Played: keys {event.keys} at {event.timestamp:.2f}s")
            
        except Exception as e:
            print(f"Error during playback: {e}")
        finally:
            self.is_playing = False
            print("Playback completed")
    
    def _play_event_audio(self, event: TouchEvent):
        """
        이벤트의 오디오 재생 (AudioManager 연결 필요)
        
        Args:
            event: 재생할 터치 이벤트
        """
        # 실제 구현에서는 AudioManager 인스턴스 필요
        # 현재는 콘솔 출력으로 대체
        for key in event.keys:
            # audio_manager.play_note(key, event.duration)
            pass
    
    def stop_playback(self):
        """재생 중지"""
        if self.is_playing:
            self.is_playing = False
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=1.0)
            print("Playback stopped")
    
    def delete_recording(self, recording_name: str) -> bool:
        """
        녹음 삭제
        
        Args:
            recording_name: 삭제할 녹음 이름
            
        Returns:
            삭제 성공 여부
        """
        if recording_name not in self.saved_recordings:
            print(f"Recording not found: {recording_name}")
            return False
        
        try:
            # 파일 삭제
            filename = f"{recording_name}.json"
            filepath = os.path.join(self.recordings_dir, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # 메모리에서 제거
            del self.saved_recordings[recording_name]
            
            print(f"Recording deleted: {recording_name}")
            return True
            
        except Exception as e:
            print(f"Error deleting recording: {e}")
            return False
    
    def list_recordings(self) -> List[str]:
        """저장된 녹음 목록 반환"""
        return list(self.saved_recordings.keys())
    
    def get_recording_info(self, recording_name: str) -> Optional[Dict]:
        """
        녹음 정보 반환
        
        Args:
            recording_name: 녹음 이름
            
        Returns:
            녹음 정보 딕셔너리
        """
        if recording_name not in self.saved_recordings:
            return None
        
        recording = self.saved_recordings[recording_name]
        return {
            'name': recording.name,
            'created_at': recording.created_at,
            'duration': recording.duration,
            'event_count': len(recording.events),
            'unique_keys': len(set(key for event in recording.events for key in event.keys)),
            'metadata': recording.metadata
        }
    
    def export_recording(self, recording_name: str, export_format: str = 'midi') -> bool:
        """
        녹음을 다른 형식으로 내보내기
        
        Args:
            recording_name: 내보낼 녹음 이름
            export_format: 내보내기 형식 ('midi', 'csv' 등)
            
        Returns:
            내보내기 성공 여부
        """
        if recording_name not in self.saved_recordings:
            print(f"Recording not found: {recording_name}")
            return False
        
        recording = self.saved_recordings[recording_name]
        
        try:
            if export_format.lower() == 'csv':
                return self._export_to_csv(recording)
            elif export_format.lower() == 'midi':
                return self._export_to_midi(recording)
            else:
                print(f"Unsupported export format: {export_format}")
                return False
                
        except Exception as e:
            print(f"Error exporting recording: {e}")
            return False
    
    def _export_to_csv(self, recording: Recording) -> bool:
        """CSV 형식으로 내보내기"""
        import csv
        
        filename = f"{recording.name}.csv"
        filepath = os.path.join(self.recordings_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'keys', 'duration'])
            
            for event in recording.events:
                writer.writerow([event.timestamp, ','.join(map(str, event.keys)), event.duration])
        
        print(f"Recording exported to CSV: {filepath}")
        return True
    
    def _export_to_midi(self, recording: Recording) -> bool:
        """MIDI 형식으로 내보내기 (기본 구현)"""
        # MIDI 라이브러리가 필요 (예: python-midi, mido 등)
        # 여기서는 기본적인 텍스트 형태로만 구현
        
        filename = f"{recording.name}_midi.txt"
        filepath = os.path.join(self.recordings_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"MIDI Export for {recording.name}\n")
            f.write(f"Duration: {recording.duration:.2f}s\n")
            f.write("Events:\n")
            
            for event in recording.events:
                f.write(f"Time: {event.timestamp:.3f}s, Notes: {event.keys}, Duration: {event.duration:.3f}s\n")
        
        print(f"Recording exported to MIDI format: {filepath}")
        return True
    
    def is_recording(self) -> bool:
        """현재 녹음 중인지 확인"""
        return self.is_recording_active
    
    def is_playing_back(self) -> bool:
        """현재 재생 중인지 확인"""
        return self.is_playing
    
    def get_recording_status(self) -> Dict:
        """현재 녹음/재생 상태 반환"""
        status = {
            'is_recording': self.is_recording_active,
            'is_playing': self.is_playing,
            'current_recording_name': self.current_recording.name if self.current_recording else None,
            'saved_recordings_count': len(self.saved_recordings),
            'total_events': len(self.touch_events) if self.is_recording_active else 0
        }
        
        if self.is_recording_active:
            status['recording_duration'] = time.time() - self.recording_start_time
        
        return status