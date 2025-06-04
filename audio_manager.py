"""
오디오 관리 모듈
pygame을 사용하여 피아노 음성을 생성하고 재생합니다.

개선사항:
1. 실시간 사인파 생성으로 자연스러운 피아노 음색 구현
2. ADSR 엔벨로프 적용으로 realistic한 음색 표현
3. 다중 음성 동시 재생 (화음) 지원
4. 음량 조절 및 페이드 효과
"""

import pygame
import numpy as np
import threading
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import math

@dataclass
class Note:
    """음표 정보 클래스"""
    frequency: float
    name: str
    octave: int
    duration: float = 0.0
    volume: float = 1.0

class ADSREnvelope:
    """
    ADSR (Attack, Decay, Sustain, Release) 엔벨로프 클래스
    자연스러운 피아노 음색을 위한 음량 변화 곡선
    """
    
    def __init__(self, attack: float = 0.01, decay: float = 0.1, 
                 sustain: float = 0.7, release: float = 0.5):
        self.attack = attack    # 공격 시간
        self.decay = decay      # 감쇠 시간
        self.sustain = sustain  # 지속 음량
        self.release = release  # 해제 시간
    
    def get_amplitude(self, t: float, note_duration: float) -> float:
        """
        시간 t에서의 음량 계산
        
        Args:
            t: 현재 시간 (초)
            note_duration: 음표 지속 시간
            
        Returns:
            0.0~1.0 사이의 음량 값
        """
        if t < 0:
            return 0.0
        
        # Attack phase
        if t < self.attack:
            return t / self.attack
        
        # Decay phase
        elif t < self.attack + self.decay:
            decay_progress = (t - self.attack) / self.decay
            return 1.0 - (1.0 - self.sustain) * decay_progress
        
        # Sustain phase
        elif t < note_duration - self.release:
            return self.sustain
        
        # Release phase
        elif t < note_duration:
            release_progress = (t - (note_duration - self.release)) / self.release
            return self.sustain * (1.0 - release_progress)
        
        # End
        else:
            return 0.0

class AudioManager:
    """
    오디오 재생 관리 클래스
    pygame mixer를 사용하여 실시간 오디오 생성 및 재생
    """
    
    def __init__(self, config):
        self.config = config
        
        # pygame mixer 초기화
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # 오디오 설정
        self.sample_rate = 44100
        self.channels = 2
        self.bit_depth = 16
        
        # 음표 주파수 매핑 (2옥타브)
        self.setup_note_frequencies()
        
        # ADSR 엔벨로프
        self.envelope = ADSREnvelope()
        
        # 활성 음표 관리
        self.active_notes: Dict[int, Dict] = {}
        self.note_threads: Dict[int, threading.Thread] = {}
        
        # 마스터 볼륨
        self.master_volume = 0.3
        
        # 음색 설정
        self.harmonics = [1.0, 0.5, 0.25, 0.125]  # 피아노 음색을 위한 하모닉스
        
        print("AudioManager initialized")
        print(f"Sample rate: {self.sample_rate}Hz")
        print(f"Audio format: {self.bit_depth}bit, {self.channels} channels")
    
    def setup_note_frequencies(self):
        """음표 주파수 설정"""
        # C4 = 261.63Hz를 기준으로 계산 (4번째 옥타브)
        base_freq = 261.63
        
        # 반음 간격 (12 톤 균등 분할)
        semitone_ratio = 2.0 ** (1.0 / 12.0)
        
        self.note_frequencies = {}
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for i in range(self.config.NUM_KEYS):
            # 음표 이름과 옥타브 계산
            note_index = i % 12
            octave = 4 + (i // 12)
            note_name = note_names[note_index]
            
            # 주파수 계산
            frequency = base_freq * (semitone_ratio ** i)
            
            self.note_frequencies[i] = Note(
                frequency=frequency,
                name=note_name,
                octave=octave
            )
            
            print(f"Key {i}: {note_name}{octave} = {frequency:.2f}Hz")
    
    def generate_tone(self, frequency: float, duration: float, volume: float = 1.0) -> np.ndarray:
        """
        지정된 주파수와 지속시간으로 음성 생성
        
        Args:
            frequency: 주파수 (Hz)
            duration: 지속시간 (초)
            volume: 음량 (0.0~1.0)
            
        Returns:
            생성된 오디오 데이터
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, False)
        
        # 기본 사인파 + 하모닉스로 피아노 음색 생성
        wave = np.zeros(num_samples)
        for i, harmonic_amp in enumerate(self.harmonics):
            harmonic_freq = frequency * (i + 1)
            harmonic_wave = harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t)
            wave += harmonic_wave
        
        # ADSR 엔벨로프 적용
        envelope_values = np.array([self.envelope.get_amplitude(time, duration) for time in t])
        wave *= envelope_values
        
        # 볼륨 조절
        wave *= volume * self.master_volume
        
        # 클리핑 방지
        wave = np.clip(wave, -1.0, 1.0)
        
        # 스테레오 변환 (16비트)
        stereo_wave = np.array([wave, wave]).T
        audio_data = (stereo_wave * 32767).astype(np.int16)
        
        return audio_data
    
    def play_note(self, key_index: int, duration: float = 1.0):
        """
        지정된 키의 음표 재생
        
        Args:
            key_index: 키 인덱스
            duration: 지속시간 (초)
        """
        if key_index not in self.note_frequencies:
            return
        
        # 이미 재생 중인 같은 키는 중복 재생 방지
        if key_index in self.active_notes:
            return
        
        note = self.note_frequencies[key_index]
        
        # 별도 스레드에서 음성 생성 및 재생
        def play_thread():
            try:
                # 음성 생성
                audio_data = self.generate_tone(note.frequency, duration)
                
                # pygame sound 객체 생성
                sound_array = pygame.sndarray.make_sound(audio_data)
                
                # 재생
                channel = pygame.mixer.find_channel()
                if channel:
                    self.active_notes[key_index] = {
                        'channel': channel,
                        'start_time': time.time(),
                        'note': note
                    }
                    
                    channel.play(sound_array)
                    
                    # 재생 완료까지 대기
                    while channel.get_busy():
                        time.sleep(0.01)
                    
                    # 활성 노트에서 제거
                    if key_index in self.active_notes:
                        del self.active_notes[key_index]
                
            except Exception as e:
                print(f"Error playing note {key_index}: {e}")
                if key_index in self.active_notes:
                    del self.active_notes[key_index]
        
        # 스레드 시작
        thread = threading.Thread(target=play_thread, daemon=True)
        thread.start()
        self.note_threads[key_index] = thread
    
    def stop_note(self, key_index: int):
        """특정 키의 음표 재생 중지"""
        if key_index in self.active_notes:
            channel = self.active_notes[key_index]['channel']
            channel.stop()
            del self.active_notes[key_index]
    
    def stop_all_notes(self):
        """모든 음표 재생 중지"""
        pygame.mixer.stop()
        self.active_notes.clear()
    
    def set_master_volume(self, volume: float):
        """
        마스터 볼륨 설정
        
        Args:
            volume: 볼륨 (0.0~1.0)
        """
        self.master_volume = max(0.0, min(1.0, volume))
        pygame.mixer.set_num_channels(32)  # 충분한 채널 확보
    
    def get_active_notes(self) -> List[int]:
        """현재 재생 중인 음표 리스트 반환"""
        return list(self.active_notes.keys())
    
    def play_chord(self, key_indices: List[int], duration: float = 1.0):
        """
        화음 재생
        
        Args:
            key_indices: 키 인덱스 리스트
            duration: 지속시간 (초)
        """
        for key_index in key_indices:
            self.play_note(key_index, duration)
    
    def cleanup(self):
        """리소스 정리"""
        print("Cleaning up audio resources...")
        self.stop_all_notes()
        
        # 모든 스레드 종료 대기
        for thread in self.note_threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        pygame.mixer.quit()
        print("Audio cleanup completed")