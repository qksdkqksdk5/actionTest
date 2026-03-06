import cv2
import numpy as np
import os
import time
import threading
from queue import Queue
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
from .base_detector import BaseDetector  # BaseDetector가 같은 폴더에 있다고 가정

class ReverseDetector(BaseDetector):
    def __init__(self, cctv_name, url, lat=37.5, lng=127.0, socketio=None, db=None, ResultModel=None, ReverseModel=None, conf=None, app=None):
        # 부모 클래스(BaseDetector) 초기화
        super().__init__(cctv_name, url, app=app, socketio=socketio, db=db, ResultModel=ResultModel)
        
        self.lat = lat  
        self.lng = lng 
        self.ReverseModel = ReverseModel
        
        # 환경 변수 및 설정
        env_conf = os.getenv('CONFIDENCE_THRESHOLD')
        self.conf = float(env_conf) if env_conf else (conf if conf else 0.66)
        
        # 모델 및 캡처 설정
        self.model = YOLO("best_DW.pt")
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 버퍼 최소화
        
        # 상태 제어 변수
        self.alerted_ids = set()
        
        # --- [역주행 알고리즘 관련 변수] ---
        self.prev_frame_gray = None
        self.SCENE_THRESHOLD = 30.0
        self.GRID_SIZE = 15
        self.COS_THRESHOLD = -0.6
        self.VELOCITY_WINDOW = 5
        self.LEARNING_FRAMES = 150
        self.frame_count = 0
        self.learning_done = False
        
        self.flow_map = None
        self.flow_count = None
        self.trajectories = defaultdict(list)
        self.wrong_way_count = defaultdict(int)
        self.wrong_way_ids = set()

        # 모델 세이브/로드 설정
        safe_name = cctv_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        self.save_dir = "learned_models"
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_file = os.path.join(self.save_dir, f"flow_{safe_name}.npy")
        
        # 학습된 데이터 로드
        self.load_flow_map()

    def load_flow_map(self):
        """저장된 이동 경로 맵 로드"""
        if os.path.exists(self.model_file):
            try:
                data = np.load(self.model_file, allow_pickle=True).item()
                self.flow_map = data["flow"]
                self.flow_count = data["count"]
                self.learning_done = True
                # 로드된 데이터 기반으로 그리드 크기 계산
                self.grid_h, self.grid_w = self.flow_map.shape[:2]
                print(f"✅ [{self.cctv_name}] 학습 모델 로드 완료")
            except Exception as e:
                print(f"⚠️ [{self.cctv_name}] 모델 로드 실패: {e}")
                self.learning_done = False

    def save_flow_map(self):
        """학습된 이동 경로 맵 저장"""
        if self.flow_map is not None:
            np.save(self.model_file, {"flow": self.flow_map, "count": self.flow_count})
            print(f"💾 [{self.cctv_name}] 학습 완료 및 모델 저장됨")

    def init_grid(self, w, h):
        """그리드 초기화"""
        self.grid_h = h // self.GRID_SIZE + 1
        self.grid_w = w // self.GRID_SIZE + 1
        self.flow_map = np.zeros((self.grid_h, self.grid_w, 2), dtype=np.float32)
        self.flow_count = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)

    def apply_spatial_smoothing(self):
        """학습 데이터 보정 (주변 벡터 참조)"""
        new_flow = self.flow_map.copy()
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if self.flow_count[y, x] < 3:
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                                if self.flow_count[ny, nx] >= 3:
                                    neighbors.append(self.flow_map[ny, nx])
                    if neighbors:
                        new_flow[y, x] = np.mean(neighbors, axis=0)
        self.flow_map = new_flow

    def reset_learning(self, w, h):
        """화면 전환 시 학습 데이터 초기화"""
        print(f"🔄 [{self.cctv_name}] 화면 전환 감지: 재학습 시작")
        self.frame_count = 0
        self.learning_done = False
        self.init_grid(w, h)
        self.trajectories.clear()
        self.wrong_way_count.clear()
        self.wrong_way_ids.clear()

    def process_alert(self, data):
        """[Worker Thread] 비동기 DB 저장 및 알림 처리"""
        frame, alert_time, track_id = data
        try:
            with self.app.app_context():
                # 1. 공통 결과 테이블 저장 (DetectionResult)
                new_alert = self.ResultModel(
                    event_type="reverse", address=self.cctv_name,
                    latitude=self.lat, longitude=self.lng,
                    detected_at=alert_time, is_simulation=False, is_resolved=False
                )
                self.db.session.add(new_alert)
                self.db.session.flush()

                # 2. 이미지 파일 저장
                ts = alert_time.strftime("%Y%m%d_%H%M%S")
                filename = f"reverse_real_{new_alert.id}_{ts}.jpg"
                save_path = os.path.join(self.app.root_path, "static", "captures")
                os.makedirs(save_path, exist_ok=True)
                filepath = os.path.join(save_path, filename)
                cv2.imwrite(filepath, frame)

                # 3. 역주행 상세 테이블 저장 (ReverseResult)
                from models import ReverseResult
                reverse_detail = ReverseResult(
                    result_id=new_alert.id,
                    image_path=f"/static/captures/{filename}",
                    vehicle_info=f"ID:{track_id} 실시간 탐지"
                )
                self.db.session.add(reverse_detail)
                self.db.session.commit()

                # 4. 소켓 알림 전송
                if self.socketio:
                    self.socketio.emit('anomaly_detected', {
                        "alert_id": new_alert.id, "type": "역주행", 
                        "address": self.cctv_name, "lat": float(self.lat), "lng": float(self.lng),
                        "video_origin": "realtime_its", "is_simulation": False,
                        "image_url": f"/static/captures/{filename}"
                    })
                print(f"🚨 [역주행 알람 완료] {self.cctv_name} - ID:{track_id}")
        except Exception as e:
            self.db.session.rollback()
            print(f"❌ 역주행 비동기 저장 에러: {e}")

    def run(self):
        """[Main Thread] 백그라운드 분석 루프"""
        print(f"🚗 [{self.cctv_name}] 역주행 분석 시작")
        while self.is_running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                time.sleep(1)
                continue

            h, w = frame.shape[:2]
            
            # 초기화 체크
            if self.flow_map is None or not hasattr(self, 'grid_h'):
                self.init_grid(w, h)

            # 화면 전환 감지 로직
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            if self.prev_frame_gray is not None and self.prev_frame_gray.shape == gray_blurred.shape:
                diff_score = np.mean(cv2.absdiff(self.prev_frame_gray, gray_blurred))
                if diff_score > self.SCENE_THRESHOLD:
                    self.reset_learning(w, h)
            self.prev_frame_gray = gray_blurred
            self.frame_count += 1

            # YOLO 추적
            results = self.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", conf=self.conf)

            if hasattr(self, 'grid_h') and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for i, track_id in enumerate(boxes.id.int().tolist()):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    self.trajectories[track_id].append((cx, cy))
                    if len(self.trajectories[track_id]) > 30: self.trajectories[track_id].pop(0)

                    traj = self.trajectories[track_id]
                    if len(traj) >= self.VELOCITY_WINDOW:
                        vdx = traj[-1][0] - traj[-self.VELOCITY_WINDOW][0]
                        vdy = traj[-1][1] - traj[-self.VELOCITY_WINDOW][1]
                        mag = np.sqrt(vdx**2 + vdy**2)
                        
                        if mag > 5: # 일정 속도 이상일 때만 판단
                            ndx, ndy = vdx / mag, vdy / mag
                            gx, gy = int(cx // self.GRID_SIZE), int(cy // self.GRID_SIZE)
                            
                            if 0 <= gy < self.grid_h and 0 <= gx < self.grid_w:
                                if not self.learning_done:
                                    # 학습 모드: 이동 방향 기록
                                    self.flow_map[gy, gx] += [ndx, ndy]
                                    self.flow_count[gy, gx] += 1
                                else:
                                    # 판별 모드: 벡터 유사도 계산
                                    is_wrong_step = False
                                    ref_vec = self.flow_map[gy, gx]
                                    ref_mag = np.sqrt(ref_vec[0]**2 + ref_vec[1]**2)
                                    if ref_mag > 0:
                                        dot = (ndx * ref_vec[0] + ndy * ref_vec[1]) / ref_mag
                                        if dot < self.COS_THRESHOLD: is_wrong_step = True

                                    if is_wrong_step:
                                        self.wrong_way_count[track_id] += 1
                                    else:
                                        self.wrong_way_count[track_id] = max(0, self.wrong_way_count[track_id] - 1)

                                    # 역주행 확정 및 큐에 전송
                                    if self.wrong_way_count[track_id] > 15 and track_id not in self.alerted_ids:
                                        self.alerted_ids.add(track_id)
                                        self.wrong_way_ids.add(track_id)
                                        # 비동기 워커로 데이터 던지기
                                        self.alert_queue.put((frame.copy(), datetime.now(), track_id))

                    # 시각화
                    is_confirmed = track_id in self.wrong_way_ids
                    color = (0, 0, 255) if is_confirmed else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = "WRONG-WAY" if is_confirmed else f"ID:{track_id}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 학습 완료 시점 처리
            if not self.learning_done and self.frame_count >= self.LEARNING_FRAMES:
                self.apply_spatial_smoothing()
                self.save_flow_map()
                self.learning_done = True

            self.latest_frame = frame
            time.sleep(0.01)

    def stop(self):
        super().stop() # 부모의 is_running = False 실행
        if self.cap.isOpened():
            self.cap.release()