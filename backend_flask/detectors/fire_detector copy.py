import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
import threading
from .base_detector import BaseDetector  # BaseDetector가 같은 폴더에 있다고 가정

class FireDetector(BaseDetector):
    def __init__(self, cctv_name, url, lat=37.5, lng=127.0, socketio=None, db=None, ResultModel=None, app=None):
        # 1. 부모 클래스(BaseDetector) 초기화 (비동기 큐 및 워커 자동 시작)
        super().__init__(cctv_name, url, app=app, socketio=socketio, db=db, ResultModel=ResultModel)
        
        self.lat = lat
        self.lng = lng
        
        # 2. 모델 로드 및 GPU 할당
        # imgsz=320 등으로 최적화하여 분석 속도 향상 가능
        self.model = YOLO("best_SB.pt").to('cuda') 

        # 3. FFMPEG 및 버퍼 설정 (지연 시간 최소화)
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 상태 제어 변수
        self.is_alerting = False 
        self.frame_count = 0
        self.frame_skip = 2  # 필요 시 프레임 스킵 (1이면 모든 프레임 읽음)

    def process_alert(self, data):
        """[Worker Thread] 부모의 큐에서 데이터를 받아 비동기로 DB/파일 저장"""
        frame, alert_time = data
        try:
            with self.app.app_context():
                # 1️⃣ 공통 결과 테이블 저장 (DetectionResult)
                new_alert = self.ResultModel(
                    event_type="fire",
                    address=self.cctv_name,
                    latitude=self.lat, 
                    longitude=self.lng,
                    detected_at=alert_time,
                    is_simulation=False,
                    is_resolved=False
                )
                self.db.session.add(new_alert)
                self.db.session.flush() # ID 확보용

                # 2️⃣ 이미지 파일 저장
                ts = alert_time.strftime("%Y%m%d_%H%M%S")
                filename = f"fire_real_{new_alert.id}_{ts}.jpg"
                save_path = os.path.join(self.app.root_path, "static", "captures")
                os.makedirs(save_path, exist_ok=True)
                filepath = os.path.join(save_path, filename)
                cv2.imwrite(filepath, frame)

                # 3️⃣ 화재 상세 테이블 저장 (FireResult)
                from models import FireResult 
                fire_detail = FireResult(
                    result_id=new_alert.id,
                    image_path=f"/static/captures/{filename}",
                    fire_severity="중간"  # 기본값 설정
                )
                self.db.session.add(fire_detail)
                self.db.session.commit()

                # 4️⃣ 소켓 알림 전송
                if self.socketio:
                    self.socketio.emit('anomaly_detected', {
                        "alert_id": new_alert.id, 
                        "type": "화재", 
                        "address": self.cctv_name,
                        "lat": float(self.lat), 
                        "lng": float(self.lng),
                        "video_origin": "realtime_its",
                        "is_simulation": False,
                        "image_url": f"/static/captures/{filename}"
                    })
                print(f"🔥 [화재 알람 완료] {self.cctv_name} - ID:{new_alert.id}")

        except Exception as e:
            self.db.session.rollback()
            print(f"❌ 화재 비동기 저장 에러: {e}")

    def run(self):
        """[Main Thread] 백그라운드 분석 루프"""
        print(f"🔥 [{self.cctv_name}] 화재 분석 스레드 시작")

        while self.is_running and self.cap.isOpened():
            # 프레임 스킵 로직 (네트워크 지연 방지)
            if self.frame_count % self.frame_skip != 0:
                self.cap.grab()
                self.frame_count += 1
                continue

            success, frame = self.cap.read()
            if not success:
                print(f"📡 [{self.cctv_name}] 연결 끊김. 재연결 시도...")
                self.cap.open(self.url)
                time.sleep(1)
                continue

            self.frame_count += 1

            # YOLO 분석 (최적화 파라미터 적용)
            results = self.model.predict(frame, verbose=False, conf=0.45, imgsz=320, device=0)
            
            fire_found_this_frame = False
            
            if len(results[0].boxes) > 0:
                fire_found_this_frame = True
                for box in results[0].boxes:
                    b = box.xyxy[0].cpu().numpy()
                    # 시각화 (빨간색 박스)
                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
                    cv2.putText(frame, "FIRE", (int(b[0]), int(b[1])-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 트리거 로직: 화재가 처음 발견되었을 때만 큐에 전송 (중복 저장 방지)
            if fire_found_this_frame and not self.is_alerting:
                # 분석 중인 현재 프레임과 시간을 부모 큐로 전송
                self.alert_queue.put((frame.copy(), datetime.now()))
                self.is_alerting = True
            elif not fire_found_this_frame:
                # 화재가 화면에서 사라지면 다시 탐지 가능 상태로 변경
                self.is_alerting = False

            # 최신 프레임 업데이트 (MJPEG 스트리밍용)
            self.latest_frame = frame
            
            # CPU 점유율 조절을 위한 미세 대기
            time.sleep(0.01)

    def stop(self):
        """자원 해제 및 스레드 종료"""
        super().stop()  # 부모의 is_running = False 호출
        if self.cap.isOpened():
            self.cap.release()