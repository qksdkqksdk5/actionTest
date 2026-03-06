import os
import requests
import random
from dotenv import load_dotenv
from flask import Blueprint, jsonify, Response, request, current_app
from detectors.reverse_detector import ReverseDetector
from detectors.fire_detector import FireDetector
from detectors.manager import detector_manager

# 환경 변수 로드
load_dotenv()

its_bp = Blueprint('its', __name__)
ITS_API_KEY = os.getenv('ITS_API_KEY', '22f088a782aa49f6a441b24c2b36d4ec')

# CCTV 리스트 캐시 (서버가 켜져 있는 동안 유지)
cached_cctv_list = []

@its_bp.route('/get_cctv_url', methods=['GET'])
def get_cctv_url():
    global cached_cctv_list
    
    # 1. 캐시된 데이터가 있다면 즉시 반환
    if cached_cctv_list:
        print("♻️ [캐시 데이터 반환] 기존 CCTV 리스트를 사용합니다.")
        return jsonify({
            "success": True,
            "cctvData": cached_cctv_list
        })

    # 2. 캐시가 없는 경우 ITS API 호출
    params = {
        'apiKey': ITS_API_KEY,
        'type': 'ex',
        'cctvType': '1',
        'minX': '126.8',
        'maxX': '127.89',
        'minY': '36.8',
        'maxY': '37.0',
        'getType': 'json'
    }
    
    target_url = "https://openapi.its.go.kr:9443/cctvInfo"
    
    try:
        response = requests.get(target_url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            cctv_list = data.get("response", {}).get("data", [])

            if cctv_list:
                # 랜덤하게 4개 추출
                sample_count = min(len(cctv_list), 4)
                random_cctvs = random.sample(cctv_list, sample_count)
                
                # 가공하여 캐시에 저장
                cached_cctv_list = [{
                    "url": item['cctvurl'],
                    "name": item['cctvname'],
                    "lat": float(item['coordy']),
                    "lng": float(item['coordx'])
                } for item in random_cctvs]

                print(f"📡 [API 호출 성공] {len(cached_cctv_list)}개의 CCTV를 고정합니다.")
                return jsonify({
                    "success": True,
                    "cctvData": cached_cctv_list
                })
        
        raise Exception("API 응답이 올바르지 않습니다.")

    except Exception as e:
        print(f"📡 ITS 연결 실패: {e}")
        # 실패 시 테스트 데이터 제공
        test_url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
        cached_cctv_list = [
            {"url": test_url, "name": f"테스트 채널 {i+1}", "lat": 37.5, "lng": 127.0} 
            for i in range(4)
        ]
        return jsonify({
            "success": True,
            "cctvData": cached_cctv_list
        })

@its_bp.route('/video_feed')
def video_feed():
    """역주행 탐지 스트리밍 라우트"""
    # 1. 파라미터 추출
    url = request.args.get('url')
    name = request.args.get('name', 'default')
    lat = float(request.args.get('lat', 37.5))
    lng = float(request.args.get('lng', 127.0))
    conf_val = float(request.args.get('conf', 0.66))

    # 2. 인스턴스 식별을 위한 유니크 네임 생성
    unique_name = f"{name}_reverse"

    # 3. 필수 객체 및 모델 임포트
    socketio = current_app.extensions['socketio']
    app_obj = current_app._get_current_object() 
    from models import db as db_inst, DetectionResult, ReverseResult

    # 4. 매니저를 통해 분석기 실행 및 획득
    detector = detector_manager.get_or_create(
        unique_name, 
        ReverseDetector, 
        url=url, 
        lat=lat, 
        lng=lng,
        socketio=socketio, 
        db=db_inst,
        ResultModel=DetectionResult,
        ReverseModel=ReverseResult,
        conf=conf_val,
        app=app_obj
    )

    # 5. MJPEG 스트리밍 응답 (BaseDetector의 generate_frames 호출)
    return Response(
        detector.generate_frames(), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@its_bp.route('/fire_feed')
def fire_feed():
    """화재 탐지 스트리밍 라우트"""
    # 1. 파라미터 추출
    url = request.args.get('url')
    name = request.args.get('name', 'fire_cctv')
    lat = float(request.args.get('lat', 37.5))
    lng = float(request.args.get('lng', 127.0))

    # 2. 인스턴스 식별을 위한 유니크 네임 생성
    unique_name = f"{name}_fire"

    # 3. 필수 객체 임포트
    socketio = current_app.extensions['socketio']
    app_obj = current_app._get_current_object()
    from models import db as db_inst, DetectionResult

    # 4. 매니저를 통해 분석기 실행 및 획득
    detector = detector_manager.get_or_create(
        unique_name,
        FireDetector,
        url=url, 
        lat=lat, 
        lng=lng, 
        socketio=socketio, 
        db=db_inst, 
        ResultModel=DetectionResult, 
        app=app_obj
    )

    # 5. MJPEG 스트리밍 응답
    return Response(
        detector.generate_frames(), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )