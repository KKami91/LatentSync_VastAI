import os
import random
import sys
import base64
import torch
import tempfile
import glob
import logging
from typing import Sequence, Mapping, Any, Union
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import cv2
import numpy as np
import subprocess

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    """
    if path is None:
        path = os.getcwd()

    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        logger.info(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)

    if parent_directory == path:
        return None

    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """Add 'ComfyUI' to the sys.path"""
    comfyui_path = "/workspace/ComfyUI"
    if os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        logger.info(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path."""
    try:
        from main import load_extra_path_config
    except ImportError:
        logger.info("Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead.")
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        logger.info("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS"""
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    init_extra_nodes()


def setup_environment():
    """Setup the ComfyUI environment"""
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    import_custom_nodes()


def process_latentsync(video_data: bytes, audio_data: bytes, video_name: str, custom_width_: int, custom_height_: int):
    from nodes import NODE_CLASS_MAPPINGS

    video_name_without_ext = os.path.splitext(video_name)[0]
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 입력 파일 설정
            video_path = os.path.join(temp_dir, "input_video.mp4")
            audio_path = os.path.join(temp_dir, "input_audio.wav")
            output_filename = f"convert_{video_name_without_ext}"
            
            # 입력 파일 저장
            with open(video_path, "wb") as f:
                f.write(video_data)
            with open(audio_path, "wb") as f:
                f.write(audio_data)

            logger.info("LatentSync 처리 시작...")
            
            try:
                with torch.inference_mode():
                    # LoadAudio
                    loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
                    loadaudio_37 = loadaudio.load(audio=audio_path)

                    # LoadVideo
                    vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
                    vhs_loadvideo_40 = vhs_loadvideo.load_video(
                        video=video_path,
                        force_rate=25,
                        custom_width=custom_width_,
                        custom_height=custom_height_,
                        frame_load_cap=0,
                        skip_first_frames=0,
                        select_every_nth=1,
                        format="AnimateDiff",
                        unique_id=12015943199208297010,
                    )

                    d_videolengthadjuster = NODE_CLASS_MAPPINGS["D_VideoLengthAdjuster"]()
                    d_latentsyncnode = NODE_CLASS_MAPPINGS["D_LatentSyncNode"]()
                    vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
                    
                    d_videolengthadjuster_53 = d_videolengthadjuster.adjust(
                        mode="pingpong",
                        fps=25,
                        silent_padding_sec=0.5,
                        images=get_value_at_index(vhs_loadvideo_40, 0),
                        audio=get_value_at_index(loadaudio_37, 0),
                    )

                    try:
                        # LatentSync 처리 시도
                        logger.info("LatentSync 처리 시도 중...")
                        d_latentsyncnode_43 = d_latentsyncnode.inference(
                            seed=random.randint(1, 2**32 - 1),
                            images=get_value_at_index(d_videolengthadjuster_53, 0),
                            audio=get_value_at_index(d_videolengthadjuster_53, 1),
                        )
                        
                        logger.info(f"처리 성공, 파일명으로 비디오 생성 중: {output_filename}")
                        
                        result = vhs_videocombine.combine_video(
                            frame_rate=25,
                            loop_count=0,
                            filename_prefix=output_filename,
                            format="video/h264-mp4",
                            pix_fmt="yuv420p",
                            crf=19,
                            save_metadata=True,
                            trim_to_audio=False,
                            pingpong=False,
                            save_output=False,
                            images=get_value_at_index(d_latentsyncnode_43, 0),
                            audio=get_value_at_index(d_latentsyncnode_43, 1),
                            unique_id=7599875590960303900,
                        )
                        
                    except Exception as e:
                        # 특정 얼굴 감지 오류에 대한 개선된 오류 처리
                        error_msg = str(e)
                        logger.warning(f"LatentSync 처리 오류: {error_msg}")

                        # 얼굴 감지 관련 오류인지 확인
                        face_detection_error = any(phrase in error_msg for phrase in [
                            "No face detected", 
                            "expected Tensor", 
                            "얼굴 감지 실패", 
                            "얼굴을 찾을 수 없습니다",
                            "but got NoneType"
                        ])
                        
                        # 특정 얼굴 감지 오류 확인
                        if face_detection_error:
                            logger.info("얼굴 감지 관련 오류 발생: 원본 비디오에 오디오만 추가합니다")
                        else:
                            logger.warning(f"기타 오류 발생: {error_msg}")
                            
                        # 원본 영상과 음성 결합
                        logger.info("대체 처리: 원본 비디오에 오디오만 추가합니다")
                        result = vhs_videocombine.combine_video(
                            frame_rate=25,
                            loop_count=0,
                            filename_prefix=output_filename,
                            format="video/h264-mp4",
                            pix_fmt="yuv420p",
                            crf=19,
                            save_metadata=True,
                            trim_to_audio=False,
                            pingpong=False,
                            save_output=False,
                            images=get_value_at_index(d_videolengthadjuster_53, 0),
                            audio=get_value_at_index(d_videolengthadjuster_53, 1),
                            unique_id=7599875590960303900,
                        )

                    # 결과에서 파일 경로 가져오기
                    if isinstance(result, dict) and 'result' in result:
                        saved_files = result['result'][0][1]
                    else:
                        saved_files = result[0][1]

                    if not saved_files:
                        raise Exception("출력 파일이 생성되지 않았습니다")

                    result_path = saved_files[-1]  # 마지막 파일이 최종 결과물
                    logger.info(f"결과 파일 경로: {result_path}")

                    if not os.path.exists(result_path):
                        raise FileNotFoundError(f"다음 경로에서 출력 파일을 찾을 수 없습니다: {result_path}")

                    # 결과 파일 영구 저장 (선택 사항)
                    permanent_output_path = os.path.join(output_dir, f"{output_filename}.mp4")
                    with open(result_path, "rb") as src, open(permanent_output_path, "wb") as dst:
                        dst.write(src.read())

                    with open(result_path, "rb") as f:
                        output_data = f.read()

                    logger.info("비디오 처리 성공")
                    
                    return {
                        "success": True,
                        "output": {
                            "video_data": base64.b64encode(output_data).decode('utf-8'),
                            "video_name": f"{output_filename}.mp4",
                            "file_path": permanent_output_path
                        }
                    }

            except Exception as e:
                logger.error(f"LatentSync 처리 중 오류: {str(e)}")
                return {"success": False, "error": str(e)}

        except Exception as e:
            logger.error(f"파일 처리 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}

        finally:
            # 임시 파일 정리
            cleanup_patterns = [
                os.path.join(temp_dir, "*.mp4"),
                os.path.join(temp_dir, "*.wav")
            ]
            
            for pattern in cleanup_patterns:
                try:
                    files = glob.glob(pattern)
                    for file in files:
                        try:
                            os.remove(file)
                            logger.info(f"파일 삭제: {file}")
                        except Exception as e:
                            logger.error(f"{file} 삭제 중 오류: {str(e)}")
                except Exception as e:
                    logger.error(f"패턴 {pattern} 처리 중 오류: {str(e)}")


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "LatentSync API",
        "status": "running",
        "usage": {
            "endpoint": "/latentsync",
            "method": "POST",
            "required_fields": ["video", "audio", "video_name"],
            "optional_fields": ["custom_width", "custom_height"]
        }
    })


@app.route('/latentsync', methods=['POST'])
def handle_latentsync():
    try:
        # API 요청에서 데이터 추출
        if not request.is_json:
            return jsonify({"success": False, "error": "Request must be JSON"}), 400
        
        data = request.json
        
        # 필수 필드 검증
        if 'video' not in data or 'audio' not in data or 'video_name' not in data:
            return jsonify({"success": False, "error": "Missing required fields (video, audio, video_name)"}), 400
        
        # 선택적 필드 설정
        custom_width = int(data.get('custom_width', 64))
        custom_height = int(data.get('custom_height', 64))
        
        # 데이터 디코딩
        try:
            video_data = base64.b64decode(data['video'])
            audio_data = base64.b64decode(data['audio'])
        except:
            return jsonify({"success": False, "error": "Invalid base64 encoding"}), 400
        
        # 처리
        result = process_latentsync(
            video_data, 
            audio_data, 
            data['video_name'], 
            custom_width, 
            custom_height
        )
        
        # 에러 처리
        if not result.get('success', True):
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in API handler: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """파일 다운로드 엔드포인트"""
    try:
        output_dir = os.path.join(os.getcwd(), "output")
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "File not found"}), 404
            
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error in download handler: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # 환경 설정
    logger.info("Setting up ComfyUI environment...")
    setup_environment()
    logger.info("Environment setup complete")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.join(os.getcwd(), "output"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), "temp"), exist_ok=True)
    
    # 서버 실행
    logger.info("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000)