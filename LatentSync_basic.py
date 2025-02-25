import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import runpod
import base64
from typing import Sequence, Mapping, Any, Union
from io import BytesIO
import glob
from runpod.serverless.utils.rp_cleanup import clean

#로컬 전용
import tempfile

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = "/workspace/ComfyUI"
    if os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


# add_comfyui_directory_to_sys_path()
# add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()




def setup_environment():
    """Setup the ComfyUI environment"""
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    import_custom_nodes()


# def process_latentsync(video_data: bytes, audio_data: bytes, video_name: str):
#     from nodes import NODE_CLASS_MAPPINGS

#     # 파일명에서 확장자 제거
#     video_name_without_ext = os.path.splitext(video_name)[0]
#     # output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
#     # if not os.path.exists(output_dir):
#     #     os.makedirs(output_dir, exist_ok=True)

#     output_dir = os.path.join(os.getcwd(), "output")
#     os.makedirs(output_dir, exist_ok=True)

#     #로컬 전용 테스트
#     # with tempfile.TemporaryDirectory() as temp_dir:
#     #     # 임시 파일 경로 생성
#     #     video_path = os.path.join(temp_dir, "input_video.mp4")
#     #     audio_path = os.path.join(temp_dir, "input_audio.wav")
        
#     #     # 파일 저장
#     #     with open(video_path, "wb") as f:
#     #         f.write(video_data)
#     #     with open(audio_path, "wb") as f:
#     #         f.write(audio_data)

#     with tempfile.TemporaryDirectory() as temp_dir:
#         # 임시 파일 경로 생성
#         video_path = os.path.join(temp_dir, "input_video.mp4")
#         audio_path = os.path.join(temp_dir, "input_audio.wav")
#         #output_path = os.path.join(output_dir, f"convert_{video_name}")
        
#         # # 입력 파일 저장
#         # with open(video_path, "wb") as f:
#         #     f.write(video_data)
#         # with open(audio_path, "wb") as f:
#         #     f.write(audio_data)

#         with tempfile.TemporaryDirectory() as temp_dir:
#             video_path = os.path.join(temp_dir, "input_video.mp4")
#             audio_path = os.path.join(temp_dir, "input_audio.wav")
            
#             # 출력 파일명 설정
#             output_filename = f"convert_{os.path.splitext(video_name)[0]}"
#             print('초기 output_filename : ',output_filename)
#             output_path = os.path.join(output_dir, output_filename)
#             print('초기 output_path : ',output_path)
            
#             with open(video_path, "wb") as f:
#                 f.write(video_data)
#             with open(audio_path, "wb") as f:
#                 f.write(audio_data)

#             try:
#                 with torch.inference_mode():
#                     loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
#                     loadaudio_37 = loadaudio.load(audio=audio_path)

#                     vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
#                     vhs_loadvideo_40 = vhs_loadvideo.load_video(
#                         video=video_path,
#                         force_rate=25,
#                         custom_width=64, #배포 0, 로컬 해상도 조절 필요
#                         custom_height=64, #배포 0, 로컬 해상도 조절 필요
#                         frame_load_cap=0,
#                         skip_first_frames=0,
#                         select_every_nth=1,
#                         format="AnimateDiff",
#                         unique_id=12015943199208297010,
#                     )

#                     d_videolengthadjuster = NODE_CLASS_MAPPINGS["D_VideoLengthAdjuster"]()
#                     d_latentsyncnode = NODE_CLASS_MAPPINGS["D_LatentSyncNode"]()
#                     vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
                    
#                     d_videolengthadjuster_53 = d_videolengthadjuster.adjust(
#                         mode="pingpong",
#                         fps=25,
#                         silent_padding_sec=0.5,
#                         images=get_value_at_index(vhs_loadvideo_40, 0),
#                         audio=get_value_at_index(loadaudio_37, 0),
#                     )

#                     d_latentsyncnode_43 = d_latentsyncnode.inference(
#                         seed=random.randint(1, 2**32 - 1),
#                         images=get_value_at_index(d_videolengthadjuster_53, 0),
#                         audio=get_value_at_index(d_videolengthadjuster_53, 1),
#                     )

#                     filename_prefix = output_filename
#                     print("videocombine filename_prefix : ",filename_prefix)
#                     print("before videocombine output_filename : ", output_filename)

#                     vhs_videocombine_41 = vhs_videocombine.combine_video(
#                         frame_rate=25,
#                         loop_count=0,
#                         #filename_prefix=f"{output_filename}"[:-4],
#                         filename_prefix=filename_prefix,
#                         format="video/h264-mp4",
#                         pix_fmt="yuv420p",
#                         crf=19,
#                         save_metadata=True,
#                         trim_to_audio=False,
#                         pingpong=False,
#                         save_output=True, # 임시 디렉토리에는 무조건 저장이 되나?
#                         images=get_value_at_index(d_latentsyncnode_43, 0),
#                         audio=get_value_at_index(d_latentsyncnode_43, 1),
#                         unique_id=7599875590960303900,
#                     )

#                     output_file = f"{output_filename}" # 
#                     print(f'output_file : {output_file}')
#                     actual_output_path = os.path.join(output_dir, output_file)
#                     print(f'actual_output_path : {actual_output_path}')

#                     if not os.path.exists(actual_output_path):
#                         print('................ not exist file ....... : ', actual_output_path)
#                         raise FileNotFoundError(f"Output file not found at: {actual_output_path}")                    

#                     print('파일 읽기 전')
#                     # 결과 파일 읽기
#                     with open(output_path, "rb") as f:
#                         print('파일 읽기 성공, read 전')
#                         output_data = f.read()
                    
#                     #print('output_data : ', output_data)
#                     return {
#                         "output": {
#                             "video_data": base64.b64encode(output_data).decode('utf-8'),
#                             "video_name": "output_file"
#                         }
#                     }

#             except Exception as e:
#                 return {"error": str(e)}
        
#             finally:
#                 # 임시 파일 정리
#                 if os.path.exists(video_path):
#                     os.remove(video_path)
#                 if os.path.exists(audio_path):
#                     os.remove(audio_path)
#                 if os.path.exists(output_path):
#                     os.remove(output_path)

def process_latentsync(video_data: bytes, audio_data: bytes, video_name: str, custom_width_: int, custom_height_: int):
    from nodes import NODE_CLASS_MAPPINGS
    import os
    import tempfile
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    video_name_without_ext = os.path.splitext(video_name)[0]
    
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

            logger.info("Starting LatentSync processing...")
            
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

                    d_latentsyncnode_43 = d_latentsyncnode.inference(
                        seed=random.randint(1, 2**32 - 1),
                        images=get_value_at_index(d_videolengthadjuster_53, 0),
                        audio=get_value_at_index(d_videolengthadjuster_53, 1),
                    )

                    logger.info(f"Processing video combine with filename: {output_filename}")
                    
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

                    # 결과에서 파일 경로 가져오기
                    if isinstance(result, dict) and 'result' in result:
                        saved_files = result['result'][0][1]
                    else:
                        saved_files = result[0][1]

                    if not saved_files:
                        raise Exception("No output files were generated")

                    result_path = saved_files[-1]  # 마지막 파일이 최종 결과물
                    logger.info(f"Result file path: {result_path}")

                    if not os.path.exists(result_path):
                        raise FileNotFoundError(f"Output file not found at: {result_path}")

                    with open(result_path, "rb") as f:
                        output_data = f.read()

                    logger.info("Successfully processed video")
                    
                    return {
                        "output": {
                            "video_data": base64.b64encode(output_data).decode('utf-8'),
                            "video_name": f"{output_filename}.mp4"
                        }
                    }

            except Exception as e:
                logger.error(f"Error during LatentSync processing: {str(e)}")
                return {"error": str(e)}

        except Exception as e:
            logger.error(f"Error in file handling: {str(e)}")
            return {"error": str(e)}
        

        # finally:
        #     print('in finally....')
        #     temp_path = os.path.dirname(result_path)
        #     comfyui_path = os.path.abspath(os.path.join(temp_path, ".."))
        #     if os.path.exists(temp_path):
        #         print('.mp4 제거..?')
        #         [os.remove(mp4_file) for mp4_file in glob.glob(os.path.join(temp_path, "*.mp4"))]
        #     if os.path.exists(comfyui_path):
        #         print('.wav 제거..?')aaaaa
        #         [os.remove(wav_file) for wav_file in glob.glob(os.path.join(comfyui_path, "*.wav"))]

            

def handler(event):
    """Runpod serverless handler"""
    import os
    import glob

    print('handler 시작?')
    try:
        # 입력 데이터 검증
        if 'input' not in event or 'video' not in event['input'] or 'audio' not in event['input']:
            raise ValueError("Missing required input fields (video and/or audio)")

        # 입력 데이터 디코딩
        video_data = base64.b64decode(event['input']['video'])
        audio_data = base64.b64decode(event['input']['audio'])
        video_name = event['input']['video_name']
        custom_width_ = event['input']['custom_width_']
        custom_height_ = event['input']['custom_height_']       
        # 환경 설정
        setup_environment()
        
        # 처리
        result = process_latentsync(video_data, audio_data, video_name, custom_width_, custom_height_)
        

        

        
        print("Cleanup completed")
        return result
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"error": str(e)}
    finally:
        # 추가적인 cleanup이 필요한 경우 여기서 처리
                # Cleanup 처리
        print("Cleaning up temporary files...")
        temp_dir = os.path.join(os.getcwd(), "temp")
        output_dir = os.path.join(os.getcwd(), "output")
        
        # 임시 파일들 정리
        cleanup_patterns = [
            os.path.join(temp_dir, "*.mp4"),
            os.path.join(temp_dir, "*.wav"),
            os.path.join(output_dir, "*.mp4"),
            os.path.join(output_dir, "*.wav"),
            os.path.join(os.getcwd(), "*.wav"),  # 루트 디렉토리의 wav 파일
        ]
        for pattern in cleanup_patterns:
            try:
                files = glob.glob(pattern)
                for file in files:
                    try:
                        os.remove(file)
                        print(f"Removed: {file}")
                    except Exception as e:
                        print(f"Error removing {file}: {str(e)}")
            except Exception as e:
                print(f"Error processing pattern {pattern}: {str(e)}")
        print("Handler completed")

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})