# utils.py
import subprocess
import os

def convert_video(input_path, output_path):
    """
    使用 ffmpeg 將影片轉換成 mp4 格式 (用於輸入影片的預處理)
    """
    cmd = [
         "ffmpeg",
         "-y",  # 自動覆蓋輸出檔案
         "-i", input_path,
         "-c:v", "libx264",
         "-preset", "fast",
         "-crf", "23",
         "-c:a", "copy",
         output_path
    ]
    print("Running ffmpeg command: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def convert_to_common_mp4(input_path, output_path):
    """
    使用 ffmpeg 將輸出影片轉換成通用 mp4 格式，
    加入參數 -profile:v high -level 4.0 與 -pix_fmt yuv420p 以提高播放兼容性
    """
    cmd = [
         "ffmpeg",
         "-y",  # 自動覆蓋輸出檔案
         "-i", input_path,
         "-c:v", "libx264",
         "-profile:v", "high",
         "-level", "4.0",
         "-pix_fmt", "yuv420p",
         output_path
    ]
    print("Running ffmpeg for output conversion: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
