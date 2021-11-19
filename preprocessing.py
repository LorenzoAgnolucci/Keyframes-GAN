import glob
import os
from pathlib import Path
import subprocess
from fractions import Fraction
import dlib
import numpy as np
import re
import argparse
import cv2

from face_aligner import FaceAligner, shape_to_np


def compress_videos(path, crf):
    Path(f"{path}/compressed_{crf}").mkdir(parents=True, exist_ok=True)
    for file in sorted(glob.glob(f"{path}/original/*.mp4")):
        file_name = os.path.basename(file)
        if not os.path.exists(f"{path}/compressed_{crf}/{file_name}"):
            print(f"\nProcessing file {file_name}\n")
            os.system(f"ffmpeg -i {path}/original/{file_name} -c:v libx264 -crf {crf} -an {path}/compressed_{crf}/{file_name}")


def extract_inference_frames(base_path):
    for file in sorted(glob.glob(f"{base_path}/*.mp4")):
        file_name = os.path.basename(file)
        print(f"{file_name}")
        path_frames = f"{base_path}/{file_name[:-4]}/frames"
        Path(path_frames).mkdir(parents=True, exist_ok=True)
        output_fps = subprocess.check_output(f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {file}", shell=True)
        output_fps = output_fps.decode("utf-8").strip("\n")
        fps = round(Fraction(output_fps))
        os.system(f"ffmpeg -i {file} -qscale:v 2 {path_frames}/%00d.jpg")
        num_frames = len(glob.glob(f"{path_frames}/*.jpg"))
        for idx in range(1, num_frames, fps):
            os.system(f"mv {path_frames}/{idx}.jpg {path_frames}/{idx}_key.jpg ")


def crop_and_align(base_path, crf):
    path_compressed = f"{base_path}/compressed_{crf}"
    path_original = f"{base_path}/original"

    face_detector = dlib.cnn_face_detection_model_v1("pretrained_models/dlib_weights.dat")
    landmark_detector = dlib.shape_predictor("pretrained_models/dlib_shape_predictor_68_face_landmarks.dat")
    face_aligner = FaceAligner(landmark_detector, desiredLeftEye=(0.38, 0.45), desiredFaceWidth=256)

    seconds_limit = 30  # Stop cropping after seconds_limit of video (i.e. seconds_limit*fps frames)

    for video_dir in sorted(glob.glob(f"{path_compressed}/*/")):
        video_name = video_dir[:-1].split("/")[-1]
        print(video_name)

        Path(f"{path_compressed}/{video_name}/crops").mkdir(parents=True, exist_ok=True)
        Path(f"{path_compressed}/{video_name}/binary_landmarks").mkdir(parents=True, exist_ok=True)
        Path(f"{path_compressed}/{video_name}/landmarks").mkdir(parents=True, exist_ok=True)
        Path(f"{path_compressed}/{video_name}/transform_matrices").mkdir(parents=True, exist_ok=True)
        Path(f"{path_original}/{video_name}/crops").mkdir(parents=True, exist_ok=True)

        keyframes = sorted([int(el[:-8]) for el in os.listdir(f"{path_compressed}/{video_name}/frames") if "key" in el])    # Take only keyframes and remove "_key.jpg" to cast to int
        fps = keyframes[1] - keyframes[0]

        last_rect = None
        last_landmarks = None

        frames = os.listdir(f"{path_compressed}/{video_name}/frames/")
        frames.sort(key=lambda f: int(re.sub('\D', '', f)))  # Sort frames correctly to crop the first 30 seconds of each video
        frames = [f"{path_compressed}/{video_name}/frames/{frame}" for frame in frames]
        for i, frame in enumerate(frames):
            frame_name = os.path.basename(frame)
            print(f"{video_name}: {frame_name}")

            frame_compressed = cv2.imread(frame)
            frame_original = cv2.imread(f"{path_original}/{video_name}/frames/{frame_name}")
            gray_frame_compressed = cv2.cvtColor(frame_compressed, cv2.COLOR_BGR2GRAY)

            face_compressed = face_detector(gray_frame_compressed, 1)
            transform_landmarks = False

            if len(face_compressed) != 0:
                last_rect = face_compressed[0].rect
                landmarks = landmark_detector(gray_frame_compressed, last_rect)
                if landmarks is not None:
                    last_landmarks = shape_to_np(landmarks)
                    transform_landmarks = True
            crop_compressed, crop_original, transformed_landmarks, transform_matrix = face_aligner.align(frame_compressed, gray_frame_compressed,
                                                                                                         rect=last_rect,
                                                                                                         compressed_image=frame_original,
                                                                                                         landmarks=last_landmarks)
            if transform_landmarks:
                last_landmarks = transformed_landmarks

            landmark_img = generate_landmark_binary_image(crop_compressed.shape, last_landmarks)

            cv2.imwrite(f"{path_compressed}/{video_name}/crops/{frame_name}", crop_compressed)
            cv2.imwrite(f"{path_compressed}/{video_name}/binary_landmarks/{frame_name}", landmark_img)
            np.save(f"{path_compressed}/{video_name}/landmarks/{frame_name[:-4]}.npy", last_landmarks)
            np.save(f"{path_compressed}/{video_name}/transform_matrices/{frame_name[:-4]}.npy", transform_matrix)

            cv2.imwrite(f"{path_original}/{video_name}/crops/{frame_name}", crop_original)

            if i >= fps * seconds_limit:
                break


def generate_landmark_binary_image(frame_shape, landmarks):
    frame_height, frame_width = frame_shape[:2]
    binary_image = np.zeros((frame_height, frame_width, 1))
    for i in range(landmarks.shape[0]):
        landmark_x = min(max(landmarks[i][0], 0), frame_height - 1)
        landmark_y = min(max(landmarks[i][1], 0), frame_width - 1)
        binary_image[landmark_y, landmark_x, 0] = 255

    return binary_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="HQ videos should be in {BASE_PATH}/original")
    parser.add_argument("--crf", type=int, default=42, help="Constant Rate Factor")
    args = parser.parse_args()

    BASE_PATH = args.base_path  # HQ videos should be in {BASE_PATH}/original
    CRF = args.crf
    compress_videos(BASE_PATH, CRF)
    extract_inference_frames(f"{BASE_PATH}/original")
    extract_inference_frames(f"{BASE_PATH}/compressed_{CRF}")
    crop_and_align(BASE_PATH, CRF)
