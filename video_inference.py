import dlib
import torch
import glob
import os
import numpy as np
from pathlib import Path
import cv2
import re
from timeit import default_timer as timer
import pandas as pd
import argparse

from BasicSR.basicsr.archs.dmsasff_arch import DMSASFFNet
from basicsr.utils import img2tensor, tensor2img

from face_aligner import shape_to_np
from mls_face_warping import mls_affine_deformation


def warp_reference(reference_img, reference_landmarks, compressed_landmarks):
    height, width, _ = reference_img.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)
    affine = mls_affine_deformation(vy, vx, reference_landmarks, compressed_landmarks, alpha=1)
    warped_mls = np.ones_like(reference_img)
    warped_mls[vx, vy] = reference_img[tuple(affine)]

    return warped_mls


def choose_best_keyframes(keyframes, keyframes_landmarks, keyframes_landmark_distances, keyframes_count_usage,
                          frame_name, reference_landmarks, max_keyframes=5, metric="euclidean"):
    if len(keyframes) == max_keyframes:

        min_index = np.argmin(keyframes_count_usage)
        keyframes_count_usage = np.delete(keyframes_count_usage, min_index)
        keyframes_count_usage = np.hstack((keyframes_count_usage, 0))
        keyframes.pop(min_index)
        keyframes_landmarks.pop(min_index)
        keyframes.append(frame_name)
        keyframes_landmarks.append(reference_landmarks)
    else:
        for i in range(len(keyframes)):
            new_distance = compute_landmarks_distance(reference_landmarks, keyframes_landmarks[i], metric)
            keyframes_landmark_distances[i, len(keyframes)] = new_distance
            keyframes_landmark_distances[len(keyframes), i] = new_distance
        keyframes.append(frame_name)
        keyframes_landmarks.append(reference_landmarks)

    keyframes_count_usage /= 2  # Exponential decay
    return keyframes, keyframes_landmarks, keyframes_landmark_distances, keyframes_count_usage


def choose_reference(compressed_landmarks, keyframes_landmarks, metric="euclidean"):
    min_value = np.inf
    min_index = 0
    for i in range(len(keyframes_landmarks)):
        dist = compute_landmarks_distance(compressed_landmarks, keyframes_landmarks[i], metric)
        if dist < min_value:
            min_value = dist
            min_index = i
    return min_index


def compute_landmarks_distance(first_landmarks, second_landmarks, metric="euclidean"):
    distance = 0
    if metric == "euclidean":
        distance = np.sqrt(np.sum((first_landmarks - second_landmarks) ** 2))
    return distance


def generate_videos(results_path, compressed_frame_path, frame_shape, nose_coordinates, fps, crop_size=(512, 512)):
    height, width = frame_shape[:2]
    output_frames = os.listdir(f"{results_path}/5_restored_frame/")
    output_frames.sort(key=lambda f: int(re.sub('\D', '', f)))  # Sort frames correctly
    output_frames = [f"{results_path}/5_restored_frame/{frame}" for frame in output_frames]

    output = cv2.VideoWriter(f"{results_path}/output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    cropped_output = cv2.VideoWriter(f"{results_path}/cropped_output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, crop_size)
    stacked_output = cv2.VideoWriter(f"{results_path}/stacked_output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                     (crop_size[0] * 2, crop_size[1]))

    range_x = crop_size[0] // 2
    range_y = crop_size[1] // 2
    nose_coordinates = [max(nose_coordinates[0], range_x), max(nose_coordinates[1], range_y)]
    for output_frame_path in output_frames:
        frame_name = os.path.basename(output_frame_path)
        output_frame = cv2.imread(output_frame_path)
        output.write(output_frame)
        cropped_output_frame = output_frame[nose_coordinates[1] - range_y: nose_coordinates[1] + range_y,
                               nose_coordinates[0] - range_x: nose_coordinates[0] + range_x, :]
        cropped_output.write(cropped_output_frame)
        input_frame = cv2.imread(f"{compressed_frame_path}/{frame_name}")
        cropped_input_frame = input_frame[nose_coordinates[1] - range_y: nose_coordinates[1] + range_y,
                              nose_coordinates[0] - range_x: nose_coordinates[0] + range_x, :]
        stacked_frame = np.hstack((cropped_input_frame, cropped_output_frame))
        stacked_output.write(stacked_frame)
    output.release()
    cropped_output.release()
    stacked_output.release()


def paste_restored_crop(restored_crop, input, transform_matrix):
    """
    Code borrowed from https://github.com/csxmli2016/DFDNet/blob/8eb80619638a745de6c72f0047cbe79dfb109039/test_FaceDict.py#L61
    :param restored_crop:
    :param input:
    :param transform_matrix:
    :return:
    """
    h, w, _ = input.shape
    inv_M = cv2.invertAffineTransform(transform_matrix)
    inv_crop_img = cv2.warpAffine(restored_crop, inv_M, (w, h))
    mask = np.ones(restored_crop.shape, dtype=np.float32)
    inv_mask = cv2.warpAffine(mask, inv_M, (w, h))
    inv_mask_erosion_removeborder = cv2.erode(inv_mask, np.ones((2, 2), np.uint8))  # to remove the black border
    inv_crop_img_removeborder = inv_mask_erosion_removeborder * inv_crop_img
    total_face_area = np.sum(inv_mask_erosion_removeborder) // 3
    w_edge = int(total_face_area ** 0.5) // 20  # compute the fusion edge based on the area of face
    erosion_radius = w_edge * 2
    inv_mask_center = cv2.erode(inv_mask_erosion_removeborder, np.ones((erosion_radius, erosion_radius), np.uint8))
    blur_size = w_edge * 2
    inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
    merge_img = inv_soft_mask * inv_crop_img_removeborder + (1 - inv_soft_mask) * input

    return merge_img


def restore_video(base_path, crf, max_keyframes):
    seconds_limit = 30

    column_names = ["Video name", "Num frames", "Total inference time (s)", "Mean inference time (s)", "Num keyframes",
                    "Total keyframes choice time (s)", "Mean keyframes choice time (s)"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = "DMSASFFNet"
    model_checkpoint_path = f"pretrained_models/net_g_55000.pth"
    print("Loading DMSASFFNet model...")
    net = DMSASFFNet().to(device)
    checkpoint = torch.load(model_checkpoint_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['params'], strict=True)
    net.eval()
    print("Model loaded")

    face_detector = dlib.cnn_face_detection_model_v1("pretrained_models/dlib_weights.dat")
    landmark_detector = dlib.shape_predictor("pretrained_models/dlib_shape_predictor_68_face_landmarks.dat")

    results_path = f"inference/{model_name}/max_keyframes_{max_keyframes}/LFU"
    Path(results_path).mkdir(parents=True, exist_ok=True)

    for video_dir in sorted(glob.glob(f"{base_path}/compressed_{crf}/*/")):
        video_name = video_dir[:-1].split("/")[-1]
        print(video_name)

        total_inference_time = 0
        total_keyframes_choice_time = 0

        results_dir = f"{results_path}/{video_name}"
        Path(f"{results_dir}/1_original").mkdir(parents=True, exist_ok=True)
        Path(f"{results_dir}/2_reference").mkdir(parents=True, exist_ok=True)
        Path(f"{results_dir}/3_warped").mkdir(parents=True, exist_ok=True)
        Path(f"{results_dir}/4_restored_crop").mkdir(parents=True, exist_ok=True)
        Path(f"{results_dir}/5_restored_frame").mkdir(parents=True, exist_ok=True)

        keyframes = sorted([int(el[:-8]) for el in os.listdir(f"{video_dir}crops") if
                            "key" in el])  # Take only keyframes and remove "_key.jpg" to cast to int
        fps = keyframes[1] - keyframes[0]

        keyframes = []
        keyframes_landmarks = []
        keyframes_landmark_distances = np.zeros((max_keyframes, max_keyframes))
        keyframes_count_usage = np.zeros(max_keyframes)

        video_shape = None
        nose_coordinates = None
        count_keyframes = 0

        crops = os.listdir(f"{video_dir}crops/")
        crops.sort(key=lambda f: int(re.sub('\D', '', f)))  # Sort frames correctly
        crops = [f"{video_dir}crops/{crop}" for crop in crops]
        for i, compressed_crop_path in enumerate(crops):
            frame_name = os.path.basename(compressed_crop_path)
            frame_num = int(frame_name.split(".")[0].split("_")[0])
            print(f"{video_name}: processing {frame_name}")

            compressed_crop = cv2.imread(compressed_crop_path)
            compressed_frame = cv2.imread(f"{video_dir}frames/{frame_name}")
            binary_crop = cv2.imread(f"{video_dir}binary_landmarks/{frame_name}")
            landmarks = np.load(f"{video_dir}landmarks/{frame_name[:-4]}.npy")
            transform_matrix = np.load(f"{video_dir}transform_matrices/{frame_name[:-4]}.npy")

            if "key" in frame_name:
                count_keyframes += 1
                original_crop = cv2.imread(f"{base_path}/original/{video_name}/crops/{frame_name}")
                cv2.imwrite(f"{results_dir}/1_original/{frame_name}", original_crop)

                if max_keyframes != 1:
                    start = timer()
                    keyframes, keyframes_landmarks, keyframes_landmark_distances, keyframes_count_usage = \
                        choose_best_keyframes(keyframes,
                                              keyframes_landmarks,
                                              keyframes_landmark_distances,
                                              keyframes_count_usage,
                                              frame_name,
                                              landmarks,
                                              max_keyframes)
                    end = timer()
                    total_keyframes_choice_time += (end - start)
                else:
                    keyframes = [frame_name]
                    keyframes_landmarks = [landmarks]
            if frame_num == 1:
                video_shape = compressed_frame.shape
                gray_compressed_frame = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2GRAY)
                face = face_detector(gray_compressed_frame, 1)[0].rect
                frame_landmarks = landmark_detector(gray_compressed_frame, face)
                frame_landmarks = shape_to_np(frame_landmarks)
                nose_coordinates = frame_landmarks[34]

            reference_index = choose_reference(landmarks, keyframes_landmarks)
            reference_crop = cv2.imread(f"{results_dir}/1_original/{keyframes[reference_index]}")
            reference_landmarks = keyframes_landmarks[reference_index]
            keyframes_count_usage[reference_index] += 1
            cv2.imwrite(f"{results_dir}/2_reference/{frame_name}", reference_crop)

            warped_crop = warp_reference(reference_crop, reference_landmarks, landmarks)
            cv2.imwrite(f"{results_dir}/3_warped/{frame_name}", warped_crop)

            compressed_crop, warped_crop, binary_crop = img2tensor(
                [compressed_crop / 255, warped_crop / 255, binary_crop / 255], bgr2rgb=True, float32=True)
            compressed_crop = compressed_crop.unsqueeze(0).to(device)
            warped_crop = warped_crop.unsqueeze(0).to(device)
            binary_crop = binary_crop.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    start = timer()
                    output = net(compressed_crop, warped_crop, binary_crop)
                    end = timer()
                    total_inference_time += (end - start)
                    restored_crop = tensor2img(output)
                del output
                torch.cuda.empty_cache()
            except Exception as e:
                print(f'DMSASFFNet inference fail: {e}')
                restored_crop = tensor2img(compressed_crop)

            cv2.imwrite(f"{results_dir}/4_restored_crop/{frame_name}", restored_crop)

            restored_frame = paste_restored_crop(restored_crop, compressed_frame, transform_matrix)
            cv2.imwrite(f"{results_dir}/5_restored_frame/{frame_name}", restored_frame)

            if i >= fps * seconds_limit:
                break

        mean_inference_time = total_inference_time / len(crops)
        mean_keyframes_choice_time = total_keyframes_choice_time / count_keyframes

        result = [[video_name, len(crops), total_inference_time, mean_inference_time, count_keyframes,
                   total_keyframes_choice_time, mean_keyframes_choice_time]]
        df = pd.DataFrame(result)

        if not os.path.isfile(f"{results_path}/time_measurement.csv"):
            df.columns = column_names
            df.to_csv(f"{results_path}/time_measurement.csv", index=False)
        else:
            df.to_csv(f"{results_path}/time_measurement.csv", header=False, mode="a", index=False)

        print(f"Generating output video for file: ", video_name)
        generate_videos(results_dir, f"{base_path}/compressed_{crf}/{video_name}/frames", video_shape, nose_coordinates,
                        fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="HQ videos should be in {BASE_PATH}/original")
    parser.add_argument("--crf", type=int, default=42, help="Constant Rate Factor")
    parser.add_argument("--max_keyframes", type=int, default=5, help="Max cardinality of the set of keyframes")
    args = parser.parse_args()

    base_path = args.base_path
    crf = args.crf
    max_keyframes = args.max_keyframes
    restore_video(base_path, crf, max_keyframes)
