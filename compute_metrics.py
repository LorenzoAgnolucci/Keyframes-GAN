import os
import glob
import cv2
import lpips as LPIPS
from brisque import BRISQUE
import argparse

from BasicSR.basicsr.metrics import lpips, psnr_ssim


def compute_metrics(output_base_path, gt_path):
    metrics_list = ["PSNR", "SSIM", "LPIPS", "BRISQUE"]
    total_metric_results = {metric: 0 for metric in metrics_list}

    lpips_net = LPIPS.LPIPS(net='alex').cuda()
    brisque = BRISQUE()

    count_videos = 0

    for video_dir in sorted(glob.glob(f"{output_base_path}/*/")):
        video_name = video_dir[:-1].split("/")[-1]
        print(video_name)

        count_videos += 1
        video_path = f"{output_base_path}/{video_name}"
        video_metric_results = {metric: 0 for metric in metrics_list}
        count_frames = 0

        for output_img_path in sorted(glob.glob(f"{video_path}/4_restored_crop/*.jpg")):
            count_frames += 1
            img_name = os.path.basename(output_img_path)
            output_img = cv2.imread(output_img_path)
            output_img = cv2.resize(output_img, (256, 256), cv2.INTER_CUBIC)

            gt_img = cv2.imread(f"{gt_path}/{video_name}/crops/{img_name}")

            if "PSNR" in metrics_list:
                video_metric_results["PSNR"] += psnr_ssim.calculate_psnr(output_img, gt_img, crop_border=0)

            if "SSIM" in metrics_list:
                video_metric_results["SSIM"] += psnr_ssim.calculate_ssim(output_img, gt_img, crop_border=0)

            if "LPIPS" in metrics_list:
                video_metric_results["LPIPS"] += lpips.calculate_lpips(output_img, gt_img, lpips_net)

            if "BRISQUE" in metrics_list:
                video_metric_results["BRISQUE"] += brisque.get_score(output_img)

        for metric in video_metric_results.keys():
            video_metric_results[metric] /= count_frames

        print("\n")
        for metric_name, metric_value in video_metric_results.items():
            total_metric_results[metric_name] += metric_value
            print(f"{metric_name}: {metric_value}")
        print("\n")

        with open(f"{output_base_path}/metric_results.txt", "a") as f:
            f.write(f"{video_name}\n")
            for metric_name, metric_value in video_metric_results.items():
                f.write(f"{metric_name}: {metric_value}\n")
            f.write("\n\n")

    for metric in total_metric_results.keys():
        total_metric_results[metric] /= count_videos

    print("Total metric results: \n")
    for metric_name, metric_value in total_metric_results.items():
        print(f"{metric_name}: {metric_value}")

    with open(f"{output_base_path}/metric_results.txt", "a") as f:
        f.write(f"Total metric results\n")
        for metric_name, metric_value in total_metric_results.items():
            f.write(f"{metric_name}: {metric_value}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, help="Path of directory with the HQ videos")
    parser.add_argument("--inference_path", type=str, default="inference/DMSASFFNet/max_keyframes_5/LFU", help="Inference path")
    args = parser.parse_args()

    gt_path = args.gt_path
    output_base_path = args.inference_path
    compute_metrics(output_base_path, gt_path)


