import os
import sys

# 获取项目根目录并添加到模块搜索路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "algorithm", "core"))
sys.path.insert(0, project_root)

import argparse


import numpy as np
import torch
from PIL import Image
import clip
import algorithm.core.transforms as T
from models import build_model
from predefined_keypoints import *
from algorithm.core.util import box_ops
from algorithm.core.util.config import Config
from algorithm.core.util.utils import clean_state_dict
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from torchvision.ops import nms
import math
import atexit
import signal
import sys
import yaml





# 定义一个全局变量来跟踪是否需要清理
cleanup_done = False

def text_encoding(instance_names, keypoints_names, model, device):
    ins_text_embeddings = []
    for cat in instance_names:
        instance_description = f"a photo of {cat.lower().replace('_', ' ').replace('-', ' ')}"
        text = clip.tokenize(instance_description).to(device)
        text_features = model.encode_text(text)  # 1*512
        ins_text_embeddings.append(text_features)
    ins_text_embeddings = torch.cat(ins_text_embeddings, dim=0)

    kpt_text_embeddings = []
    for kpt in keypoints_names:
        kpt_description = f"a photo of {kpt.lower().replace('_', ' ')}"
        text = clip.tokenize(kpt_description).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)  # 1*512
        kpt_text_embeddings.append(text_features)
    kpt_text_embeddings = torch.cat(kpt_text_embeddings, dim=0)
    return ins_text_embeddings, kpt_text_embeddings




def calculate_yaw2(left_eye_x, left_eye_y, right_eye_x, right_eye_y, foot_x, foot_y):
    """
    计算 yaw2 角度。
    基于鼻子垂足到两眼之间的比值，映射到 -90 到 90 度。
    """
    if foot_x is None or foot_y is None:
        assert False

    if abs(right_eye_x-left_eye_x) > abs(right_eye_y-left_eye_y):
        ratio = (foot_x-left_eye_x)/(right_eye_x-left_eye_x)
    else:
        ratio = (foot_y-left_eye_y)/(right_eye_y-left_eye_y)
    # 映射比值到 -90 到 90 度
    yaw2 = (ratio - 0.5) * 180  # 将 [0, 1] 映射到 [-90, 90]
    return yaw2



def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = Config.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

    


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")  # load image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image




def get_pose_output(model, image, instance_text_prompt, keypoint_text_prompt, box_threshold, iou_threshold, with_logits=True, cpu_only=False):
    instance_list = instance_text_prompt.split(',')
    device = "cuda" if not cpu_only else "cpu"
    ins_text_embeddings, kpt_text_embeddings = text_encoding(instance_list, keypoint_text_prompt, model.clip_model, device)
    target = {}
    target["instance_text_prompt"] = instance_list
    target["keypoint_text_prompt"] = keypoint_text_prompt
    target["object_embeddings_text"] = ins_text_embeddings.float()
    kpt_text_embeddings = kpt_text_embeddings.float()
    kpts_embeddings_text_pad = torch.zeros(100 - kpt_text_embeddings.shape[0], 512, device=device)
    target["kpts_embeddings_text"] = torch.cat((kpt_text_embeddings, kpts_embeddings_text_pad), dim=0)
    kpt_vis_text = torch.ones(kpt_text_embeddings.shape[0], device=device)
    kpt_vis_text_pad = torch.zeros(kpts_embeddings_text_pad.shape[0], device=device)
    target["kpt_vis_text"] = torch.cat((kpt_vis_text, kpt_vis_text_pad), dim=0)
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], [target])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    keypoints = outputs["pred_keypoints"][0][:, :2 * len(keypoint_text_prompt)]  # (nq, n_kpts * 2)
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    keypoints_filt = keypoints.cpu().clone()

    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    keypoints_filt = keypoints_filt[filt_mask]  # num_filt, n_kpts * 2

    keep_indices = nms(box_ops.box_cxcywh_to_xyxy(boxes_filt), logits_filt.max(dim=1)[0], iou_threshold=iou_threshold)
    filtered_boxes = boxes_filt[keep_indices]
    filtered_keypoints = keypoints_filt[keep_indices]

    # 只保留置信度最高的框
    if filtered_boxes.shape[0] > 0:
        best_box_idx = torch.argmax(logits_filt.max(dim=1)[0][keep_indices])  # 找到置信度最高的框索引
        filtered_boxes = filtered_boxes[best_box_idx].unsqueeze(0)  # 保留唯一的框
        filtered_keypoints = filtered_keypoints[best_box_idx].unsqueeze(0)  # 对应的关键点

    return filtered_boxes, filtered_keypoints

def perpendicular_foot(px, py, x1, y1, x2, y2):
    # 用于计算点 (px,py) 到直线 (x1,y1) 到 (x2,y2) 的垂足坐标
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    if A**2 + B**2 == 0:
        return None, None
    foot_x = (B * (B * px - A * py) - A * C) / (A**2 + B**2)
    foot_y = (A * (-B * px + A * py) - B * C) / (A**2 + B**2)
    return foot_x, foot_y

def save_positions(keypoints_filt, keypoint_text_prompt, output_dir, image_file, size):
    """
    提取识别到的眼睛和鼻子位置，计算翻滚角、偏航角和俯仰角。
    Args:
        keypoints_filt (torch.Tensor): 模型预测的关键点坐标 (num_boxes, num_keypoints * 2)。
        keypoint_text_prompt (list): 关键点名称列表。
        output_dir (str): 输出目录路径。
        image_file (str): 当前处理的图片文件名。
    Returns:
        list: 包含每个框的眼睛、鼻子和角度信息的列表。
    """
    img_W, img_H = size[0], size[1]
    eye_indices = [i for i, name in enumerate(keypoint_text_prompt) if "eye" in name.lower()]
    nose_index = next((i for i, name in enumerate(keypoint_text_prompt) if "nose" in name.lower()), None)
    if len(eye_indices) < 2 or nose_index is None:
        print(f"Not enough keypoints found in the keypoint text prompt for {image_file}.")
        return []

    eye_positions = []
    for box_idx, keypoints in enumerate(keypoints_filt):
        left_eye_x, left_eye_y = None, None
        right_eye_x, right_eye_y = None, None
        nose_x, nose_y = None, None
        for idx in eye_indices:
            x, y = float(keypoints[2 * idx]), float(keypoints[2 * idx + 1])
            if "left" in keypoint_text_prompt[idx].lower():
                left_eye_x, left_eye_y = x, y
            elif "right" in keypoint_text_prompt[idx].lower():
                right_eye_x, right_eye_y = x, y
        if nose_index is not None:
            nose_x, nose_y = float(keypoints[2 * nose_index]), float(keypoints[2 * nose_index + 1])


        if left_eye_x is None or right_eye_x is None or nose_x is None:
            print("Error: Missing required keypoints (left eye, right eye, or nose).")
            return []

        # 计算翻滚角
        roll_angle = None
        if left_eye_x is not None and right_eye_x is not None:
            if left_eye_x > right_eye_x:
                left_eye_x, right_eye_x = right_eye_x, left_eye_x
                left_eye_y, right_eye_y = right_eye_y, left_eye_y
            delta_x_roll = right_eye_x - left_eye_x
            delta_y_roll = right_eye_y - left_eye_y
            if abs(delta_x_roll) >= 0.01:
                roll_angle = math.degrees(math.atan2(delta_y_roll, delta_x_roll))
                roll_angle = max(-90, min(90, roll_angle))

        # 计算偏航角
        yaw_angle = None
        if left_eye_x is not None and right_eye_x is not None and nose_x is not None:
            mid_eye_x = (left_eye_x + right_eye_x) / 2
            mid_eye_y = (left_eye_y + right_eye_y) / 2
            delta_x_yaw = nose_x - mid_eye_x
            delta_y_yaw = nose_y - mid_eye_y
            if abs(delta_y_yaw) >= 0.01:
                yaw_angle = math.degrees(math.atan2(delta_x_yaw, delta_y_yaw))
                yaw_angle = max(-90, min(90, yaw_angle))

        # 计算 yaw2 角度
        yaw2_angle = None
        foot_x, foot_y = None, None
        if left_eye_x is not None and right_eye_x is not None and nose_x is not None:
            # 计算垂足坐标
            foot_x, foot_y = perpendicular_foot(nose_x*img_W, nose_y*img_H, left_eye_x*img_W, left_eye_y*img_H, right_eye_x*img_W, right_eye_y*img_H)
            yaw2_angle = calculate_yaw2(left_eye_x*img_W, left_eye_y*img_H, right_eye_x*img_W, right_eye_y*img_H, foot_x, foot_y)
        # 保存结果
        box_positions = {
            "image_file": image_file,
            "left_eye": {"x": left_eye_x, "y": left_eye_y},
            "right_eye": {"x": right_eye_x, "y": right_eye_y},
            "nose": {"x": nose_x, "y": nose_y},
            "roll_angle": roll_angle,
            "yaw_angle": yaw_angle,
            "yaw2_angle": yaw2_angle,
            "foot_point": {"x": foot_x, "y": foot_y},  # 保存垂足坐标
            "注释": (
                "roll_angle 表示头部倾斜角度（绕 Z 轴旋转），正值表示右眼高于左眼，负值表示左眼高于右眼。\n"
                "yaw_angle 表示头部偏转角度（绕 Y 轴旋转，基于垂直面），正值表示鼻子偏向右侧，负值表示鼻子偏向左侧。\n"
                "yaw2_angle 是基于鼻子垂足到两眼之间的比值计算的角度，正值表示向右转头，负值表示向左转头。\n"
                "如果值为 null，表示关键点位置异常或无法计算角度。"
            )
        }
        eye_positions.append({"box_index": box_idx, "eyes": box_positions})

    return eye_positions



def plot_on_image(image_pil, tgt, keypoint_skeleton, keypoint_text_prompt, output_dir, frame_idx,
                  roll_angle=None, yaw_angle=None, yaw2_angle=None, foot_point=None):
    """
    在图像上绘制关键点、骨骼连线、边界框、角度信息以及垂足。
    
    Args:
        image_pil (PIL.Image): 输入图像。
        tgt (dict): 包含预测框和关键点的目标字典。
        keypoint_skeleton (list): 骨骼连线定义。
        keypoint_text_prompt (list): 关键点名称列表。
        output_dir (str): 输出目录路径。
        frame_idx (int): 当前帧索引。
        roll_angle (float): 翻滚角（可选）。
        yaw_angle (float): 偏航角（可选）。
        yaw2_angle (float): Yaw2 角度（可选）。
        foot_point (dict): 垂足坐标（可选）。
    """
    num_kpts = len(keypoint_text_prompt)
    H, W = tgt["size"]  # 图像高度和宽度
    fig = plt.figure(frameon=False)
    dpi = plt.gcf().dpi
    fig.set_size_inches(W / dpi, H / dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = plt.gca()
    ax.imshow(image_pil, aspect='equal')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')

    # 定义关键点颜色
    color_kpt = [
        [0.00, 0.00, 0.00], [1.00, 1.00, 1.00], [1.00, 0.00, 0.00], [1.00, 1.00, 0.00],
        [0.50, 0.16, 0.16], [0.00, 0.00, 1.00], [0.69, 0.88, 0.90], [0.00, 1.00, 0.00],
        [0.63, 0.13, 0.94], [0.82, 0.71, 0.55], [1.00, 0.38, 0.00], [0.53, 0.15, 0.34],
        [1.00, 0.39, 0.28], [1.00, 0.00, 1.00], [0.04, 0.09, 0.27], [0.20, 0.63, 0.79],
        [0.94, 0.90, 0.55], [0.33, 0.42, 0.18], [0.53, 0.81, 0.92], [0.71, 0.49, 0.86],
        [0.25, 0.88, 0.82], [0.5, 0.0, 0.0], [0.0, 0.3, 0.3], [1.0, 0.85, 0.73],
        [0.29, 0.0, 0.51], [0.7, 0.5, 0.35], [0.44, 0.5, 0.56], [0.25, 0.41, 0.88],
        [0.0, 0.5, 0.0], [0.56, 0.27, 0.52], [1.0, 0.84, 0.0], [1.0, 0.5, 0.31],
        [0.85, 0.57, 0.94]
    ]

    # 绘制边界框
    polygons = []
    color_box = [0.53, 0.81, 0.92]  # 边界框颜色
    for box in tgt['boxes'].cpu():
        unnormbbox = box * torch.Tensor([W, H, W, H])
        unnormbbox[:2] -= unnormbbox[2:] / 2
        [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))

    # 添加填充透明度和边框虚线
    p = PatchCollection(polygons, facecolor=color_box, linewidths=0, alpha=0.3)  # 填充透明度
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', linestyle="--", edgecolors=color_box, linewidths=3)  # 虚线边框
    ax.add_collection(p)

    # 绘制关键点和骨骼连线
    if 'keypoints' in tgt:
        sks = np.array(keypoint_skeleton)
        if sks != []:
            if sks.min() == 1:
                sks = sks - 1  # 将骨骼索引从 1-based 转为 0-based
        for idx, ann in enumerate(tgt['keypoints']):
            kp = np.array(ann.cpu())
            Z = kp[:num_kpts * 2] * np.array([W, H] * num_kpts)  # 归一化关键点坐标
            x = Z[0::2]
            y = Z[1::2]
            c = color_box

            # 绘制骨骼连线
            for sk in sks:
                plt.plot(x[sk], y[sk], linewidth=1, color=c)

            # 绘制关键点
            for i in range(num_kpts):
                c_kpt = color_kpt[i]
                plt.plot(x[i], y[i], 'o', markersize=4, markerfacecolor=c_kpt, markeredgecolor='k', markeredgewidth=0.5)



    # 添加角度信息到图片右上角
    if roll_angle is not None or yaw_angle is not None or yaw2_angle is not None:
        text = ""
        if roll_angle is not None:
            text += f"Roll: {roll_angle:.2f}°\n"
        if yaw_angle is not None:
            text += f"Yaw: {yaw_angle:.2f}°\n"
        if yaw2_angle is not None:
            text += f"Yaw2: {yaw2_angle:.2f}°\n"

        # 文本位置调整
        margin_x = 0  # 文本右边缘到图像右边框的距离
        margin_y = 0  # 文本顶部到图像顶部的距离
        plt.text(W - margin_x, margin_y, text, fontsize=30, color='white',
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='black', alpha=0.8))

    # 保存图像
    ax.set_axis_off()
    savename = os.path.join(output_dir, f"frame{int(frame_idx)}.jpg")  # 动态生成文件名
    print("savename: {}".format(savename))
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    plt.savefig(savename, dpi=dpi)
    plt.close()




def release_resources():
    """
    释放资源的函数，例如关闭文件、保存日志等。
    """
    global cleanup_done
    if not cleanup_done:
        print("\nReleasing resources and cleaning up...")
        # 在这里添加具体的资源释放逻辑，例如：
        # 1. 关闭打开的文件
        # 2. 保存未完成的任务
        # 3. 释放模型或其他占用的内存
        cleanup_done = True
        print("Cleanup completed.")

# 使用 atexit 注册清理函数
atexit.register(release_resources)

# 捕获键盘中断信号 (Ctrl+C)
def handle_keyboard_interrupt(signal, frame):
    print("\nKeyboardInterrupt detected. Exiting gracefully...")
    release_resources()
    sys.exit(0)

# 注册信号处理函数
signal.signal(signal.SIGINT, handle_keyboard_interrupt)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



if __name__ == "__main__":
    # try:
        parser = argparse.ArgumentParser("Pose Inference", add_help=True)
        # parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
        # parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file")
        parser.add_argument("--image_dir", "-d", type=str, default=None, help="path to input image directory")
        parser.add_argument("--instance_text_prompt", "-t", type=str, required=True, help="instance text prompt")
        parser.add_argument("--keypoint_text_example", "-k", type=str, default=None, help="keypoint text prompt")
        parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
        parser.add_argument("--box_threshold", type=float, default=0, help="box threshold")
        parser.add_argument("--iou_threshold", type=float, default=0.9, help="box threshold")
        parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
        args = parser.parse_args()

        # 加载配置文件
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构造 config.yml 的相对路径
        config_path = os.path.join(current_dir, "algorithm", "config.yml")
        config = load_config(config_path)

        # config = load_config("config.yml")
        # 从 config.yml 中读取模型路径
        config_file = config.get("config_file")
        checkpoint_path = config.get("checkpoint_path")
 


        # config_file = args.config_file
        # checkpoint_path = args.checkpoint_path

        instance_text_prompt = args.instance_text_prompt
        keypoint_text_example = args.keypoint_text_example
        output_dir = args.output_dir
        box_threshold = args.box_threshold
        iou_threshold = args.iou_threshold

        # 加载关键点和骨骼信息
        if keypoint_text_example in globals():
            keypoint_dict = globals()[keypoint_text_example]
            keypoint_text_prompt = keypoint_dict.get("keypoints")
            keypoint_skeleton = keypoint_dict.get("skeleton")
        elif instance_text_prompt in globals():
            keypoint_dict = globals()[instance_text_prompt]
            keypoint_text_prompt = keypoint_dict.get("keypoints")
            keypoint_skeleton = keypoint_dict.get("skeleton")
        else:
            keypoint_dict = globals()["animal"]
            keypoint_text_prompt = keypoint_dict.get("keypoints")
            keypoint_skeleton = keypoint_dict.get("skeleton")

        os.makedirs(output_dir, exist_ok=True)
        model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

        frame_idx = 1
        if args.image_dir:
            image_dir = args.image_dir
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            # 按文件名排序，确保连续性
            image_files = sorted(
                [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions],
                key=lambda x: x.lower()  # 按文件名的字典顺序排序（不区分大小写）
            )
            for idx, image_file in enumerate(image_files):
                image_path = os.path.join(image_dir, image_file)
                print(f"Processing image {idx + 1}/{len(image_files)}: {image_path}")
                # try:
                image_pil, image = load_image(image_path)
                boxes_filt, keypoints_filt = get_pose_output(
                    model, image, instance_text_prompt, keypoint_text_prompt, box_threshold, iou_threshold,
                    cpu_only=args.cpu_only
                )
                size = image_pil.size
                pred_dict = {
                    "boxes": boxes_filt,
                    "keypoints": keypoints_filt,
                    "size": [size[1], size[0]]
                }
                eye_data = save_positions(keypoints_filt, keypoint_text_prompt, output_dir, image_file, size)
                roll_angle = yaw_angle = yaw2_angle = foot_point = None
                if len(eye_data) > 0:
                    roll_angle = eye_data[0]["eyes"]["roll_angle"]
                    yaw_angle = eye_data[0]["eyes"]["yaw_angle"]
                    yaw2_angle = eye_data[0]["eyes"]["yaw2_angle"]
                    foot_point = eye_data[0]["eyes"]["foot_point"]
                plot_on_image(
                    image_pil, pred_dict, keypoint_skeleton, keypoint_text_prompt, output_dir, frame_idx,
                    roll_angle=roll_angle, yaw_angle=yaw_angle, yaw2_angle=yaw2_angle, foot_point=foot_point
                )
                frame_idx += 1
                # except Exception as e:
                #     print(f"Error processing image {image_path}: {e}")

    # finally:
        # 确保在程序结束时调用清理函数
        release_resources()