import os
import sys
from algorithm.core.predefined_keypoints import *

# 将项目根目录添加到模块搜索路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入算法包
import algorithm


def test_pose_estimation():
    # 配置路径（相对路径）
    config_path = "algorithm/config.yml"  # 配置文件路径
    checkpoint_path = "algorithm/core/weights/Unipose_swint.pth"  # 模型权重路径
    image_dir = "algorithm/core/input/video-cat-updown"  # 输入图像文件夹路径
    output_dir = "algorithm/core/output/video-cat-updown"  # 输出文件夹路径

    # 参数设置
    instance_text_prompt = "cat"  # 实例文本提示
    keypoint_text_example = None  # 关键点文本提示（可选）
    box_threshold = 0  # 边界框阈值
    iou_threshold = 0.9  # IOU 阈值
    cpu_only = False  # 是否仅使用 CPU

    # 加载配置
    config = algorithm.load_config(config_path)

    # 加载模型
    model = algorithm.load_model(config["config_file"], checkpoint_path, cpu_only=cpu_only)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

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

    # 获取图像文件列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = sorted(
        [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions],
        key=lambda x: x.lower()  # 按文件名排序
    )

    # 批量处理图像
    frame_idx = 1
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        print(f"Processing image {idx + 1}/{len(image_files)}: {image_path}")

        # 加载图像
        image_pil, image_tensor = algorithm.load_image(image_path)

        # 获取姿态估计结果
        boxes_filt, keypoints_filt = algorithm.get_pose_output(
            model, image_tensor, instance_text_prompt, keypoint_text_prompt,
            box_threshold, iou_threshold, cpu_only=cpu_only
        )

        # 保存关键点位置和角度信息
        size = image_pil.size
        eye_data = algorithm.save_positions(keypoints_filt, keypoint_text_prompt, output_dir, image_file, size)

        # 提取角度信息
        roll_angle = yaw_angle = yaw2_angle = foot_point = None
        if len(eye_data) > 0:
            roll_angle = eye_data[0]["eyes"]["roll_angle"]
            yaw_angle = eye_data[0]["eyes"]["yaw_angle"]
            yaw2_angle = eye_data[0]["eyes"]["yaw2_angle"]
            foot_point = eye_data[0]["eyes"]["foot_point"]

        # 在图像上绘制结果
        algorithm.plot_on_image(
            image_pil,
            {"boxes": boxes_filt, "keypoints": keypoints_filt, "size": [size[1], size[0]]},
            keypoint_skeleton,
            keypoint_text_prompt,
            output_dir,
            frame_idx,
            roll_angle=roll_angle,
            yaw_angle=yaw_angle,
            yaw2_angle=yaw2_angle,
            foot_point=foot_point
        )

        frame_idx += 1

    print("Pose estimation completed.")


if __name__ == "__main__":
    test_pose_estimation()