import os
import argparse
import numpy as np
from tqdm import tqdm

import mediapy as media
import cv2


def process_all(root_path, pose_dir_path, fps):
    cam_all_pose_dir_path = os.path.join(pose_dir_path, 'ALL')
    
    if os.path.exists(cam_all_pose_dir_path):
        pose_files = sorted([file for file in os.listdir(cam_all_pose_dir_path) if not file.endswith('_top.png')])
        pose_top_files = sorted([file for file in os.listdir(cam_all_pose_dir_path) if file.endswith('_top.png')])
        
        # Get width and height of the first image
        if len(pose_files):
            first_image = cv2.imread(os.path.join(cam_all_pose_dir_path, pose_files[0]))
            height, width, _ = first_image.shape
            
            filename = 'iter_pose_all.mp4'
            
            with media.VideoWriter(
                path=os.path.join(root_path, filename), fps=fps, shape=(height, width*2)
            ) as writer:
                for pose_file, pose_top_file in tqdm(zip(pose_files, pose_top_files), desc="Writing pose all videos", total=len(pose_files), leave=False):
                    pose = cv2.imread(os.path.join(cam_all_pose_dir_path, pose_file))
                    pose_rgb = cv2.cvtColor(pose, cv2.COLOR_BGR2RGB)
                    pose_rgb = cv2.resize(pose_rgb, (width, height))
                    
                    pose_top = cv2.imread(os.path.join(cam_all_pose_dir_path, pose_top_file))
                    pose_top_rgb = cv2.cvtColor(pose_top, cv2.COLOR_BGR2RGB)
                    pose_top_rgb = cv2.resize(pose_top_rgb, (width, height))
                    cv2.putText(pose_top_rgb, "TOP VIEW", (int(width/2)-80, height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    
                    # Concatenate image_rgb and pose_rgb horizontally
                    combined_image = cv2.hconcat([pose_rgb, pose_top_rgb])
                    
                    # Add the combined image to the video writer
                    writer.add_image(combined_image)

def process_trajectory(root_path, pose_dir_path, fps):
    cam_trj_pose_dir_path = os.path.join(pose_dir_path, 'Trajectory')
    
    if os.path.exists(cam_trj_pose_dir_path):
        pose_files = sorted([file for file in os.listdir(cam_trj_pose_dir_path) if not file.endswith('_top.png')])
        pose_top_files = sorted([file for file in os.listdir(cam_trj_pose_dir_path) if file.endswith('_top.png')])
        
        # Get width and height of the first image
        if len(pose_files):
            first_image = cv2.imread(os.path.join(cam_trj_pose_dir_path, pose_files[0]))
            height, width, _ = first_image.shape
            
            filename = 'iter_trajectory.mp4'
            
            with media.VideoWriter(
                path=os.path.join(root_path, filename), fps=fps, shape=(height, width*2)
            ) as writer:
                for pose_file, pose_top_file in tqdm(zip(pose_files, pose_top_files), desc="Writing pose all videos", total=len(pose_files), leave=False):
                    pose = cv2.imread(os.path.join(cam_trj_pose_dir_path, pose_file))
                    pose_rgb = cv2.cvtColor(pose, cv2.COLOR_BGR2RGB)
                    pose_rgb = cv2.resize(pose_rgb, (width, height))
                    
                    pose_top = cv2.imread(os.path.join(cam_trj_pose_dir_path, pose_top_file))
                    pose_top_rgb = cv2.cvtColor(pose_top, cv2.COLOR_BGR2RGB)
                    pose_top_rgb = cv2.resize(pose_top_rgb, (width, height))
                    cv2.putText(pose_top_rgb, "TOP VIEW", (int(width/2)-80, height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    
                    # Concatenate image_rgb and pose_rgb horizontally
                    combined_image = cv2.hconcat([pose_rgb, pose_top_rgb])
                    
                    # Add the combined image to the video writer
                    writer.add_image(combined_image)
                
def process_cameras(root_path, image_dir_path, pose_dir_path, fps, depth):
    RESULT_CAM_STRs = sorted([os.path.basename(cam_dir) for cam_dir in os.listdir(image_dir_path)])
    print("Cameras are in : ", RESULT_CAM_STRs)
    
    for idx, cam_str in enumerate(tqdm(RESULT_CAM_STRs, desc="Processing cameras")):        
        cam_image_dir_path = os.path.join(image_dir_path, cam_str)
        cam_pose_dir_path = os.path.join(pose_dir_path, cam_str)
        
        # Get the list of image files in the directory
        image_files = sorted(os.listdir(cam_image_dir_path))
        pose_files = sorted([file for file in os.listdir(cam_pose_dir_path) if not (file.endswith('_front.png') or file.endswith('_side.png'))])
        pose_front_files = sorted([file for file in os.listdir(cam_pose_dir_path) if file.endswith('_front.png')])
        pose_side_files = sorted([file for file in os.listdir(cam_pose_dir_path) if file.endswith('_side.png')])
        
        # if idx == 0:
        # Get width and height of the first image
        first_image_path = os.path.join(cam_image_dir_path, image_files[-1])
        first_image = cv2.imread(first_image_path)
        height, width, _ = first_image.shape
                
        # height, width, _ = first_image.shape
        width_pose1 = int(width/3)
        width_pose2 = width_pose1
        width_pose3 = width - width_pose1*2
        
        filename = f'iter_depth_w_pose_{int(cam_str[3:])}.mp4' if depth else f'iter_img_w_pose_{cam_str[3:]}.mp4'
        
        with media.VideoWriter(
            path=os.path.join(root_path, filename), fps=fps, shape=(height*2, width)
        ) as writer:
            for image_f, pose_f, pose_front_f, pose_side_f in tqdm(zip(image_files, pose_files, pose_front_files, pose_side_files), desc="Writing videos", total=len(image_files), leave=False):
                img = cv2.imread(os.path.join(cam_image_dir_path, image_f))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # if cam_id == 2 or cam_id == 3:
                    # img = img[height:, :]
                img = cv2.resize(img, (width, height))
                
                # Add string to the image
                str = "Iteration: " + image_f.split('.')[0]
                cv2.putText(img, str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3) # whtie
                            
                pose = cv2.imread(os.path.join(cam_pose_dir_path, pose_f))
                pose_rgb = cv2.cvtColor(pose, cv2.COLOR_BGR2RGB)
                pose_rgb = pose_rgb[:, 100:-100]
                pose_rgb = cv2.resize(pose_rgb, (width_pose1, height))
                
                pose_front = cv2.imread(os.path.join(cam_pose_dir_path, pose_front_f))
                pose_front_rgb = cv2.cvtColor(pose_front, cv2.COLOR_BGR2RGB)
                pose_front_rgb = pose_front_rgb[:, 100:-100]
                pose_front_rgb = cv2.resize(pose_front_rgb, (width_pose2, height))
                cv2.putText(pose_front_rgb, "FRONT VIEW", (int(width_pose2/2)-80, height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                pose_side = cv2.imread(os.path.join(cam_pose_dir_path, pose_side_f))
                pose_side_rgb = cv2.cvtColor(pose_side, cv2.COLOR_BGR2RGB)
                pose_side_rgb = pose_side_rgb[:, 100:-100]
                pose_side_rgb = cv2.resize(pose_side_rgb, (width_pose3, height))
                cv2.putText(pose_side_rgb, "SIDE VIEW", (int(width_pose3/2)-60, height-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # Concatenate image_rgb and pose_rgb horizontally
                combined_image = cv2.hconcat([pose_rgb, pose_front_rgb, pose_side_rgb])
                combined_image = cv2.vconcat([img, combined_image])
                
                # Add the combined image to the video writer
                writer.add_image(combined_image)
                
def process_video(root_path, fps, depth, cam_num):
    image_dir_path = os.path.join(root_path, 'iter_depth' if depth else 'iter_image')
    pose_dir_path = os.path.join(root_path, 'iter_pose')
    
    print("Camera number: ", cam_num)
    
    process_all(root_path, pose_dir_path, fps)
    
    process_trajectory(root_path, pose_dir_path, fps)
                
    process_cameras(root_path, image_dir_path, pose_dir_path, fps, depth)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Make video from rendering images",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--result_path", "-i", type=str, required=True, help="Path to the rendering images directory")
    parser.add_argument("--fps", "-f", type=int, default=10, help="Frames per second")
    parser.add_argument("--depth", "-d", action="store_true", help="Use depth images")
    
    args = parser.parse_args()
    
    process_video(args.result_path, args.fps, args.depth, 4)
