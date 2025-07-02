import cv2
import numpy as np
import yaml
import re
import os
import argparse
from datetime import datetime
import uuid
import glob

def generate_session_id():
    """Generate a unique session ID based on timestamp and UUID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"session_{timestamp}_{short_uuid}"

def create_session_directory(output_dir, session_id=None):
    """
    Create a session directory with Detection and Results subdirectories.
    
    Args:
        output_dir: Base output directory
        session_id: Optional session ID, if None a new one will be generated
        
    Returns:
        session_dir: Path to the created session directory
        detection_dir: Path to Detection directory
        results_dir: Path to Results directory
        session_id: The session ID used
    """
    if session_id is None:
        session_id = generate_session_id()
    
    # Create charuco_detection subdirectory structure
    session_dir = os.path.join(output_dir, "charuco_detection", session_id)
    detection_dir = os.path.join(session_dir, "Detection")
    results_dir = os.path.join(session_dir, "Results")
    
    # Create directory structure
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(detection_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create camera subdirectories in Detection
    os.makedirs(os.path.join(detection_dir, "Camera_0"), exist_ok=True)
    os.makedirs(os.path.join(detection_dir, "Camera_1"), exist_ok=True)
    
    print(f"Created session: {session_id}")
    print(f"Session directory: {session_dir}")
    print(f"Detection directory: {detection_dir}")
    print(f"Results directory: {results_dir}")
    
    return session_dir, detection_dir, results_dir, session_id

def load_camera_params(yaml_file, camera_id=0):
    """
    Load camera parameters from a YAML file.
    
    Args:
        yaml_file: Path to the YAML file containing camera parameters
        camera_id: Camera ID (0 or 1)
        
    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    """
    fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode(f'camera_{camera_id}').getNode('camera_matrix').mat()
    dist_coeffs = fs.getNode(f'camera_{camera_id}').getNode('distortion_vector').mat()
    fs.release()
    
    return camera_matrix, dist_coeffs

def find_image_files(image_dir, pattern="*.png"):
    """
    Find all image files in a directory matching the pattern.
    
    Args:
        image_dir: Directory containing images
        pattern: File pattern to match (default: "*.png")
        
    Returns:
        List of image file paths sorted by filename
    """
    if not os.path.exists(image_dir):
        print(f"Warning: Directory {image_dir} does not exist")
        return []
    
    image_files = glob.glob(os.path.join(image_dir, pattern))
    
    if not image_files:
        print(f"No images found in {image_dir} with pattern {pattern}")
        return []
    
    # Sort by filename to ensure consistent order
    image_files.sort()
    
    print(f"Found {len(image_files)} images in {image_dir}")
    return image_files

def find_matching_image_pairs(camera0_dir, camera1_dir, pattern="*.png"):
    """
    Find matching image pairs between two camera directories.
    
    Args:
        camera0_dir: Directory containing Camera_0 images
        camera1_dir: Directory containing Camera_1 images
        pattern: File pattern to match (default: "*.png")
        
    Returns:
        List of tuples (camera0_path, camera1_path, image_name)
    """
    camera0_files = find_image_files(camera0_dir, pattern)
    camera1_files = find_image_files(camera1_dir, pattern)
    
    # Extract image names without path and extension
    camera0_names = {os.path.splitext(os.path.basename(img))[0]: img for img in camera0_files}
    camera1_names = {os.path.splitext(os.path.basename(img))[0]: img for img in camera1_files}
    
    # Find common image names
    common_names = set(camera0_names.keys()) & set(camera1_names.keys())
    
    if not common_names:
        print("Warning: No matching image pairs found!")
        return []
    
    # Sort by image name for consistent processing order
    common_names = sorted(common_names)
    
    image_pairs = []
    for name in common_names:
        image_pairs.append((camera0_names[name], camera1_names[name], name))
    
    print(f"Found {len(image_pairs)} matching image pairs")
    return image_pairs

def save_charuco_points_yaml(all_results, results_dir, camera_id):
    """Save charuco points for a camera to YAML file."""
    points_file = os.path.join(results_dir, f"camera_{camera_id}_charuco_points.yml")
    
    # Filter successful detections
    successful_results = [r for r in all_results if r['charuco_corners'] is not None]
    
    if not successful_results:
        print(f"No successful detections for camera {camera_id}")
        return
    
    # Write OpenCV YAML format
    with open(points_file, 'w') as f:
        f.write("%YAML:1.0\n---\n")
        f.write(f"camera_{camera_id}:\n")
        
        for result in successful_results:
            image_name = result['image_name']
            charuco_corners = result['charuco_corners']
            
            # Prepare points matrix (2xN format)
            points = charuco_corners.reshape(-1, 2)
            points_matrix = np.vstack((points[:, 0], points[:, 1]))
            
            f.write(f"  {image_name}:\n")
            f.write(f"    points: !!opencv-matrix\n")
            f.write(f"      rows: {points_matrix.shape[0]}\n")
            f.write(f"      cols: {points_matrix.shape[1]}\n")
            f.write(f"      dt: f\n")
            f.write(f"      data: {points_matrix.flatten().tolist()}\n")
    
    print(f"Charuco points saved: {points_file}")

def save_charuco_poses_yaml(all_results, results_dir, camera_id):
    """Save charuco poses for a camera to YAML file."""
    poses_file = os.path.join(results_dir, f"camera_{camera_id}_charuco_poses.yml")
    
    # Filter successful pose estimations
    successful_poses = [r for r in all_results if r['pose_valid']]
    
    if not successful_poses:
        print(f"No successful pose estimations for camera {camera_id}")
        return
    
    # Write OpenCV YAML format
    with open(poses_file, 'w') as f:
        f.write("%YAML:1.0\n---\n")
        f.write(f"camera_{camera_id}:\n")
        
        for result in successful_poses:
            image_name = result['image_name']
            rvec = result['rvec']
            tvec = result['tvec']
            
            f.write(f"  {image_name}:\n")
            f.write(f"    rotation_vector: !!opencv-matrix\n")
            f.write(f"      rows: 3\n")
            f.write(f"      cols: 1\n")
            f.write(f"      dt: f\n")
            f.write(f"      data: {rvec.flatten().tolist()}\n")
            f.write(f"    translation_vector: !!opencv-matrix\n")
            f.write(f"      rows: 3\n")
            f.write(f"      cols: 1\n")
            f.write(f"      dt: f\n")
            f.write(f"      data: {tvec.flatten().tolist()}\n")
    
    print(f"Charuco poses saved: {poses_file}")

def save_detection_summary_yaml(all_results, results_dir, camera_id):
    """Save detection summary for a camera to YAML file."""
    # Bu fonksiyon artık kullanılmayacak
    pass

def detect_charuco_board_single(image_path, calibration_file, camera_id, detection_dir):
    """
    Detect ChArUco board in a single image.
    
    Args:
        image_path: Path to the input image
        calibration_file: Path to the calibration file
        camera_id: Camera ID (0 or 1)
        detection_dir: Directory to save detection visualization
        
    Returns:
        Dictionary containing detection results
    """
    # Board parameters (Updated based on MC-Calib dataset)
    number_x_square = 7
    number_y_square = 5
    # Test different square sizes to fix pose estimation
    square_size = 0.02      # 2 cm - more reasonable size
    length_marker = 0.015   # 1.5 cm - proportional to square size
    
    # Load camera parameters
    camera_matrix, dist_coeffs = load_camera_params(calibration_file, camera_id)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Create ChArUco board
    board = cv2.aruco.CharucoBoard((number_x_square, number_y_square), square_size, length_marker, aruco_dict)
    
    # Initialize the detector parameters
    parameters = cv2.aruco.DetectorParameters()
    
    # Create detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Prepare result structure
    result = {
        'image_path': image_path,
        'image_name': os.path.splitext(os.path.basename(image_path))[0],
        'camera_id': camera_id,
        'timestamp': datetime.now().isoformat(),
        'aruco_corners': corners,
        'aruco_ids': ids,
        'charuco_corners': None,
        'charuco_ids': None,
        'num_charuco_corners': 0,
        'pose_valid': False,
        'rvec': None,
        'tvec': None
    }
    
    output_image = image.copy()
    
    if ids is not None:
        # Draw detected markers WITHOUT IDs (only marker boundaries)
        cv2.aruco.drawDetectedMarkers(output_image, corners)  # No ids parameter
        
        print(f"Camera {camera_id} - {result['image_name']}: Detected {len(ids)} ArUco markers")
        
        # Refine detection and get ChArUco corners and IDs
        response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        
        # If ChArUco corners detected
        if response > 0:
            result['charuco_corners'] = charuco_corners
            result['charuco_ids'] = charuco_ids
            result['num_charuco_corners'] = response
            
            print(f"Camera {camera_id} - {result['image_name']}: Detected {response} ChArUco corners")
            
            # Draw ChArUco corners WITH IDs (only checkerboard corner IDs)
            cv2.aruco.drawDetectedCornersCharuco(output_image, charuco_corners, charuco_ids)
            
            # Estimate board pose
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
            )
            
            if valid:
                result['pose_valid'] = True
                result['rvec'] = rvec
                result['tvec'] = tvec
                
                print(f"Camera {camera_id} - {result['image_name']}: Pose estimation successful")
                
                # Draw coordinate system with shorter axes
                cv2.drawFrameAxes(output_image, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
            else:
                print(f"Camera {camera_id} - {result['image_name']}: Pose estimation failed")
        else:
            print(f"Camera {camera_id} - {result['image_name']}: No ChArUco corners detected")
    else:
        print(f"Camera {camera_id} - {result['image_name']}: No ArUco markers detected")
    
    # Save detection visualization
    camera_detection_dir = os.path.join(detection_dir, f"Camera_{camera_id}")
    output_filename = f"{result['image_name']}_detection.jpg"
    output_path = os.path.join(camera_detection_dir, output_filename)
    cv2.imwrite(output_path, output_image)
    
    return result

def batch_process_stereo_images(camera0_dir, camera1_dir, calibration_file, output_dir="data/output", pattern="*.png"):
    """
    Process matching image pairs from both cameras using batch processing.
    
    Args:
        camera0_dir: Directory containing Camera_0 images
        camera1_dir: Directory containing Camera_1 images
        calibration_file: Path to the calibration file
        output_dir: Base output directory
        pattern: File pattern to match (default: "*.png")
    """
    # Find matching image pairs
    image_pairs = find_matching_image_pairs(camera0_dir, camera1_dir, pattern)
    
    if not image_pairs:
        print("No matching image pairs found for batch processing")
        return
    
    # Create session directory structure
    session_dir, detection_dir, results_dir, session_id = create_session_directory(output_dir)
    
    # Initialize results storage
    camera0_results = []
    camera1_results = []
    
    print(f"\nStarting batch processing of {len(image_pairs)} image pairs...")
    print("=" * 80)
    
    # Process each image pair
    for i, (camera0_path, camera1_path, image_name) in enumerate(image_pairs):
        print(f"\nProcessing pair {i+1}/{len(image_pairs)}: {image_name}")
        print("-" * 50)
        
        # Process Camera_0
        result0 = detect_charuco_board_single(camera0_path, calibration_file, 0, detection_dir)
        if result0:
            camera0_results.append(result0)
        
        # Process Camera_1
        result1 = detect_charuco_board_single(camera1_path, calibration_file, 1, detection_dir)
        if result1:
            camera1_results.append(result1)
        
        print(f"Pair {i+1} completed")
    
    # Save results for both cameras
    print(f"\nSaving results...")
    print("-" * 30)
    
    # Save Camera_0 results
    save_charuco_points_yaml(camera0_results, results_dir, 0)
    save_charuco_poses_yaml(camera0_results, results_dir, 0)
    
    # Save Camera_1 results
    save_charuco_points_yaml(camera1_results, results_dir, 1)
    save_charuco_poses_yaml(camera1_results, results_dir, 1)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Session ID: {session_id}")
    print(f"Total image pairs processed: {len(image_pairs)}")
    
    camera0_successful_detections = sum(1 for r in camera0_results if r['num_charuco_corners'] > 0)
    camera1_successful_detections = sum(1 for r in camera1_results if r['num_charuco_corners'] > 0)
    camera0_successful_poses = sum(1 for r in camera0_results if r['pose_valid'])
    camera1_successful_poses = sum(1 for r in camera1_results if r['pose_valid'])
    
    print(f"Camera 0 successful detections: {camera0_successful_detections}/{len(image_pairs)}")
    print(f"Camera 1 successful detections: {camera1_successful_detections}/{len(image_pairs)}")
    print(f"Camera 0 successful poses: {camera0_successful_poses}/{len(image_pairs)}")
    print(f"Camera 1 successful poses: {camera1_successful_poses}/{len(image_pairs)}")
    print(f"Results saved in: {session_dir}")
    print("=" * 80)
    
    return session_id

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect CharUco markers in stereo camera images')
    parser.add_argument('--calibration_file', type=str, 
                        default="data/input/calibrated_cameras_data.yml",
                        help='Path to calibration file')
    parser.add_argument('--camera0_dir', type=str, default="data/input/Camera_0",
                        help='Directory containing Camera_0 images')
    parser.add_argument('--camera1_dir', type=str, default="data/input/Camera_1",
                        help='Directory containing Camera_1 images')
    parser.add_argument('--output_dir', type=str, default="data/output",
                        help='Directory to save output files')
    parser.add_argument('--pattern', type=str, default="*.png",
                        help='File pattern for batch processing (default: *.png)')
    parser.add_argument('--batch', action='store_true', default=True,
                        help='Enable batch processing mode (default: True)')
    
    args = parser.parse_args()
    
    # Check if calibration file exists
    if not os.path.exists(args.calibration_file):
        print(f"Error: Calibration file {args.calibration_file} does not exist")
        exit(1)
    
    # Check if camera directories exist
    if not os.path.exists(args.camera0_dir):
        print(f"Error: Camera_0 directory {args.camera0_dir} does not exist")
        exit(1)
    
    if not os.path.exists(args.camera1_dir):
        print(f"Error: Camera_1 directory {args.camera1_dir} does not exist")
        exit(1)
    
    # Run batch processing
    session_id = batch_process_stereo_images(
        args.camera0_dir,
        args.camera1_dir, 
        args.calibration_file, 
        args.output_dir, 
        args.pattern
    )