import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import torch
import logging

# Configure logging
# logging.basicConfig(level=logging.INFO) # Commented out to avoid duplicate handlers
logger = logging.getLogger(__name__)

# Define keypoint indices for upper body tracking
UPPER_BODY_JOINTS = [
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist'
]

KEYPOINT_MAP = {
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10
}

SKELETON_CONNECTIONS = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_elbow', 'right_wrist')
]

class PoseMemory:
    """
    Enhanced memory for pose tracking with improved stability to reduce flinching.
    """
    def __init__(self, memory_frames: int = 15, min_keypoint_conf: float = 0.3):
        self.person_poses: Dict[int, List[Dict]] = {}
        self.memory_frames = memory_frames
        self.min_keypoint_conf = min_keypoint_conf
        # Add temporal smoothing parameters
        self.smoothing_factor = 1  # Lower = more smoothing (0.0-1.0)
        self.velocity_dampening = 1  # Dampen sudden movements (0.0-1.0)
        # Track previous velocities for each joint
        self.person_velocities: Dict[int, Dict[str, np.ndarray]] = {}

    def update_pose(self, person_id: int, pose: Dict[str, np.ndarray]) -> None:
        """
        Update pose history for a person with temporal smoothing.
        
        Args:
            person_id: Person identifier
            pose: Dictionary mapping joint names to their positions
        """
        if person_id not in self.person_poses:
            self.person_poses[person_id] = []
            self.person_velocities[person_id] = {}
            
        # Store the original pose first (for reference)
        original_pose = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in pose.items()}
        
        # Apply temporal smoothing if we have previous poses
        if len(self.person_poses[person_id]) > 0:
            smoothed_pose = {}
            
            # Get the most recent pose
            last_pose = self.person_poses[person_id][0]
            
            for joint_name, position in pose.items():
                if joint_name in last_pose:
                    last_pos = last_pose[joint_name]
                    
                    # Calculate current velocity
                    current_velocity = position - last_pos
                    
                    # Get previous velocity (if available)
                    if joint_name in self.person_velocities[person_id]:
                        prev_velocity = self.person_velocities[person_id][joint_name]
                        
                        # Apply velocity dampening to reduce sudden movements
                        damped_velocity = prev_velocity * (1 - self.velocity_dampening) + current_velocity * self.velocity_dampening
                        
                        # Store the damped velocity for next frame
                        self.person_velocities[person_id][joint_name] = damped_velocity
                        
                        # Apply the damped velocity to get a smoothed position
                        smoothed_position = last_pos + damped_velocity
                    else:
                        # No previous velocity, initialize it
                        self.person_velocities[person_id][joint_name] = current_velocity
                        smoothed_position = position.copy()
                    
                    # Apply temporal smoothing between original position and smoothed position
                    alpha = self.smoothing_factor
                    final_position = smoothed_position * (1 - alpha) + position * alpha
                    
                    smoothed_pose[joint_name] = final_position
                else:
                    # No previous position for this joint, use as is
                    smoothed_pose[joint_name] = position.copy()
                    
            # Use the smoothed pose instead of the original
            self.person_poses[person_id].insert(0, smoothed_pose)
        else:
            # First pose for this person, no smoothing
            self.person_poses[person_id].insert(0, original_pose)
            
        # Keep only the specified number of frames
        if len(self.person_poses[person_id]) > self.memory_frames:
            self.person_poses[person_id].pop()

    def get_last_valid_position(self, person_id: int, joint_name: str) -> Optional[np.ndarray]:
        """
        Get last valid position for a joint with improved stability.
        """
        if person_id not in self.person_poses:
            return None
            
        # Try to find the joint in recent poses
        for pose in self.person_poses[person_id]:
            if joint_name in pose:
                return pose[joint_name].copy()
                
        return None


# Additional fix for detect_upper_body_pose to reduce "flinching"
def detect_upper_body_pose(
    frame: np.ndarray,
    results: torch.Tensor,
    pose_memory: Optional[PoseMemory] = None,
    config: Optional[object] = None,
    original_frame_size=None
) -> List[Tuple[int, Dict[str, np.ndarray]]]:
    """
    Detect upper body poses with improved stability to reduce flinching.
    """
    poses = []

    if results is None or not hasattr(results, 'keypoints') or results.keypoints is None:
        return poses

    frame_shape = frame.shape[:2]
    min_keypoint_conf = 0.15 if config is None else getattr(config, 'yolo_min_keypoint_conf', 0.15)
    
    # Calculate scaling factors if original size is provided
    scale_x, scale_y = 1.0, 1.0
    if original_frame_size is not None:
        orig_width, orig_height = original_frame_size
        scale_x = orig_width / frame.shape[1]
        scale_y = orig_height / frame.shape[0]

    try:
        # Validate YOLO results
        if results is None or not hasattr(results, 'keypoints') or results.keypoints is None:
            return poses

        keypoints_data = results.keypoints.data
        if len(keypoints_data) == 0:
            return poses

        # Get tracking IDs if available
        if hasattr(results, 'boxes') and hasattr(results.boxes, 'id') and results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)
        else:
            # Fallback ID generation
            ids = np.arange(len(results.keypoints.data))

        # First, create a mapping of person IDs to their keypoints
        person_keypoints = {}
        for person_idx in range(len(keypoints_data)):
            try:
                if person_idx >= len(keypoints_data) or person_idx >= len(ids):
                    continue

                keypoints = keypoints_data[person_idx].cpu().numpy()
                person_id = int(ids[person_idx])
                
                # Store all keypoints for this person
                person_keypoints[person_id] = keypoints
            except Exception:
                continue

        # Now process each person with improved temporal stability
        for person_id, keypoints in person_keypoints.items():
            try:
                pose = {}
                
                # Process each joint
                for joint_name, idx in KEYPOINT_MAP.items():
                    if idx < len(keypoints):
                        position = keypoints[idx][:2].copy()
                        confidence = keypoints[idx][2]

                        # Basic validation
                        if not isinstance(position, np.ndarray) or len(position) != 2:
                            continue

                        if np.any(np.isnan(position)) or np.any(np.isinf(position)):
                            continue
                            
                        # Apply scaling to match original frame
                        if original_frame_size is not None:
                            position[0] *= scale_x
                            position[1] *= scale_y

                        # Add to pose if confidence is good
                        if confidence >= min_keypoint_conf:
                            pose[joint_name] = position
                        elif pose_memory:
                            # Try to get position from memory
                            last_pos = pose_memory.get_last_valid_position(person_id, joint_name)
                            if last_pos is not None:
                                # Apply scaling to memory positions
                                if original_frame_size is not None:
                                    scaled_pos = last_pos.copy()
                                    scaled_pos[0] *= scale_x
                                    scaled_pos[1] *= scale_y
                                    pose[joint_name] = scaled_pos
                                else:
                                    pose[joint_name] = last_pos

                # Only add pose if we have at least one valid joint
                if pose:
                    poses.append((person_id, pose))
                    
                    # For memory, store unscaled positions
                    if pose_memory:
                        unscaled_pose = {}
                        for joint_name, position in pose.items():
                            if original_frame_size is not None:
                                unscaled_pos = position.copy()
                                unscaled_pos[0] /= scale_x
                                unscaled_pos[1] /= scale_y
                                unscaled_pose[joint_name] = unscaled_pos
                            else:
                                unscaled_pose[joint_name] = position.copy()
                                
                        # Update pose memory with unscaled positions
                        pose_memory.update_pose(person_id, unscaled_pose)

            except Exception as e:
                logger.warning(f"Error processing person {person_id}: {e}")
                continue

        return poses

    except Exception as e:
        logger.error(f"Error in detect_upper_body_pose: {str(e)}")
        return poses

def draw_poses(
    frame: np.ndarray,
    poses: List[Tuple[int, Dict[str, np.ndarray]]],
    pose_memory: Optional[PoseMemory] = None,
    original_frame_size=None  # Add parameter to track scaling
) -> np.ndarray:
    """
    Draw detected poses on frame with proper scaling.
    
    Args:
        frame: The original frame to draw on
        poses: List of detected poses (scaled to match frame)
        pose_memory: Optional pose memory for temporal consistency
        original_frame_size: Optional original frame size used for detection
        
    Returns:
        Frame with poses drawn
    """
    # Colors for different people - improved visibility
    colors = [
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 255),  # Purple
        (255, 128, 0)   # Orange
    ]

    # Create a copy of the frame
    output_frame = frame.copy()
    
    # Calculate scaling if needed
    scale_x, scale_y = 1.0, 1.0
    if original_frame_size is not None:
        # If we're drawing on the original frame but poses were detected on a resized frame
        orig_width, orig_height = original_frame_size
        curr_height, curr_width = frame.shape[:2]
        
        if curr_width != orig_width or curr_height != orig_height:
            scale_x = curr_width / orig_width
            scale_y = curr_height / orig_height

    # Add a title that shows number of people detected
    cv2.putText(output_frame, f"People detected: {len(poses)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Skip overlay - it makes pose detection harder to see
    # Instead, just draw poses on the original frame

    for person_id, pose in poses:
        color = colors[person_id % len(colors)]

        # Draw joints with better visibility
        for joint_name, position in pose.items():
            try:
                # Apply any additional scaling if needed
                scaled_position = position.copy()
                if scale_x != 1.0 or scale_y != 1.0:
                    scaled_position[0] *= scale_x
                    scaled_position[1] *= scale_y
                
                # Convert to integer coordinates and draw circles
                pt = tuple(scaled_position.astype(int))
                
                # Improved visibility of joints
                cv2.circle(output_frame, pt, 8, color, -1)  # Filled circle
                cv2.circle(output_frame, pt, 10, (255, 255, 255), 1)  # White border

                # Add joint label for main joints with better visibility
                if joint_name in ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist']:
                    label = joint_name.split('_')[0][0] + joint_name.split('_')[1][0]  # e.g., "ls" for left_shoulder
                    cv2.putText(output_frame, label,
                               (pt[0] + 5, pt[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black background/outline
                    cv2.putText(output_frame, label,
                               (pt[0] + 5, pt[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Colored text
            except Exception as e:
                logger.debug(f"Error drawing joint {joint_name}: {e}")
                continue

        # Draw connections with improved visibility
        for joint1_name, joint2_name in SKELETON_CONNECTIONS:
            if joint1_name in pose and joint2_name in pose:
                try:
                    pos1 = pose[joint1_name].copy()
                    pos2 = pose[joint2_name].copy()
                    
                    # Apply any additional scaling
                    if scale_x != 1.0 or scale_y != 1.0:
                        pos1[0] *= scale_x
                        pos1[1] *= scale_y
                        pos2[0] *= scale_x
                        pos2[1] *= scale_y
                        
                    pt1 = tuple(pos1.astype(int))
                    pt2 = tuple(pos2.astype(int))
                    
                    # Draw lines with white borders for better visibility
                    cv2.line(output_frame, pt1, pt2, (0, 0, 0), 5)  # Black background
                    cv2.line(output_frame, pt1, pt2, (255, 255, 255), 3)  # White middle
                    cv2.line(output_frame, pt1, pt2, color, 2)  # Colored line
                except Exception as e:
                    logger.debug(f"Error drawing connection: {e}")
                    continue

        # Add ID label with improved visibility
        try:
            center = None
            if 'right_shoulder' in pose:
                center_pos = pose['right_shoulder'].copy()
            elif 'left_shoulder' in pose:
                center_pos = pose['left_shoulder'].copy()
            else:
                center_pos = next(iter(pose.values())).copy()
                
            # Apply scaling
            if scale_x != 1.0 or scale_y != 1.0:
                center_pos[0] *= scale_x
                center_pos[1] *= scale_y
                
            center = tuple(center_pos.astype(int))

            # Draw text with background for better visibility
            text = f"ID: {person_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(output_frame,
                         (center[0] - 5, center[1] - text_size[1] - 15),
                         (center[0] + text_size[0] + 5, center[1] - 5),
                         (0, 0, 0), -1)
            cv2.putText(output_frame, text,
                       (center[0], center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception as e:
            logger.debug(f"Error adding ID label: {e}")
            pass

    return output_frame

def calculate_displacement(current_pose, prev_pose, max_displacement=3.0):
    # Input validation
    if not isinstance(current_pose, dict) or not isinstance(prev_pose, dict):
        return None
    if not current_pose or not prev_pose:
        return None
    
    common_joints = set(current_pose.keys()).intersection(set(prev_pose.keys()))
    if not common_joints:
        return None

    joint_names = ['left_shoulder', 'right_shoulder', 'left_elbow',
                   'right_elbow', 'left_wrist', 'right_wrist']
    displacements = [np.nan] * len(joint_names)
    
    try:
        # Calculate scale factor based on shoulder width
        scale_factor = 100.0  # Default scale factor
        if 'left_shoulder' in common_joints and 'right_shoulder' in common_joints:
            try:
                shoulder_width = np.linalg.norm(
                    np.array(current_pose['left_shoulder']) - 
                    np.array(current_pose['right_shoulder'])
                )
                if shoulder_width > 1e-6:
                    scale_factor = shoulder_width
            except Exception as e:
                logger.debug(f"Error calculating shoulder width: {e}")
        
        valid_joints = 0
        displacement_stats = [] # Keep track of valid displacements for outlier check

        for i, joint in enumerate(joint_names):
            if joint not in common_joints:
                # Joint missing in one or both frames, leave as np.nan
                continue

            try:
                curr_pos = np.array(current_pose[joint])
                prev_pos = np.array(prev_pose[joint])

                # Validate positions
                if not (np.all(np.isfinite(curr_pos)) and np.all(np.isfinite(prev_pos))):
                    # Position data itself is invalid, leave as np.nan
                    continue

                # Calculate raw displacement
                raw_displacement = np.linalg.norm(curr_pos - prev_pos)

                # Normalize by scale factor
                normalized_displacement = raw_displacement / scale_factor

                # Clamp displacement
                if normalized_displacement > max_displacement:
                    # Optionally log clamping: logger.debug(...)
                    normalized_displacement = max_displacement

                displacements[i] = float(normalized_displacement)
                displacement_stats.append(normalized_displacement) # Add valid displacement
                valid_joints += 1

            except Exception as e:
                logger.debug(f"Error processing joint {joint}: {e}")
                # Error during processing, leave as np.nan
                continue

        if valid_joints == 0:
            # No valid joints could be processed
            return None
        
        # Final validation and outlier removal
        if displacement_stats:
            mean_disp = np.mean(displacement_stats)
            std_disp = np.std(displacement_stats)
            
            # Clip extremely large values (more than 3 standard deviations)
            if std_disp > 0:
                for i, val in enumerate(displacements):
                    if val > mean_disp + 3 * std_disp:
                        displacements[i] = mean_disp + 3 * std_disp
                        logger.debug(f"Clipping outlier displacement at index {i}")
        
        # Ensure all values are finite
        for i, val in enumerate(displacements):
            if not np.isfinite(val):
                displacements[i] = 0.0
        
        return displacements
        
    except Exception as e:
        logger.error(f"Error in calculate_displacement: {str(e)}")
        return [np.nan] * len(joint_names)

def normalize_joints(pose: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    """
    Normalize joint positions relative to a reference point and scale.
    """
    # Input validation
    if not isinstance(pose, dict) or not pose:
        return None

    try:
        # Step 1: Validate and clean individual joint positions
        valid_joints = {}
        for joint_name, position in pose.items():
            try:
                # Convert to numpy array if needed
                if not isinstance(position, np.ndarray):
                    position = np.array(position, dtype=np.float32)

                # Must be 2D coordinates
                if position.shape != (2,):
                    continue

                # Must contain finite values
                if not np.all(np.isfinite(position)):
                    continue

                valid_joints[joint_name] = position
            except Exception:
                continue

        # Need at least one valid joint
        if not valid_joints:
            return None

        # Step 2: Determine reference point and scale

        # Strategy 1: Use shoulder midpoint and width (ideal case)
        if 'left_shoulder' in valid_joints and 'right_shoulder' in valid_joints:
            try:
                reference_point = (valid_joints['left_shoulder'] + valid_joints['right_shoulder']) / 2
                scale = np.linalg.norm(valid_joints['left_shoulder'] - valid_joints['right_shoulder'])
                if scale < 1e-6:
                    scale = 100.0  # Default fallback
            except Exception:
                # If calculation fails, use default strategy
                reference_point = valid_joints['left_shoulder']
                scale = 100.0

        # Strategy 2: Use single shoulder if available
        elif 'left_shoulder' in valid_joints:
            reference_point = valid_joints['left_shoulder']
            scale = 100.0
        elif 'right_shoulder' in valid_joints:
            reference_point = valid_joints['right_shoulder']
            scale = 100.0

        # Strategy 3: Use any available joint as reference
        else:
            joint_name = next(iter(valid_joints.keys()))
            reference_point = valid_joints[joint_name]
            scale = 100.0

        # Step 3: Normalize all valid joints
        normalized_joints = {}
        for joint_name, position in valid_joints.items():
            try:
                # Apply normalization
                normalized_pos = (position - reference_point) / scale

                # Validate result
                if np.all(np.isfinite(normalized_pos)):
                    normalized_joints[joint_name] = normalized_pos
            except Exception:
                continue

        # Step 4: Final validation - need at least one normalized joint
        if normalized_joints:
            return normalized_joints
        else:
            return None

    except Exception as e:
        logger.error(f"Error in normalize_joints: {str(e)}")
        return None