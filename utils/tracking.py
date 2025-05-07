import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_tracking_errors_and_correct(movement_data, window_size=5, correction_threshold=0.7):
    """
    Identify and correct technical tracking errors.
    
    Args:
        movement_data: Dictionary of movement data by person ID
        window_size: Window size for analysis
        correction_threshold: Threshold for considering a value as an error
        
    Returns:
        Corrected movement data
    """
    result_data = {}
    
    for person_id, data in movement_data.items():
        result_data[person_id] = data.copy()
        
        if 'scores' in data and len(data['scores']) > window_size:
            scores = np.array(data['scores'])
            corrected_scores = scores.copy()
            
            # Detect and correct outliers
            for i in range(1, len(scores)-1):
                # Calculate rate of change
                if i > 0 and i < len(scores) - 1:
                    current_score = scores[i]
                    
                    # Avoid division by zero
                    prev_score = max(0.1, scores[i-1])
                    next_score = max(0.1, scores[i+1])
                    
                    # Calculate drop and recovery rates
                    drop_rate = (current_score - prev_score) / prev_score
                    recovery_rate = (next_score - current_score) / current_score if current_score > 0.1 else 0
                    
                    # Correct if sudden drop followed by recovery
                    if drop_rate < -correction_threshold and recovery_rate > correction_threshold:
                        # Interpolate the value
                        corrected_scores[i] = (scores[i-1] + scores[i+1]) / 2
                        logger.info(f"Corrected tracking error for person {person_id} at time {data['times'][i]:.2f}s: {current_score:.2f} -> {corrected_scores[i]:.2f}")
            
            # Apply median smoothing to remove jitter
            kernel_size = 3
            for i in range(kernel_size // 2, len(corrected_scores) - kernel_size // 2):
                window = corrected_scores[i - kernel_size // 2:i + kernel_size // 2 + 1]
                corrected_scores[i] = np.median(window)
            
            # Store both original and corrected scores
            result_data[person_id]['scores'] = corrected_scores.tolist()
            result_data[person_id]['original_scores'] = scores.tolist()
            
            # Log the amount of correction applied
            correction_diff = np.abs(scores - corrected_scores)
            if np.any(correction_diff > 0):
                logger.info(f"Applied corrections to person {person_id}: Average change = {np.mean(correction_diff):.2f}, Max change = {np.max(correction_diff):.2f}")
    
    return result_data


def filter_movement_data(movement_data, min_detections=10, max_persons=5):
    """
    Identify which persons have sufficient data for reliable analysis.
    Does not modify the data - just returns which IDs should be considered.
    
    Args:
        movement_data: Dictionary of movement data by person ID
        min_detections: Minimum number of detections to consider reliable
        max_persons: Maximum number of persons to track
        
    Returns:
        List of person IDs that have sufficient data
    """
    # Skip if no data
    if not movement_data:
        return []
    
    # Count valid detections for each person
    person_counts = {}
    
    for person_id, data in movement_data.items():
        if 'scores' in data and len(data['scores']) >= min_detections:
            # Count the number of valid detections (non-zero scores)
            valid_count = sum(1 for score in data['scores'] if score > 0)
            person_counts[person_id] = valid_count
    
    # Sort persons by detection count and keep top max_persons
    sorted_persons = sorted(person_counts.items(), key=lambda x: x[1], reverse=True)
    reliable_ids = [person_id for person_id, _ in sorted_persons[:max_persons] 
                   if person_counts[person_id] >= min_detections]
    
    if reliable_ids:
        logger.info(f"Identified {len(reliable_ids)} persons with sufficient data for analysis")
        for i, person_id in enumerate(reliable_ids):
            logger.info(f"Person {person_id}: {person_counts[person_id]} valid detections")
    else:
        logger.info("No persons have sufficient detections for reliable analysis")
    
    return reliable_ids


def get_raw_movement_data(movement_data):
    """
    Ensure we're using raw movement data without any smoothing or filtering.
    If original data exists, restores it.
    
    Args:
        movement_data: Dictionary of movement data by person ID
        
    Returns:
        Movement data with raw scores
    """
    raw_data = {}
    
    for person_id, data in movement_data.items():
        raw_data[person_id] = data.copy()
        
        # If original scores were saved, restore them
        if 'original_scores' in data:
            raw_data[person_id]['scores'] = data['original_scores'].copy()
            logger.info(f"Restored original raw data for person {person_id}")
    
    return raw_data