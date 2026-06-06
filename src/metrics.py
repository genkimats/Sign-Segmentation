import numpy as np
from sklearn.metrics import f1_score

def extract_segments(bio_sequence):
    """
    Decodes a 1D array of BIO tags (0: Out, 1: In, 2: Begin) 
    into a list of segment tuples: [(start_frame, end_frame), ...]
    """
    segments = []
    start = -1
    
    for i, tag in enumerate(bio_sequence):
        if tag == 2:  # B-tag (Begin)
            if start != -1:
                # Close the previous segment if a new one begins immediately
                segments.append((start, i - 1))
            start = i
        elif tag == 0:  # O-tag (Out)
            if start != -1:
                # Close the current segment
                segments.append((start, i - 1))
                start = -1
                
    # If the sequence ends while a segment is still open
    if start != -1:
        segments.append((start, len(bio_sequence) - 1))
        
    return segments

def calculate_1d_iou(seg1, seg2):
    """Calculates the Intersection over Union (IoU) of two 1D temporal segments."""
    start1, end1 = seg1
    start2, end2 = seg2
    
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    if intersection_start > intersection_end:
        return 0.0 # No overlap
        
    intersection = intersection_end - intersection_start + 1
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
    
    return intersection / union

def calculate_segment_metrics(pred_sequence, gt_sequence, iou_threshold=0.5):
    """
    Calculates the Percentage of Segments (Often called F1@k in literature).
    A predicted segment is a True Positive if it overlaps a ground truth segment by > iou_threshold.
    """
    pred_segments = extract_segments(pred_sequence)
    gt_segments = extract_segments(gt_sequence)
    
    if len(gt_segments) == 0 and len(pred_segments) == 0:
        return 1.0 # Perfect agreement on empty sequence
    if len(gt_segments) == 0 or len(pred_segments) == 0:
        return 0.0
        
    true_positives = 0
    
    # For every predicted segment, check if it matches a ground truth segment
    for p_seg in pred_segments:
        best_iou = 0
        for g_seg in gt_segments:
            iou = calculate_1d_iou(p_seg, g_seg)
            if iou > best_iou:
                best_iou = iou
                
        if best_iou >= iou_threshold:
            true_positives += 1
            
    # Calculate Precision and Recall for segments
    precision = true_positives / len(pred_segments) if len(pred_segments) > 0 else 0
    recall = true_positives / len(gt_segments) if len(gt_segments) > 0 else 0
    
    # Calculate Segment F1 (Percentage of Segments)
    if precision + recall == 0:
        return 0.0
        
    segment_f1 = 2 * (precision * recall) / (precision + recall)
    return segment_f1

def evaluate_batch(predictions, targets):
    """
    Takes batched logits and targets, and returns the three core metrics.
    predictions: (Batch, SequenceLength) -> Argmaxed predictions
    targets: (Batch, SequenceLength) -> Ground truth
    """
    # 1. Frame-Level F1 (Macro average across all 3 classes: B, I, O)
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    frame_f1 = f1_score(target_flat, pred_flat, average='macro', zero_division=0)
    
    # 2. Segment Metrics (IoU and % of Segments)
    batch_size = predictions.shape[0]
    total_iou = 0.0
    total_seg_f1 = 0.0
    
    for i in range(batch_size):
        pred_seq = predictions[i].tolist()
        gt_seq = targets[i].tolist()
        
        # Segment F1 at standard 0.5 overlap threshold
        total_seg_f1 += calculate_segment_metrics(pred_seq, gt_seq, iou_threshold=0.5)
        
        # Calculate Average IoU for this sequence
        p_segs = extract_segments(pred_seq)
        g_segs = extract_segments(gt_seq)
        
        seq_iou = 0.0
        if len(p_segs) > 0 and len(g_segs) > 0:
            for p_seg in p_segs:
                best_iou = max([calculate_1d_iou(p_seg, g_seg) for g_seg in g_segs], default=0.0)
                seq_iou += best_iou
            seq_iou /= len(p_segs) # Average IoU of predicted segments
        elif len(p_segs) == 0 and len(g_segs) == 0:
            seq_iou = 1.0 # Both correctly predicted no segments
            
        total_iou += seq_iou
        
    return {
        'Frame_F1': frame_f1,
        'Mean_IoU': total_iou / batch_size,
        'Segment_F1_05': total_seg_f1 / batch_size
    }