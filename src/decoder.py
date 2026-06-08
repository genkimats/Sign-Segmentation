import torch
import torch.nn.functional as F

def decode_predictions(logits, strategy="argmax", threshold=0.60):
    """
    Decodes the raw logits from the model into final class predictions.
    Assuming Classes: 0 = Outside (O), 1 = Inside (I), 2 = Begin (B)
    """
    # Convert logits to probabilities: Shape (Batch, Classes, Time)
    probs = F.softmax(logits, dim=1)
    
    # Baseline: Simple Argmax
    raw_preds = torch.argmax(probs, dim=1)
    
    if strategy == "argmax":
        return raw_preds
        
    elif strategy == "threshold":
        # Strategy from "Hands-On" (Low et al., 2025)
        # Rule: A boundary (Begin) is only accepted if its probability strictly 
        # exceeds a high confidence threshold. Otherwise, we ignore it to prevent false positives.
        
        b_probs = probs[:, 2, :] # Probabilities of the 'Begin' class
        mask_confident_b = b_probs >= threshold
        
        # If it's NOT a confident Begin, fall back to comparing just Outside vs Inside
        fallback_preds = torch.argmax(probs[:, :2, :], dim=1)
        
        # Apply the threshold mask
        preds = torch.where((raw_preds == 2) & ~mask_confident_b, fallback_preds, raw_preds)
        return preds
        
    elif strategy == "linguistic":
        # Strategy from "Linguistically Motivated..." 
        # Rule 1: You cannot physically transition from Outside (0) directly to Inside (1). 
        #         You MUST pass through a Begin (2) state first.
        # Rule 2: Hysteresis. Do not change the current state unless the new state's 
        #         probability breaks the strict threshold (eliminates flickering).
        
        B_size, C, T = logits.shape
        final_preds = torch.zeros_like(raw_preds)
        
        for b in range(B_size):
            curr_state = 0 # All sequences start in the 'Outside' state
            
            for t in range(T):
                target_state = raw_preds[b, t].item()
                target_prob = probs[b, target_state, t].item()
                p_B = probs[b, 2, t].item()
                
                # Rule 1: Block illegal O -> I transitions
                if curr_state == 0 and target_state == 1:
                    # If 'Begin' is a close second choice, assume it's a Begin. Else, stay Outside.
                    if p_B > (1.0 - threshold): 
                        next_state = 2
                    else:
                        next_state = 0
                        
                # Rule 2: State Hysteresis (Sticky States)
                else:
                    if target_state != curr_state and target_prob < threshold:
                        # Ignore the sudden low-confidence flicker; stay where we are
                        next_state = curr_state 
                    else:
                        # Confident enough to transition
                        next_state = target_state
                        
                final_preds[b, t] = next_state
                curr_state = next_state
                
        return final_preds
        
    else:
        raise ValueError(f"Unknown decoder strategy: {strategy}")