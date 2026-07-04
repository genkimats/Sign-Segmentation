import json
import os

QUEUE_FILE = "train_queue.json"

def add_to_queue(job_config):
    """Appends a new hyperparameter configuration to the queue."""
    queue = []
    
    # Load existing queue if it exists
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as f:
            try:
                queue = json.load(f)
            except json.JSONDecodeError:
                queue = []
                
    queue.append(job_config)
    
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
        
    print(f"✅ Added job to queue. Total jobs in queue: {len(queue)}")
    print(f"   Description: {job_config.get('description', 'No description')}")

if __name__ == "__main__":
    # Define your specific hyperparameter combination here
    new_job = {
        "batch_size": 16,
        "epochs": 50,
        "learning_rate": 0.0001,
        "window_size": 128,
        "overlap": 50,
        "num_vertices": 65,
        "tolerance_window": 5,
        "loss_function": "weighted_ce",
        "class_weights": [0.1, 0.3, 1.0],
        "use_full_length": False,
        "base_features": ["x-cord", "y-cord", "z-cord"],
        "kinematic_features": ["velocity", "acceleration", "jerk", "velocity-mag", "angular-vel"],
        "in_channels": 14,
        "decoder_strategy": "threshold",
        "decoder_threshold": 0.5,
        "d_model": 256,
        "n_layers": 4,
        "focal_loss_gamma": 2.0,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "basename": "stgcn_mamba",
        "description": "Full 14-Channel SOTA Validation Run"
    }
    
    # You can also copy/paste and modify `new_job` to add multiple 
    # jobs in one go by calling `add_to_queue` multiple times!
    add_to_queue(new_job)