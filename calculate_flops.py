import os
import json
import torch
import argparse
from torch.utils.flop_counter import FlopCounterMode

# Import your custom modules
from src.models import PureMambaBaseline, BiMambaBaseline, STGCN_Mamba, STGCN_BiMamba

# ==============================================================================
# 🎛️ DEFAULT CONFIGURATION
# ==============================================================================
# Change this variable to test different models 
# Options: "pure_mamba", "bi_mamba", "stgcn_mamba", "stgcn_bimamba"
CHOSEN_MODEL = "stgcn_mamba"  

# Default prefix to use if no terminal arguments are provided
DEFAULT_PREFIX = 39 

MODEL_REGISTRY = {
    "pure_mamba": PureMambaBaseline,
    "bi_mamba": BiMambaBaseline,
    "stgcn_mamba": STGCN_Mamba,
    "stgcn_bimamba": STGCN_BiMamba
}
# ==============================================================================

def main():
    # 1. Setup Argument Parsing
    parser = argparse.ArgumentParser(description="Calculate FLOPs for trained models.")
    parser.add_argument(
        "--prefix", 
        type=int, 
        default=DEFAULT_PREFIX, 
        help="The experiment run prefix (e.g., 39 for run -39)."
    )
    args = parser.parse_args()
    
    prefix_formatted = f"{args.prefix:02d}"
    
    # 2. Locate Hyperparameters
    hyperparameter_path = f"experiments/{CHOSEN_MODEL}-{prefix_formatted}/hyperparameters.json"
    
    if not os.path.exists(hyperparameter_path):
        raise FileNotFoundError(f"❌ Error: Could not find hyperparameters at {hyperparameter_path}")

    with open(hyperparameter_path, 'r') as f:
        hp = json.load(f)

    # 3. Extract necessary variables to build the dummy tensor
    # We use Batch Size = 1 because FLOPs are traditionally reported 'Per Sequence' in research papers.
    BATCH_SIZE = 1 
    WINDOW_SIZE = hp.get("window_size")
    NUM_VERTICES = hp.get("num_vertices")
    IN_CHANNELS = hp.get("in_channels")
    D_MODEL = hp.get("d_model")
    N_LAYERS = hp.get("n_layers")

    print("="*60)
    print(f"🧮 FLOP CALCULATOR: {CHOSEN_MODEL.upper()} (Run {prefix_formatted})")
    print("="*60)
    print(f"📄 Loaded Hyperparameters:")
    print(f"   Window Size: {WINDOW_SIZE} | Vertices: {NUM_VERTICES} | Channels: {IN_CHANNELS}")
    
    # 4. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_class = MODEL_REGISTRY[CHOSEN_MODEL]
    
    model = model_class(
        num_vertices=NUM_VERTICES, 
        in_channels=IN_CHANNELS, 
        d_model=D_MODEL, 
        n_layers=N_LAYERS
    ).to(device)
    
    model.eval()

    # 5. Create Dummy Input (Shape: Batch, Channels, Frames, Vertices)
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, WINDOW_SIZE, NUM_VERTICES).to(device)
    print(f"📦 Dummy Input Shape: {dummy_input.shape} (Batch size fixed to 1 for standard metric)\n")

    # 6. Calculate FLOPs natively using PyTorch's Dispatcher
    flop_counter = FlopCounterMode(display=False)
    
    with torch.no_grad():
        with flop_counter:
            _ = model(dummy_input)

    # Calculate Totals
    total_flops = flop_counter.get_total_flops()
    
    # Convert to GigaFLOPs (GFLOPs) for clean reporting
    gflops = total_flops / 1e9

    print("="*60)
    print(f"🚀 RESULTS FOR '{CHOSEN_MODEL}':")
    print(f"   Total FLOPs:  {total_flops:,}")
    print(f"   Total GFLOPs: {gflops:.4f} GFLOPs per {WINDOW_SIZE}-frame sequence")
    print("="*60)

if __name__ == "__main__":
    main()