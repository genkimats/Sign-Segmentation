import torch
import torch.nn as nn
import math
from torch.utils.flop_counter import FlopCounterMode

# ==============================================================================
# 1. "Hands-On" Transformer Proxy Architecture
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Sequence, Feature)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

class HandsOnTransformerProxy(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, d_model=256, n_layers=4, nhead=8, num_classes=3, apply_downsampling=True):
        super().__init__()
        self.apply_downsampling = apply_downsampling
        input_dim = num_vertices * in_channels
        
        # 1. Auxiliary Module (3-layer MLP as described in paper)
        self.aux_mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 2. Temporal Downsampling (Factor of 2)
        if self.apply_downsampling:
            self.downsample = nn.Conv1d(d_model, d_model, kernel_size=2, stride=2)
            # Needed to return the sequence to full length for frame-level BIO tagging
            self.upsample = nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2)
            
        # 3. Multi-Modal Mixer (Mocked here as the final 3-layer MLP before Transformer)
        self.mixer_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 4. Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (B, C, T, V)
        B, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, T, V * C)
        
        x = self.aux_mlp(x)
        
        if self.apply_downsampling:
            x = x.permute(0, 2, 1) # (B, d_model, T)
            x = self.downsample(x)
            x = x.permute(0, 2, 1) # (B, T/2, d_model)
            
        x = self.mixer_mlp(x)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # The O(N^2) bottleneck
        
        if self.apply_downsampling:
            x = x.permute(0, 2, 1) 
            x = self.upsample(x)
            x = x.permute(0, 2, 1) 
            
        logits = self.classifier(x)
        return logits.permute(0, 2, 1)

# ==============================================================================
# 2. FLOP Calculation Sweep
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We test two versions: The exact paper model (Downsampled), and a Pure Transformer
    models_to_test = {
        "Hands-On Transformer (with T/2 Downsampling)": True,
        "Pure Transformer (No Downsampling)": False
    }
    
    window_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    for model_name, do_downsample in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"🧮 {model_name}")
        print(f"{'='*60}")
        print("Window Size | GFLOPs")
        print("-" * 25)
        
        model = HandsOnTransformerProxy(
            num_vertices=65, in_channels=3, d_model=256, n_layers=4, apply_downsampling=do_downsample
        ).to(device)
        model.eval()

        for w in window_sizes:
            dummy_input = torch.randn(1, 3, w, 65).to(device)
            flop_counter = FlopCounterMode(display=False)
            
            try:
                with torch.no_grad():
                    with flop_counter:
                        _ = model(dummy_input)
                
                # Fixed to handle different PyTorch versions (int vs dict return types)
                flops_res = flop_counter.get_total_flops()
                total_flops = sum(flops_res.values()) if isinstance(flops_res, dict) else flops_res
                
                gflops = total_flops / 1e9
                print(f"{w:<11} | {gflops:.4f}")
                
            except RuntimeError as e:
                # Explicitly catch true CUDA memory issues
                if "out of memory" in str(e).lower():
                    print(f"{w:<11} | OUT OF MEMORY (OOM)")
                else:
                    print(f"{w:<11} | FAILED: {str(e)}")
                torch.cuda.empty_cache() 
            except Exception as e:
                # Print actual trace issues instead of hiding them under an OOM label
                print(f"{w:<11} | FAILED: {str(e)}")
                torch.cuda.empty_cache()