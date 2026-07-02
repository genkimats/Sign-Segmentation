import torch
import torch.nn as nn
import math
from torch.utils.flop_counter import FlopCounterMode

# ==============================================================================
# 1. "Hands-On" Transformer Proxy Architecture (NAIVE / EXPLICIT MATMUL)
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
        return x + self.pe[:x.size(1)].transpose(0, 1)

class NaiveAttention(nn.Module):
    """
    We MUST use a naive attention implementation. PyTorch's default nn.MultiheadAttention 
    uses C++ fused kernels (SDPA/FlashAttention) which hide the O(N^2) FLOPs from the 
    FlopCounterMode. This explicit version forces the counter to register the quadratic math.
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, D = x.shape
        # Project and reshape into (3, B, nhead, T, d_k)
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.nhead, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # EXPLICIT O(N^2) Matmul: (B, nhead, T, d_k) @ (B, nhead, d_k, T) -> (B, nhead, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # EXPLICIT O(N^2) Matmul: (B, nhead, T, T) @ (B, nhead, T, d_k) -> (B, nhead, T, d_k)
        attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(attn_output)

class NaiveTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = NaiveAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.linear2(self.activation(self.linear1(self.norm2(x))))
        return x

class HandsOnTransformerProxy(nn.Module):
    def __init__(self, num_vertices=65, in_channels=3, d_model=256, n_layers=4, nhead=8, num_classes=3, apply_downsampling=True):
        super().__init__()
        self.apply_downsampling = apply_downsampling
        input_dim = num_vertices * in_channels
        
        self.aux_mlp = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        if self.apply_downsampling:
            self.downsample = nn.Conv1d(d_model, d_model, kernel_size=2, stride=2)
            self.upsample = nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2)
            
        self.mixer_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.pos_encoder = PositionalEncoding(d_model)
        
        # Replace the opaque TransformerEncoder with our fully-exposed Naive layers
        self.transformer_encoder = nn.Sequential(
            *[NaiveTransformerLayer(d_model, nhead) for _ in range(n_layers)]
        )
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, T, V * C)
        
        x = self.aux_mlp(x)
        
        if self.apply_downsampling:
            x = x.permute(0, 2, 1) 
            x = self.downsample(x)
            x = x.permute(0, 2, 1) 
            
        x = self.mixer_mlp(x)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) 
        
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
                
                flops_res = flop_counter.get_total_flops()
                total_flops = sum(flops_res.values()) if isinstance(flops_res, dict) else flops_res
                
                gflops = total_flops / 1e9
                print(f"{w:<11} | {gflops:.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{w:<11} | OUT OF MEMORY (OOM)")
                else:
                    print(f"{w:<11} | FAILED: {str(e)}")
                torch.cuda.empty_cache() 
            except Exception as e:
                print(f"{w:<11} | FAILED: {str(e)}")
                torch.cuda.empty_cache()