import torch
import torch.nn as nn
import math
from torch.utils.flop_counter import FlopCounterMode

# ==============================================================================
# 1. "Hands-On" True Pipeline Architecture 
# (4096d HaMeR / 150d Angles -> 512d -> 1024d Concat -> 1024d Transformer)
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
    """Explicit Attention to force FLOP counter to register O(N^2) math."""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.nhead, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(attn_output)

class NaiveTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = NaiveAttention(d_model, nhead)
        # Standard Transformer FFN expands by 4x internally
        self.linear1 = nn.Linear(d_model, d_model * 4) 
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.linear2(self.activation(self.linear1(self.norm2(x))))
        return x

class TrueHandsOnPipeline(nn.Module):
    def __init__(self, hamer_dim=4096, angle_dim=150, proj_dim=512, n_layers=4, nhead=8, num_classes=3, apply_downsampling=True):
        super().__init__()
        self.apply_downsampling = apply_downsampling
        
        # 1. Feature-Specific Auxiliary Modules (Up-projects to 512d)
        self.hamer_mlp = nn.Sequential(
            nn.Linear(hamer_dim, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.angle_mlp = nn.Sequential(
            nn.Linear(angle_dim, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # 2. Temporal Downsampling (Factor of 2)
        if self.apply_downsampling:
            self.hamer_downsample = nn.Conv1d(proj_dim, proj_dim, kernel_size=2, stride=2)
            self.angle_downsample = nn.Conv1d(proj_dim, proj_dim, kernel_size=2, stride=2)
            # 1024d Upsampler for final output mapping
            self.upsample = nn.ConvTranspose1d(proj_dim * 2, proj_dim * 2, kernel_size=2, stride=2)
            
        # 3. Multi-Modal Mixer (1024d)
        d_model = proj_dim * 2 # 1024 dimensions
        
        self.mixer_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 4. Transformer Encoder (Processing at the full 1024d concat dimension)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.Sequential(
            *[NaiveTransformerLayer(d_model, nhead) for _ in range(n_layers)]
        )
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, hamer_feat, angle_feat):
        # Processing independent modalities
        h_f = self.hamer_mlp(hamer_feat)
        a_f = self.angle_mlp(angle_feat)
        
        if self.apply_downsampling:
            h_f = h_f.permute(0, 2, 1)
            h_f = self.hamer_downsample(h_f).permute(0, 2, 1)
            
            a_f = a_f.permute(0, 2, 1)
            a_f = self.angle_downsample(a_f).permute(0, 2, 1)
            
        # Concatenate into M in R^1024
        combined_feat = torch.cat([h_f, a_f], dim=-1)
        
        mixed_feat = self.mixer_mlp(combined_feat)
        
        mixed_feat = self.pos_encoder(mixed_feat)
        mixed_feat = self.transformer_encoder(mixed_feat) 
        
        if self.apply_downsampling:
            mixed_feat = mixed_feat.permute(0, 2, 1) 
            mixed_feat = self.upsample(mixed_feat).permute(0, 2, 1) 
            
        logits = self.classifier(mixed_feat)
        return logits.permute(0, 2, 1)

# ==============================================================================
# 2. Fair FLOP Calculation Sweep
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    models_to_test = {
        "True Hands-On Pipeline (T/2 Downsampled)": True,
        "True Hands-On Pipeline (Pure, No Downsampling)": False
    }
    
    window_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    for model_name, do_downsample in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"🧮 {model_name}")
        print(f"{'='*60}")
        print("Window Size | GFLOPs")
        print("-" * 25)
        
        model = TrueHandsOnPipeline(
            hamer_dim=4096, angle_dim=150, proj_dim=512, n_layers=4, apply_downsampling=do_downsample
        ).to(device)
        model.eval()

        for w in window_sizes:
            # Explicitly modeling the frozen feature vectors rather than a generic 3D tensor
            dummy_hamer = torch.randn(1, w, 4096).to(device)
            dummy_angles = torch.randn(1, w, 150).to(device)
            
            flop_counter = FlopCounterMode(display=False)
            
            try:
                with torch.no_grad():
                    with flop_counter:
                        _ = model(dummy_hamer, dummy_angles)
                
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