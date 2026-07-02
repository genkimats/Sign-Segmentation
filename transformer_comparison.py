import torch
import torch.nn as nn
import math
from torch.utils.flop_counter import FlopCounterMode

# ==============================================================================
# 1. "Hands-On" Exact Pipeline Architecture (512d, 1024d concat, T/2)
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
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.linear2(self.activation(self.linear1(self.norm2(x))))
        return x

class ExactHandsOnPipeline(nn.Module):
    def __init__(self, num_body_vertices=33, num_hand_vertices=42, in_channels=3, d_model=512, n_layers=4, nhead=8, num_classes=3, apply_downsampling=True):
        super().__init__()
        self.apply_downsampling = apply_downsampling
        self.num_body_vertices = num_body_vertices
        self.num_hand_vertices = num_hand_vertices
        
        # 1. Two Separate Auxiliary Modules (3-layer MLPs up-projecting to 512d)
        self.body_mlp = nn.Sequential(
            nn.Linear(num_body_vertices * in_channels, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.hand_mlp = nn.Sequential(
            nn.Linear(num_hand_vertices * in_channels, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 2. Temporal Downsampling (Factor of 2)
        if self.apply_downsampling:
            self.body_downsample = nn.Conv1d(d_model, d_model, kernel_size=2, stride=2)
            self.hand_downsample = nn.Conv1d(d_model, d_model, kernel_size=2, stride=2)
            self.upsample = nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2)
            
        # 3. Multi-Modal Mixer (1024d -> 1024d -> 512d)
        self.mixer_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model) # Project back to 512d for the Transformer
        )

        # 4. Transformer Encoder (d_model=512)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.Sequential(
            *[NaiveTransformerLayer(d_model, nhead) for _ in range(n_layers)]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (B, C, T, V) where V = 75
        B, C, T, V = x.shape
        
        body_x = x[:, :, :, :self.num_body_vertices]
        hand_x = x[:, :, :, self.num_body_vertices:]
        
        # Flatten spatial and channel dimensions
        body_feat = body_x.permute(0, 2, 3, 1).contiguous().view(B, T, self.num_body_vertices * C)
        hand_feat = hand_x.permute(0, 2, 3, 1).contiguous().view(B, T, self.num_hand_vertices * C)
        
        body_feat = self.body_mlp(body_feat)
        hand_feat = self.hand_mlp(hand_feat)
        
        if self.apply_downsampling:
            body_feat = body_feat.permute(0, 2, 1)
            body_feat = self.body_downsample(body_feat).permute(0, 2, 1)
            
            hand_feat = hand_feat.permute(0, 2, 1)
            hand_feat = self.hand_downsample(hand_feat).permute(0, 2, 1)
            
        # Concatenate to 1024 dimensions
        combined_feat = torch.cat([body_feat, hand_feat], dim=-1)
        
        # Mix and map down to 512 dimensions
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
        "Exact Hands-On Pipeline (T/2 Downsampled)": True,
        "Exact Hands-On Pipeline (Pure, No Downsampling)": False
    }
    
    window_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    for model_name, do_downsample in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"🧮 {model_name}")
        print(f"{'='*60}")
        print("Window Size | GFLOPs")
        print("-" * 25)
        
        model = ExactHandsOnPipeline(
            num_body_vertices=33, num_hand_vertices=42, in_channels=3, d_model=512, n_layers=4, apply_downsampling=do_downsample
        ).to(device)
        model.eval()

        for w in window_sizes:
            # Note: 75 vertices to match the Hands-On paper
            dummy_input = torch.randn(1, 3, w, 75).to(device)
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