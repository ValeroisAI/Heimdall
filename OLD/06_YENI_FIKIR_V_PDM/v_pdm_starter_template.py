"""
v_pdm_starter_template.py
==========================
Valerois Predictive Delta Memory (V-PDM) Başlangıç Kodu
Okul laboratuvarında doğrudan geliştirme ve test için hazır şablon.

Temel İlke:
  1. Tahmin Et: v_hat = (q @ S_{t-1}) / sqrt(d)
  2. Sürprizi Hesapla: delta = v - v_hat
  3. Sadece Sürprizi Yaz: S_t = Lambda * S_{t-1} + alpha * (k (x) delta)
  4. Oku ve Çıkar: y = RMSNorm(q @ S_t) * silu(gate)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight

class ValeroisPredictiveDeltaMemory(nn.Module):
    """
    Attention yerine sıfırdan çalışan,
    sadece bilgi sürprizini (delta) hafızaya kazıyan tekil katman.
    """
    def __init__(self, d_model=256, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projeksiyon matrisleri
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_gate = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Öğrenilebilir Dinamik Kapılar
        # 1. Lambda: Çok-ölçekli çürüme (Decay gate)
        self.w_decay = nn.Linear(d_model, num_heads, bias=True)
        # 2. Alpha: Sürpriz yazma gücü (Write gate)
        self.w_alpha = nn.Linear(d_model, num_heads, bias=True)

        # Başlatma: Yarısı hızlı unutur (yerel sözdizimi), yarısı yavaş unutur (global değişkenler)
        slow_decay = torch.linspace(2.5, 4.5, num_heads // 2)
        fast_decay = torch.linspace(0.5, 1.5, num_heads - num_heads // 2)
        self.w_decay.bias.data.copy_(torch.cat([slow_decay, fast_decay]))
        nn.init.constant_(self.w_alpha.bias, 1.0)

        self.norm = RMSNorm(self.head_dim)

    def forward_step(self, x_t, state):
        """
        Tek bir token için çıkarım adımı (O(1) sabit bellek).
        x_t: [B, D]
        state: [B, H, Dh, Dh]
        """
        B, D = x_t.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(x_t).view(B, H, Dh)
        k = self.k_proj(x_t).view(B, H, Dh)
        v = self.v_proj(x_t).view(B, H, Dh)
        gate = F.silu(self.out_gate(x_t))

        # Lambda: Çürüme katsayısı [B, H, 1, 1]
        decay = torch.sigmoid(self.w_decay(x_t)).view(B, H, 1, 1)
        # Alpha: Sürpriz yazma katsayısı [B, H, 1]
        alpha = torch.sigmoid(self.w_alpha(x_t)).view(B, H, 1)

        # 1. MEVCUT HAFIZADAN TAHMİN ET (Predict)
        # q: [B, H, 1, Dh] @ state: [B, H, Dh, Dh] -> v_hat: [B, H, Dh]
        v_hat = (q.unsqueeze(2) @ state).squeeze(2) / math.sqrt(Dh)

        # 2. SÜRPİZİ HESAPLA (Predictive Delta)
        # Gerçek gelen değer ile tahmin arasındaki fark
        delta = v - v_hat

        # 3. SADECE SÜRPİZİ HAFIZAYA KAZI (Delta Outer Product)
        # k: [B, H, Dh, 1] x delta: [B, H, 1, Dh] -> [B, H, Dh, Dh]
        delta_update = (k.unsqueeze(-1) @ delta.unsqueeze(-2)) * alpha.unsqueeze(-1)
        new_state = state * decay + delta_update

        # 4. OKU VE ÇIKAR
        readout = (q.unsqueeze(2) @ new_state).squeeze(2)
        readout = self.norm(readout).view(B, D)
        out = self.out_proj(readout * gate)

        return out, new_state, delta.norm(dim=-1).mean().item()

    def forward_sequence(self, x):
        """
        Eğitim / dizi akışı döngüsü.
        x: [B, T, D]
        """
        B, T, D = x.shape
        state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        outputs = []
        deltas = []

        for t in range(T):
            x_t = x[:, t, :]
            out_t, state, delta_norm = self.forward_step(x_t, state)
            outputs.append(out_t)
            deltas.append(delta_norm)

        out = torch.stack(outputs, dim=1)
        return out, state, deltas


# =============================================================================
# LABORATUVAR TEST BLOĞU
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Cihaz: {device}")
    print("[*] Valerois Predictive Delta Memory (V-PDM) Başlatılıyor...")

    d_model = 256
    seq_len = 100
    batch_size = 2

    layer = ValeroisPredictiveDeltaMemory(d_model=d_model, num_heads=4).to(device)
    total_params = sum(p.numel() for p in layer.parameters())
    print(f"[*] Katman Parametre Sayısı: {total_params:,}")

    # Rastgele veri ile test
    dummy_input = torch.randn(batch_size, seq_len, d_model, device=device)
    output, final_state, deltas = layer.forward_sequence(dummy_input)

    print(f"[*] Girdi Boyutu        : {dummy_input.shape}")
    print(f"[*] Çıktı Boyutu        : {output.shape}")
    print(f"[*] O(1) Sabit Durum    : {final_state.shape} ({final_state.nelement()*final_state.element_size()/1024:.2f} KB)")
    print(f"[*] Ortalama Sürpriz Normu: {sum(deltas)/len(deltas):.4f}")
    print("\n[✓] V-PDM Katmanı Başarıyla Çalıştı! Lab dersinde geliştirmeye hazırsın.")
