"""ROCm/GPU teşhisi: hangi işlem 'no kernel image' veriyor? Her test ayrı süreçte (bozulan GPU bağlamı diğerini etkilemesin).
    python bench/rocm_teshis.py
    TORCH_BLAS_PREFER_HIPBLASLT=0 python bench/rocm_teshis.py      # diğer BLAS kütüphanesiyle
"""
import subprocess
import sys

TESTS = {
    "bf16 GEMM 512x5120x32768":   "a=torch.randn(512,5120,device=d,dtype=torch.bfloat16); b=torch.randn(32768,5120,device=d,dtype=torch.bfloat16); y=F.linear(a,b)",
    "bf16 GEMM 4096x5120x32768":  "a=torch.randn(4096,5120,device=d,dtype=torch.bfloat16); b=torch.randn(32768,5120,device=d,dtype=torch.bfloat16); y=F.linear(a,b)",
    "bf16 GEMM 512x5120x248320":  "a=torch.randn(512,5120,device=d,dtype=torch.bfloat16); b=torch.randn(248320,5120,device=d,dtype=torch.bfloat16); y=F.linear(a,b)",
    "log_softmax fp32 512x248320": "y=F.log_softmax(torch.randn(512,248320,device=d),-1)",
    "log_softmax fp32 512x32768":  "y=F.log_softmax(torch.randn(512,32768,device=d),-1)",
    "cross_entropy 512x248320":    "y=F.cross_entropy(torch.randn(512,248320,device=d), torch.randint(0,248320,(512,),device=d))",
    "topk fp32 512x248320 k=32":   "y=torch.randn(512,248320,device=d).topk(32,-1).values",
    "max/exp/sum 512x248320":      "x=torch.randn(512,248320,device=d); m=x.amax(-1,keepdim=True); y=(x-m).exp().sum(-1).log()",
}
for name, code in TESTS.items():
    prog = f"import torch, torch.nn.functional as F; d='cuda'; {code}; torch.cuda.synchronize(); print('ok')"
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
    ok = r.returncode == 0 and "ok" in r.stdout
    err = "" if ok else (r.stderr.strip().splitlines() or ["?"])[-1][:110]
    print(f"{'OK  ' if ok else 'HATA'} {name:30s} {err}", flush=True)
