import os, time, torch, psutil, logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("rag_performance.log"), logging.StreamHandler()]
)
# ─────────────────────────────────────────────────────────────
GEN_MODEL = os.getenv("GEN_MODEL", "Qwen/Qwen2-1.5B-Instruct")
HF_TOKEN  = os.getenv("HF_TOKEN")

tok = AutoTokenizer.from_pretrained(GEN_MODEL, token=HF_TOKEN, trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL,
    token=HF_TOKEN,
    dtype=torch.float32,   # CPU 안정
    trust_remote_code=True,
)
llm.to("cpu")

def generate(prompt: str, max_new_tokens=200, temperature=0.3):
    """LLM 답변 생성 + 성능 로그 기록"""
    start_time = time.perf_counter()

    # 실행 전 리소스 상태
    process = psutil.Process(os.getpid())
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB

    inputs = tok(prompt, return_tensors="pt")

    with torch.inference_mode():
        out = llm.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    # 실행 후 리소스 상태
    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 ** 2)
    elapsed = time.perf_counter() - start_time

    # 로그 출력 및 파일 기록
    logging.info(
        f"Prompt: {prompt[:50]}..."
        f"\n⏱️ Elapsed: {elapsed:.2f}s | "
        f"CPU(before→after): {cpu_before:.1f}%→{cpu_after:.1f}% | "
        f"Memory(before→after): {mem_before:.1f}MB→{mem_after:.1f}MB"
    )

    return tok.decode(out[0], skip_special_tokens=True)
