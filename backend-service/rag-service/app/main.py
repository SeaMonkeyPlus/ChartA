import sys
from datetime import datetime
from fastapi import FastAPI
from app.models import QuestionRequest
from app.services.retriever import search_similar_chunks
from app.services.generator import generate
from app.utils.prompt import build_prompt
from app.services.supabase_client import supa
from app.services.embeddings import embed_text

app = FastAPI(title="Supabase Mini RAG with Events")

@app.post("/ask")
def ask(req: QuestionRequest):
    docs = search_similar_chunks(req.query)
    prompt = build_prompt(req.query, docs)
    answer = generate(prompt)
    return {"answer": answer, "refs": docs}

@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    # 1️⃣ 더미 이벤트 삽입
    print("✅ Supabase에 이벤트 데이터 삽입 중...")
    dummy_events = [
        {
            "event_time": datetime(2025, 1, 3, 10, 0).isoformat(),
            "source": "Yonhap News",
            "title": "삼성전자, 반도체 수요 회복으로 실적 반등",
            "content": "삼성전자는 2025년 1분기 반도체 부문에서 수요 회복세를 보이며 주가가 상승했다."
        },
        {
            "event_time": datetime(2025, 2, 15, 9, 30).isoformat(),
            "source": "Bloomberg",
            "title": "엔비디아, AI 수요 폭발로 사상 최고가 경신",
            "content": "AI 반도체 수요 급증으로 엔비디아 주가가 20% 급등하며 시장을 주도했다."
        },
        {
            "event_time": datetime(2025, 3, 10, 14, 0).isoformat(),
            "source": "한국경제",
            "title": "한국은행, 기준금리 동결 발표",
            "content": "한국은행은 물가 안정세를 이유로 기준금리를 3.5%로 동결한다고 발표했다."
        }
    ]

    # events insert
    for ev in dummy_events:
        res = supa.table("events").insert(ev).execute()
        event_id = res.data[0]["event_id"]
        print(f"📰 이벤트 삽입 완료 → {ev['title']} (id={event_id})")

        # 2️⃣ rag_chunks 로 임베딩 삽입
        vec = embed_text(ev["content"])
        supa.table("rag_chunks").insert({
            "event_id": event_id,
            "text": ev["content"],
            "embedding": vec
        }).execute()
        print(f"   ↳ 임베딩 저장 완료 ✅")

    print("✅ 전체 이벤트 → 임베딩 완료")

    # 3️⃣ CLI 질의
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        docs = search_similar_chunks(query)
        prompt = build_prompt(query, docs)
        print("🧠 질문:", query)
        print("📄 참고 문서:", docs)
        print("💬 답변:\n", generate(prompt))
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8080)
