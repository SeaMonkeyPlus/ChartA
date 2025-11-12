def build_prompt(query: str, docs: list[str]):
    context = "\n".join([f"- {d}" for d in docs]) if docs else "(관련 문서 없음)"
    return f"질문: {query}\n\n참고 문서:\n{context}\n\n답변:"