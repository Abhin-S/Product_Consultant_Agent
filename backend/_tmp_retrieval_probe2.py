from retrieval.retriever import retrieve_local

queries = [
    "How should an early startup calculate CAC, LTV, and payback period?",
    "Seed funding strategy and runway planning for AI startup",
    "MVP roadmap and product-market fit experiments",
    "Brand positioning for a new D2C startup",
]

for q in queries:
    docs, low, diag = retrieve_local(q, 5)
    print("---")
    print("QUERY:", q)
    print("LOW_CONFIDENCE:", low)
    print("MAX_SIM:", max((d.similarity_score for d in docs), default=0.0))
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.source} | sim={d.similarity_score:.3f}")
