import json
from graphify.build import build_from_json
from graphify.cluster import score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from pathlib import Path

extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text(encoding='utf-8'))
detection  = json.loads(Path('graphify-out/.graphify_detect.json').read_text(encoding='utf-8'))
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text(encoding='utf-8'))

G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
cohesion = {int(k): v for k, v in analysis['cohesion'].items()}
tokens = {'input': 0, 'output': 0}

labels = {
    0: "Retrieval & Statistics",
    1: "Embedding Backend Core",
    2: "Backend Abstractions",
    3: "Backend Factories",
    4: "LLM Backends",
    5: "Answer Evaluation",
    6: "OpenAI Embeddings",
    7: "RAG Pipeline Concepts",
    8: "LLM Implementations",
    9: "HotpotQA Dataset",
    10: "Sample Corpus Data",
    11: "Evaluation Metrics",
    12: "Benchmark Results",
    13: "HotpotQA Loader",
    14: "Sample Data Builder",
    15: "RAGAS Adapter",
    16: "Backends Init",
    17: "Enhanced Chunk Overlap",
    18: "Naive Chunk Overlap",
    19: "Evaluation Init",
    20: "Evaluation Module",
    21: "Package Init",
    22: "Retrieval Init",
    23: "Retrieval Module",
}

questions = suggest_questions(G, communities, labels)
report = generate(G, communities, cohesion, labels, analysis['gods'], analysis['surprises'], detection, tokens, '.', suggested_questions=questions)
Path('graphify-out/GRAPH_REPORT.md').write_text(report, encoding='utf-8')
Path('graphify-out/.graphify_labels.json').write_text(json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False), encoding='utf-8')
print('Report updated with community labels')
print('Suggested questions:')
for q in questions[:5]:
    print(' -', q)
