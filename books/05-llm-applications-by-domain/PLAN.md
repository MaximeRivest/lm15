# Book 5 — LLM Applications by Domain with lm15

**Analog:** Text Mining with R / Tidy Modeling with R (combined)
**Audience:** Domain practitioners applying LLMs to real workloads
**Length:** ~300 pages
**Prerequisite:** Book 1 Parts I–IV (complete, model, tools, multimodal) or equivalent
**Each part is standalone** — readers pick the domain they need

---

## Thesis

LLMs are general-purpose, but applying them well is domain-specific. This book is a collection of five self-contained domains. Each part starts with the problem, builds the tools, shows the lm15 code, and ends with a production-ready pattern you can adapt.

---

## Part I — Text Processing (Chapters 1–4)

### Chapter 1: Summarization

- Single-document summarization: `lm15.complete(model, f"Summarize:\n{text}")`
- Controlling length: `max_tokens`, system prompt instructions
- Extractive vs abstractive: prompt design for each
- Long documents: chunking strategies, map-reduce pattern
- Multi-document summarization: concatenate vs sequential refinement
- Evaluation: ROUGE-like manual checks, self-critique pipeline

### Chapter 2: Information Extraction

- Named entities, dates, amounts, relationships
- Structured output via system prompt: "Return JSON with fields..."
- `prefill="{"` — forcing JSON start
- Schema enforcement: describe the schema in the system prompt
- Batch extraction: config-from-dict pattern over many documents
- Error handling: malformed JSON, missing fields, hallucinated values

### Chapter 3: Classification and Labeling

- Binary classification: sentiment, spam, relevance
- Multi-class: topic, intent, severity
- Prompt patterns: zero-shot, few-shot (examples in system prompt)
- Confidence via reasoning: `reasoning=True` → read `resp.thinking` for rationale
- Batch classification: list comprehension with `lm15.complete(**config, prompt=text)`
- Cost optimization: cheap model (nano/flash) for classification, expensive for edge cases

### Chapter 4: Text Transformation

- Translation: source/target language in system prompt
- Style transfer: formal ↔ casual, technical ↔ layperson
- Rewriting: "Rewrite for clarity", "Make this shorter"
- Template filling: structured input → prose output
- Chained transformations: draft → critique → revise pipeline

---

## Part II — Code Generation (Chapters 5–8)

### Chapter 5: Code Writing

- Prompt design for code: be specific about language, framework, constraints
- System prompt: "You are a Python developer. Output only code. No explanations."
- `prefill="```python\n"` — steering output format
- Testing generated code: capture output, run it, verify
- Iterative refinement: feed errors back, let the model fix

### Chapter 6: Code Review

- Feed code + review guidelines via system prompt
- Structured output: severity, location, description, suggested fix
- Multi-file review: read files as tools, iterate
- Diff-based review: show only changed lines, ask for issues
- Using reasoning: `reasoning=True` for deep analysis of complex code

### Chapter 7: Refactoring

- "Refactor this function to use X pattern"
- Before/after pattern: show original, get transformed version
- Multi-step refactoring agent: read → plan → execute → test
- Large-scale: identify all instances, transform each, verify consistency
- Cross-model verification: model A refactors, model B reviews

### Chapter 8: Migration

- Language migration: "Convert this Python to TypeScript"
- Framework migration: "Convert this Flask app to FastAPI"
- API migration: "Update all calls from v1 to v2 API"
- Agent pattern: scan codebase → identify migration targets → transform → test
- Preserving behavior: generate tests before migrating, run after

---

## Part III — Document Intelligence (Chapters 9–11)

### Chapter 9: PDF Analysis

- `Part.document(data=..., media_type="application/pdf")` — send PDF directly
- `lm15.upload(model, "file.pdf")` — upload for large documents
- Page-by-page analysis: chunk by page, query each
- Prompt caching: `Part.document(..., cache=True)` — ask many questions about one PDF
- Extraction targets: tables, clauses, figures, metadata

### Chapter 10: Contract Review

- System prompt: legal analysis guidelines, risk categories
- Tools: `search_clause(term)`, `compare_sections(a, b)`
- Multi-document: compare two contracts, find differences
- Structured output: clause-by-clause risk assessment with severity
- Agent pattern: scan → flag risks → draft summary → review

### Chapter 11: Multi-Document QA

- Upload multiple documents, ask cross-cutting questions
- RAG-lite: no vector DB needed for small document sets
- Conversation pattern: `lm15.model()` with documents in first message
- Prompt caching: cache the documents, vary only the question
- Source attribution: instruct model to cite document + page

---

## Part IV — Vision (Chapters 12–14)

### Chapter 12: Image Description and Analysis

- `Part.image(url=...)` and `Part.image(data=...)`
- Describing photos, diagrams, charts, screenshots
- Structured output: "List all objects with position and color"
- Comparing images: send two images, ask for differences
- Provider strengths: Gemini for general vision, Claude for document images

### Chapter 13: Diagram to Code

- Architecture diagrams → infrastructure-as-code
- UI mockups → HTML/CSS
- Flowcharts → state machines
- Prompt pattern: describe → plan → generate → verify
- Cross-model: Gemini describes diagram, Claude generates code

### Chapter 14: Visual QA Pipelines

- Dashboard screenshot → "Is revenue up or down?"
- Product image → "Does this match the description?"
- Multi-image comparison: before/after, A/B variants
- Pipeline: image analysis → text reasoning → structured answer
- Chaining: `resp1 = gemini([image, "Describe"])` → `resp2 = claude(f"Analyze: {resp1.text}")`

---

## Part V — Data Workflows (Chapters 15–17)

### Chapter 15: LLM-in-the-Loop ETL

- Schema inference: send sample rows, get column types and descriptions
- Data cleaning: "Fix these addresses", "Normalize these dates"
- Deduplication: "Are these two records the same entity?"
- Batch processing: iterate over rows, classify/transform each
- Cost control: use cheapest model that achieves accuracy target

### Chapter 16: Data Exploration and Analysis

- Send CSV/table data as text, ask analytical questions
- "What are the outliers?", "What trends do you see?"
- Generating SQL from natural language questions
- Generating Python/pandas code for analysis
- Self-verifying: generate code, run it (via tool), check output

### Chapter 17: Schema Design and Documentation

- "Design a database schema for X"
- Structured output: table definitions, relationships, constraints
- Generating migration scripts from natural language spec
- Documenting existing schemas: send DDL, get human-readable docs
- Iterative refinement: generate → review → revise

---

## Appendices

- A: Prompt templates by domain (copy-paste starters)
- B: Model selection guide by task type and cost
- C: Batch processing patterns (parallel, sequential, map-reduce)
- D: Cost estimation per domain (typical token counts)
- E: Quality evaluation patterns (self-critique, cross-model verification)
