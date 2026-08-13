# 콘텐츠 작성 가이드

> 코드는 건드리지 않습니다. 아래 3개 파일만 수정하면 사이트가 업데이트됩니다.

---

## 어떤 파일을 수정하면 되나요?

```
hyeonwoo/
├── data/
│   ├── knowledge.json   ← 지식 그래프 (개념 추가, 연결, 학습 기록)
│   ├── research.json    ← 연구 비전 페이지
│   └── notes/
│       └── *.md         ← 개념별 상세 노트
```

---

## 1. 새 개념 추가 → `knowledge.json`

`"nodes"` 배열에 한 줄 추가:

```json
{
  "id": "flash_attention",
  "label": "Flash Attention",
  "category": "Architecture",
  "confidence": 1,
  "studyCount": 0,
  "note": "data/notes/flash_attention.md"
}
```

**카테고리**: `Generative` / `Architecture` / `Language Model` / `Multimodal` / `Training` / `RL` / `Math & Stats` / `Systems` / `Algorithm`

**자신감(confidence)**: `0`모름 → `1`기초 → `2`중급 → `3`고급 → `4`전문가

---

## 2. 개념 간 연결 추가 → `knowledge.json`

`"edges"` 배열에 추가:

```json
{
  "source": "attention",
  "target": "flash_attention",
  "relation": "leads_to",
  "weight": 3,
  "insight": "Attention 연산을 IO-aware하게 최적화"
}
```

**관계 종류**: `basis_of` / `uses` / `part_of` / `leads_to` / `comparison` / `enables`

---

## 3. 오늘 공부한 내용 기록 → `knowledge.json`

`"sessions"` 배열 맨 뒤에 추가 (Activity 히트맵에 반영):

```json
{
  "date": "2026-08-13",
  "topics": ["transformer", "attention", "flash_attention"],
  "note": "Attention 메커니즘 + Flash Attention 최적화 공부"
}
```

---

## 4. 개념 노트 작성 → `data/notes/개념명.md`

```markdown
# Flash Attention

## 핵심 아이디어
IO-aware한 Attention 구현으로 메모리 사용량과 속도를 동시에 개선.

## 수식
$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

## 직관
HBM ↔ SRAM 데이터 이동을 최소화하여 메모리 병목 해결.

## 연결 개념
- Attention: 이 최적화의 대상
- KV Cache: 함께 사용되는 메모리 최적화 기법
```

---

## 5. 연구 비전 수정 → `data/research.json`

```json
{
  "tagline": "나는 어떤 세상을 만들고 싶은가",
  "sections": {
    "worldview": "세상의 흐름을 여기에...",
    "need": "사회적 필요를 여기에...",
    "value": "가치 창출을 여기에...",
    "research": "연구 과제를 여기에..."
  }
}
```

> ⚠️ research.json은 브라우저 localStorage가 있으면 덮어쓰입니다.
> 반영하려면 브라우저 콘솔에서: `localStorage.removeItem('hyeonwoo_research_v1')` 후 새로고침

---

## AI한테 부탁하는 방법

이 폴더(`hyeonwoo/`)를 열고 새 세션에서 자연어로 말하면 됩니다:

```
"Flash Attention 개념 추가해줘. Architecture 카테고리, Attention이랑 연결."
"오늘 Transformer, BERT 공부했어. 세션 기록해줘."
"research.json 세상의 흐름 섹션 이렇게 바꿔줘: ..."
"MLE 노트에 수식이랑 직관 추가해줘."
```

AI는 `.agents/GEMINI.md`를 자동으로 읽어서 형식에 맞게 파일만 수정해줍니다.

---

## 수정 후 GitHub 업로드

```bash
cd ~/Desktop/대학원준비_2027/hyeonwoo
git add data/
git commit -m "📝 Flash Attention 추가"
git push origin main
```
