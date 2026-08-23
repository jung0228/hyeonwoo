# 정현우의 지식 지도 — AI 어시스턴트 작업 규칙

이 프로젝트는 정현우의 AI/ML 지식 관리 웹사이트입니다.
**코드(app.js, style.css, index.html)는 절대 수정하지 않습니다.**
콘텐츠 업데이트는 아래 3개 파일만 수정합니다.

---

## 📁 수정 가능한 파일

| 파일 | 역할 |
|---|---|
| `data/knowledge.json` | 지식 그래프 (노드·엣지·학습 세션) |
| `data/research.json` | 연구 비전 (세상의 흐름·사회적 필요·가치 창출·연구 과제) |
| `data/notes/*.md` | 개념별 상세 노트 |

---

## 1. 지식 그래프 (`data/knowledge.json`)

### 노드(개념) 추가

`"nodes"` 배열에 추가합니다.

```json
{
  "id": "영문_소문자_언더스코어",
  "label": "화면에 표시될 이름",
  "category": "카테고리명",
  "confidence": 0,
  "studyCount": 0,
  "note": "data/notes/파일명.md"
}
```

**`category` 선택지 (반드시 이 중 하나)**

| category | 클러스터 | 의미 |
|---|---|---|
| `"Generative"` | 🤖 딥러닝 | 생성 모델 (VAE, GAN, Diffusion) |
| `"Architecture"` | 🤖 딥러닝 | 네트워크 구조 (Transformer, ResNet) |
| `"Language Model"` | 🤖 딥러닝 | 언어 모델 (GPT, BERT, LLM) |
| `"Multimodal"` | 🤖 딥러닝 | 멀티모달 (Vision-Language, Audio) |
| `"Training"` | 🤖 딥러닝 | 학습 기법 (LoRA, KV Cache, SFT) |
| `"RL"` | 📐 머신러닝 | 강화학습 기초 |
| `"Math & Stats"` | 📐 머신러닝 | 수학·통계 (MLE, PCA, Cross-Entropy) |
| `"Systems"` | 💻 시스템 | 컴퓨터 시스템 (OS, 캐시, 파이프라인) |
| `"Algorithm"` | 🔢 알고리즘 | 알고리즘 (DP, BFS, Sliding Window) |

**`confidence` 값**

| 값 | 의미 |
|---|---|
| `0` | 모름 / 처음 접함 |
| `1` | 기초 — 개념만 앎 |
| `2` | 중급 — 설명 가능 |
| `3` | 고급 — 응용 가능 |
| `4` | 전문가 — 깊게 이해 |

---

### 엣지(연결) 추가

`"edges"` 배열에 추가합니다.

```json
{
  "source": "출발_노드_id",
  "target": "도착_노드_id",
  "relation": "관계_타입",
  "weight": 3,
  "insight": "왜 연결되는지 한 줄 설명 (선택)"
}
```

**`relation` 선택지**

| relation | 의미 |
|---|---|
| `"basis_of"` | A가 B의 이론적 기반 |
| `"uses"` | A가 B를 사용/활용 |
| `"part_of"` | A가 B의 구성 요소 |
| `"leads_to"` | A를 이해하면 B로 이어짐 |
| `"comparison"` | A와 B를 비교/대조 |
| `"enables"` | A가 있어야 B가 가능 |

**`weight`**: 1(약한 연결) ~ 5(강한 연결), 기본값 `3`

---

### 학습 세션 추가

`"sessions"` 배열에 추가합니다. (Activity 탭 히트맵에 반영)

```json
{
  "date": "YYYY-MM-DD",
  "topics": ["노드_id1", "노드_id2", "노드_id3"],
  "note": "오늘 학습 내용 한 줄 요약"
}
```

- `topics`에는 오늘 공부한 노드 id를 나열합니다
- `date`는 `"2026-08-13"` 형식
- 최신 날짜가 배열 뒤에 오도록 추가합니다

---

## 2. 연구 비전 (`data/research.json`)

```json
{
  "tagline": "한 줄 연구 슬로건",
  "sections": {
    "worldview": "세상의 흐름 — 여러 줄 가능 (\\n으로 줄바꿈)",
    "need": "사회적 필요",
    "value": "가치 창출",
    "research": "연구 과제"
  },
  "agenda": [
    {
      "id": 1,
      "title": "연구 주제",
      "priority": "high | medium | low",
      "status": "idea | exploring | active | done",
      "tags": ["태그1", "태그2"],
      "note": "상세 설명"
    }
  ],
  "keywords": ["키워드1", "키워드2"]
}
```

> ⚠️ `research.json`은 **최초 접속 시 기본값**으로만 사용됩니다.
> 브라우저에서 직접 편집한 내용이 localStorage에 저장되어 우선 적용됩니다.
> localStorage 초기화 후 반영하려면 브라우저에서 `localStorage.removeItem('hyeonwoo_research_v1')` 실행.

---

## 3. 개념 노트 (`data/notes/*.md`)

각 노드의 `"note"` 필드가 가리키는 마크다운 파일입니다.
노드를 클릭하면 우측 패널에 렌더링되어 표시됩니다.

**파일명 규칙**: `data/notes/{노드_id}.md`

**기본 형식**:

```markdown
# 개념 이름

## 핵심 아이디어
한두 줄로 핵심만.

## 수식 (선택)
$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

## 직관적 설명
비유나 예시로 설명.

## 연결 개념
- [[관련_개념1]]
- [[관련_개념2]]

## 참고
- 논문명 / 출처
```

- KaTeX 수식 지원: `$인라인$`, `$$블록$$`
- 마크다운 전체 문법 지원

---

## 📋 작업 예시

### "Flash Attention 개념 추가해줘"
→ `knowledge.json`의 `nodes`에 추가:
```json
{ "id": "flash_attention", "label": "Flash Attention", "category": "Architecture", "confidence": 1, "studyCount": 0, "note": "data/notes/flash_attention.md" }
```
→ `edges`에 Attention과 연결 추가
→ `data/notes/flash_attention.md` 생성

### "오늘 Transformer랑 Attention 공부했어"
→ `knowledge.json`의 `sessions`에 추가:
```json
{ "date": "2026-08-13", "topics": ["transformer", "attention"], "note": "Transformer 구조 복습" }
```

### "연구 비전 세상의 흐름 섹션 수정해줘"
→ `research.json`의 `sections.worldview` 수정

---

## 4. 칼럼 관리 (`data/columns.json` 및 `data/columns/*.md`)

정현우의 깊은 지적 질문을 아카이브로 남기는 공간입니다.

### 칼럼 정보 추가 (`data/columns.json`)
```json
{
  "id": "ai_data_center_necessity",
  "title": "이 시대의 인프라, AI 데이터 센터는 왜 필요한가",
  "date": "YYYY-MM-DD",
  "author": "정현우",
  "summary": "한 줄 요약 헤드라인 내용",
  "file": "data/columns/파일명.md",
  "category": "분야명(예: Infrastructure, AI, Society 등)",
  "readTime": "5 min read"
}
```

### 칼럼 본문 작성 (`data/columns/파일명.md`)
* 마크다운 형식으로 작성하며 첫 줄의 제목(H1)은 신문 헤더와 중복되므로 렌더링 시 자동으로 제외됩니다.
* 수식 연산(KaTeX) 및 다단 그리드가 완벽 지원됩니다.

---


## ✅ 작업 완료 후 반드시

```bash
cd ~/Desktop/대학원준비_2027/hyeonwoo
git add data/
git commit -m "📝 내용 설명"
git push origin main
```

코드 파일(app.js, style.css, index.html, assets/vendor/)은 **절대 수정하지 않습니다.**


---

## 5. 🧠 똑똑한 개발자 6대 코드 룰 및 4단계 방어 프로토콜 (AI 필수 준수)

1. **짐작 금지 (Never Guess Code/Schema/Paths)**:
   - "안 보인다", "오작동한다"는 요청 시 조급히 짐작하지 말고, `view_file` 또는 `grep_search`로 RAW 소스코드를 전수 정밀 조회 후 작업한다.
2. **에러 근본 원인 파악 (No Superficial Patches)**:
   - 겉증상만 가리는 임시 패치(`try-except` 예외 삼킴 등)를 절대 하지 않으며, Full Log/RAW 소스를 통해 주석 미닫힘, 오타 등 근본 원인을 해결한다.
3. **무지성 반복 시도 금지 (Analyze Before Retrying)**:
   - 실패한 명령이나 탭 조정을 원인 분석 없이 똑같이 재시도하지 않는다.
4. **실증 검증 없는 성공 선언 금지 (Verify Before Claiming Success)**:
   - 파일 수정만으로 "완료했다"고 말하지 않으며, `git diff`, HTML 주석 닫힘 검사, `curl -I` 200 OK 수치적 검증 후 성공을 선언한다.
5. **레이아웃 스코프 격리 & 사이드 이펙트 방지 (Preserve Isolation)**:
   - 새로운 UI/탭 추가 시 기존 탭의 `display`, `position`, `flex` 레이아웃에 충돌이나 밀림 현상이 없는지 사전에 점검한다.
6. **작업 완료 후 무조건 Git 동기화**:
   - `git add data/ app.js style.css index.html`, `git commit`, `git push origin main`으로 항상 배포 서버까지 최신화한다.
