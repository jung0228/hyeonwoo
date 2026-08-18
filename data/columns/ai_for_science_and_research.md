# AI는 과연 연구(Research)를 할 수 있는가: 방법론, 실제 사례, 그리고 오토 리서치 아키텍처

## 서론: 문제 풀이(Search)를 넘어 미지의 발견(Discovery)으로

2026년 현재, 인공지능이 인간 연구자의 조수를 넘어 **'연구의 주체'**가 될 수 있는가에 대한 논쟁은 더 이상 철학적 담론에 머무르지 않는다. 사카나 AI(Sakana AI)의 *The AI Scientist*가 2026년 3월 *Nature* 본지에 시스템 아키텍처를 게재하고, 딥마인드의 *FunSearch*가 수십 년간 미해결이었던 극값 조합론(Cap Set Problem)의 수학적 경계를 갱신하면서, AI는 이미 실질적인 과학적 발견을 만들어내기 시작했다.

하지만 연구 현장에서 AI를 진정으로 활용하기 위해서는 막연한 기대나 회의론을 넘어 구체적인 질문에 답해야 한다:
1. 연구의 본질은 무엇이며, 좋은 연구 문제는 어떻게 정의되는가?
2. AI는 실제로 어떤 메커니즘으로 과학 연구를 수행해 왔는가?
3. 인간은 주제를 어디까지 던져주며, 시스템은 무엇을 어떻게 수집·자동화하는가?
4. 무한한 가설의 바다에서 어떻게 '독창적인 연구 아이디어'를 체계적으로 도출할 것인가?
5. 인간 연구자는 어디에 집중해야 인지 부채(Cognitive Debt)를 피하고 본질적 가치를 남길 수 있는가?

이 글은 2027 대학원 진학 및 향후 독자적인 AI/ML 연구를 준비하며, AI 기반 연구 자동화(Automated Scientific Research)의 방법론과 엔지니어링 청사진을 집대성한 기록이다.

---

## 1. 연구 문제 발굴의 3차원 좌표계

학계에서 수많은 논문이 쏟아져 나오지만, 소수의 논문만이 거대한 파급력을 갖는 이유는 기술적 기교의 차이가 아니라 **"문제를 정의하는 시야의 깊이와 당위성"** 때문이다. 모든 위대한 연구는 다음 3가지 차원의 교차점에서 출발한다.

| 차원 (Dimension) | 핵심 질문 | 연구자가 파고들어야 할 지점 | 실전 예시 |
|---|---|---|---|
| **축 1: 거시적 당위성<br>(Macro Why)** | "이 문제를 풀면 세상과 AI에 어떤 변화가 오는가?" | 이 병목이 안 풀렸을 때 전체 시스템(로봇, 에이전트)이 멈춰 서는 결정적 지점 | *"1분짜리 영상도 VRAM이 터져 뚝뚝 끊어 보는 문제를 못 풀면 실시간 자율주행과 수술 로봇은 불가능하다."* |
| **축 2: 기존 연구의 결함<br>(Prior Art Pathology)** | "과거 SOTA는 왜 이 문제 앞에서 실패했는가?" | 단순 데이터/컴퓨팅 부족이 아닌, 알고리즘의 **근본적인 가정(Assumption)의 오류** 규명 | *"균등 프레임 샘플링은 연산량은 줄이지만, 1초 미만의 결정적 액션을 통째로 누락하는 태생적 결함이 있다."* |
| **축 3: 최신 논문의 맹점<br>(Frontier Blind Spot)** | "올해 나온 최신 논문들은 어디서 멈췄는가?" | 논문의 Abstract보다 **`Limitations` 섹션과 `Failure Cases` 부록**을 집요하게 분석 | *"LongVALE는 옴니모달을 달성했으나 영상이 30분을 넘으면 전후 인과관계 추론에서 환각을 일으킨다."* |

---

## 2. 실제로 AI가 연구를 수행한 5대 실전 사례

AI가 단순히 계산을 보조한 수준을 넘어, **가설을 세우고 코드를 작성하여 새로운 과학적 결과를 도출한 대표 사례**들은 다음과 같다.

| 프로젝트 | 연구 기관 | 적용 도메인 | 핵심 메커니즘 | 실질적 연구 성과 |
|---|---|---|---|---|
| **FunSearch**<br>(Nature 2023) | Google DeepMind | 극값 조합론 &<br>알고리즘 최적화 | LLM 프로그램 생성 $\leftrightarrow$ 외부 샌드박스 평가기 유전 피드백 루프 | 20년간 미해결이던 Cap Set 문제의 수학적 하한 갱신 |
| **The AI Scientist**<br>(Nature 2026) | Sakana AI &<br>Oxford Univ. | 머신러닝 시스템 &<br>딥러닝 아키텍처 | Semantic Scholar 문헌 조사 $\rightarrow$ 코드 수정 $\rightarrow$ GPU 훈련 $\rightarrow$ 논문 작성 $\rightarrow$ 자동 피어리뷰 | $15 미만 비용으로 ICLR 워크숍 수준 연구 논문 자동 완결 |
| **GNoME & A-Lab**<br>(Nature 2023) | Google DeepMind &<br>Berkeley Lab | 무기 결정 재료학 &<br>자동 무인 합성 | 그래프 신경망(GNN) 결정 구조 예측 $\rightarrow$ 무인 로봇 실험실 연동 | 220만 종 신물질 발견 및 736종 실제 로봇 합성 성공 |
| **AlphaProof &<br>AlphaGeometry 2**<br>(2024) | Google DeepMind | 형식 수학(Formal Math)<br>& 정리 증명 | 형식 수학 언어(Lean 4) + 강화학습 트리 탐색 | IMO 2024 은/금메달 수준의 고난도 복합 정리 증명 완료 |
| **PaperQA2**<br>(2024) | FutureHouse | 생명과학 문헌 분석 &<br>지식 그래프 합성 | 인용 네트워크 추적 + 고정밀 과학 RAG + 모순점 자동 탐지 | 박사급 연구원 능가하는 초인적 문헌 검색 및 WikiCrow 생성 |

---

## 3. 오토 리서치는 어떻게 작동하는가: 입력, 수집, 실행 파이프라인

연구 자동화는 추상적인 개념이 아닌 정교한 엔지니어링 아키텍처다.

### 1) 인간은 주제를 "어디까지" 던져주는가?

| 레벨 (Level) | 인간 연구자가 제공하는 입력 | AI 시스템의 역할 | 대표 사례 |
|---|---|---|---|
| **Level 1: 템플릿 기반<br>(Template-driven)** | • 실행 가능한 베이스라인 코드 (`experiment.py`)<br>• 결과 시각화 코드 (`plot.py`)<br>• 배경 및 시드 아이디어 2~3개 (`prompt.json`)<br>• 타겟 메트릭 (Val Loss, Latency) | 코드 공간을 체계적으로 변형(Ablation)하며 수십 개의 파생 가설 자가 실험 | Sakana AI v1,<br>FunSearch |
| **Level 2: 목표 지향형<br>(Goal-oriented)** | • 자연어 연구 질문<br>• 타겟 벤치마크 및 하드웨어/비용 제약 조건 | arXiv/GitHub에서 적절한 베이스라인을 직접 검색·조립하여 실험 파이프라인 구성 | Sakana AI v2,<br>Co-Scientist |
| **Level 3: 개방형 탐색<br>(Open-ended Discovery)** | • 광범위한 도메인 관심사 (예: "비디오 LLM 효율화") | 최신 학회 논문들의 한계점을 읽고 스스로 풀 가치가 있는 문제와 벤치마크 정의 | 차세대 연구 프런티어 |

---

### 2) 무엇을 어떻게 수집하고 구조화하는가?

성공적인 연구 자동화 시스템은 **문헌 지식, 오픈소스 코드베이스, 벤치마크 데이터셋**을 수집하여 구조화된 그래프 형태로 인덱싱한다.

#### 🗄️ 논문 수집 시 반드시 저장해야 할 6대 핵심 엔티티 스키마

```json
{
  "paper_id": "arxiv_2501_xxxxx",
  "title": "StreamKV: Towards Streaming Long-Context Video LLMs",
  "domain": "Multimodal / Video-LLM",
  "problem_formulation": {
    "target_task": "Long Video Streaming Understanding",
    "unaddressed_bottleneck": "O(T) KV Cache memory explosion in continuous frame generation",
    "core_limitation_of_prior_art": "Uniform eviction loses crucial temporal anchors"
  },
  "core_hypothesis": "Recent tokens provide temporal locality, while high-attention sink tokens maintain global semantics. Dynamic Top-K eviction bounded by fixed budget preserves 95% accuracy.",
  "mathematical_operators": [
    "S_i = \\sum_{\\tau=t-W}^t \\sum_h A_{h, \\tau, i}",
    "\\mathcal{S}^* = \\text{TopK}(S_i, K_{\\text{budget}} - K_{\\text{recent}}) \\cup \\mathcal{S}_{\\text{recent}}"
  ],
  "experimental_setup": {
    "baselines": ["StreamingLLM", "H2O", "Full Attention"],
    "benchmarks": ["Video-ChatGPT", "LongVideoBench", "LVBench"],
    "hardware": "8x NVIDIA H100 (80GB)",
    "metrics": {"latency_reduction": "4.2x", "memory_saving": "68%"}
  },
  "limitations_and_failure_modes": [
    "Fails when key visual clue occurs in a single frame with low initial attention score",
    "Does not compress Query/Key dimensions, only sequence length"
  ],
  "graph_edges": [
    {"target": "StreamingLLM", "relation": "extends"},
    {"target": "Momentseeker", "relation": "tested_on"}
  ]
}
```

---

### 3) 5단계 연구 자동화 폐루프 (The Closed-Loop Engine)

| 단계 | 담당 에이전트 | 핵심 동작 |
|---|---|---|
| **1. 가설 생성** | Ideation Agent | 템플릿 코드와 문헌 지식 그래프를 결합해 구체적인 개선 가설 수립 |
| **2. 독창성 검증** | Novelty Filter | Semantic Scholar API로 자동 검색 $\rightarrow$ 기존 논문과 **유사도 85% 이상 시 중복으로 판정하고 즉시 폐기** |
| **3. 코드 실행 & 디버깅** | Code & Experiment Agent | 코딩 에이전트가 `experiment.py`를 수정하고 Docker/GPU 샌드박스에서 실행 $\rightarrow$ 에러 발생 시 Traceback을 읽고 자가 디버깅(Self-Debugging) 반복 |
| **4. 통계적 검증** | Ablation Verifier | 동일 시드 반복 실험(3~5회)으로 성능 향상이 통계적으로 유의미한지($p < 0.05$) 검증 |
| **5. 논문 작성 & 심사** | Drafting & Review Agent | LaTeX 템플릿에 수치와 차트를 자동 바인딩하고, ICLR 기준 심사 에이전트가 점수 및 비평 제공 |

---

## 4. 독창적인 연구 아이디어를 발굴하는 4대 벡터

아이디어는 단순한 영감이 아니라 **구조화된 사고 연산(Systematic Operators)**의 결과다.

| 발굴 벡터 (Vector) | 핵심 원리 | 실전 적용 예시 |
|---|---|---|
| **1. 이종 결합<br>(Cross-Pollination)** | A 분야에서 검증된 강력한 기법을 아직 도입되지 않은 B 분야의 병목에 이식 | LLM 추론의 KV Cache 압축 기법 $\rightarrow$ 비디오 모델의 장기 메모리 병목에 적용 (*StreamKV*) |
| **2. 기저 가정 파괴<br>(Assumption Inversion)** | 업계 연구자들이 "당연하다"고 믿고 있는 암묵적 상식을 정면으로 의심 | *"비디오 프레임은 왜 항상 일정한 fps로 샘플링해야 하는가?"* $\rightarrow$ 사건 기반 가변 프레임 샘플링 도출 |
| **3. 병목 역전<br>(Bottleneck Targeting)** | 프로파일링을 통해 시스템 비용/지연의 80%를 차지하는 단 하나의 물리적 병목 공략 | GPU 연산 속도가 아닌 SRAM-HBM 메모리 접근 IO 병목 타겟팅 $\rightarrow$ *FlashAttention* |
| **4. 실패 모드 군집화<br>(Failure Cluster Synthesis)** | 벤치마크 오답 500개를 수집해 클러스터링하고, 가장 큰 오답 군집 하나를 해결하는 메커니즘 설계 | Momentseeker 분석 결과 찰나의 액션에서 오답 집중 $\rightarrow$ High-frame-rate Temporal Gating 설계 |

---

## 5. 지능의 본질: "압축(Compression)"과 "좋은 정의(Good Definition)"

그랜트 샌더슨(3Blue1Brown)과 정보이론 학자들이 역설하듯, **지능의 본질은 무한한 자연 데이터에서 본질적인 수학적 규칙을 추출하는 '압축'**에 있다.

$$K(x) = \min_{p} \left\{ |p| : U(p) = x \right\}$$

AI는 방대한 가설 공간을 탐색하고 패턴을 내삽(Interpolation)하는 데 압도적이다. 그러나 **"어떤 문제를 풀어야 가치가 있는가?"**, **"무엇이 인류에게 의미 있는 좋은 정의(Definition)인가?"**라는 질문은 AI 스스로 내릴 수 없다.

* 미적분학의 엄밀성을 세운 코시와 바이어슈트라스의 $\epsilon$-$\delta$ 정의
* 정보 시대를 연 클로드 섀넌의 엔트로피 정의 $H(X) = -\sum P(x) \log P(x)$
* 컴퓨터 과학의 기틀을 다진 앨런 튜링의 튜링 기계 정의

이 모든 위대한 도약은 단순한 '계산'의 산물이 아니라, 혼란스러운 현상에 **새로운 수학적 깃발을 꽂은 인간 연구자의 '의도(Intent)'와 '가치 판단'**이었다.

---

## 결론: 인간 연구자와 AI의 하이브리드 오케스트레이션

AI 시대에 연구자가 된다는 것은 AI와 경쟁하는 것이 아니다. **AI를 '가장 강력한 연구 실행 하네스'로 부리는 지적 지휘자(Orchestrator)**가 되는 것이다.

| 연구 단계 | 주체 | 구체적인 역할 및 책임 |
|---|---|---|
| **1. 거시적 질문 던지기** | 👤 인간 연구자 | 세상의 결정적 병목 정의 (*"긴 비디오의 연속적 시간 이해"*) |
| **2. 문헌 수집 & 맹점 분석** | 🤖 AI 시스템 | arXiv 수백 편 논문의 Limitations 섹션 자동 추출 및 분류 |
| **3. 가설 브레인스토밍** | 👤+🤖 협업 | 4대 벡터(이종 결합, 가정 파괴 등)를 활용한 후보 가설 생성 |
| **4. 코드 구현 및 실험 반복** | 🤖 AI 시스템 | 베이스라인 코드 수정, GPU 학습 실행, 에러 자가 디버깅 |
| **5. 결과 해석 및 가치 판단** | 👤 인간 연구자 | *"이 수치가 진짜 의미 있는가? 새로운 정의인가?"* 비판적 평가 |

1. **자동화할 것**: 논문 크롤링, 메타데이터 구조화, 베이스라인 코드 수정, GPU 학습 실행, 에러 디버깅, 차트 생성.
2. **인간 연구자로서 끝까지 붙들 것**:
   - 도메인 현상의 본질을 꿰뚫는 **'날카로운 문제 정의(Macro Why)'**
   - 벤치마크 오답 속에서 인과적 법칙을 발견하는 **'수학적 직관'**
   - AI가 만들어낸 수천 개의 결과물 중 인류의 지식을 확장할 진짜 통찰을 골라내는 **'심미안과 비판적 평가'**

지식 지도와 연구 아카이브는 단순한 노트 정리가 아니다. 내가 AI와 함께 사고하고, AI를 지휘하여 새로운 연구를 창출해 낼 **미래 연구소의 운영체제(OS)**다.
