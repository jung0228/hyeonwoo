# AI는 과연 연구(Research)를 할 수 있는가: 방법론, 실제 사례, 그리고 오토 리서치 아키텍처

## 서론: 문제 풀이(Search)를 넘어 미지의 발견(Discovery)으로

2026년 현재, 인공지능이 인간 연구자의 조수를 넘어 **'연구의 주체'**가 될 수 있는가에 대한 논쟁은 더 이상 철학적 담론에 머무르지 않는다. 사카나 AI(Sakana AI)의 *The AI Scientist*가 2026년 3월 *Nature* 본지에 시스템 아키텍처를 게재하고, 딥마인드의 *FunSearch*가 수십 년간 미해결이었던 극값 조합론(Cap Set Problem)의 수학적 경계를 갱신하면서, AI는 이미 실질적인 과학적 발견을 만들어내기 시작했다.

하지만 연구 현장에서 AI를 진정으로 활용하기 위해서는 막연한 기대나 회의론을 넘어 구체적인 질문에 답해야 한다:
1. **연구의 본질은 무엇이며, 좋은 연구 문제는 어떻게 정의되는가?**
2. **AI는 실제로 어떤 메커니즘으로 과학 연구를 수행해 왔는가? (실제 성공 사례)**
3. **인간은 주제를 어디까지 던져주며, 시스템은 무엇을 어떻게 수집·자동화하는가?**
4. **무한한 가설의 바다에서 어떻게 '독창적인 연구 아이디어'를 체계적으로 도출할 것인가?**
5. **인간 연구자는 어디에 집중해야 인지 부채(Cognitive Debt)를 피하고 본질적 가치를 남길 수 있는가?**

이 글은 2027 대학원 진학 및 향후 독자적인 AI/ML 연구를 준비하며, AI 기반 연구 자동화(Automated Scientific Research)의 방법론과 엔지니어링 청사진을 집대성한 기록이다.

---

## 1. 연구 문제 발굴의 3차원 좌표계 (The 3D Coordinate of Research)

학계에서 수많은 논문이 쏟아져 나오지만, 90% 이상의 논문이 잊히고 소수의 논문만이 거대한 파급력을 갖는 이유는 기술적 기교의 차이가 아니라 **"문제를 정의하는 시야의 깊이와 당위성"** 때문이다. 모든 위대한 연구는 다음 **3차원 좌표계의 교차점**에서 출발한다.

```
                       [ 축 1: 거시적 당위성 (Macro Why) ]
                   "이 문제를 풀면 세상/AI에 어떤 변화가 오는가?"
                                       │
                                       │
                                       ├───────────────────────────────┐
                                       │                               │
                                       ▼                               ▼
       [ 축 2: 기존 연구의 구조적 결함 ]                  [ 축 3: 최신 논문의 미해결 틈새 ]
        "과거 SOTA는 왜 여기서 실패했는가?"                "최신 논문은 무엇을 한계로 남겨두었는가?"
    (단순 스케일링으로 안 풀리는 구조적 병목)                 (Limitations & Failure Table의 맹점)
```

### 1) 축 1: 거시적 당위성 (Macro "Why It Matters")
> *"단순히 벤치마크 점수를 0.5% 올리는 것이 아니라, 왜 지금 이 문제를 반드시 풀어야 하는가?"*
- **접근법**: 이 병목이 해결되지 않았을 때 전체 AI 시스템(로보틱스, 에이전트, 멀티모달)이 어디서 멈춰 서는지 파악한다.
- **예시**: *"비디오 모델에서 프레임 처리 속도를 조금 올리자"*가 아니라, *"현재 AI는 VRAM 폭증 때문에 1분짜리 영상도 프레임을 뚝뚝 끊어서 본다. 이 시간적 연속성(Temporal Continuity)을 해결하지 못하면 실시간 자율주행도, 수술 보조 로봇도 불가능하다."*

### 2) 축 2: 기존 방법론의 구조적 결함 (Prior Art Pathology)
> *"과거의 쟁쟁한 모델들은 왜 이 문제 앞에서 무너졌는가?"*
- **접근법**: 이전 연구들의 실패 원인이 단순한 '데이터/컴퓨팅 부족' 때문인지, 아니면 **알고리즘의 근본적인 가정(Assumption)이 틀렸기 때문인지** 규명한다.

### 3) 축 3: 최신 논문의 미해결 틈새 (Frontier Blind Spots)
> *"올해 발표된 최신 SOTA 논문들은 어디까지 풀었고, 어디서 멈췄는가?"*
- **접근법**: 최신 논문의 Abstract보다 **`Limitations(한계점)` 섹션과 `Failure Cases(오답 분석)` 부록**을 집요하게 파고든다. 모든 위대한 후속 연구는 이전 논문의 한계점 문단에서 탄생한다.

---

## 2. 실제로 AI가 연구를 수행한 5대 실전 사례

AI가 단순히 계산을 보조한 수준을 넘어, **가설을 세우고 코드를 작성하여 새로운 과학적 결과를 도출한 대표 사례**들은 다음과 같다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AI 과학 연구의 5대 대표 사례                          │
├──────────────────┬──────────────────┬──────────────────┬────────────────────┤
│     프로젝트     │       기관       │    적용 도메인   │    핵심 연구 성과  │
├──────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ FunSearch (2023) │ Google DeepMind  │ 극값 조합론/수학 │ Cap Set 문제 해결, │
│                  │                  │ & 알고리즘 최적화│ 온라인 Bin-packing │
├──────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ The AI Scientist │ Sakana AI /      │ 머신러닝 시스템  │ 가설→코드→실험→논문│
│ (2024-2026)      │ Oxford Univ.     │ & 딥러닝 아키텍처│ 전 과정 자동화 ($15│
├──────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ GNoME (2023)     │ Google DeepMind  │ 무기 결정 재료학 │ 220만 종 신물질 발견│
│ & A-Lab          │ / Berkeley Lab   │ & 자동 무인 합성 │ 736종 실제 로봇합성│
├──────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ AlphaProof /     │ Google DeepMind  │ 형식 수학(Lean 4)│ IMO 2024 은/금메달 │
│ AlphaGeometry 2  │                  │ 및 정리 증명     │ 수준 형식 증명 완료│
├──────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ PaperQA2 (2024)  │ FutureHouse      │ 생명과학 문헌 분석│ 박사급 문헌 검색 및│
│ & WikiCrow       │                  │ 및 모순점 탐지   │ 초인적 과학 위키생성│
└──────────────────┴──────────────────┴──────────────────┴────────────────────┘
```

* **FunSearch (Nature 2023)**: LLM이 코드로 표현된 수학 함수를 생성하고 외부 평가기(Evaluator)가 점수를 매기는 유전적 피드백 루프로 20년간 풀리지 않던 Cap Set 문제의 수학적 하한을 갱신.
* **The AI Scientist (Nature 2026)**: Semantic Scholar 기반 문헌 조사 $\rightarrow$ 아이디어 생성 $\rightarrow$ PyTorch 코드 수정 및 GPU 훈련 $\rightarrow$ 시각화 $\rightarrow$ LaTeX 작성 $\rightarrow$ 자동 피어리뷰 전 과정을 $15 미만으로 자동 완결.
* **GNoME & A-Lab (Nature 2023)**: 그래프 신경망(GNN)으로 220만 개 신규 결정 구조를 예측하고, 무인 로봇 실험실(A-Lab)이 인간 개입 없이 736개 신소재를 합성.

---

## 3. 오토 리서치는 어떻게 작동하는가: 입력, 수집, 실행 파이프라인

연구 자동화는 추상적인 개념이 아닌 정교한 엔지니어링 아키텍처다.

### 🎯 1) 인간은 주제를 "어디까지" 던져주는가? (3가지 입력 레벨)

```
[ Level 1: 템플릿 기반 (Template-driven) ]  <-- 현재 가장 안정적인 실전 방식
인간 제공: 실행 가능한 베이스라인 코드 + 타겟 메트릭 + 초기 시드 아이디어 2~3개
AI의 역할: 코드 공간을 변형(Ablation)하며 수십 개의 파생 가설 자가 실험

[ Level 2: 목표 지향형 (Goal-oriented) ]     <-- AI Scientist v2 방식
인간 제공: 자연어 연구 질문 + 타겟 벤치마크 및 제약 조건
AI의 역할: arXiv/GitHub에서 적절한 베이스라인을 직접 검색·조립하여 실험

[ Level 3: 개방형 탐색 (Open-ended Discovery) ] <-- 차세대 프런티어
인간 제공: 광범위한 도메인 관심사 (예: "비디오 LLM의 추론 효율화")
AI의 역할: 최신 논문들의 '한계점'을 읽고 스스로 풀 문제를 정의
```

가장 성공적인 실전 방식인 **Level 1 템플릿**에서 인간 연구자가 시스템에 제공하는 인풋은 다음과 같다:
* `experiment.py`: 모델 학습 및 평가가 완결되는 베이스라인 코드
* `plot.py`: 결과를 시각화하는 차트 생성 코드
* `prompt.json`: 연구 배경 및 **시드 아이디어(Seed Ideas)** 2~3개  
  *(예: "Learning rate scheduler를 코사인 대신 지수형으로 변경", "어텐션 레이어 사이에 Skip-connection 추가")*
* **목표 메트릭(Target Metric)**: Validation Loss 최소화, 추론 속도 2배 향상 등

---

### 📥 2) 무엇을 어떻게 수집하고 구조화하는가?

성공적인 오토 리서치는 3가지 자산을 수집해 **연구 지식 그래프(GraphRAG)**로 인덱싱한다.

```
  [1. 최신 논문 & 인용 그래프]      [2. 실행 가능한 오픈소스 코드]      [3. 벤치마크 & 평가 데이터]
  arXiv API / Semantic Scholar     GitHub / Papers with Code        HuggingFace Datasets
               │                                │                                │
               └────────────────────────────────┼────────────────────────────────┘
                                                ▼
                                  [ 연구 지식 그래프 (GraphRAG) ]
                                  - 기존 방법의 구조적 한계점
                                  - 수식 및 아키텍처 모듈
                                  - 벤치마크별 SOTA 수치
```

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

### ⚙️ 3) 5단계 연구 자동화 폐루프 (The Closed-Loop Engine)

```
   [1. Ideation Agent] ──▶ [2. Novelty Filter] ──▶ [3. Code/Experiment Agent]
            ▲                                                  │
            │ (반복 개선)                                        ▼
   [5. Self-Review Agent] ◀── [4. Manuscript Generator] ◀── [GPU 샌드박스 실행]
```

1. **가설 생성 (Ideation)**: LLM이 템플릿과 문헌 그래프를 조합해 새로운 연구 가설을 수립.
2. **독창성 필터 (Novelty Filter - 핵심!)**: Semantic Scholar API로 자동 검색을 수행하여, 기존 논문들과의 임베딩 유사도가 **85% 이상이면 "이미 존재하는 연구"로 판정하고 폐기**.
3. **코드 수정 및 자가 디버깅 (Code & Self-Debugging)**: 코딩 에이전트가 `experiment.py`를 수정하고 Docker/GPU 샌드박스에서 실행. 에러 발생 시 Traceback을 읽고 스스로 수정(Self-Debugging) 반복.
4. **통계적 검증 (Ablation Verifier)**: 시드 반복 실험을 통해 성능 향상이 통계적으로 유의미한지($p < 0.05$) 검증.
5. **논문 작성 및 피어리뷰 (Drafting & Review)**: LaTeX 템플릿에 수치와 차트를 자동 바인딩하고, Reviewer Agent(ICLR 심사위원 프롬프트)가 채점 및 피드백 제공.

---

## 4. 독창적인 연구 아이디어를 발굴하는 4대 벡터

아이디어는 단순한 영감이 아니라 **구조화된 사고 연산(Systematic Operators)**의 결과다.

```
       [ 1. Cross-Pollination ]             [ 2. Assumption Inversion ]
     도메인 A의 검증된 해법 ──▶ 도메인 B       기존 관행/상식 의심 ──▶ 제약 완화
           (예: LLM KV Cache ──▶ Video LLM)         (예: 꼭 Dense Grid 토큰이어야 하는가?)
                         │                               │
                         ├───────────────┬───────────────┤
                         │               │
       [ 3. Bottleneck Targeting ]          [ 4. Failure Cluster Synthesis ]
     시스템의 80% 비용/지연 병목 공략         SOTA 벤치마크 오답 패턴 군집화
     (예: Prefill/Decode 분리, 광인터커넥트)      (예: Fine-grained 시간 추론 실패 전담)
```

1. **이종 결합 (Cross-Pollination)**: A 분야의 검증된 해법을 B 분야의 병목에 이식 (*예: LLM KV Cache 압축 $\rightarrow$ Video-LLM StreamKV*).
2. **기저 가정 파괴 (Assumption Inversion)**: 업계가 당연하게 여기는 상식을 의심 (*예: "비디오 토큰은 왜 항상 동일한 fps로 샘플링해야 하는가?"*).
3. **병목 역전 (Bottleneck Targeting)**: 프로파일링을 통해 시스템 비용/지연의 80%를 차지하는 단 하나의 물리적 병목을 타겟팅 (*예: FlashAttention*).
4. **실패 모드 군집화 (Failure Cluster Synthesis)**: 벤치마크 오답 500개를 분석해 가장 큰 오답 군집 하나를 해결하는 전용 메커니즘을 역설계.

---

## 5. 지능의 본질: "압축(Compression)"과 "좋은 정의(Good Definition)"

그랜트 샌더슨(3Blue1Brown)과 정보이론 학자들이 역설하듯, **지능의 본질은 무한한 자연 데이터에서 본질적인 수학적 규칙을 추출하는 '압축'**에 있다.

$$K(x) = \min_{p} \left\{ |p| : U(p) = x \right\}$$

AI는 방대한 가설 공간을 탐색하고 패턴을 내삽(Interpolation)하는 데 압도적이다. 그러나 **"어떤 문제를 풀어야 가치가 있는가?"**, **"무엇이 인류에게 의미 있는 좋은 정의(Definition)인가?"**라는 질문은 AI 스스로 내릴 수 없다.

* 미적분학의 엄밀성을 세운 $\epsilon$-$\delta$ 정의
* 정보 시대를 연 섀넌의 엔트로피 $H(X) = -\sum P(x) \log P(x)$
* 컴퓨터 과학을 정의한 앨런 튜링의 튜링 기계

이 모든 위대한 도약은 '계산'의 산물이 아니라, 혼란스러운 현상에 **새로운 수학적 깃발을 꽂은 인간 연구자의 '의도(Intent)'와 '가치 판단'**이었다.

---

## 결론: 인간 연구자와 AI의 하이브리드 오케스트레이션

AI 시대에 연구자가 된다는 것은 AI와 경쟁하는 것이 아니다. **AI를 '가장 강력한 연구 실행 하네스'로 부리는 지적 지휘자(Orchestrator)**가 되는 것이다.

```
┌───────────────────────────┬────────────────────────────────────────────────────────┐
│        단계 (Phase)       │                     역할 분담 (Role)                   │
├───────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. 거시적 질문 던지기     │ 👤 인간 (현우 님): 세상의 병목 정의 ("긴 비디오 메모리") │
│ 2. 문헌 수집 & 한계점 추출│ 🤖 AI: arXiv 100편 논문의 Limitations 섹션 자동 추출   │
│ 3. 가설 브레인스토밍      │ 👤+🤖 협업: 4대 벡터(이종 결합, 가정 파괴)로 아이디어 생성│
│ 4. 코드 구현 및 실험 반복 │ 🤖 AI: 베이스라인 코드 수정, GPU 학습 실행, 에러 디버깅│
│ 5. 결과 해석 및 가치 판단 │ 👤 인간: "이 수치가 진짜 의미가 있는가? 새로운 정의인가?"│
└───────────────────────────┴────────────────────────────────────────────────────────┘
```

1. **자동화할 것**: 논문 크롤링, 메타데이터 구조화, 베이스라인 코드 수정, GPU 학습 실행, 에러 디버깅, 차트 생성.
2. **인간 연구자로서 끝까지 붙들 것**:
   - 도메인 현상의 본질을 꿰뚫는 **'날카로운 문제 정의(Macro Why)'**
   - 벤치마크 오답 속에서 인과적 법칙을 발견하는 **'수학적 직관'**
   - AI가 만들어낸 수천 개의 결과물 중 인류의 지식을 확장할 진짜 통찰을 골라내는 **'심미안과 비판적 평가'**

지식 지도와 연구 아카이브는 단순한 노트 정리가 아니다. 내가 AI와 함께 사고하고, AI를 지휘하여 새로운 연구를 창출해 낼 **미래 연구소의 운영체제(OS)**다.
