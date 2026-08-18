# AI는 과연 연구(Research)를 할 수 있는가: 방법론, 실제 사례, 그리고 오토 리서치 아키텍처

## 서론: 문제 풀이(Search)를 넘어 미지의 발견(Discovery)으로

2026년 현재, 인공지능이 인간 연구자의 조수를 넘어 **'연구의 주체'**가 될 수 있는가에 대한 논쟁은 더 이상 철학적 담론에 머무르지 않는다. 사카나 AI(Sakana AI)의 *The AI Scientist*가 2026년 3월 *Nature* 본지에 시스템 아키텍처를 게재하고, 딥마인드의 *FunSearch*가 수십 년간 미해결이었던 극값 조합론(Cap Set Problem)의 수학적 경계를 갱신하면서, AI는 이미 실질적인 과학적 발견을 만들어내기 시작했다.

하지만 연구 현장에서 AI를 진정으로 활용하기 위해서는 막연한 기대나 회의론을 넘어 구체적인 질문에 답해야 한다:
1. **AI는 실제로 어떤 메커니즘으로 과학 연구를 수행하는가?**
2. **실제로 AI가 인류 지식의 프런티어를 확장한 구체적 성공 사례는 무엇인가?**
3. **연구의 전 과정을 자동화하려면 어떤 파이프라인과 하네스(Harness)를 구축해야 하는가?**
4. **논문을 어떻게 수집하고, 데이터베이스에 무엇을 어떤 구조로 저장해야 하는가?**
5. **무한한 가설의 바다에서 어떻게 '독창적인 연구 아이디어'를 체계적으로 도출할 것인가?**

이 글은 2027 대학원 진학 및 향후 독자적인 AI/ML 연구를 준비하며, AI 기반 연구 자동화(Automated Scientific Research)의 방법론과 청사진을 집대성한 기록이다.

---

## 1. 실제로 AI가 연구를 수행한 5대 실전 사례

AI가 단순히 계산기를 두드린 수준이 아니라, **기존 문헌을 분석하고 가설을 세워 새로운 과학적 결과를 도출한 실제 사례**들은 다음과 같다.

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

### 1) FunSearch (Nature 2023) — 코드 공간에서의 함수 탐색과 검증기
딥마인드의 FunSearch는 자연어로 수식을 직접 푸는 대신, **'문제를 해결하는 컴퓨터 프로그램(함수)'을 생성하고 평가(Evaluator)하는 유전적 피드백 루프**를 설계했다.
- **메커니즘**: 사전학습 LLM이 프로그램 후보를 작성 $\rightarrow$ 외부 샌드박스에서 컴파일 및 점수 측정 $\rightarrow$ 최고 득점 코드를 다음 프롬프트의 Exemplar로 주입.
- **성과**: 20년간 정체되어 있던 고차원 Cap Set 문제의 점근적 하한을 인류 최초로 갱신했으며, 인간이 읽고 해석할 수 있는 알고리즘 코드를 산출했다.

### 2) The AI Scientist (Sakana AI, Nature 2026) — 엔드투엔드 연구 수명주기 자동화
머신러닝 연구자가 아이디어를 내고 논문을 제출하기까지의 모든 과정을 단일 에이전트 시스템으로 통합했다.
- **메커니즘**: Semantic Scholar 검색을 통한 문헌 조사 $\rightarrow$ 아이디어 생성 및 필터링 $\rightarrow$ PyTorch 코드 수정 및 GPU 훈련 $\rightarrow$ 결과 시각화 $\rightarrow$ LaTeX 논문 작성 $\rightarrow$ LLM 자동 피어리뷰.
- **의의**: 단 $15 미만의 비용으로 실제 ICLR 워크숍 수준의 논문을 생성할 수 있음을 증명하며 '오토 리서치'의 개념적 기준을 세웠다.

### 3) GNoME & A-Lab (Nature 2023) — 물질 탐색과 로봇 실험실의 결합
그래프 신경망(GNN)을 활용해 결정 구조의 안정성을 예측하는 모델(GNoME)을 통해 220만 개의 신규 결정 구조를 발견했으며, 이 중 38만 개를 안정 후보군으로 분류했다. 버클리 연구소의 A-Lab(로봇 무인 합성 실험실)은 이 가설을 받아 인간 개입 없이 736개의 신소재를 실제로 합성해 냈다.

---

## 2. 연구 자동화(Auto-Research) 시스템의 5단계 파이프라인

체계적인 AI 연구 자동화 시스템은 다음과 같은 **5개 모듈의 폐루프(Closed-Loop)**로 구성된다.

```
       ┌────────────────────────────────────────────────────────┐
       │             [1. Literature Ingestion Engine]           │
       │    arXiv, OpenAlex, Semantic Scholar ──▶ GraphRAG     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             [2. Hypothesis & Ideation Agent]           │
       │    4대 아이디어 추출 벡터 (가정 파괴, 이종 결합, 병목 역전)  │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             [3. Execution & Experiment Harness]        │
       │    Docker 샌드박스 + GPU 스케줄러 + WandB 메트릭 로깅   │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             [4. Verifier & Evaluator (검증기)]         │
       │    Ground Truth, Unit Tests, Lean 4 형식 검증, Ablation│
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             [5. Synthesis & Manuscript Drafting]       │
       │    자동 LaTeX 작성, Matplotlib 차트, 한계점 명시      │
       └────────────────────────────────────────────────────────┘
```

### 1단계: 문헌 수집 및 지식 그래프 구축 (Literature Ingestion)
- 단순 텍스트 RAG가 아닌, 논문 간의 **인용 관계, 비교 기준(Baseline), 태스크-메트릭 튜플**을 구조화된 그래프로 저장.

### 2단계: 가설 생성기 (Hypothesis Generator)
- LLM에 무작정 "좋은 아이디어 내줘"라고 프롬프팅하는 것이 아니라, 문헌 그래프에서 발견된 **"기존 SOTA의 실패 케이스"**와 **"타 도메인의 성공 기법"**을 결합하도록 강제하는 제약 조건(Constrained Prompting) 부여.

### 3단계: 실행 및 실험 하네스 (Experiment Harness)
- 코드를 작성하고 격리된 Docker 컨테이너에서 GPU를 할당하여 학습 및 평가 수행.
- 에러 발생 시 Traceback을 읽고 자가 수정(Self-Debugging)하는 루프 포함.

### 4단계: 검증기 및 비평가 (Verifier & Critic)
- LLM의 환각(Hallucination)을 원천 차단하기 위해, 평가는 LLM의 자의적 판단이 아닌 **'외부 코드 실행 결과(Metric / Loss / Accuracy / Speedup)'** 또는 **'형식 검증기(Formal Verifier)'**로만 점수화.

### 5단계: 논문 작성 및 공유 (Synthesis & Dissemination)
- 실험 수치와 차트를 템플릿 LaTeX에 자동 바인딩하고, 주장의 과장을 막는 보수적 비평 에이전트(Devil's Advocate Critic)를 거쳐 완성.

---

## 3. 논문을 어떻게 수집하고, 무엇을 저장해야 하는가?

성공적인 연구 자동화의 80%는 **'데이터베이스에 논문을 어떤 형태로 저장해 두었는가'**에서 결정된다. 수천 편의 PDF를 그대로 벡터 DB에 넣는 것은 잡음만 늘릴 뿐이다.

### 📥 1) 논문 자동 수집 파이프라인
* **arXiv API**: 매일 특정 카테고리(`cs.AI`, `cs.CV`, `cs.CL`, `cs.LG`)의 최신 논문 메타데이터 및 전문(PDF / LaTeX 소스) 자동 크롤링.
* **Semantic Scholar / OpenAlex API**: 인용 수, 영향력 있는 피인용(Influential Citations), 저자 소속 연구실 추출.
* **Hugging Face Papers / Twitter/X AI 피드**: 커뮤니티에서 실시간으로 화제가 되는 코드 및 데모 추적.

### 🗄️ 2) 반드시 구조화해서 저장해야 할 6대 핵심 엔티티

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

> 💡 **가장 중요한 것은 `limitations_and_failure_modes` (한계와 실패 모드)의 저장이다.**  
> 모든 위대한 후속 연구는 이전 논문의 '한계점' 문단에서 출발한다.

---

## 4. 독창적인 연구 아이디어를 체계적으로 발굴하는 4대 벡터

아이디어는 하늘에서 떨어지는 영감이 아니라, **구조화된 사고 연산(Systematic Operators)**의 결과물이다.

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

### 1) 이종 결합 (Cross-Pollination / Domain Transfer)
* **원리**: A 분야에서 이미 검증된 강력한 기술을, 아직 그 기술이 도입되지 않은 B 분야의 병목에 이식한다.
* **실제 예시**:
  * LLM 텍스트 추론의 KV Cache 압축 기법 $\rightarrow$ 비디오 모델의 긴 시퀀스 메모리 병목에 적용 (*StreamKV*)
  * 소프트웨어의 작업 큐/메모리 계층 구조 $\rightarrow$ Web Agent의 과거 탐색 경험 재사용에 적용 (*Agent Workflow Memory*)

### 2) 기저 가정 파괴 (Assumption Inversion)
* **원리**: 업계의 모든 연구자들이 "당연하다"고 받아들이고 있는 암묵적 가정을 정면으로 반박해 본다.
* **실제 예시**:
  * "시각 토큰은 항상 고정된 $14 \times 14$ 패치 그리드로 쪼개야 하는가?" $\rightarrow$ 객체 중심 또는 시공간 가변 패치 연구 도출
  * "LLM은 모든 레이어에서 동일한 수의 파라미터를 계산해야 하는가?" $\rightarrow$ Early-exit 및 Mixture of Depths 연구 도출

### 3) 병목 역전 (Bottleneck Targeting)
* **원리**: 시스템 프로파일링을 통해 연산 시간, 메모리 대역폭, 전력 소모의 80%가 어디서 발생하는지 계측하고, 그 한 지점만을 극적으로 줄이는 전용 아키텍처를 설계한다.
* **실제 예시**:
  * FlashAttention (GPU SRAM과 HBM 간의 메모리 접근 IO 병목을 타겟팅)
  * CPO 및 실리콘 포토닉스 (구리선 인터커넥트의 대역폭·발열 병목 타겟팅)

### 4) 실패 모드 군집화 (Failure Cluster Synthesis)
* **원리**: 최신 SOTA 모델들이 특정 벤치마크에서 틀린 문제 500개를 수집하여 클러스터링(Error Analysis)한다. 가장 큰 오답 군집 하나를 정의하고, 이를 해결하기 위한 메커니즘을 역설계한다.
* **실제 예시**:
  * Momentseeker 벤치마크 분석 결과, MLLM들이 "빠르게 지나가는 찰나의 액션"에서 일관되게 시간 구간 예측에 실패함 $\rightarrow$ High-frame-rate Temporal Gating 메커니즘 가설 도출.

---

## 5. 지능의 본질: "압축(Compression)"과 "좋은 정의(Good Definition)"

3Blue1Brown(그랜트 샌더슨)과 정보이론 학자들이 강조하듯, **지능의 본질은 무한한 자연 데이터에서 본질적인 수학적 규칙을 추출하는 '압축'**에 있다.

$$K(x) = \min_{p} \{ |p| : U(p) = x \}$$

AI는 패턴을 내삽(Interpolation)하고 방대한 가설 공간을 초고속으로 탐색하는 데 탁월하다. 그러나 **"어떤 문제를 풀어야 하는가?"**, **"무엇이 인류에게 가치 있는 좋은 정의(Definition)인가?"**라는 질문은 AI 스스로 도출할 수 없다.

* 미적분학의 엄밀성을 세운 $\epsilon$-$\delta$ 정의
* 정보 시대를 연 섀넌의 엔트로피 $H(X) = -\sum P(x) \log P(x)$
* 컴퓨터 과학을 정의한 튜링 기계

이 모든 위대한 도약은 '계산'의 산물이 아니라, 혼란스러운 현상에 **새로운 수학적 깃발을 꽂은 인간 연구자의 '의도(Intent)'와 '가치 판단'**이었다.

---

## 결론: 2027년을 준비하는 나(정현우)의 연구 아키텍처

AI 시대에 연구자가 된다는 것은 AI와 경쟁하는 것이 아니다. **AI를 '가장 강력한 연구 실행 하네스'로 부리는 지적 지휘자(Orchestrator)**가 되는 것이다.

1. **자동화할 것**: 논문 크롤링, 메타데이터 구조화, 베이스라인 코드 실행, 하이퍼파라미터 튜닝, 정량적 메트릭 시각화.
2. **인간 연구자로서 끝까지 붙들 것**:
   - 도메인 현상의 본질을 꿰뚫는 **'날카로운 문제 정의'**
   - 벤치마크 오답 속에서 인과적 법칙을 발견하는 **'수학적 직관'**
   - AI가 만들어낸 수천 개의 결과물 중 인류의 지식을 확장할 진짜 통찰을 골라내는 **'심미안과 비판적 평가'**

지식 지도와 연구 아카이브는 단순한 노트 정리가 아니다. 내가 AI와 함께 사고하고, AI를 지휘하여 새로운 연구를 창출해 낼 **미래 연구소의 운영체제(OS)**다.
