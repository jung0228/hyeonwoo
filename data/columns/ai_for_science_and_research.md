# AI가 연구를 할 수 있는가

## 프롤로그: IMO 금메달의 착시와 두 가지 지능

2026년, 인공지능은 인류 최고의 두뇌들이 겨루는 국제수학올림피아드(IMO) 문제를 실시간으로 풀어내며 금메달 수준의 성취를 증명했다. 1억 개가 넘는 단백질 3차원 구조를 예측하고(AlphaFold), 수백만 줄의 소프트웨어 버그를 자동으로 패치하는 에이전트가 일상이 되었다.

그러나 학계와 최전선 연구소의 질문은 이제 전혀 다른 차원으로 향하고 있다.

> **"체스와 수학 올림피아드 문제를 완벽히 푸는 AI는, 과연 뉴턴이나 아인슈타인처럼 인류의 새로운 패러다임을 여는 '연구(Research)'를 해낼 수 있는가?"**

답을 내리기 위해서는 먼저 **'정해진 룰 안에서의 탐색(Search)'**과 **'새로운 개념과 세계관을 창조하는 발견(Discovery)'** 사이의 근본적인 간극을 직시해야 한다.

---

## 1장. 지능의 본질: 왜 연구는 '압축'과 '정의'의 싸움인가?

수학 교육자 그랜트 샌더슨(3Blue1Brown)과 이론물리학자들은 지능의 본질을 **'압축(Compression)'**이라는 한 단어로 요약한다.

자연계는 매초 무한대에 가까운 고차원 데이터를 쏟아낸다. 행성들의 불규칙한 궤적 데이터 속에서 케플러는 3가지 법칙을 뽑아냈고, 뉴턴은 이를 중력 방정식 $F = G \frac{m_1 m_2}{r^2}$이라는 단 한 줄의 수식으로 압축했다.

$$K(x) = \min_{p} \left\{ |p| : U(p) = x \right\}$$

콜모고로프 복잡도 $K(x)$ 관점에서 현대 딥러닝(LLM) 역시 인터넷의 방대한 인류 지식을 수천억 개의 파라미터로 가장 손실 없이 압축하는 거대한 압축기다.

하지만 여기서 결정적인 차이가 발생한다:
* **내삽적 압축(Interpolation)**: 기존 데이터 분포 내에서 가장 가능성 높은 패턴을 조합하고 요약하는 일 $\rightarrow$ *현재의 AI가 신(God)의 경지에 오른 영역*
* **외삽적 정의(Extrapolation & Definition)**: 기존 공리계를 깨뜨리고 관측된 적 없는 새로운 개념 체계를 선언하는 일 $\rightarrow$ *진정한 과학적 발견(Scientific Discovery)*

```
[ 내삽적 압축 (AI의 영역) ] ──▶ 수만 개의 논문과 실험 데이터에서 패턴 추출
                                                │
                                                ▼ (지적 도약의 순간)
[ 외삽적 정의 (인간의 영역) ] ──▶ 혼란스러운 현상에 '새로운 깃발(개념)'을 꽂는 행위
```

수학사와 과학사를 돌아보면, 문명을 바꾼 위대한 도약은 복잡한 계산을 잘 해냈을 때가 아니라 **'새롭고 우아한 정의(Good Definition)'**를 도입했을 때 일어났다.

| 역사적 대발견 | 도입된 '좋은 정의(Good Definition)' | 인류 지식에 미친 영향 |
|---|---|---|
| **미적분학의 엄밀화** | 코시·바이어슈트라스의 **$\epsilon$-$\delta$ 극한 정의** | 직관에 의존하던 무한소 개념을 엄밀한 수학 체계로 확립 |
| **정보 시대의 개막** | 클로드 섀넌의 **엔트로피 정의** $H(X) = -\sum P(x) \log P(x)$ | 불확실성을 정량화하여 현대 통신, 압축, 딥러닝 손실 함수의 근간 완성 |
| **컴퓨터 과학의 탄생** | 앨런 튜링의 **튜링 기계(Turing Machine) 정의** | '계산이란 무엇인가'를 수학적으로 규정하여 컴퓨터 문명 개막 |

이러한 정의들은 방대한 계산의 부산물이 아니라, **"수많은 잡음 속에서 인류가 앞으로 어떤 개념에 집중해야 하는가"**를 선언한 인간 연구자의 의도(Intent)와 가치 판단이었다.

---

## 2장. 오토 리서치의 실전: AI는 지금 어디까지 진짜 연구를 해냈는가?

그렇다면 AI는 과학 연구에서 무력한가? 결코 그렇지 않다. **인간이 '명확한 검증기(Verifier)'와 '베이스라인 템플릿'을 제공했을 때, AI는 인간이 수십 년 걸릴 가설 탐색 공간을 단 며칠 만에 돌파한다.**

| 프로젝트 | 연구 기관 | 적용 도메인 | 핵심 메커니즘 | 실질적 연구 성과 |
|---|---|---|---|---|
| **FunSearch**<br>(Nature 2023) | Google DeepMind | 극값 조합론 &<br>알고리즘 최적화 | LLM 프로그램 생성 $\leftrightarrow$ 외부 샌드박스 평가기 유전 피드백 루프 | 20년간 미해결이던 Cap Set 문제의 수학적 하한 갱신 |
| **The AI Scientist**<br>(Nature 2026) | Sakana AI &<br>Oxford Univ. | 머신러닝 시스템 &<br>딥러닝 아키텍처 | Semantic Scholar 문헌 조사 $\rightarrow$ 코드 수정 $\rightarrow$ GPU 훈련 $\rightarrow$ 논문 작성 $\rightarrow$ 자동 피어리뷰 | $15 미만 비용으로 ICLR 워크숍 수준 연구 논문 자동 완결 |
| **GNoME & A-Lab**<br>(Nature 2023) | Google DeepMind &<br>Berkeley Lab | 무기 결정 재료학 &<br>자동 무인 합성 | 그래프 신경망(GNN) 결정 구조 예측 $\rightarrow$ 무인 로봇 실험실 연동 | 220만 종 신물질 발견 및 736종 실제 로봇 합성 성공 |
| **AlphaProof &<br>AlphaGeometry 2**<br>(2024) | Google DeepMind | 형식 수학(Formal Math)<br>& 정리 증명 | 형식 수학 언어(Lean 4) + 강화학습 트리 탐색 | IMO 2024 은/금메달 수준의 고난도 복합 정리 증명 완료 |
| **PaperQA2**<br>(2024) | FutureHouse | 생명과학 문헌 분석 &<br>지식 그래프 합성 | 인용 네트워크 추적 + 고정밀 과학 RAG + 모순점 자동 탐지 | 박사급 연구원 능가하는 초인적 문헌 검색 및 WikiCrow 생성 |

---

## 3장. 오토 리서치 엔지니어링: 인간의 입력 레벨과 5단계 폐루프

연구 자동화는 추상적 비유가 아닌, 입력과 출력이 정의된 소프트웨어 파이프라인이다.

### 1) 인간은 주제를 "어디까지" 던져주는가?

| 레벨 (Level) | 인간 연구자가 제공하는 입력 | AI 시스템의 역할 | 대표 사례 |
|---|---|---|---|
| **Level 1: 템플릿 기반<br>(Template-driven)** | • 실행 가능한 베이스라인 코드 (`experiment.py`)<br>• 결과 시각화 코드 (`plot.py`)<br>• 배경 및 시드 아이디어 2~3개 (`prompt.json`)<br>• 타겟 메트릭 (Val Loss, Latency) | 코드 공간을 체계적으로 변형(Ablation)하며 수십 개의 파생 가설 자가 실험 | Sakana AI v1,<br>FunSearch |
| **Level 2: 목표 지향형<br>(Goal-oriented)** | • 자연어 연구 질문<br>• 타겟 벤치마크 및 하드웨어/비용 제약 조건 | arXiv/GitHub에서 적절한 베이스라인을 직접 검색·조립하여 실험 파이프라인 구성 | Sakana AI v2,<br>Co-Scientist |
| **Level 3: 개방형 탐색<br>(Open-ended Discovery)** | • 광범위한 도메인 관심사 (예: "비디오 LLM 효율화") | 최신 학회 논문들의 한계점을 읽고 스스로 풀 가치가 있는 문제와 벤치마크 정의 | 차세대 연구 프런티어 |

---

### 2) 5단계 연구 자동화 폐루프 (The Closed-Loop Engine)

```
   [1. Ideation Agent] ──▶ [2. Novelty Filter] ──▶ [3. Code/Experiment Agent]
            ▲                                                  │
            │ (반복 개선)                                        ▼
   [5. Self-Review Agent] ◀── [4. Manuscript Generator] ◀── [GPU 샌드박스 실행]
```

| 단계 | 담당 에이전트 | 핵심 동작 및 자동화 원리 |
|---|---|---|
| **1. 가설 생성** | Ideation Agent | 템플릿 코드와 문헌 지식 그래프를 결합해 구체적인 개선 가설 수립 |
| **2. 독창성 검증** | Novelty Filter | Semantic Scholar API로 자동 검색 $\rightarrow$ 기존 논문과 **임베딩 유사도 85% 이상 시 중복으로 판정하고 즉시 폐기** |
| **3. 코드 실행 & 디버깅** | Code & Experiment Agent | 코딩 에이전트가 `experiment.py`를 수정하고 Docker/GPU 샌드박스에서 실행 $\rightarrow$ 에러 발생 시 Traceback을 읽고 자가 디버깅(Self-Debugging) 반복 |
| **4. 통계적 검증** | Ablation Verifier | 동일 시드 반복 실험(3~5회)으로 성능 향상이 통계적으로 유의미한지($p < 0.05$) 검증 |
| **5. 논문 작성 & 심사** | Drafting & Review Agent | LaTeX 템플릿에 수치와 차트를 자동 바인딩하고, ICLR 기준 심사 에이전트가 점수 및 비평 제공 |

---

## 4장. 연구자의 실전 나침반: 3차원 좌표계와 4대 발굴 벡터

AI에게 '무엇을 탐색시킬 것인가'를 결정하는 것은 연구자의 몫이다. 훌륭한 문제는 다음 3가지 좌표축과 4대 사고 연산자에서 나온다.

### 1) 연구 문제 발굴의 3차원 좌표계

| 차원 (Dimension) | 핵심 질문 | 연구자가 파고들어야 할 지점 | 실전 예시 |
|---|---|---|---|
| **축 1: 거시적 당위성<br>(Macro Why)** | "이 문제를 풀면 세상과 AI에 어떤 변화가 오는가?" | 이 병목이 안 풀렸을 때 전체 시스템(로봇, 에이전트)이 멈춰 서는 결정적 지점 | *"1분짜리 영상도 VRAM이 터져 뚝뚝 끊어 보는 문제를 못 풀면 실시간 자율주행과 수술 로봇은 불가능하다."* |
| **축 2: 기존 연구의 결함<br>(Prior Art Pathology)** | "과거 SOTA는 왜 이 문제 앞에서 실패했는가?" | 단순 데이터/컴퓨팅 부족이 아닌, 알고리즘의 **근본적인 가정(Assumption)의 오류** 규명 | *"균등 프레임 샘플링은 연산량은 줄이지만, 1초 미만의 결정적 액션을 통째로 누락하는 태생적 결함이 있다."* |
| **축 3: 최신 논문의 맹점<br>(Frontier Blind Spot)** | "올해 나온 최신 논문들은 어디서 멈췄는가?" | 논문의 Abstract보다 **`Limitations` 섹션과 `Failure Cases` 부록**을 집요하게 분석 | *"LongVALE는 옴니모달을 달성했으나 영상이 30분을 넘으면 전후 인과관계 추론에서 환각을 일으킨다."* |

---

### 2) 독창적인 연구 아이디어를 발굴하는 4대 벡터

| 발굴 벡터 (Vector) | 핵심 원리 | 실전 적용 예시 |
|---|---|---|
| **1. 이종 결합<br>(Cross-Pollination)** | A 분야에서 검증된 강력한 기법을 아직 도입되지 않은 B 분야의 병목에 이식 | LLM 추론의 KV Cache 압축 기법 $\rightarrow$ 비디오 모델의 장기 메모리 병목에 적용 (*StreamKV*) |
| **2. 기저 가정 파괴<br>(Assumption Inversion)** | 업계 연구자들이 "당연하다"고 믿고 있는 암묵적 상식을 정면으로 의심 | *"비디오 프레임은 왜 항상 일정한 fps로 샘플링해야 하는가?"* $\rightarrow$ 사건 기반 가변 프레임 샘플링 도출 |
| **3. 병목 역전<br>(Bottleneck Targeting)** | 프로파일링을 통해 시스템 비용/지연의 80%를 차지하는 단 하나의 물리적 병목 공략 | GPU 연산 속도가 아닌 SRAM-HBM 메모리 접근 IO 병목 타겟팅 $\rightarrow$ *FlashAttention* |
| **4. 실패 모드 군집화<br>(Failure Cluster Synthesis)** | 벤치마크 오답 500개를 수집해 클러스터링하고, 가장 큰 오답 군집 하나를 해결하는 메커니즘 설계 | Momentseeker 분석 결과 찰나의 액션에서 오답 집중 $\rightarrow$ High-frame-rate Temporal Gating 설계 |

---

## 5장. 연구자의 데이터베이스: 무엇을 수집하고 저장할 것인가?

수천 편의 논문 PDF를 그대로 저장하는 것은 잡음만 늘릴 뿐이다. **모든 위대한 후속 연구는 이전 논문의 `Limitations(한계점)`에서 출발한다.**

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

## 에필로그: 지적 지휘자(Orchestrator)의 탄생

AI가 연구의 전 과정을 잠식해 들어가는 것처럼 보이는 시대일수록, 연구자의 본질은 더욱 명확해진다:

> **"실행(Execution)의 가치는 제로에 수렴하고, 의도(Intent)와 평가(Evaluation)의 가치는 무한대로 수렴한다."**

| 연구 단계 | 주체 | 구체적인 역할 및 책임 |
|---|---|---|
| **1. 거시적 질문 던지기** | 👤 인간 연구자 | 세상의 결정적 병목 정의 (*"긴 비디오의 연속적 시간 이해"*) |
| **2. 문헌 수집 & 맹점 분석** | 🤖 AI 시스템 | arXiv 수백 편 논문의 Limitations 섹션 자동 추출 및 분류 |
| **3. 가설 브레인스토밍** | 👤+🤖 협업 | 4대 벡터(이종 결합, 가정 파괴 등)를 활용한 후보 가설 생성 |
| **4. 코드 구현 및 실험 반복** | 🤖 AI 시스템 | 베이스라인 코드 수정, GPU 학습 실행, 에러 자가 디버깅 |
| **5. 결과 해석 및 가치 판단** | 👤 인간 연구자 | *"이 수치가 진짜 의미 있는가? 새로운 정의인가?"* 비판적 평가 |

대학원 진학을 준비하며 AI/ML을 연구하는 나의 여정은 단순히 최신 모델을 사용하는 사용자에 머무르는 것이 아니다. AI를 가장 강력한 지적 탐색의 하네스로 부리고, **인간의 인지적 한계를 확장하는 가장 우아한 질문과 정의를 던지는 지휘자(Orchestrator)**로 성장하는 것—이것이 내가 지식 지도를 구축하고 연구를 이어가는 궁극적인 이유다.
