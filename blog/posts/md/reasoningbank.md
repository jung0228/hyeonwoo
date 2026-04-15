---
title: "ReasoningBank: 실패에서 배우는 에이전트 메모리"
dek: "성공과 실패를 모두 추상화해 재사용 가능한 추론 전략을 쌓는 메모리 프레임워크 — 그리고 메모리와 테스트타임 스케일링의 시너지"
tags: [Agent, LLM]
date: 2026-04-15
readtime: 14
slug: reasoningbank
---

LLM 에이전트가 웹 브라우징이나 소프트웨어 엔지니어링 같은 복잡한 태스크를 수행할 때, 매번 새로운 태스크를 마치 처음 보는 것처럼 다룬다. 지난주에 같은 실수를 했어도, 그 경험은 어디에도 남지 않는다. **ReasoningBank** (ICLR 2026)는 이 문제를 정면으로 다룬다. 과거 성공과 실패 경험을 모두 추상화해 재사용 가능한 추론 전략으로 저장하고, 새 태스크에서 꺼내 쓰는 메모리 프레임워크다.

## 왜 기존 메모리 방식은 부족한가

기존 에이전트 메모리 연구는 크게 두 갈래로 나뉜다.

**Raw Trajectory 저장** (Synapse 등): 과거 에이전트의 행동 궤적을 통째로 저장해 유사 태스크에서 참조한다. 궤적이 너무 길고 노이즈가 많아 실제 유용한 패턴을 찾기 어렵다.

**Workflow/Procedure 저장** (AWM 등): 성공한 경험에서 절차를 추출해 저장한다. 하지만 *성공만* 저장하기 때문에, 실패에서 얻을 수 있는 "이건 하지 마라"는 교훈을 완전히 버린다.

두 방식 모두 **높은 수준의 전이 가능한 추론 패턴**을 뽑아내지 못한다는 공통적인 한계가 있다.

## ReasoningBank: 메모리 프레임워크

<figure>
<img src="img/reasoningbank/ReasoningBank.png" alt="ReasoningBank 개요">
<figcaption>ReasoningBank의 전체 흐름. 경험/궤적에서 구조화된 메모리 아이템을 추출하고, 새 태스크에서 검색해 활용한다.</figcaption>
</figure>

### 메모리 스키마

ReasoningBank의 메모리 아이템은 저수준 실행 세부사항을 추상화하고 전이 가능한 전략만 담도록 설계됐다. 각 아이템은 세 필드로 구성된다.

- **Title**: 핵심 전략을 한 줄로 요약하는 식별자 (예: "User-Specific Information Navigation")
- **Description**: 메모리 아이템에 대한 한 문장 요약
- **Content**: 추출된 추론 단계, 의사결정 근거, 운용 인사이트

이 구조 덕분에 메모리는 사람이 읽을 수 있으면서, LLM의 시스템 프롬프트에 주입하기도 쉽다.

### 세 단계의 클로즈드 루프

에이전트가 ReasoningBank와 함께 동작하는 방식은 세 단계로 이루어진다.

**① Memory Retrieval (검색)**: 새 태스크가 들어오면, 태스크 컨텍스트로 ReasoningBank를 쿼리해 임베딩 유사도 기반 top-k 메모리 아이템을 검색한다. 검색된 아이템들은 에이전트의 시스템 인스트럭션에 추가된다.

**② Memory Extraction (추출)**: 태스크가 끝나면 궤적에서 새 메모리 아이템을 추출한다. 이때 핵심이 바로 **LLM-as-a-Judge**: 정답 레이블 없이 에이전트 스스로 쿼리와 궤적을 보고 성공/실패를 판단한다. 성공 경험에서는 검증된 전략을, 실패 경험에서는 반례 신호(counterfactual signal)와 함정을 추출한다.

**③ Memory Consolidation (통합)**: 새로 추출된 아이템들을 ReasoningBank에 추가한다. 이 세 단계가 반복되며 에이전트는 태스크를 거칠수록 점점 더 강해진다.

<figure>
<img src="img/reasoningbank/intro.png" alt="누적 성공률 비교">
<figcaption>WebArena-Admin 서브셋에서 태스크가 늘어날수록 ReasoningBank를 사용한 에이전트의 누적 성공률이 "No Memory" 기준선을 꾸준히 앞선다.</figcaption>
</figure>

## MaTTS: 메모리-aware 테스트타임 스케일링

ReasoningBank가 "경험의 질"을 높인다면, **MaTTS (Memory-aware Test-Time Scaling)**는 "경험의 양"도 함께 키운다.

테스트타임 스케일링(TTS)의 아이디어는 단순하다 — 같은 태스크에 더 많은 계산을 투입해 더 많은 궤적을 생성하면 성능이 올라간다. 그런데 단순히 더 많은 궤적을 독립적으로 ReasoningBank에 집어넣는 것(Vanilla TTS)은 **대조 신호**를 활용하지 못해 비효율적이다.

MaTTS는 이 대조 신호를 적극 활용하도록 설계된 두 가지 방식을 제안한다.

<figure>
<img src="img/reasoningbank/matts.png" alt="MaTTS 비교">
<figcaption>(a) Vanilla TTS vs (b) Parallel Scaling vs (c) Sequential Scaling. MaTTS는 여러 궤적 사이의 대조 신호를 메모리 추출에 활용한다.</figcaption>
</figure>

### Parallel Scaling (병렬 스케일링)

같은 태스크에 대해 여러 궤적을 **동시에** 생성한다. 검색된 메모리 아이템의 가이던스 하에 각기 다른 탐색 경로를 생성하고, 이 궤적들을 **Self-Contrast** — 서로 비교하고 대조하는 방식 — 로 분석해 일관된 패턴은 강화하고 스퓨리어스한 해법은 걸러낸다.

### Sequential Scaling (순차 스케일링)

**Self-Refinement** 원칙에 따라 단일 궤적을 반복적으로 개선한다. 초기 완료 후 에이전트가 자신의 궤적을 재검토하고 다음 단계에서 개선을 시도한다. 이 과정에서 생성되는 중간 노트 — 추론 시도, 수정, 인사이트 — 도 메모리 신호로 활용된다.

## 실험 결과

### WebArena

<div class="callout">

**WebArena 전체 성공률 (Gemini-2.5-pro 기준)**

| 방법 | Overall SR | Avg Steps |
|------|-----------|-----------|
| No Memory | 46.7 | 8.8 |
| Synapse | 47.7 | 8.5 |
| AWM | 47.6 | 8.7 |
| **ReasoningBank** | **53.9** | **7.4** |
| + MaTTS | **56.3** | **7.1** |

ReasoningBank는 메모리 없는 기준선 대비 **+7.2 SR** 향상, 단계 수는 **1.4회 감소**.

</div>

세 가지 LLM 백본(Gemini-2.5-flash, Gemini-2.5-pro, Claude-3.7-sonnet) 모두에서 일관된 향상을 보인다는 점이 중요하다. AWM은 Multi-domain 서브셋에서 오히려 성능이 떨어지기도 하는데, ReasoningBank는 이 도메인 전이 설정에서도 강건하다.

### SWE-Bench-Verified

소프트웨어 엔지니어링 태스크에서도 동일한 패턴이 확인된다. Gemini-2.5-pro 기준 Resolve Rate가 54.0 → **57.4**로 향상되고, 평균 단계 수도 21.1 → **19.8**로 줄었다.

### MaTTS 스케일링 효과

<figure>
<img src="img/reasoningbank/scaling_mats.png" alt="스케일링 팩터 k에 따른 성능 변화">
<figcaption>Scaling factor k가 증가할수록 MaTTS의 성능이 안정적으로 향상된다. 메모리 없는 TTS는 훨씬 낮은 수준에서 불안정하게 증가한다.</figcaption>
</figure>

Parallel Scaling에서 k=1(49.7) → k=5(55.1), Sequential Scaling에서 k=1(49.7) → k=5(54.5)로 향상된다. 메모리 없는 TTS는 같은 k=5에서 겨우 42.2에 머문다.

흥미로운 관찰: Sequential Scaling은 소규모 k에서 빠르게 올라오지만 곧 포화된다. Parallel Scaling은 더 다양한 rollout을 제공하기 때문에 k가 커질수록 앞서간다.

## 분석

### 실패에서 실제로 배우는가?

<figure>
<img src="img/reasoningbank/ablation_failure.png" alt="실패 궤적 활용 ablation">
<figcaption>성공 궤적만 사용(초록)과 실패 포함(분홍) 비교. ReasoningBank만이 실패 포함 시 성능이 크게 향상된다.</figcaption>
</figure>

Synapse는 실패 추가 시 40.6 → 41.7로 미미하게 증가하고, AWM은 오히려 44.4 → 42.2로 **하락**한다. 이들은 실패를 활용하는 메커니즘 자체가 없기 때문이다. 반면 ReasoningBank는 46.5 → **49.7**로 대폭 향상된다. 실패를 노이즈가 아닌 구조화된 교훈으로 변환하는 설계 덕분이다.

### Emergent Behaviors: 전략이 진화한다

<figure>
<img src="img/reasoningbank/evolving_strategy.png" alt="전략의 진화 사례">
<figcaption>"User-Specific Information Navigation" 메모리 아이템이 태스크를 거치며 어떻게 진화하는지를 보여주는 케이스 스터디.</figcaption>
</figure>

ReasoningBank에서 가장 흥미로운 현상은 메모리 아이템이 단순히 쌓이는 게 아니라 **진화**한다는 점이다.

동일한 전략 "User-Specific Information Navigation"의 메모리 아이템이 시간에 따라 어떻게 변화하는지를 추적하면:

1. **초기** — 단순한 절차적 실행 전략 ("Next Page 링크를 찾아 클릭")
2. **중기** — 적응적 자기성찰 ("식별자를 재확인해 단순 실수 줄이기")
3. **후기** — 체계적 점검 전략 ("검색/필터 기능을 먼저 확인 후 결과 도출")
4. **성숙** — 복합적 추론 전략 ("태스크 요구사항과 교차 참조해 옵션 재평가")

저수준 액션 규칙에서 고수준 추론 패턴으로 발전하는 이 과정은 강화학습에서 나타나는 학습 역학과 유사하다고 논문은 해석한다.

### LLM-as-a-Judge의 노이즈 강건성

성공/실패 판단을 정답 레이블 없이 LLM 스스로 한다는 점이 불안할 수 있다. 실제 judge 정확도는 72.7%로 측정됐다. 논문은 judge 정확도를 50%~100%로 시뮬레이션한 실험에서, **70%~90% 범위에서는 성능 차이가 거의 없음**을 보인다. 즉 완벽하지 않은 judge라도 실용적으로 충분하다.

### 효율성의 원천

성공 케이스와 실패 케이스의 단계 수를 따로 분석하면, ReasoningBank의 단계 감소가 주로 **성공한 경우에서 훨씬 크게 나타난다** (최대 26.9% 감소). 이는 단순히 빨리 포기하는 게 아니라, 성공 경로를 더 직접적으로 따라가는 능력이 향상됐음을 의미한다.

## 핵심 기여 요약

ReasoningBank는 세 가지 점에서 기존 에이전트 메모리 연구와 구별된다.

**실패의 활용**: 성공만 저장하던 기존 방식과 달리, 실패 경험을 구조화된 반례 신호로 변환한다. 실패를 포함할 때 성능이 크게 향상되는 반면, 기존 방법들은 오히려 악화되는 점이 이를 방증한다.

**추상화 수준**: Raw trajectory나 절차(workflow) 대신 전이 가능한 추론 패턴을 title+description+content 스키마로 저장한다. 새 태스크에서도 재사용 가능한 고수준 전략이 된다.

**메모리와 스케일링의 시너지**: MaTTS는 더 많은 계산이 더 좋은 메모리를 만들고, 더 좋은 메모리가 스케일링을 더 효과적으로 만드는 선순환을 확립한다. 이를 논문은 *"memory-driven experience scaling as a new scaling dimension, enabling agents to self-evolve with emergent behaviors naturally arise"* 라고 표현한다.
