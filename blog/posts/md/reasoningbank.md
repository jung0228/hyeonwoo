---
title: "ReasoningBank: 실패에서 배우는 에이전트 메모리"
dek: "성공과 실패를 모두 추상화해 재사용 가능한 추론 전략을 쌓는 메모리 프레임워크 — 그리고 메모리와 테스트타임 스케일링의 시너지"
desc: "raw trajectory나 성공 workflow 대신, 실패까지 포함한 고수준 reasoning strategy를 메모리로 저장하면 무엇이 달라질까? ReasoningBank는 그 질문에 가장 직접적으로 답한다."
tags: [Agent, LLM]
date: 2026-04-15
readtime: 15 min read
slug: reasoningbank
---

## 왜 기존 메모리는 에이전트를 진화시키지 못할까

<p><strong>흐름:</strong> 에이전트의 반복적 실패 → raw trajectory / workflow 메모리의 한계 → ReasoningBank의 문제 재정의</p>

### 메모리가 있어도 에이전트는 같은 실수를 반복한다

<mark>ReasoningBank의 출발점은 단순하다. 오늘 실패한 에이전트가 내일도 비슷하게 실패한다면, 그 시스템은 사실상 경험을 축적하지 못하고 있는 것이다.</mark>

웹 브라우징, 소프트웨어 엔지니어링, 컴퓨터 사용 같은 long-horizon agent task에서 LLM 에이전트는 이미 꽤 인상적인 성능을 보인다. 하지만 이런 에이전트들은 대개 태스크를 episode 단위로 처리한다. 지난 태스크에서 얻은 통찰이 다음 태스크의 정책으로 제대로 이어지지 않는다.

논문은 이 문제를 아주 직접적으로 짚는다.

*"By approaching each new task in isolation, they are doomed to repeat similar errors observed in the past."*

문제는 메모리가 없어서가 아니다. 이미 많은 연구가 과거 경험을 저장한다. 하지만 저장 방식이 약하다. 한쪽은 raw trajectory를 그대로 쌓아두고, 다른 한쪽은 성공한 workflow만 정리한다. 전자는 너무 길고 시끄럽고, 후자는 실패에서 배울 기회를 버린다.

### raw trajectory와 success-only workflow는 둘 다 반쪽짜리다

<mark>ReasoningBank가 겨냥하는 건 "무엇을 저장할까"가 아니라 "어떤 수준으로 추상화해서 저장할까"다.</mark>

기존 trajectory memory는 기록으로서 충실하지만 전략으로서 둔탁하다. 성공과 무관한 우회, 시행착오, 환경 잡음까지 함께 딸려온다. 반대로 workflow memory는 더 정제돼 있지만, 성공한 절차에 치우쳐 있어서 "이렇게 하면 실패한다"는 교훈을 거의 남기지 못한다.

논문은 이를 두 가지 한계로 요약한다.

*"They lack the ability to distill higher-level, transferable reasoning patterns."*

*"They leave the valuable lessons from an agent's own failures largely underexplored."*

즉 중요한 건 특정 웹페이지에서 어떤 버튼을 눌렀느냐가 아니라, 어떤 reasoning pattern이 미래 태스크로 옮겨갈 수 있느냐다. ReasoningBank는 바로 그 수준의 메모리를 만들겠다고 선언한다.

<figure>
<img src="img/reasoningbank/intro.jpg" alt="Figure 1">
<figcaption><strong>Figure 1</strong> — WebArena-Admin 서브셋에서 ReasoningBank를 쓴 에이전트는 태스크가 쌓일수록 누적 성공률이 기준선보다 더 빠르게 올라간다. 이 논문의 메시지는 "메모리가 있으면 좋다"가 아니라, 경험을 reusable reasoning strategy로 바꾸는 메모리만이 agent를 실제로 진화시킨다는 것이다.</figcaption>
</figure>

## ReasoningBank는 무엇을 저장하나

<p><strong>흐름:</strong> memory schema → retrieval/extraction/consolidation의 closed loop → 실패를 구조적 신호로 쓰는 방식</p>

### 메모리 아이템은 행동 기록이 아니라 추론 전략이다

<mark>ReasoningBank의 핵심 설계는 메모리 단위를 trajectory가 아니라 reasoning item으로 바꾼다는 데 있다.</mark>

메모리 아이템은 세 필드로 구성된다.

- `Title`: 전략을 압축한 이름
- `Description`: 한 문장 요약
- `Content`: 실제 reasoning hint, decision rationale, operational insight

이 구조는 중요하다. 너무 짧으면 검색은 쉽지만 실제 행동을 못 바꾸고, 너무 길면 raw trajectory와 다를 바가 없다. ReasoningBank는 이 중간지점을 노린다. 사람이 읽어도 이해 가능하고, LLM system instruction 안에 넣어도 부담이 적은 형태다.

*"Memory items in ReasoningBank are designed ... to abstract away low-level execution details while preserving transferrable reasoning patterns and strategies."*

### 이 시스템은 retrieval보다 closed loop가 더 중요하다

<mark>ReasoningBank는 정적인 메모리 저장소가 아니라, retrieval → extraction → consolidation이 반복되는 self-improving loop다.</mark>

새 태스크가 들어오면 먼저 현재 쿼리와 유사한 메모리 아이템을 검색해 system instruction에 주입한다. 태스크가 끝나면 이번 trajectory를 다시 읽고 새로운 memory item을 뽑는다. 마지막으로 그 아이템을 bank에 합친다. 이 세 단계가 순환하면서 메모리 자체가 agent의 장기적인 경험 저장소가 된다.

논문은 이 단계를 아주 명료하게 정의한다.

*"The integration proceeds in three steps: (i) memory retrieval, (ii) memory extraction, and (iii) memory consolidation."*

즉 ReasoningBank는 "과거를 꺼내 보는 장치"일 뿐 아니라, "현재 경험을 미래 전략으로 바꾸는 장치"다.

<figure>
<img src="img/reasoningbank/ReasoningBank.jpg" alt="Figure 2">
<figcaption><strong>Figure 2</strong> — ReasoningBank의 전체 구조. 새 태스크에서는 관련 memory item을 검색해 쓰고, 태스크 종료 후에는 trajectory를 다시 분석해 success insight 또는 failure reflection을 memory item으로 바꾼다. 이 item이 다시 bank에 추가되면서, 메모리는 정적 기록이 아니라 시간이 갈수록 더 풍부해지는 폐루프 시스템이 된다.</figcaption>
</figure>

### 실패를 저장하는 게 아니라, 실패에서 교훈을 추출한다

<mark>이 논문에서 가장 중요한 차별점은 실패를 단순한 negative sample이 아니라 counterfactual signal로 다룬다는 점이다.</mark>

성공 trajectory는 "무엇이 먹혔는가"를 알려준다. 실패 trajectory는 "어디서 reasoning이 잘못 꺾였는가"를 보여준다. ReasoningBank는 이 둘을 모두 메모리 원천으로 본다. 다만 실패를 raw form으로 저장하지 않는다. 대신 LLM-as-a-Judge가 trajectory를 success/failure로 판정하고, failure의 경우에는 pitfall과 preventive lesson을 추출한다.

이 관점 전환이 중요하다. 실패 사례를 그대로 넣으면 노이즈가 많지만, 실패를 반사실적 교훈으로 바꾸면 오히려 future guardrail이 된다.

*"Successful experiences contribute validated strategies, while failed ones supply counterfactual signals and pitfalls that help sharpen guardrails."*

## MaTTS: 메모리와 테스트타임 스케일링을 묶는 법

<p><strong>흐름:</strong> vanilla TTS의 한계 → parallel self-contrast → sequential self-refinement → memory-driven scaling이라는 관점</p>

### 더 많은 rollout이 자동으로 더 좋은 메모리를 만들지는 않는다

<mark>ReasoningBank만으로도 강하지만, 논문은 여기서 한 걸음 더 나가 test-time scaling과 메모리를 결합한다.</mark>

테스트타임 스케일링(TTS)은 같은 태스크에 더 많은 추론 자원을 투입해 성능을 높이는 방식이다. 그런데 interactive agent setting에서는 여러 trajectory를 많이 생성하는 것만으로는 충분하지 않다. 같은 문제에 대해 왜 어떤 경로는 실패하고 어떤 경로는 성공했는지, 그 contrastive signal을 메모리 생성에 활용해야 한다.

논문은 vanilla TTS를 비판적으로 본다.

*"This vanilla form is suboptimal because it does not leverage inherent contrastive signal that arises from redundant exploration on the same problem."*

그래서 Memory-aware Test-Time Scaling, 즉 MaTTS가 등장한다.

### Parallel Scaling은 여러 trajectory를 서로 비교하게 만든다

<mark>Parallel MaTTS의 아이디어는 간단하다. 같은 태스크의 여러 시도를 병렬로 만들고, 그 차이를 통해 더 믿을 만한 memory를 추출하는 것이다.</mark>

같은 쿼리에서 여러 trajectory를 생성하면, 어떤 reasoning pattern은 일관되게 성공 쪽에 붙고 어떤 패턴은 실패 쪽에 붙는다. Parallel Scaling은 이 self-contrast를 이용해 우연한 해법은 걸러내고, 여러 경로에서 반복되는 전략을 memory item으로 승격시킨다.

즉 양적인 scaling이 곧바로 질적인 memory curation으로 이어지도록 만드는 장치다.

### Sequential Scaling은 단일 trajectory 내부에서 refinement를 일으킨다

<mark>Sequential MaTTS는 여러 시도를 비교하는 대신, 한 trajectory를 반복적으로 재검토하며 intermediate reasoning signal까지 회수한다.</mark>

이 방식은 self-refinement에 가깝다. 에이전트는 한 번 생성한 trajectory를 다시 읽고, 무엇이 부족했는지, 어떤 지점을 수정해야 하는지 점검한다. 이때 최종 답뿐 아니라 중간 correction note 자체가 memory signal이 된다.

둘의 차이는 분명하다.

- Parallel: 여러 경로 사이의 차이에서 일반화 가능한 패턴 추출
- Sequential: 한 경로 안의 수정 과정에서 정제된 reasoning signal 추출

<figure>
<img src="img/reasoningbank/matts.jpg" alt="Figure 3">
<figcaption><strong>Figure 3</strong> — (a) vanilla TTS는 rollout 수만 늘릴 뿐 trajectory 간 대조 신호를 거의 쓰지 못한다. (b) Parallel MaTTS는 여러 trajectory를 self-contrast해 더 신뢰도 높은 memory를 만든다. (c) Sequential MaTTS는 같은 trajectory를 반복적으로 self-refine하며 correction signal을 memory로 회수한다.</figcaption>
</figure>

## 실험에서 가장 중요한 결과

<p><strong>흐름:</strong> WebArena / SWE-Bench 성능 → MaTTS의 scaling 효과 → memory와 scaling의 시너지</p>

### ReasoningBank 자체만으로도 baseline을 안정적으로 넘는다

<mark>이 논문은 단지 "메모리가 있으면 조금 낫다" 수준이 아니라, 여러 backbone과 여러 태스크에서 일관된 우위를 보여준다.</mark>

WebArena에서 Gemini-2.5-pro 기준 전체 성공률은 다음과 같다.

<div class="table-caption">표 1. WebArena 전체 성능 요약 (Gemini-2.5-pro)</div>
<table style="width:100%;border-collapse:collapse;margin:1rem 0;font-size:0.92rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="text-align:left;padding:10px;border:1px solid #ddd;">방법</th>
      <th style="text-align:center;padding:10px;border:1px solid #ddd;">Overall SR</th>
      <th style="text-align:center;padding:10px;border:1px solid #ddd;">Avg Steps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:10px;border:1px solid #ddd;">No Memory</td>
      <td style="text-align:center;padding:10px;border:1px solid #ddd;">46.7</td>
      <td style="text-align:center;padding:10px;border:1px solid #ddd;">8.8</td>
    </tr>
    <tr>
      <td style="padding:10px;border:1px solid #ddd;">Synapse</td>
      <td style="text-align:center;padding:10px;border:1px solid #ddd;">47.7</td>
      <td style="text-align:center;padding:10px;border:1px solid #ddd;">8.5</td>
    </tr>
    <tr>
      <td style="padding:10px;border:1px solid #ddd;">AWM</td>
      <td style="text-align:center;padding:10px;border:1px solid #ddd;">47.6</td>
      <td style="text-align:center;padding:10px;border:1px solid #ddd;">8.7</td>
    </tr>
    <tr style="background:#fef9c3;font-weight:700;">
      <td style="padding:10px;border:1px solid #ddd;">ReasoningBank</td>
      <td style="text-align:center;padding:10px;border:1px solid #ddd;">53.9</td>
      <td style="text-align:center;padding:10px;border:1px solid #ddd;">7.4</td>
    </tr>
    <tr style="background:#eef5ff;">
      <td style="padding:10px;border:1px solid #ddd;">ReasoningBank + MaTTS</td>
      <td style="text-align:center;padding:10px;border:1px solid #ddd;">56.3</td>
      <td style="text-align:center;padding:10px;border:1px solid #ddd;">7.1</td>
    </tr>
  </tbody>
</table>

포인트는 두 가지다.

- 성공률이 baseline보다 뚜렷하게 오른다.
- step 수까지 줄어든다.

즉 ReasoningBank는 단순히 오래 시도해서 많이 맞히는 메커니즘이 아니라, 더 직접적인 reasoning path를 따라가게 만든다. SWE-Bench-Verified에서도 같은 패턴이 나온다. Gemini-2.5-pro 기준 resolve rate가 54.0에서 57.4로 오르고, 평균 step은 21.1에서 19.8로 줄어든다.

### MaTTS는 scaling을 "메모리 품질 개선"으로 바꾼다

<mark>이 논문의 두 번째 큰 결과는 scaling factor를 늘릴수록 MaTTS가 memory 없는 scaling보다 훨씬 안정적으로 이득을 낸다는 점이다.</mark>

Parallel Scaling에서 k를 늘리면 성능이 안정적으로 오른다. Sequential도 소규모 k에서는 잘 오르지만, 큰 k에서는 parallel이 더 강하다. 논문 해석대로라면 parallel은 더 다양한 trajectory를 확보하기 때문에 contrastive signal이 richer하다.

<figure>
<img src="img/reasoningbank/scaling_mats.jpg" alt="Figure 4">
<figcaption><strong>Figure 4</strong> — scaling factor k가 증가할수록 MaTTS의 성능은 비교적 안정적으로 올라간다. 특히 parallel setting은 더 많은 diverse rollout을 contrastive signal로 활용할 수 있어 큰 k에서 더 강한 경향을 보인다.</figcaption>
</figure>

논문은 여기서 더 나아가, 동일한 k=5 조건에서 어떤 memory mechanism을 붙였는지에 따라 MaTTS의 효과가 얼마나 달라지는지도 비교한다. 결론은 ReasoningBank가 가장 큰 상승폭을 만든다는 것이다. 즉 좋은 scaling은 좋은 memory를 필요로 하고, 좋은 memory는 scaling이 만든 더 많은 경험을 다시 흡수한다.

이 논문이 말하는 "memory-driven experience scaling"은 바로 이 선순환을 가리킨다.

## 왜 이 방법이 실제로 먹히는가

<p><strong>흐름:</strong> failure 활용의 차이 → emergent strategy → judge 노이즈 강건성 → 효율성 해석</p>

### 실패를 넣는다고 다 배우는 건 아니다

<mark>ReasoningBank가 특별한 이유는 실패를 사용해서가 아니라, 실패를 구조화된 reasoning lesson으로 바꿔서 사용하기 때문이다.</mark>

실패 trajectory를 memory induction에 포함했을 때의 ablation이 이를 잘 보여준다. Synapse는 실패를 넣어도 효과가 거의 없고, AWM은 오히려 성능이 떨어진다. 반면 ReasoningBank는 실패를 넣었을 때 성능이 확실히 오른다.

<figure>
<img src="img/reasoningbank/ablation_failure.jpg" alt="Figure 5">
<figcaption><strong>Figure 5</strong> — 성공 trajectory만 쓸 때보다, 실패 trajectory까지 포함해 memory를 만들었을 때 ReasoningBank의 성능은 더 크게 오른다. 같은 failure signal을 넣어도 baseline이 잘 못 쓰는 이유는, 실패를 reusable lesson으로 재구성하는 추상화 단계가 없기 때문이다.</figcaption>
</figure>

이 결과는 생각보다 중요하다. 많은 memory 연구가 "실패도 중요하다"라고 말하지만, 실제로 실패를 넣으면 noisy exemplar만 늘어나는 경우가 많다. ReasoningBank는 failure reflection을 별도의 reasoning unit로 바꾸기 때문에 그 함정을 피한다.

### 메모리는 그냥 쌓이는 게 아니라 진화한다

<mark>이 논문에서 가장 흥미로운 장면은 memory item이 시간이 지나며 더 높은 수준의 전략으로 진화하는 case study다.</mark>

동일한 전략 아이템이 처음에는 매우 절차적이다. 예를 들면 "Next Page를 눌러라", "Load More 링크를 찾아라" 같은 원자적 규칙이다. 그런데 시간이 지나면 이 메모리는 "현재 뷰와 태스크 요구사항을 교차검증하라", "검색/필터 기능을 먼저 확인해 completeness를 보장하라" 같은 더 일반적이고 더 강한 전략으로 발전한다.

*"By abstracting experiences into reusable reasoning units, ReasoningBank enables agents to ... learn from failures, providing richer guidance for test-time learning."*

이 문장을 가장 잘 시각화한 그림이 바로 아래 케이스다.

<figure>
<img src="img/reasoningbank/evolving_strategy.jpg" alt="Figure 6">
<figcaption><strong>Figure 6</strong> — 하나의 memory strategy가 시간축을 따라 어떻게 진화하는지를 보여준다. 초반에는 execution-level tip에 가깝다가, 점차 re-checking, cross-referencing, expectation alignment 같은 더 일반적인 reasoning strategy로 올라간다. 이 논문이 말하는 "self-evolving agent memory"가 무엇인지 가장 잘 드러나는 사례다.</figcaption>
</figure>

### LLM-as-a-Judge가 완벽하지 않아도 시스템은 버틴다

<mark>ReasoningBank는 ground-truth label 없이 success/failure를 판단하기 때문에, 자연스럽게 judge noise에 대한 의문이 생긴다.</mark>

논문은 이를 정면으로 다룬다. 실제 judge 정확도는 72.7% 수준이지만, 시뮬레이션 실험에서 70%~90% 구간에서는 최종 성능 차이가 거의 크지 않다. 즉 이 시스템은 완벽한 판정기보다 "대체로 방향이 맞는 평가자"만 있어도 충분히 돌아간다.

이건 실용적으로 꽤 큰 장점이다. interactive agent 환경에서는 정답 레이블을 항상 즉시 얻기 어렵기 때문이다.

### step 감소는 실패를 빨리 끝내서가 아니라 성공 경로를 더 직접적으로 타서 생긴다

<mark>효율성 이득이 왜 나는지도 논문은 따로 분해해서 보여준다.</mark>

성공 인스턴스와 실패 인스턴스를 나눠 step 수를 보면, ReasoningBank의 감소폭은 특히 성공 케이스에서 더 크다. 즉 agent가 빨리 포기해서 평균 step이 내려간 것이 아니라, 실제로 더 좋은 reasoning hint를 따라 더 적은 상호작용으로 문제를 푼다는 뜻이다.

이 분석은 ReasoningBank를 단순한 heuristic cache가 아니라 policy-shaping memory로 읽게 만든다.

## 이 논문을 어떻게 봐야 하나

<p><strong>흐름:</strong> 핵심 기여 정리 → 왜 중요한가 → 남는 질문</p>

### ReasoningBank의 핵심 기여는 세 가지다

<mark>이 논문은 agent memory를 "기록 보관소"에서 "경험을 재구성하는 reasoning system"으로 바꿔 놓는다.</mark>

첫째, 성공과 실패를 모두 메모리 원천으로 삼는다.  
둘째, raw trajectory나 workflow보다 한 단계 높은 reusable reasoning strategy를 저장한다.  
셋째, MaTTS를 통해 메모리와 테스트타임 스케일링을 서로 증폭시키는 구조를 제안한다.

특히 첫 번째와 두 번째가 중요하다. 실패를 메모리에 넣는 것만으로는 부족하고, 그것을 generalized lesson으로 추출해야만 실제로 agent behavior가 바뀐다. ReasoningBank는 그 점을 가장 설득력 있게 보여주는 논문 중 하나다.

### 그래서 왜 흥미로운가

<mark>ReasoningBank는 에이전트 학습의 새로운 축을 제안한다. 더 큰 모델, 더 긴 context, 더 많은 rollout 말고도 "더 나은 memory induction"이라는 축이 있다는 것이다.</mark>

이 관점은 이후 논문들과도 자연스럽게 이어진다. SkillRL처럼 experience를 skill로 추상화하는 방식, 혹은 memory를 RL과 함께 진화시키는 방식도 결국 같은 질문을 던진다. 과거 경험을 어떤 중간 표현으로 바꿔야 미래 행동을 가장 잘 바꿀 수 있는가?

ReasoningBank의 답은 명확하다. trajectory를 그대로 저장하지 말고, 성공과 실패 모두에서 reasoning strategy를 추출하라는 것이다. 메모리의 단위가 example이 아니라 principle이 될 때, 에이전트는 비로소 조금씩 "자기 경험으로 성장하는 시스템"에 가까워진다.

### 남는 질문

<mark>물론 memory extraction의 품질을 backbone LLM에 상당히 의존하고, consolidation도 아직 단순 append 수준이라는 한계는 남아 있다.</mark>

더 정교한 pruning, merging, forgetting이 들어가면 bank 품질이 더 좋아질 여지는 크다. 또 현재는 web browsing과 SWE처럼 텍스트 중심 reasoning에 잘 맞지만, 더 multimodal한 computer-use setting에서도 같은 구조가 유지될지는 추가 검증이 필요하다.

그럼에도 불구하고 이 논문은 agent memory 연구에서 분명한 전환점처럼 보인다. "무엇을 얼마나 저장할까"보다 "어떤 수준의 reasoning unit로 추상화할까"가 더 중요하다는 걸, 정성적 사례와 정량적 실험으로 함께 설득했기 때문이다.
