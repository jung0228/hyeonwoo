---
title: "Web Agents with World Models"
dek: "행동 결과를 미리 시뮬레이션해 더 나은 결정을 내리는 WMA 웹 에이전트"
desc: "LLM 기반 웹 에이전트에 World Model을 접목해 행동의 결과를 미리 시뮬레이션하고, Tree Search 수준의 성능을 훨씬 낮은 비용으로 달성한 ICLR 2025 연구."
tags: ["Agent", "LLM"]
date: "Apr 2026"
readtime: "10 min read"
slug: wma-web-agent
katex: false
---

## 왜 웹 에이전트는 인간처럼 생각하지 못할까

<p><strong>흐름:</strong> 문제 제기 → 인간과의 비교 → 논문의 출발점</p>

GPT-4 기반 웹 에이전트의 WebArena 성공률은 **14.4%**다. 인간이 같은 벤치마크에서 **78.2%**를 기록하는 것과 비교하면 엄청난 격차다. 왜 이런 차이가 생길까?

*"Humans avoid unwanted situations by considering the possible outcomes of our actions beforehand. Such awareness of actions and outcomes is referred to as the 'world model'."*

인간은 행동하기 전에 **결과를 미리 상상한다**. 환불 불가 항공권을 또 결제할 것 같으면 멈춘다. 장바구니에 이미 담긴 상품을 또 담지 않는다. 이 능력, 즉 "지금 이 행동을 하면 어떻게 될까?"를 내부에서 시뮬레이션하는 능력이 바로 **월드 모델(world model)**이다.

현재 LLM 기반 웹 에이전트에는 이 월드 모델이 없다. 그래서 trial-and-error에 의존하고, 돌이킬 수 없는 실수를 반복한다.

<figure>
<img src="img/wma-web-agent/fig0_motivation.jpg" alt="Motivating example">
<figcaption><strong>Figure 1</strong> — 월드 모델이 없는 에이전트는 행동의 결과를 예측하지 못해 치명적인 실수를 반복한다.</figcaption>
</figure>

## 사전 분석: LLM에게 정말 월드 모델이 없는가

<p><strong>흐름:</strong> 분석 I (다음 상태 예측) → 분석 II (다음 상태를 줬을 때 행동 선택) → 시사점</p>

### LLM은 다음 상태를 예측하지 못한다

논문은 먼저 "LLM이 행동의 결과를 예측할 수 있는가?"를 실험으로 확인한다. WebArena에서 100개의 사용자 지시를 뽑아 인간이 주석을 단 궤적을 수집한 뒤, 각 스텝에서 모델에게 묻는다: *"현재 상태와 이 행동이 주어졌을 때, 다음 상태 A와 B 중 어느 쪽이 맞는가?"*

<mark>GPT-4o, Claude-3.5-Sonnet 등 최신 LLM 모두 평균 54.75% 정확도 — 사실상 랜덤 수준.</mark>

*"Claude-3.5-Sonnet performs almost as badly as random guessing. These suggest that the world model, the ability to foresee the potential outcomes of actions taken, is absent in LLMs."*

<figure>
<img src="img/wma-web-agent/fig_analysis1.jpg" alt="Analysis I — next state prediction">
<figcaption><strong>Figure 2</strong> — 다음 상태 예측 정확도. 모든 LLM이 인간보다 크게 낮고, Claude-3.5-Sonnet은 랜덤에 가깝다.</figcaption>
</figure>

### 다음 상태를 알면 행동 선택이 확 좋아진다

두 번째 실험은 반대로 묻는다: *"각 행동 후보의 다음 상태를 미리 알고 있다면 올바른 행동을 선택할 수 있는가?"* 10개의 행동 후보와 각 행동의 다음 상태를 함께 제공한다.

<mark>GPT-4o: 다음 상태 없이 53% → 다음 상태 있을 때 73% (+20%p). 최대 +38%p 향상.</mark>

<figure>
<img src="img/wma-web-agent/fig_analysis2.jpg" alt="Analysis II — action selection with next state">
<figcaption><strong>Figure 3</strong> — 다음 상태 정보 유무에 따른 행동 선택 정확도 비교. 점선 막대(없을 때) vs 실선 막대(있을 때).</figcaption>
</figure>

두 분석의 결론은 명확하다. **LLM에는 월드 모델이 없다. 하지만 월드 모델을 주면 성능이 크게 오른다.** 이것이 WMA 에이전트의 출발점이다.

<div class="ornament">· · ·</div>

## WMA 프레임워크: 시뮬레이션 기반 정책 선택

<p><strong>흐름:</strong> 전체 구조 개요 → 월드 모델 학습 (3단계) → 추론 시 정책 최적화</p>

### 전체 구조

WMA(World-Model-Augmented) 웹 에이전트는 세 컴포넌트로 구성된다:

- **정책 모델 θ**: 행동 후보를 생성하는 LLM (GPT-4o). **Frozen** — 파라미터 업데이트 없음.
- **월드 모델 φ**: 행동 후보 각각의 다음 상태를 시뮬레이션하는 fine-tuned Llama-3.1-8B.
- **가치 함수 V**: 시뮬레이션된 다음 상태의 보상을 추정해 최적 행동을 고르는 fine-tuned Llama-3.1-8B.

<figure>
<img src="img/wma-web-agent/fig1_overview.jpg" alt="WMA Framework Overview">
<figcaption><strong>Figure 4</strong> — 전체 프레임워크. 상단은 월드 모델 학습 데이터 수집, 하단은 추론 시 정책 최적화 흐름.</figcaption>
</figure>

### 월드 모델 학습 — Step 1: 데이터 수집

GPT-4o-mini를 정책 모델로 사용해 WebArena 환경에서 궤적을 수집한다. LLM으로 다양한 사용자 지시 870개를 합성하고, 각 지시당 5개 궤적을 수집해 총 **14K 인스턴스**의 학습 데이터를 만든다.

각 인스턴스는 `(사용자 지시 I, 현재 관찰 o_t, 행동 a_t, 다음 관찰 o_{t+1})` 형태.

### 월드 모델 학습 — Step 2: Transition-focused Observation Abstraction

<mark>핵심 아이디어: 다음 상태 전체가 아니라 "변화한 부분"만 예측 목표로 삼는다.</mark>

왜 전체 다음 상태 예측이 문제인가?

- 웹 페이지 전환은 일부분만 바뀜 → `o_t`와 `o_{t+1}` 대부분이 동일 → **정보 이득이 낮은 학습**
- accessibility tree 평균 길이가 **4K 토큰** → 학습 비용 폭증

*"We propose to abstract raw text observations, with a focus on state transition between consecutive observations, for obtaining better training objectives."*

구체적 절차:

1. **헝가리안 알고리즘**으로 `o_t` ↔ `o_{t+1}` 요소를 매칭해 cost matrix 계산
2. `ADDED / UPDATED / DELETED` 요소 목록 `Δ(o_t, o_{t+1})` 추출
3. LLM으로 이 diff를 **자유형식 자연어 서술** `tilde_o_{t+1}`로 변환

이렇게 만든 `tilde_o`는 짧고 정보 밀도가 높다. 이것이 월드 모델의 학습 목표다.

<figure>
<img src="img/wma-web-agent/fig2_abstraction.jpg" alt="Transition-focused Observation Abstraction">
<figcaption><strong>Figure 5</strong> — Transition-focused Observation Abstraction 개요. 왼쪽: 전체 accessibility tree, 오른쪽: 변화만 담은 자연어 서술.</figcaption>
</figure>

### 월드 모델 학습 — Step 3: 환경 동역학 학습

추상화된 데이터셋 `tilde_D = {I, o_t, a_t, tilde_o_{t+1}}`로 Llama-3.1-8B-Instruct를 fine-tuning한다. 학습 목표는 next-token prediction:

<div class="callout">
<strong>학습 목표:</strong> 사용자 지시 I, 현재 관찰 o_t, 행동 a_t가 주어졌을 때 → 추상화된 다음 관찰 tilde_o_{t+1}을 생성.
</div>

### 추론 시 정책 최적화

추론 단계에서 WMA 에이전트는 다음 절차를 따른다:

1. 정책 모델 θ에서 top-p decoding으로 **k개 행동 후보** `{a_t^1, ..., a_t^k}` 샘플링
2. 월드 모델 φ로 각 후보의 **다음 상태 시뮬레이션**: `tilde_o_{t+1}^i = φ(o_t, a_t^i, I)`
3. 가치 함수 V로 각 시뮬레이션 결과의 보상 추정 → **보상이 가장 높은 행동 선택**:

`hat_a_t = argmax V(I, o_t, a_t^i, tilde_o_{t+1}^i)`

실제 환경과 상호작용 없이 시뮬레이션만으로 최적 행동을 고르기 때문에, Tree Search처럼 여러 궤적을 실제로 탐색할 필요가 없다.

<div class="ornament">· · ·</div>

## 실험 결과

<p><strong>흐름:</strong> WebArena 성능 → Mind2Web 성능 → 비용·시간 효율 비교</p>

### WebArena

<table>
<thead>
<tr><th>방법</th><th>Success Rate</th></tr>
</thead>
<tbody>
<tr><td>CoT (GPT-4o)</td><td>13.1%</td></tr>
<tr><td style="background:#fef9c3;font-weight:700;">WMA (GPT-4o)</td><td style="background:#fef9c3;font-weight:700;">16.6%</td></tr>
<tr><td>Tree Search Agent</td><td>19.2%</td></tr>
</tbody>
</table>

WMA는 CoT 대비 **+3.5%p (약 27% 향상)**을 보인다. Tree Search와 비교하면 절대 수치는 약간 낮지만, CoT 대비 성능 향상폭(+29.7% vs +28.0%)은 오히려 더 크다. GPT-4o-mini 사용 시 Gitlab에서 **+181%**, Map에서 **+92%** 성능 향상.

### Mind2Web — 새 SOTA 달성

Mind2Web에서는 이전 SOTA인 AWM을 뛰어넘어 **새 SOTA**를 달성한다. Tree Search는 오프라인 벤치마크라 환경이 없어 적용 불가한 반면, WMA는 오프라인 궤적 데이터만으로도 학습 가능하다는 강점이 있다.

### 비용과 시간 — Tree Search 대비 압도적 효율

<table>
<thead>
<tr><th>방법</th><th>시간 (초/태스크)</th><th>상대 비용</th></tr>
</thead>
<tbody>
<tr><td>Tree Search Agent</td><td>748.3초</td><td>6.8×</td></tr>
<tr style="background:#fef9c3;font-weight:700;"><td>WMA (ours)</td><td>140.3초</td><td>1×</td></tr>
</tbody>
</table>

*"WMA web agent only takes 140.3 seconds per instance by simulating the possible action candidates rather than actually executing them, which is 5.3 times faster than Tree search agent."*

Tree Search는 backtracing 시 이전 행동 시퀀스를 전부 재실행해야 해서 느리고 비싸다. WMA는 시뮬레이션만 하기 때문에 **비용 6.8배 절감, 속도 5.3배 향상**.

<div class="ornament">· · ·</div>

## 분석 및 Ablation

<p><strong>흐름:</strong> Ablation 결과 → 월드 모델 에러 유형 분석 → 탐색 예산(k) 효과</p>

### Ablation: 무엇이 중요한가

<ul>
  <li><strong>시뮬레이션된 다음 상태 없이 보상 추정</strong>: 성능 하락 → 다음 상태 정보가 가치 함수 추정에 필수적</li>
  <li><strong>fine-tuning 없이 GPT-4o-mini 프롬프팅으로 월드 모델 대체</strong>: 성능 크게 하락 → SOTA LLM도 fine-tuning 없이는 환경 동역학 지식이 부족</li>
  <li><strong>전체 accessibility tree 예측 (abstraction 없이)</strong>: 가장 나쁜 성능 → 중복 정보가 학습을 방해</li>
</ul>

### 탐색 예산 k

<figure>
<img src="img/wma-web-agent/fig_num_actions.jpg" alt="Number of sampled actions ablation">
<figcaption><strong>Figure 6</strong> — 샘플링 행동 수 k에 따른 성능 변화. k가 클수록 성능이 높아진다.</figcaption>
</figure>

k(샘플링 행동 수)가 늘어날수록 성능이 향상되는 양의 상관관계가 있다. 예산이 허용된다면 더 많은 미래 상태를 탐색할수록 유리하다.

### 월드 모델의 에러 유형

<figure>
<img src="img/wma-web-agent/fig_error.jpg" alt="Error types in world model predictions">
<figcaption><strong>Figure 7</strong> — 월드 모델의 잘못된 예측 50개를 수동 분류한 결과.</figcaption>
</figure>

<ul>
  <li><strong>반사실적 상상 (42%)</strong>: 존재하지 않는 상품이나 요소를 만들어 냄</li>
  <li><strong>웹 컴포넌트 이해 부족 (26%)</strong>: 검색창에 기존 텍스트를 지우지 않은 채 새 키워드를 입력하면 어떻게 될지 예측 실패</li>
  <li><strong>지나치게 모호한 서술 (24%)</strong>: "사용자는 다양한 기능을 볼 것입니다" 같은 무의미한 수준의 일반화</li>
  <li><strong>기타 (8%)</strong>: 현재 시점을 건너뛴 예측 등</li>
</ul>

<div class="ornament">· · ·</div>

## 정리

<mark>WMA는 LLM 웹 에이전트에 월드 모델을 접목한 최초의 연구로, "시뮬레이션 → 평가 → 선택"이라는 패러다임을 제시한다.</mark>

핵심 기여를 세 줄로 요약하면:

1. **Transition-focused Observation Abstraction**: 다음 상태 전체가 아닌 변화분만 학습해 효율적인 월드 모델 훈련
2. **Training-free 정책 최적화**: 정책 모델을 건드리지 않고 월드 모델 + 가치 함수만으로 기존 에이전트에 plug-in 가능
3. **Tree Search 대비 6.8× 비용 절감, 5.3× 속도 향상**으로 비슷한 성능

웹 에이전트 연구의 다음 단계는 월드 모델의 반사실적 환각 문제를 줄이고, 더 복잡한 상태 공간(쇼핑 등)에서의 정확도를 높이는 것이 과제로 남는다.

<div class="footnote">
원문: <a href="https://arxiv.org/abs/2410.13232">Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation</a> (ICLR 2025) &nbsp;·&nbsp; <a href="https://github.com/kyle8581/WMA-Agents">GitHub</a>
</div>
