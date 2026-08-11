---
title: 웹 에이전트 2025–2026 연구 총정리
dek: RL 혁명, Visual Grounding 전쟁, 합성 데이터 스케일업 — 1티어 학회 20+ 논문으로 보는 현재
desc: ICLR·NeurIPS·CVPR 2025와 2026 arXiv까지 — 웹 에이전트 연구의 판을 바꾼 논문들을 트렌드별로 깊게 분석한다.
tags: [Agent, LLM, Multimodal]
date: Mar 2026
readtime: 22 min read
slug: web-agent-2025-survey
katex: false
---

## 들어가며

2024년까지의 웹 에이전트 연구는 크게 두 가지 흐름이었다. 하나는 GPT-4V 같은 강력한 프론티어 모델에 잘 설계된 프롬프트를 붙이는 방식, 다른 하나는 인간이 시연한 궤적 데이터로 SFT(Supervised Fine-Tuning)를 하는 방식이었다. WebArena 기준 최고 성능이 20% 초반대에 머물렀고, "LLM이 복잡한 웹 태스크를 자율적으로 수행하기엔 아직 멀었다"는 회의론이 지배적이었다.

2025년에 판이 바뀌었다.

ICLR 2025 한 학회에서만 웹·GUI 에이전트 관련 논문이 수십 편 쏟아졌고, NeurIPS 2025에는 45편의 computer-use agent 논문이 한꺼번에 발표됐다. 성능 수치도 달라졌다 — Llama-3.1-8B짜리 소형 모델이 WebArena에서 44%를 찍으며 GPT-4o를 넘었다. 그 뒤에는 **강화학습(RL)의 본격적인 침투**, **순수 비전 기반 그라운딩의 부상**, **합성 데이터로 훈련 환경을 무한 확장하려는 시도**가 있다.

이 글에서는 2025–2026년 1티어 학회와 최신 arXiv에서 나온 주요 논문 20+편을 트렌드별로 정리하고, 내가 기술적으로 가장 흥미롭다고 생각하는 논문들은 깊게 파고든다.

<div class="ornament">· · ·</div>

## 전체 지형도: 4가지 큰 흐름

<div class="pullquote">
  <strong>2025 웹 에이전트 연구의 핵심:</strong> "프롬프트 엔지니어링의 시대"에서 "강화학습으로 직접 훈련하는 시대"로의 전환.
</div>

논문들을 관통하는 흐름은 네 가지다.

**① RL이 SFT를 대체하기 시작했다.** WebRL, WebAgent-R1 등이 보여줬듯, 환경과 직접 상호작용하며 성공 보상만으로 훈련하면 인간 레이블 데이터 없이도 훨씬 높은 성능에 도달한다. 특히 WebAgent-R1은 Llama-3.1-8B로 OpenAI o3를 넘었다.

**② DOM 없이 스크린샷만 본다.** UGround, OS-ATLAS, ShowUI 등은 텍스트 DOM에 의존하지 않고 픽셀 좌표로 직접 UI 요소를 찾는다. 실제 브라우저 자동화에서 DOM 접근이 항상 가능한 건 아니고, 사람도 화면을 보고 클릭하니까.

**③ 훈련 데이터를 자동으로 무한 생성한다.** AgentTrek, VeriEnv, WebFactory 등은 인간의 레이블링 없이 합성 궤적이나 합성 환경을 자동 생성한다. 데이터 병목이 사라지면 RL 스케일업이 가능해진다.

**④ 벤치마크가 더 현실적으로 진화했다.** Mind2Web 2, REAL, TheAgentCompany는 기존 WebArena보다 훨씬 실제 업무와 가까운 태스크를 다룬다. 최고 모델들도 여기서 30~50%밖에 못 풀고 있다.

<div class="ornament">· · ·</div>

## 심층 분석 1: RL 혁명 — WebRL과 WebAgent-R1

### WebRL: 실패에서 새 문제를 만들다 (ICLR 2025)

<div class="callout">
  <strong>논문:</strong> WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning<br>
  <strong>기관:</strong> Tsinghua / THUDM &nbsp;·&nbsp; <strong>벤치마크:</strong> WebArena-style 5개 웹사이트
</div>

WebRL이 출발한 질문은 간단하다: *"왜 웹 에이전트 RL이 어려운가?"*

논문이 진단한 병목은 세 가지였다.

1. **태스크 부족** — 웹 태스크 데이터셋은 수백~수천 개 수준이라 RL을 돌리기엔 너무 적다.
2. **희소 보상** — 웹 태스크는 최종 결과가 성공이냐 실패냐인 이진 보상이라, 에이전트가 어떤 중간 행동이 좋았는지 알기 어렵다.
3. **분포 드리프트** — RL로 정책이 업데이트되면 에이전트가 방문하는 상태 분포가 달라지는데, 초기 데이터셋은 이 새로운 분포를 커버하지 못한다.

<div class="pullquote">
  <strong>핵심 아이디어:</strong> 에이전트가 실패한 궤적 자체가 새로운 훈련 문제의 소스다.
</div>

WebRL의 해법은 **Self-Evolving Curriculum**이다. 에이전트가 어떤 태스크를 풀다가 실패하면, 그 실패 궤적을 분석해서 "지금 에이전트 수준에서 풀 수 있을 법한" 새 태스크를 자동 생성한다. 너무 쉬운 문제(항상 성공)나 너무 어려운 문제(항상 실패)는 RL 학습에 신호가 없으니, 성공률이 30~70% 사이의 "적당히 어려운" 문제를 계속 공급하는 셈이다.

<svg viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;display:block;margin:2rem auto;font-family:'Source Serif 4',serif;">
  <rect width="680" height="260" fill="#fafaf8" rx="8"/>
  <!-- Step boxes -->
  <rect x="30" y="90" width="120" height="60" rx="6" fill="#fef3c7" stroke="#d4a017" stroke-width="1.5"/>
  <text x="90" y="116" text-anchor="middle" font-size="12" fill="#78350f" font-weight="bold">에이전트 실행</text>
  <text x="90" y="132" text-anchor="middle" font-size="11" fill="#78350f">태스크 시도</text>
  <rect x="200" y="90" width="120" height="60" rx="6" fill="#ffd6d6" stroke="#b91c1c" stroke-width="1.5"/>
  <text x="260" y="116" text-anchor="middle" font-size="12" fill="#7f1d1d" font-weight="bold">실패 궤적 분석</text>
  <text x="260" y="132" text-anchor="middle" font-size="11" fill="#7f1d1d">어디서 막혔나?</text>
  <rect x="370" y="90" width="130" height="60" rx="6" fill="#d4f7d4" stroke="#15803d" stroke-width="1.5"/>
  <text x="435" y="116" text-anchor="middle" font-size="12" fill="#14532d" font-weight="bold">새 태스크 생성</text>
  <text x="435" y="132" text-anchor="middle" font-size="11" fill="#14532d">성공률 30–70% 수준</text>
  <rect x="545" y="90" width="110" height="60" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="1.5"/>
  <text x="600" y="116" text-anchor="middle" font-size="12" fill="#1e1b4b" font-weight="bold">ORM 보상</text>
  <text x="600" y="132" text-anchor="middle" font-size="11" fill="#1e1b4b">+ RL 업데이트</text>
  <!-- Arrows -->
  <line x1="150" y1="120" x2="198" y2="120" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="320" y1="120" x2="368" y2="120" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="500" y1="120" x2="543" y2="120" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- Feedback loop -->
  <path d="M 600 150 Q 600 210 90 210 Q 30 210 30 150" fill="none" stroke="#999" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arr)"/>
  <text x="340" y="230" text-anchor="middle" font-size="11" fill="#666">커리큘럼 피드백 루프</text>
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#999"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="340" y="35" text-anchor="middle" font-size="13" fill="#333" font-weight="bold">WebRL Self-Evolving Curriculum</text>
</svg>

희소 보상 문제는 **ORM(Outcome-supervised Reward Model)**으로 해결했다. 최종 성공/실패 신호만 갖고 ORM을 학습시켜서, 중간 상태에서도 "지금 방향이 맞나"를 판단하는 dense reward를 만들어낸다. 사람이 중간 과정 레이블을 달지 않아도 된다.

결과는 놀라웠다. Llama-3.1-8B가 WebArena 스타일 벤치마크에서 4.8% → **42.4%**로 뛰었고, Llama-3.1-70B는 **47.3%**에 달했다. GPT-4-Turbo가 17.6%인 것과 비교하면 오픈소스 8B 모델이 GPT-4를 2.5배 이상 능가한 셈이다.

---

### WebAgent-R1: 이진 보상만으로 OpenAI o3를 넘다 (EMNLP 2025)

<div class="callout">
  <strong>논문:</strong> WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning<br>
  <strong>핵심 결과:</strong> Llama-3.1-8B로 WebArena-Lite 44.8% — OpenAI o3(42.4%) 능가
</div>

WebAgent-R1이 흥미로운 이유는 **극도로 단순한 학습 신호**로 극단적인 성능을 뽑아냈다는 점이다. 보상 설계? 태스크 성공이면 +1, 실패면 0. 그게 전부다.

기존 RL 기반 웹 에이전트 연구들의 문제는 복잡도였다. 별도의 reward model을 학습해야 하고, off-policy 데이터 필터링 파이프라인도 있어야 하고, 여러 단계의 파이프라인을 관리해야 했다. WebAgent-R1은 이걸 다 걷어냈다.

<div class="pullquote">
  <strong>핵심 통찰:</strong> 웹 태스크의 성공/실패는 명확하게 검증 가능하다. 이 이진 신호만으로 충분하다.
</div>

아키텍처는 두 단계로 구성된다:

**1단계 — Behavior Cloning (BC) Warm-up:** 순수 RL부터 시작하면 에이전트가 너무 오래 헤매기 때문에, 먼저 전문가 궤적으로 SFT를 해서 "대충 웹을 어떻게 쓰는지"를 학습시킨다. 논문에서는 이 warm-up 단계가 없으면 RL이 수렴하지 않는다는 것을 실험으로 보였다.

**2단계 — End-to-End Multi-Turn RL:** 실제 웹 환경(WebArena-Lite의 5개 웹사이트)과 온라인 상호작용하며 GRPO(Group Relative Policy Optimization) 알고리즘으로 학습한다. 비동기 trajectory 생성 시스템으로 GPU 활용률을 높였다.

<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;display:block;margin:2rem auto;">
  <rect width="680" height="300" fill="#fafaf8" rx="8"/>
  <text x="340" y="35" text-anchor="middle" font-size="13" fill="#333" font-weight="bold">WebAgent-R1 학습 파이프라인</text>
  <!-- Phase 1 -->
  <rect x="40" y="60" width="260" height="100" rx="8" fill="#fef3c7" stroke="#d4a017" stroke-width="1.5"/>
  <text x="170" y="85" text-anchor="middle" font-size="12" fill="#78350f" font-weight="bold">Phase 1: BC Warm-up</text>
  <text x="170" y="107" text-anchor="middle" font-size="11" fill="#78350f">전문가 궤적 SFT</text>
  <text x="170" y="124" text-anchor="middle" font-size="11" fill="#78350f">"웹 사용법 기초 학습"</text>
  <text x="170" y="141" text-anchor="middle" font-size="11" fill="#78350f">없으면 RL이 수렴 안 함</text>
  <!-- Phase 2 -->
  <rect x="380" y="60" width="260" height="100" rx="8" fill="#d4f7d4" stroke="#15803d" stroke-width="1.5"/>
  <text x="510" y="85" text-anchor="middle" font-size="12" fill="#14532d" font-weight="bold">Phase 2: Online RL</text>
  <text x="510" y="107" text-anchor="middle" font-size="11" fill="#14532d">실제 웹 환경과 상호작용</text>
  <text x="510" y="124" text-anchor="middle" font-size="11" fill="#14532d">보상: 성공 +1 / 실패 0</text>
  <text x="510" y="141" text-anchor="middle" font-size="11" fill="#14532d">GRPO 알고리즘</text>
  <!-- Arrow -->
  <line x1="300" y1="110" x2="378" y2="110" stroke="#999" stroke-width="2" marker-end="url(#arr2)"/>
  <!-- Results -->
  <rect x="40" y="200" width="600" height="70" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="1.5"/>
  <text x="340" y="224" text-anchor="middle" font-size="12" fill="#1e1b4b" font-weight="bold">결과 (WebArena-Lite)</text>
  <text x="175" y="248" text-anchor="middle" font-size="11" fill="#3730a3">Qwen-2.5-3B: 6.1% → 33.9%</text>
  <text x="340" y="248" text-anchor="middle" font-size="11" fill="#3730a3">LLaMA-3.1-8B: 8.5% → 44.8%</text>
  <text x="510" y="248" text-anchor="middle" font-size="11" fill="#3730a3">OpenAI o3: 42.4% (패배)</text>
  <defs>
    <marker id="arr2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#999"/>
    </marker>
  </defs>
</svg>

논문이 강조하는 또 하나의 포인트: **WebAgent-R1-Zero**(RL만, warm-up 없음)도 실험했는데 성능이 훨씬 낮았다. DeepSeek-R1의 "zero" 세팅처럼 RL만으로 충분하다는 주장은 웹 에이전트에서는 아직 성립하지 않는다. 웹 태스크의 action space가 코딩이나 수학보다 훨씬 넓고 탐색 공간이 크기 때문이다.

<div class="ornament">· · ·</div>

## 심층 분석 2: 웹 에이전트에 World Model을 — WMA (ICLR 2025)

<div class="callout">
  <strong>논문:</strong> Web Agents with World Models (WMA)<br>
  <strong>기관:</strong> Yonsei University (한국!) &nbsp;·&nbsp; <strong>Venue:</strong> ICLR 2025
</div>

WMA가 던지는 질문은 철학적으로 흥미롭다: *"GPT-4o, Claude-3.5 같은 강력한 LLM도 웹 에이전트로 쓰면 왜 실수를 그렇게 많이 하는가?"*

논문의 답: **이 모델들에게 world model이 없기 때문이다.**

### World Model이란 무엇인가

인간이 웹을 탐색할 때 우리는 "이 버튼을 클릭하면 다음 페이지가 어떻게 바뀔지" 머릿속에서 미리 그려본다. 이게 world model이다. 불확실한 행동을 하기 전에 결과를 시뮬레이션해서 리스크를 줄인다.

반면 기존 LLM 웹 에이전트는 각 스텝에서 "지금 현재 화면을 보고, 다음 행동을 바로 선택"한다. 행동의 결과를 예측하지 않는다. 논문은 이를 실험으로 입증했다 — GPT-4o에게 "이 행동을 하면 다음 상태가 어떻게 될까요?"를 직접 물으면 예측이 매우 부정확하다.

<div class="pullquote">
  <strong>WMA의 핵심:</strong> LLM 자체를 world model로 활용해서, 행동 전에 "결과 시뮬레이션"을 수행한다.
</div>

### Transition-Focused Observation Abstraction

WMA의 핵심 기술 기여는 **transition-focused observation abstraction**이다. 전체 웹페이지 HTML/스크린샷을 다 주는 대신, "이전 상태에서 이 행동을 했더니 어떻게 변했나"에 집중하는 추상화된 표현을 만든다.

직관적으로 이해하면: 웹페이지 전체를 다 볼 필요 없이 "버튼을 눌렀더니 새 모달이 떴다"처럼 변화의 델타(delta)만 추적하면 된다. 이 추상화된 transition을 LLM이 학습하면 효율적인 world model이 된다.

<svg viewBox="0 0 680 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;display:block;margin:2rem auto;">
  <rect width="680" height="280" fill="#fafaf8" rx="8"/>
  <text x="340" y="35" text-anchor="middle" font-size="13" fill="#333" font-weight="bold">WMA: World Model로 행동 선택</text>
  <!-- Current state -->
  <rect x="30" y="60" width="130" height="70" rx="6" fill="#fef3c7" stroke="#d4a017" stroke-width="1.5"/>
  <text x="95" y="88" text-anchor="middle" font-size="11" fill="#78350f" font-weight="bold">현재 상태 S_t</text>
  <text x="95" y="106" text-anchor="middle" font-size="10" fill="#78350f">웹페이지 화면</text>
  <text x="95" y="121" text-anchor="middle" font-size="10" fill="#78350f">(추상화됨)</text>
  <!-- Action candidates -->
  <rect x="220" y="45" width="120" height="35" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="1"/>
  <text x="280" y="67" text-anchor="middle" font-size="10" fill="#1e1b4b">행동 후보 A₁</text>
  <rect x="220" y="95" width="120" height="35" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="1"/>
  <text x="280" y="117" text-anchor="middle" font-size="10" fill="#1e1b4b">행동 후보 A₂</text>
  <!-- World model -->
  <rect x="220" y="155" width="200" height="55" rx="6" fill="#d4f7d4" stroke="#15803d" stroke-width="1.5"/>
  <text x="320" y="178" text-anchor="middle" font-size="11" fill="#14532d" font-weight="bold">World Model (LLM)</text>
  <text x="320" y="196" text-anchor="middle" font-size="10" fill="#14532d">S_t + A_i → S_{t+1} 예측</text>
  <!-- Predicted states -->
  <rect x="480" y="45" width="130" height="70" rx="6" fill="#ffd6d6" stroke="#b91c1c" stroke-width="1"/>
  <text x="545" y="68" text-anchor="middle" font-size="10" fill="#7f1d1d" font-weight="bold">예측 상태 S'₁</text>
  <text x="545" y="84" text-anchor="middle" font-size="10" fill="#7f1d1d">"목표와 멀어짐"</text>
  <text x="545" y="100" text-anchor="middle" font-size="10" fill="#7f1d1d">→ 제외</text>
  <rect x="480" y="140" width="130" height="70" rx="6" fill="#d4f7d4" stroke="#15803d" stroke-width="1"/>
  <text x="545" y="163" text-anchor="middle" font-size="10" fill="#14532d" font-weight="bold">예측 상태 S'₂</text>
  <text x="545" y="179" text-anchor="middle" font-size="10" fill="#14532d">"목표에 가까움"</text>
  <text x="545" y="195" text-anchor="middle" font-size="10" fill="#14532d">→ 선택!</text>
  <!-- Arrows -->
  <line x1="160" y1="95" x2="218" y2="65" stroke="#999" stroke-width="1.2" marker-end="url(#arr3)"/>
  <line x1="160" y1="95" x2="218" y2="112" stroke="#999" stroke-width="1.2" marker-end="url(#arr3)"/>
  <line x1="280" y1="80" x2="280" y2="153" stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>
  <line x1="280" y1="112" x2="280" y2="153" stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>
  <line x1="420" y1="175" x2="478" y2="80" stroke="#999" stroke-width="1.2" marker-end="url(#arr3)"/>
  <line x1="420" y1="175" x2="478" y2="175" stroke="#999" stroke-width="1.2" marker-end="url(#arr3)"/>
  <defs>
    <marker id="arr3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#999"/>
    </marker>
  </defs>
</svg>

결과는 GPT-4o 기반 에이전트 기준 WebArena에서 13.1% → **16.6%** (+27% 상대 향상). 수치 자체는 크지 않아 보일 수 있는데, 이 방법의 가치는 수치보다 아이디어에 있다. World model 기반 look-ahead는 이후 RL 기반 플래닝과 결합될 때 훨씬 강력해질 수 있고, 실제로 2026년 arXiv 논문들에서 이 방향이 이어지고 있다.

<div class="ornament">· · ·</div>

## 심층 분석 3: Visual Grounding 전쟁

### 왜 Grounding이 중요한가

웹 에이전트가 "로그인 버튼을 클릭해"라고 결정했더라도, 실제로 화면의 어떤 픽셀을 클릭해야 하는지 아는 게 **grounding**이다. 기존 방법들은 HTML DOM에서 버튼 element를 찾아 클릭했는데, 두 가지 문제가 있다.

첫째, DOM이 항상 깔끔하지 않다. 동적 웹앱에서 shadow DOM, iframe, 캔버스 요소는 DOM으로 접근하기 어렵다.
둘째, 사람은 화면을 보고 클릭한다. 에이전트도 그럴 수 있어야 한다 — 그래야 진짜로 범용적이다.

### UGround: 역대 최대 데이터셋으로 ICLR Oral을 따낸 논문 (ICLR 2025 Oral)

<div class="callout">
  <strong>논문:</strong> Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents<br>
  <strong>기관:</strong> OSU NLP Group &nbsp;·&nbsp; <strong>Venue:</strong> ICLR 2025 Oral
</div>

UGround의 접근은 직접적이다: 충분히 많은 (스크린샷, UI 요소 설명, 픽셀 좌표) 페어로 훈련시키면 된다. 문제는 데이터였다.

기존 GUI grounding 데이터셋은 수만~수십만 개 수준이었다. UGround는 **1.3M 스크린샷, 10M+ GUI 요소**로 역대 최대 규모의 그라운딩 데이터셋을 구축했다. 핵심은 자동화된 합성 파이프라인이다 — 실제 웹페이지를 렌더링하고, 접근성 트리(accessibility tree)로부터 요소 위치를 자동 추출했다.

모델 아키텍처는 LLaVA 기반 경량 변형을 사용했고, 입력은 스크린샷 + 텍스트 쿼리("로그인 버튼이 어디에 있나요?"), 출력은 픽셀 좌표다.

결과: 기존 GUI grounding 모델 대비 최대 **+20% absolute** 향상. 특히 HTML/DOM 없이 순수 비전만으로 이전 SOTA 에이전트 성능을 능가했다.

---

### OS-ATLAS: 5개 플랫폼을 하나의 모델로 (ICLR 2025 Spotlight)

<div class="callout">
  <strong>논문:</strong> OS-ATLAS: A Foundation Action Model for Generalist GUI Agents<br>
  <strong>기관:</strong> Shanghai AI Lab &nbsp;·&nbsp; <strong>Venue:</strong> ICLR 2025 Spotlight
</div>

UGround가 "웹에서 최고의 그라운딩"을 목표로 했다면, OS-ATLAS는 **크로스 플랫폼 일반화**를 목표로 한다. Windows, Linux, macOS, Android, 웹 — 모든 플랫폼에서 동작하는 단일 기반 모델이다.

데이터 규모: **13M+ GUI 요소**를 포함한 크로스 플랫폼 코퍼스. 두 가지 모델 변형을 오픈소스로 공개했다 — OS-Atlas-Base-4B (InternVL2-4B 기반)와 OS-Atlas-7B (Qwen2-VL-7B 기반).

6개 벤치마크(모바일·데스크탑·웹)에서 이전 SOTA 대비 유의미한 개선을 보였다. 이 모델이 흥미로운 이유는 데이터 합성 툴킷을 함께 공개했다는 점 — 커뮤니티가 자체 플랫폼에 맞는 데이터를 만들 수 있다.

---

### GUI-Actor: 좌표를 예측하지 말고 가리켜라 (NeurIPS 2025)

기존 그라운딩 모델들은 클릭할 좌표를 숫자로 출력한다: "클릭 (423, 187)". GUI-Actor는 이 패러다임 자체를 바꿨다.

<div class="pullquote">
  <strong>GUI-Actor의 혁신:</strong> 좌표 예측 대신, 화면에서 직접 "이 요소"를 가리키는 coordinate-free 그라운딩.
</div>

좌표 예측의 문제는 숫자 회귀 자체가 어렵다는 것이다 — 특히 고해상도 화면에서 픽셀 단위 정밀도가 필요할 때. GUI-Actor는 이를 분류 문제로 바꿔서 요소 자체를 선택하게 했다. 7B 모델로 그보다 10배 큰 UI-TARS-72B baseline을 능가했다.

<div class="ornament">· · ·</div>

## 심층 분석 4: UI-TARS-2 — ByteDance의 괴물 모델

ByteDance가 2025년 초 공개한 UI-TARS 시리즈는 2025 웹·GUI 에이전트 연구에서 빼놓을 수 없다. Qwen2-VL 기반의 end-to-end 네이티브 GUI 에이전트로, 스크린샷만 입력받아 행동을 직접 출력한다.

**UI-TARS (Jan 2025):** 인식, 그라운딩, 추론, 반성, 태스크 완료를 한 모델에 통합. 당시 웹·모바일·데스크탑 다수 벤치마크에서 SOTA.

**UI-TARS-2 (Sep 2025):** 멀티턴 RL + **data flywheel** 구조가 핵심이다. 모델이 실제 환경에서 성공한 궤적을 다시 학습 데이터로 만들어 다음 버전을 훈련하는 선순환이다. 결과:

- **Online-Mind2Web: 88.2%**
- **OSWorld: 47.5%** (현재 최고 수준)
- **AndroidWorld: 73.3%**

70B 파라미터의 규모도 있지만, data flywheel 구조 덕에 점점 스스로 더 좋아지는 시스템을 만들었다는 점이 인상적이다.

<div class="ornament">· · ·</div>

## 데이터 합성 혁명: 환경을 자동으로 만들다

RL로 웹 에이전트를 훈련하려면 두 가지가 필요하다 — 좋은 알고리즘, 그리고 에이전트가 마음껏 실험할 수 있는 **안전한 훈련 환경**. 실제 웹사이트에서 훈련하면 실수로 뭔가를 주문하거나 계정이 잠길 수 있다. 2026년 arXiv 논문들은 이 문제를 정면으로 공격한다.

### AgentTrek: 웹 튜토리얼이 훈련 데이터가 된다 (ICLR 2025 Spotlight)

인터넷에는 "Gmail에서 라벨 만드는 법", "Slack에서 채널 아카이브하는 법" 같은 튜토리얼이 수백만 개 있다. AgentTrek은 이걸 자동으로 에이전트 궤적으로 변환한다.

파이프라인: (1) 튜토리얼 크롤링 및 필터링, (2) 구조화된 태스크 명세로 변환, (3) VLM 에이전트가 실제 환경에서 실행하며 스크린샷 + 행동 시퀀스 기록.

궤적당 비용이 $0.55 수준으로 인간 레이블 대비 극도로 저렴하다. 확장성이 핵심이다.

---

### VeriEnv: 코딩 에이전트가 웹사이트 자체를 복제한다 (arXiv Mar 2026)

VeriEnv는 더 급진적인 아이디어다. 코딩 에이전트가 대상 웹사이트를 **자동으로 완전히 복제**해서 로컬에 실행 가능한 환경을 만든다. Python SDK로 내부 상태에 직접 접근할 수 있어서 성공/실패를 결정론적으로 판단할 수 있다.

실제 웹사이트에서 훈련하면 로그인 자격 증명, 결제 정보, 다른 사용자 데이터에 노출되는 리스크가 있다. VeriEnv는 이런 위험 없이 임의로 많은 훈련 환경을 자동 생성한다.

---

### WebFactory: LLM이 웹사이트도 만든다 (arXiv Mar 2026)

WebFactory는 합성의 끝을 보여준다 — LLM이 레이아웃, 워크플로우, 콘텐츠를 포함한 **현실적인 합성 웹사이트 자체**를 자동 생성한다. 10개 합성 웹사이트만의 데이터로 훨씬 많은 인간 레이블 데이터로 훈련한 모델과 동등한 성능을 달성했다.

<div class="ornament">· · ·</div>

## 나머지 주요 논문 빠른 정리

### ICLR 2025

**Agent S (Simular AI)** — 외부 웹 검색 + 내부 경험 기억의 이중 메모리 구조 + 계층적 플래닝(장기 목표→서브태스크→단계 실행). OSWorld에서 +9.37% 향상. 오픈소스.

**LCoW (KAIST)** — 웹 페이지 이해와 의사결정을 아예 분리한 모듈. 독립 훈련된 contextualization 모듈이 raw HTML을 LLM이 소화하기 쉬운 형태로 변환한다. GPT-4o에 붙이면 +15.6%, LLaMA에 붙이면 +23.7%.

**Ferret-UI 2 (Apple)** — iPhone, Android, iPad, Webpage, Apple TV 5개 플랫폼 UI 이해 MLLM. 고해상도 adaptive scaling 탑재.

**ShowUI (NUS Show Lab) — CVPR 2025** — UI를 connected graph로 모델링해서 중복 토큰을 33% 제거하는 경량 2B VLA 모델. 제로샷 스크린샷 그라운딩 75.1%.

---

### NeurIPS 2025 벤치마크 논문들

**Mind2Web 2** — 실시간 웹 브라우징이 필요한 130개 장기 에이전틱 검색 태스크. 시간에 따라 정답이 바뀌는 문제를 평가하는 Agent-as-a-Judge 프레임워크. 최고 시스템(OpenAI Deep Research)도 인간의 50~70%.

**REAL** — 이커머스·여행·소셜 등 11개 실제 웹사이트의 결정론적 고충실도 복제본. 112개 실용 태스크. 프론티어 모델 최고 성공률 41%.

**TheAgentCompany** — 소프트웨어 회사 환경 시뮬레이션. 웹 브라우징·코딩·프로그램 실행·동료 소통 포함. 최고 에이전트 약 30% 완수.

**ScreenSpot-Pro** — CAD, 과학 도구, IDE 같은 전문 고해상도 소프트웨어 그라운딩 벤치마크. 현재 최고 SOTA(OS-Atlas-7B)가 겨우 18.9% — 미해결 과제.

**WASP** — 웹 에이전트 보안 평가. 프롬프트 인젝션 공격으로 에이전트를 속이는 데 성공률 최대 **86%**. 배포 전 반드시 알아야 할 취약성.

<div class="ornament">· · ·</div>

## 정리: 2026년에는 어디로 가는가

2025년 연구들이 확립한 것들을 정리하면:

**확립된 것들:**
- RL은 SFT보다 분명히 강하다. WebAgent-R1처럼 단순한 이진 보상으로도 충분하다.
- Visual grounding은 DOM 없이도 가능하다. 충분한 데이터와 모델이 있으면.
- 합성 환경/데이터로 훈련을 스케일업할 수 있다.

**아직 미해결:**
- 전문 고해상도 소프트웨어 그라운딩(ScreenSpot-Pro 18.9%)
- 장기 태스크에서의 신뢰성 (Mind2Web 2에서 최고 모델도 70% 수준)
- 보안 취약성 — 프롬프트 인젝션 공격 방어
- 웹 에이전트의 추론 비용 vs. 성능 트레이드오프

2026년 arXiv 논문들(VeriEnv, WebFactory, WebGym)이 공통적으로 가리키는 방향은 **"환경 자동 생성 + 대규모 RL"**이다. 인간이 만든 훈련 데이터의 한계를 넘어서, 에이전트가 스스로 환경을 만들고 그 안에서 무한히 훈련하는 방향.

2년 전에 "WebArena 50%는 불가능"이라던 연구자들이 이제 "OSWorld 50%를 어떻게 넘길까"를 고민하고 있다. 속도가 무섭다.

<div class="footnote">
  주요 참고 논문: <a href="https://openreview.net/forum?id=oVKEAFjEqv">WebRL (ICLR 2025)</a> · <a href="https://aclanthology.org/2025.emnlp-main.401/">WebAgent-R1 (EMNLP 2025)</a> · <a href="https://arxiv.org/abs/2410.13232">WMA (ICLR 2025)</a> · <a href="https://github.com/OSU-NLP-Group/UGround">UGround (ICLR 2025 Oral)</a> · <a href="https://arxiv.org/abs/2410.23218">OS-ATLAS (ICLR 2025)</a> · <a href="https://arxiv.org/abs/2501.12326">UI-TARS (arXiv 2025)</a> · <a href="https://cua.ai/blog/neurips-2025-cua-papers">NeurIPS 2025 CUA Papers 45편</a>
</div>
