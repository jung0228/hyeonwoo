---
title: "WebNavigator: Global Web Navigation via Interaction Graph Retrieval"
dek: "웹 에이전트의 '지도 없는 탐색' 문제를 Interaction Graph로 해결한 ICLR 2026 연구"
desc: "웹 에이전트의 근본 한계인 Topological Blindness를 진단하고, 오프라인 그래프 탐색 + 온라인 Retrieve-Reason-Teleport 워크플로우로 WebArena 멀티사이트 72.9% 달성."
tags: ["Agent", "LLM"]
date: "Apr 2026"
readtime: "11 min read"
slug: web-navigator
katex: false
---

## 웹 에이전트는 왜 아직도 길을 잃는가

<p><strong>흐름:</strong> 현재 에이전트의 한계 진단 → Topological Blindness 개념 도입 → 기존 두 접근법의 한계 → WebNavigator 소개</p>

GPT-4가 수학 증명과 코드 생성에서 인간을 뛰어넘었는데도, 웹 내비게이션 성공률은 여전히 인간에 한참 못 미친다. 논문은 이 격차의 원인을 모델의 추론 능력 부족으로 보지 않는다.

*"We argue that this limitation stems from Topological Blindness rather than simply insufficient model reasoning."*

### Topological Blindness란

현재 웹 에이전트는 의사결정 시 `{현재 관찰, 히스토리, LLM 내부 지식}`만 참조한다. 웹사이트 **전체 구조가 어떻게 생겼는지** — 어떤 페이지들이 존재하고, 어떤 행동이 어떤 페이지로 연결되는지 — 를 모르는 채로 탐색하는 것이다.

인간 전문가라면 웹사이트의 '지도'를 머릿속에 갖고 있다. 에이전트는 그 지도가 없다. 그 결과:

- **불안정한 계획**: 전체 구조를 모르니 엉뚱한 경로를 선택
- **시행착오 비용**: 맞는 페이지를 찾을 때까지 무한 탐색
- **조기 종료**: 길을 못 찾으면 그냥 포기

### 기존 접근법의 한계

<mark>Topological Blindness를 해결하려는 시도는 두 계열이 있었지만 둘 다 근본적 한계가 있다.</mark>

**Paradigm 1 — 온라인 탐색 기반 (Tree Search 계열)**
추론 중에 look-ahead 탐색으로 더 넓은 관찰을 확보하려 한다. 문제는 매 태스크마다 처음부터 탐색을 다시 해야 해서 — *"reinventing the wheel for every new task"* — 비용이 크고, 그 지식은 태스크가 끝나면 버려진다.

**Paradigm 2 — 학습된 내부 지식 (World Model 계열)**
WMA 같은 방법이 여기 해당된다. 모델이 환경 동역학을 내부에 담으려 하지만, 파라미터 지식은 사이트별 세부사항을 담기에 너무 희소하고, 월드 모델은 새 사이트에서의 일반화에 실패하며 오류가 누적된다.

두 패러다임의 공통 문제: 모두 **전역 환경 정보 없이** 결정을 내린다.

<div class="ornament">· · ·</div>

## WebNavigator: 탐색을 검색으로 바꾸다

<p><strong>흐름:</strong> 핵심 아이디어 → Phase I (오프라인 그래프 구축) → Phase II (온라인 검색 기반 내비게이션) → 에이전트 설계</p>

### 핵심 아이디어

<mark>웹 내비게이션을 "확률적 탐색"이 아니라 "결정론적 검색 + 경로탐색"으로 재정의한다.</mark>

사전에 웹사이트를 탐색해 **Interaction Graph**를 만들어 두고, 실제 태스크 수행 시에는 그래프에서 목적지를 검색해 최단 경로로 텔레포트한다. LLM이 "어디로 갈지"만 말하면, 나머지는 그래프가 결정론적으로 처리한다.

<figure>
<img src="img/web-navigator/fig1_overview.jpg" alt="WebNavigator Overview">
<figcaption><strong>Figure 1</strong> — WebNavigator 전체 구조. 왼쪽(Phase I): 오프라인에서 Interaction Graph를 구축하고 벡터 DB에 인덱싱. 오른쪽(Phase II): 온라인에서 Retrieve → Reason → Teleport 워크플로우로 내비게이션.</figcaption>
</figure>

### Phase I — 오프라인 Interaction Graph 구축

**Interaction Graph** `G = (V, E)`의 정의:
- **노드** `v ∈ V`: 고유한 페이지 관찰 (스크린샷 + DOM tree + accessibility tree)
- **엣지** `e = (v, a, v') ∈ E`: 노드 `v`에서 행동 `a`를 취하면 `v'`로 전이

이 그래프를 어떻게 만드나? **Heuristic Auto-Exploration Engine** — BFS 기반 탐색 엔진이 홈페이지 URL 하나만 받아서 자동으로 그래프를 구축한다.

<div class="callout">
<strong>핵심:</strong> Zero-token cost. LLM을 전혀 쓰지 않는다. 탐색에 드는 비용은 실제 브라우저 인터랙션뿐이다. 한 번 만들어두면 재사용 가능하다.
</div>

단순 BFS의 문제는 웹 인터랙션이 대부분 **부분적인 변화**만 만든다는 점이다. 부모 페이지와 자식 페이지의 요소 대부분이 동일하다. 그래서 **Adaptive BFS**를 설계했다:

1. 각 노드를 DOM tree + URL의 해시로 유일하게 식별: `id_v = MD5(H_v || url_v)`
2. 부모 노드와 자식 노드의 구조 해시 집합 차이를 계산: `ΔH_v = H_v \ H_v_parent`
3. 새로 등장한 요소(`ΔH_v`)에서만 인터랙티브 엘리먼트를 추출해 탐색

이렇게 탐색이 끝난 모든 노드는 스크린샷을 멀티모달 임베딩으로 변환해 벡터 DB에 인덱싱한다.

### Phase II — 온라인 Retrieve-Reason-Teleport

태스크가 주어지면 세 단계로 목적지 페이지로 이동한다.

**① Retrieve — 후보 검색**

에이전트가 내비게이션 의도(`intent`)를 바탕으로 쿼리 `q`를 작성한다. 예: "Edit product X's price to $50" → query: "page to edit product information"

이 쿼리를 멀티벡터 임베딩으로 변환해 DB에서 top-k 후보 페이지를 검색한다. 검색 방식은 **Late Interaction** — 쿼리 토큰과 스크린샷 토큰 사이의 fine-grained 매칭:

```
s(q, v_j) = (1/n) Σ_i max_ℓ (q_i · v_j,ℓ)
```

단일 벡터 압축이 아니라 토큰 수준 유사도를 계산해, "특정 버튼", "특정 입력 필드" 같은 세밀한 시각 정보까지 포착한다.

**② Reason — 최적 후보 선택**

top-k 후보 스크린샷들을 멀티모달 LLM에 넘겨 의도에 맞는 최적 페이지 `v*`를 고른다. 탐색(generation)이 아니라 **검증(verification)** 문제로 단순화한 것이 핵심이다.

**③ Teleport — 최단 경로 실행**

`v*`가 결정되면, Interaction Graph에서 현재 위치 → `v*`의 최단 경로를 계산해 자동 실행한다. 역시 **zero-token cost**.

에이전트 관점에서 이 전체 워크플로우는 단 하나의 액션으로 추상화된다: `navigate(domain, query)`

### 에이전트 설계 — 6개 액션으로 충분하다

<mark>WebNavigator는 비교 대상 중 가장 컴팩트한 액션 공간 — 총 6개 액션만 사용한다.</mark>

`navigate(domain, query)` 하나가 크로스도메인 계획, 탭 관리, 경로탐색을 모두 흡수한다. 나머지 5개는 로컬 실행 액션(클릭, 입력 등)이다. 기존 방법들이 `tab_focus`, `new_tab`, `tab_close`를 별도 액션으로 갖거나 사이트별 전용 액션(GitLab용 `find_commits` 등)을 하드코딩하는 것과 대비된다.

<figure>
<img src="img/web-navigator/fig2_trajectory.jpg" alt="Trajectory comparison on WebArena task 760">
<figcaption><strong>Figure 2</strong> — WebArena task 760: "Allentown, PA에서 고객 Amanda Kim이 사는 곳까지의 경로를 보여줘." ReAct 에이전트는 Topological Blindness로 조기 종료하지만, WebNavigator는 CMS에서 주소 검색 후 Map으로 텔레포트해 태스크를 완료한다.</figcaption>
</figure>

<div class="ornament">· · ·</div>

## 실험 결과

<p><strong>흐름:</strong> WebArena 메인 결과 → 멀티사이트 성능 → Online-Mind2Web → 도메인별 분석</p>

### WebArena — 새 SOTA

<table>
<thead>
<tr><th>방법</th><th>Paradigm</th><th>WebArena SR</th></tr>
</thead>
<tbody>
<tr><td>WebPilot</td><td>온라인 탐색</td><td>37.2%</td></tr>
<tr><td>Plan-and-Act</td><td>내부 지식</td><td>45.7%</td></tr>
<tr><td>WebNavigator (GPT-4o)</td><td>그래프 검색</td><td>49.9%</td></tr>
<tr><td>WebNavigator (Claude-Sonnet-4)</td><td>그래프 검색</td><td>57.1%</td></tr>
<tr style="background:#fef9c3;font-weight:700;"><td>WebNavigator (Gemini-2.5-Pro)</td><td>그래프 검색</td><td>63.3%</td></tr>
</tbody>
</table>

멀티사이트 태스크에서 성과가 특히 두드러진다. GPT-4o와 Claude-Sonnet-4 기준 **50.0%** — 이전 SOTA AgentOccam(25.0%) 대비 **2배**. Gemini-2.5-Pro로는 **72.9%**, 엔터프라이즈급 에이전트 CUGA의 2배 이상이다.

### Online-Mind2Web — 136개 실제 웹사이트

Gemini-2.5-Pro 기준 **52.7%**, GPT-4o 기준 41.3%로 SOTA 달성. WebArena(5개 사이트)가 아닌 136개 다양한 실제 웹사이트에서도 일반화가 된다는 점에서 의미가 크다.

### 왜 도메인마다 향상 폭이 다른가

Topological Blindness의 심각도가 도메인마다 다르다.

- **Reddit**: 포럼이 90개 이상인 넓고 얕은 구조. 전체 포럼 목록 없이는 맞는 포럼을 찾을 확률이 낮음 → Interaction Graph 효과 극대화
- **Shopping, CMS, GitLab**: 깊고 복잡한 구조. 관련 페이지가 표면 뒤에 숨어 있어 에이전트가 쉽게 함정에 빠짐 → 큰 향상
- **Map**: 노드 수 16개에 불과한 매우 단순한 위상 → Topological Blindness가 거의 없어 모델 간 성능 차이 없음

<div class="ornament">· · ·</div>

## 웹 위상의 실제 구조 분석

<p><strong>흐름:</strong> Topological Skeleton 가설 → Discovery Velocity 분석 → 시사점</p>

*"Prior research characterizes web environments as effectively infinite observation spaces. In contrast, we hypothesize that individual websites possess compact topological skeletons."*

논문은 웹사이트의 기능적 구조(Topological Skeleton)가 실제로 얼마나 compact한지 탐색 깊이에 따른 **Discovery Velocity**로 분석한다.

<figure>
<img src="img/web-navigator/fig_node_reddit.jpg" alt="Node exploration Reddit">
<figcaption><strong>Figure 3a</strong> — Reddit: 깊이 2에서 discovery velocity가 최고점을 찍고 급격히 감소. 주요 포럼들이 얕은 곳에 집중.</figcaption>
</figure>

<figure>
<img src="img/web-navigator/fig_node_cms.jpg" alt="Node exploration CMS">
<figcaption><strong>Figure 3b</strong> — CMS: Reddit과 비슷하게 깊이 2에서 peak. 핵심 기능 페이지들이 조기에 발견됨.</figcaption>
</figure>

<figure>
<img src="img/web-navigator/fig_node_gitlab.jpg" alt="Node exploration GitLab">
<figcaption><strong>Figure 3c</strong> — GitLab: 깊이 4까지 감소하다가 5에서 다시 상승. 대시보드·설정은 얕고, 저장소 세부 설정은 깊은 곳에 있음.</figcaption>
</figure>

<figure>
<img src="img/web-navigator/fig_node_map.jpg" alt="Node exploration Map">
<figcaption><strong>Figure 3d</strong> — Map: 노드 29개로 완전 탐색. 깊이 5에서 velocity가 0에 수렴.</figcaption>
</figure>

결론: 이론적으로 무한해 보이는 웹사이트도, **기능적 골격은 컴팩트**하다. 태스크 관련 페이지들은 얕은 깊이에 집중되어 있고, 탐색 깊이 2~3 정도면 대부분의 태스크를 커버할 수 있다.

<div class="ornament">· · ·</div>

## 심층 분석

<p><strong>흐름:</strong> Knowledge Completeness → Information Bandwidth → Task Simplification → Retrieval Granularity</p>

### 지식 완전성 (탐색 깊이)

Reddit 도메인에서 탐색 깊이를 1~4로 변화시킨 실험:

| 탐색 깊이 | 성공률 |
|---|---|
| depth 1 | 63.2% |
| depth 2 | 70.8% |
| depth 3 | 73.6% |
| depth 4 | 75.5% |

깊이 1→2 구간에서 급상승 후 수렴. **지식 완전성이 성능의 일차 결정 요인**임을 확인.

### Task Simplification — 작은 모델로도 충분

<mark>Selector를 8B 모델(Qwen3-VL-8B)로 써도 GPT-4o, Gemini-2.5-Flash와 동등한 75.5% 달성.</mark>

*"This consistency across model scales confirms that WebNavigator successfully offloads navigation complexity from model reasoning to structured knowledge retrieval."*

내비게이션 복잡도를 그래프가 흡수하기 때문에, LLM이 할 일은 top-k 후보 중 하나를 고르는 verification 문제로 단순화된다.

### Retrieval Granularity — Late Interaction의 중요성

| 검색 방식 | 성공률 |
|---|---|
| Dense embedding | 66~67% |
| **Late Interaction (ours)** | **73.6%** |

웹 내비게이션은 "특정 버튼", "특정 폼 필드"를 정확히 찾아야 하는 fine-grained 문제다. 전체 스크린샷을 단일 벡터로 압축하면 이런 세밀한 공간 정보가 사라진다. 토큰 수준 매칭이 필수적이다.

<div class="ornament">· · ·</div>

## 정리

<mark>WebNavigator는 "모델이 더 잘 추론하게 만들자"는 방향이 아니라, "에이전트에게 지도를 줘서 탐색 문제 자체를 없애자"는 방향을 택한다.</mark>

핵심 기여:

1. **Topological Blindness**: 웹 에이전트의 실패 원인을 새 개념으로 명확히 진단
2. **Zero-token cost 오프라인 탐색**: LLM 없이 Interaction Graph를 구축해 재사용 가능한 지식으로 저장
3. **Retrieve-Reason-Teleport**: 내비게이션을 검색 문제로 전환 — 모델 규모와 무관하게 작동
4. **WebArena 멀티사이트 72.9%**: 기존 SOTA 대비 2배 이상, 기업급 에이전트도 뛰어넘음

남은 과제는 동적으로 변하는 웹사이트(콘텐츠가 수시로 바뀌는 뉴스 사이트, 이커머스 등)에서 그래프를 얼마나 최신 상태로 유지할 수 있는가, 그리고 로그인이 필요한 인증 환경으로의 확장이다.

<div class="footnote">
원문: <a href="https://fate-ubw.github.io/webNavigator_homepage/">WebNavigator: Global Web Navigation via Interaction Graph Retrieval</a> (ICLR 2026) &nbsp;·&nbsp; <a href="https://github.com/fate-ubw/webNavigator">GitHub</a>
</div>
