---
title: Mind2Web — Towards a Generalist Agent for the Web
dek: 137개 실제 웹사이트, 2,350개 태스크. 진짜 웹을 항해하는 범용 에이전트를 위한 첫 번째 데이터셋.
desc: 실제 웹사이트 137개·2,350개 태스크로 구성된 Mind2Web 데이터셋과 MindAct 프레임워크 완전 분석.
tags: [Agent, LLM]
date: Mar 2026
readtime: 18 min read
slug: mind2web
katex: false
---

## 왜 이 논문이 필요했는가

웹 에이전트 연구는 사실 오래됐다. 2017년 OpenAI가 MiniWoB을, 2018년에 MiniWoB++를 공개하면서 이 분야가 시작됐다고 봐도 된다. 그런데 연구가 계속 쌓이는데도 뭔가 찜찜했다. 논문 속 에이전트들은 점점 잘하는데, 실제 웹에서 써보면 아무것도 못 한다. 왜일까?

문제는 **환경**에 있었다. 기존 벤치마크들을 뜯어보면:

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.88rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.7rem 1rem;text-align:left;border-bottom:2px solid #ddd;">데이터셋</th>
      <th style="padding:0.7rem 1rem;text-align:center;border-bottom:2px solid #ddd;">환경 수</th>
      <th style="padding:0.7rem 1rem;text-align:center;border-bottom:2px solid #ddd;">페이지당 평균 요소</th>
      <th style="padding:0.7rem 1rem;text-align:left;border-bottom:2px solid #ddd;">한계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;"><strong>MiniWoB++</strong></td>
      <td style="padding:0.7rem 1rem;text-align:center;border-bottom:1px solid #eee;">100</td>
      <td style="padding:0.7rem 1rem;text-align:center;border-bottom:1px solid #eee;">28개</td>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;">장난감 모바일 환경, 실제 웹과 무관</td>
    </tr>
    <tr>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;"><strong>WebShop</strong></td>
      <td style="padding:0.7rem 1rem;text-align:center;border-bottom:1px solid #eee;">1</td>
      <td style="padding:0.7rem 1rem;text-align:center;border-bottom:1px solid #eee;">38개</td>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;">단일 도메인 (전자상거래), 합성 데이터</td>
    </tr>
    <tr>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;"><strong>RUSS</strong></td>
      <td style="padding:0.7rem 1rem;text-align:center;border-bottom:1px solid #eee;">22</td>
      <td style="padding:0.7rem 1rem;text-align:center;border-bottom:1px solid #eee;">—</td>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;">태스크 80개뿐, 스케일 부족</td>
    </tr>
    <tr style="background:#fffbf0;">
      <td style="padding:0.7rem 1rem;"><strong>Mind2Web</strong></td>
      <td style="padding:0.7rem 1rem;text-align:center;"><strong style="background:#fef3c7;padding:2px 6px;border-radius:3px;">137개</strong></td>
      <td style="padding:0.7rem 1rem;text-align:center;"><strong style="background:#fef3c7;padding:2px 6px;border-radius:3px;">1,135개</strong></td>
      <td style="padding:0.7rem 1rem;">31개 도메인, 실제 웹사이트</td>
    </tr>
  </tbody>
</table>

MiniWoB++는 페이지당 요소가 고작 **28개**다. 실제 웹사이트는 **1,135개**. 무려 40배 차이다. 이 환경에서 학습한 에이전트가 실제 웹에서 작동하길 기대하는 건, 체스 AI가 바둑을 둘 수 있기를 기대하는 것과 비슷하다.

논문이 던지는 질문은 명확하다:

<div class="pullquote">
  <strong>"How can we build a generalist agent for the web that, given any website, can follow language instructions and carry out the corresponding tasks?"</strong>
</div>

이 질문에 답하려면 에이전트가 세 가지를 동시에 갖춰야 한다:

**첫째, 일반화 능력 (Generalizability).** 세상의 모든 웹사이트에 대해 학습 데이터를 충분히 확보하는 건 불가능하다. 따라서 학습 때 보지 못한 웹사이트, 심지어 학습 때 보지 못한 도메인에도 적용할 수 있는 일반화 능력이 필수다.

**둘째, 실제 웹의 복잡성 처리 (Real-world complexity).** 실제 웹사이트는 동적이다. 사용자 행동에 반응해 콘텐츠가 실시간으로 바뀐다. 팝업이 뜨고, 로그인 상태에 따라 UI가 달라진다. 기존 시뮬레이터는 이걸 반영하지 못한다: *"Real-world websites...are dynamic, generating and rendering different content in response to user actions."*

**셋째, 다양한 인터랙션 지원 (Diverse interactions).** 단순히 클릭만 하는 게 아니라, 텍스트를 입력하고, 드롭다운에서 옵션을 선택하고, 여러 페이지에 걸쳐 다단계 인터랙션을 수행해야 한다.

<div class="ornament">· · ·</div>

## 데이터셋 구성: 4단계 파이프라인

### 웹사이트 선정

먼저 저자들이 직접 31개 2차 도메인을 선정했다. 5개 상위 카테고리 안에 구성된다:

- **Travel** — 항공사, 호텔, 렌터카
- **Shopping** — 이커머스, 마켓플레이스
- **Service** — 금융, 헬스케어, 유틸리티
- **Entertainment** — 스트리밍, 게임, 소셜 미디어
- **Information** — 뉴스, 검색, 날씨

각 도메인에서 미국 내 인기도 기준으로 **3~5개 웹사이트**를 선택했다. 도메인당 5개로 제한한 건 특정 도메인에 쏠리는 걸 막기 위해서다. 총 137개 웹사이트.

### 1단계 — Task Proposal: 어떻게 좋은 태스크를 수집하나

Amazon Mechanical Turk (MTurk) 작업자들에게 태스크 제안을 맡긴다. 단, 완전한 백지에서 시작하면 태스크의 다양성이 떨어진다. 그래서 **ChatGPT로 미리 웹사이트별 시드 태스크 50개를 생성**해두고, 작업자에게 10개씩 보여준다. 작업자는 이걸 *참고*만 하고 자기만의 태스크 5개를 새로 제안한다.

태스크 제안 기준:
- **다양한 유형**이어야 함
- **여러 번의 인터랙션**이 필요해야 함
- **고수준 목표**로 기술해야 함 — step-by-step 지침이 아니라

> *"describe the high-level goal instead of step-by-step instructions"*

"항공권 검색하기"는 O. "검색창 클릭 → 출발지 입력 → 도착지 입력 → ..."은 X. 이 구분이 중요한 이유는, 실제 언어 지시는 항상 고수준으로 들어오기 때문이다. 우리가 사람에게 심부름을 시킬 때 모든 단계를 설명하지 않는 것처럼.

보수 구조도 흥미롭다. 태스크 제안 자체는 $0.05 명목 보상. 하지만 다음 단계인 **실제 시연에 성공하면 $0.80**을 준다. 불성실한 태스크를 제안하면 시연 단계에서 자기가 고생하는 구조다. 자연스러운 품질 인센티브다.

### 2단계 — Task Demonstration: 실제 웹에서 직접 해보기

제안된 태스크를 실제로 수행하면서 액션 시퀀스를 녹화한다. **Playwright** 기반 커스텀 어노테이션 툴을 사용하는데, 두 개의 창이 뜬다 — 대화 제어창과 실제 브라우저 창.

작업 흐름:
1. 먼저 브라우저를 자유롭게 탐색하며 어떻게 태스크를 수행할지 파악 (이 단계는 녹화 안 됨)
2. 준비되면 녹화 시작
3. DOM 요소를 선택하면 강조 표시만 되고 바로 실행되지 않음
4. 대화 창에서 오퍼레이션을 확인한 뒤 실행

지원하는 오퍼레이션은 딱 세 가지:

<div class="callout">
  <strong>세 가지 오퍼레이션</strong>
  <ul>
    <li><strong>CLICK</strong> — 클릭, 호버, 엔터 키 누르기. 추가 인수 없음.</li>
    <li><strong>TYPE</strong> — 텍스트 입력. 입력할 값(value)을 함께 기록.</li>
    <li><strong>SELECT</strong> — 드롭다운 옵션 선택. 선택된 옵션 값을 함께 기록.</li>
  </ul>
  그리고 특별한 <strong>CLICK(Fake)</strong>가 있다. 게시물 발행, 일정 예약 같이 실제로 실행하면 안 되는 상태 변경 액션에 사용한다. 작업자 개인 정보 보호 및 사이트 무결성 유지용.
</div>

중요한 품질 기준: 최소 **1,000개 이상 승인된 HIT** + **98% 이상 승인률**을 가진 작업자만 참여할 수 있다. 그리고 팝업이나 CAPTCHA는 탐색(exploration) 단계에서 미리 처리해둔다.

### 3단계 — Task Verification: 저자들의 직접 검토

수집된 2,411개 태스크를 저자들이 전부 리뷰했다. 결과:

<div style="background:#f0f4ff;border-left:4px solid #7c9ef5;padding:1.2rem 1.4rem;margin:2rem 0;border-radius:0 6px 6px 0;">
  <strong style="display:block;margin-bottom:0.6rem;">검증 결과</strong>
  <ul style="margin:0;padding-left:1.2rem;">
    <li><strong>61개</strong> 태스크 완전 폐기 (태스크 자체가 부적절)</li>
    <li><strong>390개</strong> 태스크 설명 수정 (실제 수행된 액션과 설명이 불일치)</li>
    <li><strong>187개</strong> 태스크에서 불필요한 여분 스텝 제거</li>
    <li>최종 <strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">2,350개</strong> 태스크 확정</li>
  </ul>
</div>

<div class="ornament">· · ·</div>

## 데이터 샘플 하나를 뜯어보면 (Figure 2)

백 마디 설명보다 실제 예시 하나가 낫다. 논문 Figure 2에서 BBB (Better Business Bureau) 웹사이트 예시를 그대로 가져왔다.

**태스크:** *"Show me the reviews for the auto repair business closest to 10002."*
(우편번호 10002 근처 자동차 수리점 리뷰 보여줘)

데이터 한 샘플은 **세 가지 컴포넌트**로 구성된다.

**① Task Description** — 위처럼 고수준 자연어 목표 한 문장.

**② Action Sequence** — 태스크를 완료하기 위한 (Target Element, Operation) 쌍의 시퀀스:

<table style="width:100%;border-collapse:collapse;margin:1.2rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.5rem 0.8rem;text-align:center;border-bottom:2px solid #ddd;">#</th>
      <th style="padding:0.5rem 0.8rem;text-align:left;border-bottom:2px solid #ddd;">Target Element</th>
      <th style="padding:0.5rem 0.8rem;text-align:left;border-bottom:2px solid #ddd;">Operation</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:0.45rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">1</td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;"><code>[searchbox] Find</code></td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;">TYPE: <em>auto repair</em></td></tr>
    <tr><td style="padding:0.45rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">2</td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;"><code>[button] Auto Repair</code></td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;">CLICK</td></tr>
    <tr><td style="padding:0.45rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">3</td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;"><code>[textbox] Near</code></td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;">TYPE: <em>10002</em></td></tr>
    <tr><td style="padding:0.45rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">4</td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;"><code>[button] 10002</code></td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;">CLICK</td></tr>
    <tr style="background:#fff0f0;"><td style="padding:0.45rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong>5</strong></td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;"><strong><code>[Button] Search</code></strong></td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;"><strong style="color:#e03131;">CLICK 🔴</strong></td></tr>
    <tr><td style="padding:0.45rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">6</td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;"><code>[switch] Show BBB Accredited only</code></td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;">CLICK</td></tr>
    <tr><td style="padding:0.45rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">7</td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;"><code>[vg]</code></td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;">CLICK</td></tr>
    <tr><td style="padding:0.45rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">8</td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;"><code>[button] Sort By</code></td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;">CLICK</td></tr>
    <tr style="background:#fff0f0;"><td style="padding:0.45rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong>9</strong></td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;"><strong><code>[link] Fast Lane 24 Hour Auto Repair</code></strong></td><td style="padding:0.45rem 0.8rem;border-bottom:1px solid #eee;"><strong style="color:#e03131;">CLICK 🔴</strong></td></tr>
    <tr><td style="padding:0.45rem 0.8rem;text-align:center;">10</td><td style="padding:0.45rem 0.8rem;"><code>[link] Read Reviews</code></td><td style="padding:0.45rem 0.8rem;">CLICK</td></tr>
  </tbody>
</table>

<div class="callout">
  <strong>🔴 빨간색 액션 = 새 페이지로 이동</strong><br>
  5번 Search 클릭 → 검색 결과 페이지로 전환. 9번 업체 링크 클릭 → 업체 상세 페이지로 전환. 나머지는 같은 페이지 내 인터랙션. 하나의 태스크가 <strong>여러 페이지에 걸쳐</strong> 진행된다는 게 핵심이다.
</div>

**③ Webpage Snapshots** — 각 액션 시점의 실제 웹페이지 스크린샷 + 타깃 요소의 HTML을 같이 제공한다:

- Action 1 → `<input name="Find_text" type="search">`
- Action 2 → `<em>Auto Repair</em>`
- Action 5 → `<button>Search</button>`
- Action 6 → `<button>Show BBB Accredited only</button>`
- Action 9 → `<span>Fast Lane 24 Hour Auto Repair</span>`
- Action 10 → `<a href="link-XXX">Read Reviews</a>`

스냅샷은 여러 포맷으로 제공된다: raw HTML 전체를 담은 MHTML, DOM 트리 + 레이아웃 정보를 담은 DOM snapshot, 네트워크 트래픽을 담은 HAR 파일, 재현용 trace 파일. 연구자가 원하는 방식으로 모델링할 수 있도록 풀세트를 제공하는 것이다.

에이전트는 매 스텝마다 태스크 설명 + 현재 웹페이지 + 이전 액션 이력을 받아, **다음에 어떤 요소에 무슨 오퍼레이션을 할지** 예측한다.

<div class="ornament">· · ·</div>

## 데이터셋 통계: 규모와 복잡도

<div class="callout">
  <strong>핵심 통계</strong>
  <ul>
    <li>총 태스크: <strong style="background:#fef3c7;padding:2px 6px;border-radius:3px;">2,350개</strong></li>
    <li>웹사이트: <strong>137개</strong> / 31개 도메인</li>
    <li>태스크당 평균 액션 수: <strong>7.3개</strong></li>
    <li>페이지당 평균 DOM 요소: <strong style="background:#fef3c7;padding:2px 6px;border-radius:3px;">1,135개</strong> (클리닝 후 580개, target recall 94.7% 유지)</li>
  </ul>
</div>

페이지당 1,135개 요소라는 숫자가 핵심이다. 이걸 LLM 컨텍스트에 통째로 넣으면 어떻게 될까?

> *"Raw HTML documents...could consist of thousands of elements, are either infeasible or cost-prohibitive to be directly fed into LLMs"*

GPT-4의 컨텍스트 윈도우(당시 8K~32K)로는 불가능하거나 비용이 폭발한다. 이게 MindAct의 설계 이유다.

<div class="ornament">· · ·</div>

## MindAct: 2단계 파이프라인

문제를 다시 정의하자. 에이전트는 매 스텝마다 이걸 해야 한다:

1. 현재 페이지의 수천 개 DOM 요소 중 **어떤 요소**를 건드릴지 결정
2. 그 요소에 **무슨 오퍼레이션**을 할지 결정 (CLICK / TYPE / SELECT)
3. TYPE이나 SELECT라면 **어떤 값**을 입력/선택할지 결정

MindAct는 이 세 가지를 **Stage 1 (필터링)** + **Stage 2 (예측)** 로 분리해 해결한다.

<div style="background:#f0f4ff;border-left:4px solid #7c9ef5;padding:1.2rem 1.4rem;margin:2rem 0;border-radius:0 6px 6px 0;">
  <strong style="display:block;margin-bottom:0.8rem;font-size:1.05rem;">MindAct 파이프라인 개요</strong>
  <div style="display:flex;gap:1rem;flex-wrap:wrap;">
    <div style="flex:1;min-width:200px;background:white;padding:0.9rem;border-radius:6px;border:1px solid #d0d9f5;">
      <strong>Stage 1</strong><br>
      <span style="font-size:0.88rem;color:#444;">DeBERTa-v3-base (86M)<br>1,135개 → 상위 50개 후보</span>
    </div>
    <div style="flex:1;min-width:200px;background:white;padding:0.9rem;border-radius:6px;border:1px solid #d0d9f5;">
      <strong>Stage 2</strong><br>
      <span style="font-size:0.88rem;color:#444;">Flan-T5 / GPT 계열<br>50개 후보 → 정답 요소 + 오퍼레이션</span>
    </div>
  </div>
</div>

### Stage 1: Candidate Generation — 작은 모델로 크게 걸러내기

**DeBERTa-v3-base** (8600만 파라미터, 인코더 전용)를 **cross-encoder** 구조로 파인튜닝한다.

Cross-encoder가 뭔지 간단히 설명하면 — 두 개의 텍스트를 독립적으로 인코딩하는 bi-encoder와 달리, 두 텍스트를 **하나의 시퀀스로 이어붙여** 한 번에 인코딩한다. 이렇게 하면 두 텍스트 간 세밀한 상호작용을 모델이 직접 볼 수 있다. 대신 모든 요소를 각각 쌍으로 인코딩해야 하니 느리다 — 하지만 Stage 1은 정확도보다 **재현율(Recall)** 이 중요하니 괜찮다.

입력 구성:
- **Task query** = 태스크 설명 + 이전 액션 이력
- **Element representation** = HTML 태그 + 텍스트 콘텐츠 + 속성들 + 부모/자식 컨텍스트

각 DOM 요소와 task query를 쌍으로 묶어 DeBERTa에 넣고, sigmoid 활성화로 관련성 점수(0~1)를 뽑는다. 학습은 랜덤 네거티브 샘플링 + 이진 교차 엔트로피 손실.

추론 시에는 전체 DOM 요소를 다 점수 매긴 뒤 **상위 50개**를 추린다.

<div class="callout">
  <strong>Stage 1 Recall@50 — 얼마나 잘 걸러내나</strong>
  <ul>
    <li>Cross-Task: <strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">88.9%</strong></li>
    <li>Cross-Website: <strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">85.3%</strong></li>
    <li>Cross-Domain: <strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">85.7%</strong></li>
  </ul>
  1,135개 요소를 50개로 96% 압축하면서도, 정답 요소가 그 50개 안에 포함될 확률이 85~89%다. 이 단계에서 놓치면 Stage 2가 아무리 잘해도 오답이 되므로, Recall이 핵심 지표다.
</div>

### Stage 2: Action Prediction — 객관식 선택으로 바꾸다

50개 후보를 LLM에게 통째로 주면 여전히 복잡하다. MindAct의 해결책: **다중 선택 QA (multi-choice QA)** 형태로 변환.

구체적으로:
1. 50개 후보를 **5개짜리 그룹**으로 나눈다 (나머지 하나는 "None" 옵션)
2. 각 그룹에 대해 LLM이 "정답이 이 그룹에 있는가? 있다면 어느 것?" 을 판단
3. 선택된 후보가 여러 개면 다시 그룹으로 묶어 반복

LLM에게 주는 입력에는 후보 요소들의 **Pruned HTML snippet** (요소 + 주변 이웃 요소들), 태스크 설명, 이전 액션 이력, 그리고 객관식 선택지가 들어간다.

LLM이 출력하는 것:
- **선택한 요소** (A, B, C, D, E, None 중 하나)
- **오퍼레이션** (CLICK / TYPE / SELECT)
- **값** — TYPE이면 입력 텍스트, SELECT면 선택할 옵션

논문이 이 설계에서 강조하는 핵심 원리:

<div class="pullquote">
  <strong>"Training LMs for discrimination rather than generation is more generalizable."</strong>
</div>

직접 HTML을 생성하거나 XPath를 출력하게 하는 **생성(generation)** 방식보다, 후보 중에서 고르는 **판별(discrimination)** 방식이 훨씬 잘 된다는 것이다. 그리고 이건 실험으로도 확인된다 — generation baseline은 20.2%, MindAct는 55.1%.

<div class="ornament">· · ·</div>

## 평가 설정: 갈수록 어려워지는 세 가지 분할

일반화 능력을 체계적으로 측정하기 위해 세 가지 분할을 정의한다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.88rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.7rem 1rem;text-align:left;border-bottom:2px solid #ddd;">설정</th>
      <th style="padding:0.7rem 1rem;text-align:left;border-bottom:2px solid #ddd;">훈련 데이터</th>
      <th style="padding:0.7rem 1rem;text-align:left;border-bottom:2px solid #ddd;">테스트 데이터</th>
      <th style="padding:0.7rem 1rem;text-align:left;border-bottom:2px solid #ddd;">핵심 질문</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;"><strong>Cross-Task</strong></td>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;">1,009개 태스크 (무작위 80%)</td>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;">252개 태스크 (본 사이트)</td>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;">같은 사이트, 새 태스크도 되나?</td>
    </tr>
    <tr>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;"><strong>Cross-Website</strong></td>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;">도메인별 나머지 사이트</td>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;">177개 태스크 (도메인당 10개 미공개 사이트)</td>
      <td style="padding:0.7rem 1rem;border-bottom:1px solid #eee;">같은 도메인, 새 사이트도 되나?</td>
    </tr>
    <tr>
      <td style="padding:0.7rem 1rem;"><strong>Cross-Domain</strong></td>
      <td style="padding:0.7rem 1rem;">나머지 29개 도메인</td>
      <td style="padding:0.7rem 1rem;">912개 태스크 (Information·Service 도메인 전체)</td>
      <td style="padding:0.7rem 1rem;">완전히 새로운 도메인도 되나?</td>
    </tr>
  </tbody>
</table>

**평가 지표** — BBB 예시로 뜯어보자. 3번 스텝 상황이다: 에이전트는 `[textbox] Near` 에 `10002` 를 입력해야 한다.

<div style="background:#f8f8f4;border:1px solid #e0e0d8;border-radius:8px;padding:1.2rem 1.4rem;margin:1.5rem 0;font-size:0.9rem;">
  <div style="margin-bottom:1rem;"><strong>정답 (ground truth):</strong> <code>[textbox] Near</code> → TYPE: <em>10002</em></div>
  <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
    <thead>
      <tr style="background:#eeeee8;">
        <th style="padding:0.5rem 0.8rem;text-align:left;border-bottom:1px solid #ccc;">에이전트 예측</th>
        <th style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #ccc;">Element Accuracy</th>
        <th style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #ccc;">Operation F1</th>
        <th style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #ccc;">Step SR</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:0.5rem 0.8rem;border-bottom:1px solid #eee;"><code>[textbox] Near</code> → TYPE: <em>10002</em> ✅</td>
        <td style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:1px 5px;border-radius:3px;">✅</strong></td>
        <td style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:1px 5px;border-radius:3px;">✅</strong></td>
        <td style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:1px 5px;border-radius:3px;">✅</strong></td>
      </tr>
      <tr>
        <td style="padding:0.5rem 0.8rem;border-bottom:1px solid #eee;"><code>[textbox] Near</code> → TYPE: <em>New York</em> ← 요소는 맞는데 값이 틀림</td>
        <td style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:1px 5px;border-radius:3px;">✅</strong></td>
        <td style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#ffd6d6;padding:1px 5px;border-radius:3px;">❌</strong></td>
        <td style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#ffd6d6;padding:1px 5px;border-radius:3px;">❌</strong></td>
      </tr>
      <tr>
        <td style="padding:0.5rem 0.8rem;border-bottom:1px solid #eee;"><code>[searchbox] Find</code> → TYPE: <em>10002</em> ← 값은 맞는데 요소가 틀림</td>
        <td style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#ffd6d6;padding:1px 5px;border-radius:3px;">❌</strong></td>
        <td style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#ffd6d6;padding:1px 5px;border-radius:3px;">❌</strong></td>
        <td style="padding:0.5rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#ffd6d6;padding:1px 5px;border-radius:3px;">❌</strong></td>
      </tr>
      <tr>
        <td style="padding:0.5rem 0.8rem;"><code>[textbox] Near</code> → CLICK ← 요소는 맞는데 오퍼레이션이 틀림</td>
        <td style="padding:0.5rem 0.8rem;text-align:center;"><strong style="background:#d4f7d4;padding:1px 5px;border-radius:3px;">✅</strong></td>
        <td style="padding:0.5rem 0.8rem;text-align:center;"><strong style="background:#ffd6d6;padding:1px 5px;border-radius:3px;">❌</strong></td>
        <td style="padding:0.5rem 0.8rem;text-align:center;"><strong style="background:#ffd6d6;padding:1px 5px;border-radius:3px;">❌</strong></td>
      </tr>
    </tbody>
  </table>
</div>

즉:

- **Element Accuracy** — 1,135개 DOM 요소 중 **올바른 요소를 골랐는가**만 본다. 오퍼레이션이나 값은 신경 안 씀.
- **Operation F1** — 오퍼레이션(CLICK/TYPE/SELECT) + 값(입력 텍스트, 선택 옵션)이 맞는지를 F1으로 측정. 요소 선택은 이미 맞았다고 가정하고 채점.
- **Step SR (Step Success Rate)** — 요소 + 오퍼레이션 + 값 **셋 다 맞아야** 1점. 하나라도 틀리면 0점. 스텝 전체에 걸쳐 평균낸 비율.
- **Task SR (Task Success Rate)** — 태스크의 **모든 스텝이 전부 Step SR = 1**이어야 성공. BBB 예시라면 10개 스텝을 단 하나도 안 틀리고 다 맞혀야 한다. 하나라도 틀리면 태스크 실패.

<div class="ornament">· · ·</div>

## 실험 결과: 숫자들이 말하는 것

### 전체 결과 테이블

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.85rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.8rem;text-align:left;border-bottom:2px solid #ddd;">모델</th>
      <th style="padding:0.6rem 0.8rem;text-align:left;border-bottom:2px solid #ddd;">설정</th>
      <th style="padding:0.6rem 0.8rem;text-align:center;border-bottom:2px solid #ddd;">Elem Acc</th>
      <th style="padding:0.6rem 0.8rem;text-align:center;border-bottom:2px solid #ddd;">Op F1</th>
      <th style="padding:0.6rem 0.8rem;text-align:center;border-bottom:2px solid #ddd;">Step SR</th>
      <th style="padding:0.6rem 0.8rem;text-align:center;border-bottom:2px solid #ddd;">Task SR</th>
    </tr>
  </thead>
  <tbody>
    <tr style="color:#888;">
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">DeBERTa (분류)</td>
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">Cross-Task</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">26.8%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">—</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">—</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">—</td>
    </tr>
    <tr style="color:#888;">
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">Flan-T5-B (생성)</td>
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">Cross-Task</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">20.2%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">52.0%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">17.5%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">0.0%</td>
    </tr>
    <tr style="color:#888;">
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">GPT-3.5-turbo</td>
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">Cross-Task</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">20.3%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">56.6%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">17.4%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">0.8%</td>
    </tr>
    <tr>
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;"><strong>MindAct (Flan-T5-B)</strong></td>
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">Cross-Task</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">43.6%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">76.8%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">41.0%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">4.0%</td>
    </tr>
    <tr>
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;"><strong>MindAct (Flan-T5-L)</strong></td>
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">Cross-Task</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">53.4%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">75.7%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">50.3%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">7.1%</td>
    </tr>
    <tr style="background:#fffbf0;">
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;"><strong>MindAct (Flan-T5-XL)</strong></td>
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">Cross-Task</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#fef3c7;padding:2px 5px;border-radius:3px;">55.1%</strong></td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong>75.7%</strong></td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#fef3c7;padding:2px 5px;border-radius:3px;">52.0%</strong></td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong>5.2%</strong></td>
    </tr>
    <tr style="background:#fffbf0;">
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;"><strong>MindAct (Flan-T5-XL)</strong></td>
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">Cross-Website</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#fef3c7;padding:2px 5px;border-radius:3px;">42.0%</strong></td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong>65.2%</strong></td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#fef3c7;padding:2px 5px;border-radius:3px;">38.9%</strong></td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong>5.1%</strong></td>
    </tr>
    <tr style="background:#fffbf0;">
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;"><strong>MindAct (Flan-T5-XL)</strong></td>
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">Cross-Domain</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#fef3c7;padding:2px 5px;border-radius:3px;">42.1%</strong></td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong>66.5%</strong></td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#fef3c7;padding:2px 5px;border-radius:3px;">39.6%</strong></td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;"><strong>2.9%</strong></td>
    </tr>
    <tr style="color:#666;">
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">GPT-4 (50 tasks)</td>
      <td style="padding:0.6rem 0.8rem;border-bottom:1px solid #eee;">Cross-Task</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">41.6%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">60.6%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">36.2%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;border-bottom:1px solid #eee;">2.0%</td>
    </tr>
    <tr style="color:#666;">
      <td style="padding:0.6rem 0.8rem;">GPT-4 (50 tasks)</td>
      <td style="padding:0.6rem 0.8rem;">Cross-Domain</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;">37.1%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;">46.5%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;">26.4%</td>
      <td style="padding:0.6rem 0.8rem;text-align:center;">2.0%</td>
    </tr>
  </tbody>
</table>

<div class="ornament">· · ·</div>

## 실험에서 나온 핵심 발견들

### 발견 1: 일반화 갭은 10% 이상이다

<div style="background:#fff0f3;border-left:4px solid #f87171;padding:1rem 1.2rem;margin:1.5rem 0;border-radius:0 6px 6px 0;">
  <strong>"All models perform best on the Cross-Task setting, with over 10% absolute gap (step SR) on average compared with Cross-Website and Cross-Domain settings."</strong><br>
  <span style="font-size:0.9rem;color:#555;margin-top:0.5rem;display:block;">Cross-Task 52.0% → Cross-Website 38.9%. 10%p 이상 폭락. 같은 사이트에서 새 태스크를 하는 것 자체는 잘하지만, 사이트가 바뀌면 바로 무너진다. 웹사이트별 레이아웃, 버튼 구조, 네이밍 컨벤션이 모두 다르기 때문이다.</span>
</div>

### 발견 2: Cross-Website ≈ Cross-Domain — 도메인보다 사이트 디자인이 더 어렵다

가장 놀라운 결과 중 하나다. 직관적으로는 완전히 새로운 도메인(Cross-Domain)이 같은 도메인의 새 사이트(Cross-Website)보다 훨씬 어려울 거라 예상하지만, **두 설정의 성능이 거의 같다** — Cross-Website 38.9%, Cross-Domain 39.6%.

왜일까? 저자들의 해석:

> *"Website design diversity dominates domain-specific challenges."*

여행 사이트들이 쇼핑 사이트들과 사용하는 HTML 패턴, 버튼 구조, 폼 레이아웃이 생각보다 많이 겹친다. 반면 같은 여행 도메인이라도 에어비앤비와 익스피디아는 UI가 완전히 다르다. **도메인 지식보다 UI 패턴 다양성이 더 큰 장벽**이라는 의미다.

이건 웹 에이전트 연구에서 중요한 시사점이다. "도메인 특화 에이전트"를 만드는 것보다, "다양한 UI 패턴에 강인한 에이전트"를 만드는 게 더 중요할 수 있다.

### 발견 3: Task Success Rate가 처참히 낮다 — 수학적으로 당연하다

<div style="background:#fff8ed;border-left:4px solid #f59e0b;padding:1rem 1.2rem;margin:1.5rem 0;border-radius:0 6px 6px 0;">
  <strong>최고 성능인 Cross-Task에서도 Task SR = 5.2%.</strong><br>
  <span style="font-size:0.9rem;color:#555;margin-top:0.5rem;display:block;">이건 사실 수학적으로 예상 가능하다. 스텝당 Step SR이 52%라고 하자. 태스크 평균 길이가 7.3스텝이면, 모든 스텝을 맞힐 확률은 대략 0.52^7 ≈ 0.9%. 실제 Task SR 5.2%는 이보다 높은데, 이는 스텝들이 완전히 독립적이지 않고 문맥을 공유하기 때문이다. 어쨌든, "agent often commits at least one error step in most cases"라고 논문이 직접 인정한다.</span>
</div>

이게 웹 에이전트의 근본적인 난제다. 스텝 하나하나를 잘해도, 긴 체인을 오류 없이 완주하는 건 훨씬 어렵다. 에러가 누적되고 전파된다.

### 발견 4: GPT-4가 생각보다 강하다

GPT-4는 파인튜닝 없이 in-context learning만으로 Cross-Task에서 Element Accuracy 41.6%를 기록했다. 파인튜닝된 Flan-T5-XL의 55.1%보다는 낮지만, **GPT-3.5-turbo(20.3%)와는 비교가 안 될 만큼 높다.**

단, 비용 문제로 태스크당 샘플 수를 50개로 제한했기 때문에, 통계적 신뢰도에는 한계가 있다.

### 발견 5: 파인튜닝이 필수다

제로샷 Flan-T5-XL은 Element Accuracy **10.8%**에 불과하다. 파인튜닝 후 **52.0%**. instruction-tuned 모델이라도 이 태스크에 대해선 파인튜닝이 필수라는 뜻이다. "대형 LLM이면 그냥 쓰면 된다"는 통념이 여기선 통하지 않는다.

### 발견 6: 모델 크기 스케일링은 작동한다

Flan-T5-B (Step SR 41.0%) → Flan-T5-L (50.3%) → Flan-T5-XL (52.0%). 모델이 커질수록 성능이 일관되게 오른다. XL에서의 증분이 B→L보다 작아지긴 하지만, 여전히 스케일링 이득이 있다.

<div class="ornament">· · ·</div>

## 에러 분석: 어디서 틀리나

논문이 직접 지목하는 GPT-3.5의 대표적 실패 패턴이 흥미롭다:

> *"Model exhibits propensity to select the None option, asserting that the task cannot be finished on the current webpage."*

GPT-3.5가 자꾸 "이 페이지에서는 태스크를 완료할 수 없다"고 None을 선택한다는 것. 이게 실제로 맞는 경우도 있다 — 멀티페이지 태스크에서 다음 페이지로 넘어가야 할 때. 하지만 전반적으로 너무 보수적이어서 실패율이 높다.

전반적인 에러의 구조적 원인:

<div class="callout">
  <strong>에러가 나는 주요 원인</strong>
  <ul>
    <li><strong>Stage 1 실패 (14~15%):</strong> 정답 요소가 top-50에 포함되지 않으면, Stage 2가 아무리 잘해도 끝</li>
    <li><strong>유사 요소 혼동:</strong> 비슷한 역할의 버튼이나 링크가 여러 개일 때 잘못된 것을 선택</li>
    <li><strong>오퍼레이션 예측 오류:</strong> 요소는 맞혔는데 CLICK 해야 할 걸 TYPE으로 예측</li>
    <li><strong>값 예측 오류:</strong> SELECT에서 "뉴욕"을 선택해야 하는데 "NY" 또는 다른 표기를 예측</li>
    <li><strong>에러 전파:</strong> 이전 스텝 실수가 이후 스텝의 컨텍스트를 오염</li>
  </ul>
</div>

<div class="ornament">· · ·</div>

## 한계들: 논문이 스스로 인정한 것들

**1. 텍스트만 본다.** 렌더링된 웹페이지의 시각 정보를 전혀 활용하지 않는다. 아이콘만 있는 버튼, 이미지로 된 배너, CSS로 숨겨진 요소들은 처리 못 한다. 논문 자체가 멀티모달 확장을 미래 방향으로 명시한다.

**2. 오프라인 평가.** 캐시된 정적 스냅샷으로 평가한다. 실제 웹은 동적이고 살아있다. 더 큰 문제는 **대안 경로(alternative paths)**를 무시한다는 것 — 다른 경로로 똑같은 결과를 달성해도 오답 처리될 수 있다.

**3. 영어·미국 사이트 편향.** MTurk 작업자 특성상 인터넷에 익숙한 미국인 편향. 비영어권 사이트, 비서구권 웹 UI 패턴은 커버되지 않는다.

**4. 안전 문제.** 논문이 직접 지적한다:
- CAPTCHA 우회에 악용될 수 있음
- 금융 거래 자동화 위험
- 악의적 사용자에 의한 해로운 자동화

이를 위해 데이터셋 접근 조건에 용도 제한을 명시했다.

<div class="ornament">· · ·</div>

## 저자들이 제시하는 미래 방향

<div class="callout">
  <strong>논문이 직접 제시한 세 가지 방향</strong>
  <ul>
    <li><strong>멀티모달 통합:</strong> 렌더링된 웹페이지 스크린샷 + HTML을 함께 활용 (이후 SeeAct, WebVoyager가 실현)</li>
    <li><strong>온라인 강화학습:</strong> 오프라인 평가의 한계를 극복하고 실제 환경에서 피드백 반영</li>
    <li><strong>웹 에이전트 특화 LLM:</strong> 일반 instruction-tuning 대신 웹 네비게이션에 특화된 모델 개발</li>
  </ul>
</div>

<div class="ornament">· · ·</div>

## 마무리

Mind2Web은 2023년 기준으로 웹 에이전트 연구에 가장 현실적인 기준점을 제시한 논문이다. MiniWoB++의 장난감 환경에서 벗어나, 진짜 복잡한 웹을 직시하자는 선언이었다.

MindAct의 Task Success Rate 5%는 초라해 보일 수 있다. 하지만 이렇게 생각해보자 — 연구자들이 처음으로 진짜 문제의 난이도를 직면했고, 그걸 측정할 수 있는 척도를 만들었다. **"어렵다"는 걸 정확히 아는 것이 발전의 시작이다.**

논문 제목의 "Towards"가 핵심이다. 이건 완성된 해결책이 아니라 방향 설정이다.

> *"There is still substantial room for further improvement towards generalist agents for the web."*

이후 SeeAct가 시각을 더했고, WebVoyager가 GPT-4V로 실제 브라우저를 탐색했고, 지금은 Claude Computer Use, OpenAI Operator까지 왔다. 그 출발점에 Mind2Web이 있다.

<div class="footnote">
  논문: <a href="https://arxiv.org/abs/2306.06070">Mind2Web: Towards a Generalist Agent for the Web (NeurIPS 2023 Spotlight)</a><br>
  저자: Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, Yu Su (Ohio State University)
</div>
