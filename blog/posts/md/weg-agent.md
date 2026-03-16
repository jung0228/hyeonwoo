---
title: "웹 에이전트 직접 만들어보기: SoM + ReAct + LLM-as-Judge"
dek: Set-of-Mark으로 DOM을 시각화하고, Claude Vision으로 다나와 PC 견적을 자동 탐색한 실험 기록.
desc: SoM 퍼셉션, ReAct 루프, LLM-as-Judge 평가까지 — 웹 에이전트의 핵심 컴포넌트를 직접 구현하며 배운 것들.
tags: [Agent, Multimodal, LLM]
date: Mar 2026
readtime: 14 min read
slug: weg-agent
katex: false
---

## 왜 만들었나

웹 에이전트 논문은 많이 읽었는데, 실제로 돌아가는 코드를 직접 써본 적은 없었다. WebVoyager, Mind2Web, SoM 방법론들을 정리하면서 계속 드는 생각이 있었다 — *저게 실제로 어떻게 작동하는 거지?* 이론으로 아는 것과 구현이 돌아가는 것 사이에는 항상 간극이 있다.

그래서 만들어봤다. 목표는 단순하다: **다나와에서 예산 내 PC 견적을 자동으로 찾아주는 에이전트.** 예산과 사용 목적을 주면 에이전트가 직접 다나와를 탐색해 부품별 가격을 찾고, 견적을 구성해 보고한다.

이 글은 구현 과정에서 배운 것들, 그리고 실제로 작동시켜보니 어디서 막혔는지에 대한 기록이다.

<div class="ornament">· · ·</div>

## 전체 구조

에이전트는 크게 세 단계로 작동한다.

<div style="margin: 2rem 0;">
<svg viewBox="0 0 720 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:720px;display:block;margin:0 auto;font-family:'Source Serif 4',serif;">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#555"/>
    </marker>
  </defs>
  <!-- Task input -->
  <rect x="10" y="130" width="130" height="60" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
  <text x="75" y="155" text-anchor="middle" font-size="12" fill="#92400e" font-weight="bold">사용자 태스크</text>
  <text x="75" y="172" text-anchor="middle" font-size="11" fill="#78350f">예산 + 목적</text>
  <!-- arrow -->
  <line x1="140" y1="160" x2="175" y2="160" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- ReAct loop box -->
  <rect x="175" y="60" width="370" height="200" rx="10" fill="none" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="360" y="82" text-anchor="middle" font-size="11" fill="#6b7280">ReAct 루프 (최대 20스텝)</text>
  <!-- OBSERVE -->
  <rect x="195" y="95" width="100" height="52" rx="6" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="245" y="116" text-anchor="middle" font-size="11" fill="#1d4ed8" font-weight="bold">① OBSERVE</text>
  <text x="245" y="132" text-anchor="middle" font-size="10" fill="#1e40af">SoM 스크린샷</text>
  <text x="245" y="144" text-anchor="middle" font-size="10" fill="#1e40af">생성</text>
  <!-- arrow -->
  <line x1="295" y1="121" x2="322" y2="121" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- THINK -->
  <rect x="322" y="95" width="100" height="52" rx="6" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
  <text x="372" y="116" text-anchor="middle" font-size="11" fill="#065f46" font-weight="bold">② THINK</text>
  <text x="372" y="132" text-anchor="middle" font-size="10" fill="#047857">Claude Vision</text>
  <text x="372" y="144" text-anchor="middle" font-size="10" fill="#047857">→ Action 결정</text>
  <!-- arrow -->
  <line x1="422" y1="121" x2="449" y2="121" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- ACT -->
  <rect x="449" y="95" width="80" height="52" rx="6" fill="#fce7f3" stroke="#ec4899" stroke-width="1.5"/>
  <text x="489" y="116" text-anchor="middle" font-size="11" fill="#9d174d" font-weight="bold">③ ACT</text>
  <text x="489" y="132" text-anchor="middle" font-size="10" fill="#be185d">Playwright</text>
  <text x="489" y="144" text-anchor="middle" font-size="10" fill="#be185d">실행</text>
  <!-- loop back arrow -->
  <path d="M 489 147 Q 489 210 360 210 Q 231 210 231 147" fill="none" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arr)"/>
  <text x="360" y="227" text-anchor="middle" font-size="10" fill="#9ca3af">Observation → 다음 스텝</text>
  <!-- arrow out -->
  <line x1="545" y1="160" x2="580" y2="160" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- Eval -->
  <rect x="580" y="130" width="130" height="60" rx="8" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="645" y="155" text-anchor="middle" font-size="12" fill="#4c1d95" font-weight="bold">LLM-as-Judge</text>
  <text x="645" y="172" text-anchor="middle" font-size="11" fill="#5b21b6">평가 + JSON</text>
</svg>
</div>

각 단계를 하나씩 뜯어보자.

<div class="ornament">· · ·</div>

## OBSERVE: Set-of-Mark 퍼셉션

가장 흥미로운 부분이다. 에이전트가 웹페이지를 "보는" 방식을 설계하는 문제.

순수하게 HTML을 텍스트로 넘기는 방법도 있다. 실제로 Mind2Web 같은 초기 에이전트들이 이 방식을 썼다. 하지만 현대 웹페이지의 HTML은 수천 줄이고, 그 대부분은 에이전트에게 무관한 메타데이터다.

**Set-of-Mark(SoM)** 방법은 다르게 접근한다. 스크린샷에 번호 뱃지를 오버레이해서 *인터랙티브 요소만 번호로 표시*하고, Claude에게 "4번 링크를 클릭해" 같은 방식으로 좌표 없이 요소를 지정하게 한다.

<div class="callout">
  <strong>SoM이 해결하는 문제:</strong> 클릭 좌표를 직접 예측하는 건 어렵다. 페이지 레이아웃이 조금만 달라져도 틀린다. 요소에 번호를 붙이면 LLM이 "무엇을 클릭할지"만 결정하면 되고, 실제 좌표 계산은 코드가 담당한다.
</div>

구현은 두 단계로 이루어진다:

**1단계 — DOM 요소 수집 (JavaScript)**

Playwright로 페이지에 JS를 주입해서 클릭 가능한 요소들을 수집한다. `a[href]`, `button`, `input`, `select`, `textarea`, `[role="button"]` 등을 쿼리하고, 각 요소에서 `getBoundingClientRect()`로 화면 좌표와 `getXPath()`로 Playwright에서 쓸 XPath를 뽑는다.

필터링도 중요하다. 뷰포트 밖(`r.top > innerHeight`)이거나 크기가 2px 이하인 요소는 버린다. 실제로 이 필터 없이 돌려보면 숨겨진 요소가 수백 개 나온다.

**2단계 — 스크린샷 오버레이 (PIL)**

수집한 요소 bbox 위에 컬러 박스와 번호 뱃지를 그린다. 요소 종류별로 색을 다르게 해서 LLM이 구분하기 쉽게 했다:

<div style="margin: 1.5rem 0;">
<svg viewBox="0 0 500 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px;display:block;margin:0 auto;font-family:monospace;">
  <rect x="10" y="20" width="110" height="40" rx="5" fill="none" stroke="#3b82f6" stroke-width="2"/>
  <rect x="10" y="20" width="32" height="18" rx="3" fill="#3b82f6"/>
  <text x="26" y="33" text-anchor="middle" font-size="11" fill="white" font-weight="bold">#1</text>
  <text x="70" y="45" text-anchor="middle" font-size="12" fill="#1d4ed8">INPUT / TEXTAREA</text>
  <rect x="140" y="20" width="110" height="40" rx="5" fill="none" stroke="#ef4444" stroke-width="2"/>
  <rect x="140" y="20" width="30" height="18" rx="3" fill="#ef4444"/>
  <text x="155" y="33" text-anchor="middle" font-size="11" fill="white" font-weight="bold">$2</text>
  <text x="195" y="45" text-anchor="middle" font-size="12" fill="#b91c1c">BUTTON / SELECT</text>
  <rect x="270" y="20" width="100" height="40" rx="5" fill="none" stroke="#22c55e" stroke-width="2"/>
  <rect x="270" y="20" width="30" height="18" rx="3" fill="#22c55e"/>
  <text x="285" y="33" text-anchor="middle" font-size="11" fill="white" font-weight="bold">@3</text>
  <text x="320" y="45" text-anchor="middle" font-size="12" fill="#15803d">LINK (a)</text>
  <rect x="390" y="20" width="100" height="40" rx="5" fill="none" stroke="#f97316" stroke-width="2"/>
  <rect x="390" y="20" width="26" height="18" rx="3" fill="#f97316"/>
  <text x="403" y="33" text-anchor="middle" font-size="11" fill="white" font-weight="bold">4</text>
  <text x="440" y="45" text-anchor="middle" font-size="12" fill="#c2410c">기타</text>
  <text x="250" y="105" text-anchor="middle" font-size="11" fill="#6b7280">접두사로 요소 종류 구분: # (입력), $ (버튼), @ (링크)</text>
  <text x="250" y="125" text-anchor="middle" font-size="11" fill="#6b7280">Claude는 "TYPE [#3] \"검색어\"" 처럼 번호로만 지시</text>
</svg>
</div>

접두사 구분이 중요하다. `#N`은 타입할 수 있는 입력창, `$N`은 버튼/셀렉트, `@N`은 링크라는 걸 LLM이 프롬프트 없이도 번호만 보고 파악할 수 있다.

<div class="ornament">· · ·</div>

## THINK: Claude Vision으로 결정하기

매 스텝마다 Claude에게 전달하는 것은 세 가지다:

1. **SoM 오버레이 스크린샷** (base64 PNG) — 현재 페이지 화면에 번호 뱃지가 그려진 것
2. **요소 목록 텍스트** — "[@1] 다나와 로고 링크, [@2] CPU 카테고리, [#3] 검색창 ..." 형식
3. **전체 대화 히스토리** — 이전 스텝들의 Thought + Action + Observation

Claude에게 강제하는 출력 형식은 단순하다:

```
Thought: <현재 상황 분석>
Action: CLICK [@2] | TYPE [#3] "i5-13400" | SCROLL DOWN | DONE "결과 요약"
```

이 형식 강제가 꽤 중요하다. 자유롭게 답변하면 파싱이 복잡해지고, 형식이 일관될수록 regex 파싱이 안정적으로 된다.

실제로 Claude가 다나와에서 견적을 찾는 과정을 보면 꽤 자연스럽다. "CPU 카테고리로 이동 → 가격순 정렬 → 예산 범위 필터 → 제품 클릭 → 가격 확인 → DONE" 같은 흐름을 스스로 구성한다.

<div class="callout">
  <strong>시스템 프롬프트에 넣은 것들:</strong> 요소 번호 표기 규칙 (# $ @ 의미), 사용 가능한 액션 목록, "예산을 반드시 지켜라", "DONE 전에 반드시 총 가격을 계산하라" 같은 제약 조건. 이 지시들이 없으면 Claude가 예산을 초과한 견적을 당당하게 DONE하는 경우가 생긴다.
</div>

<div class="ornament">· · ·</div>

## ACT: Playwright로 실행하기

Claude의 텍스트 출력을 실제 브라우저 동작으로 변환하는 부분이다. regex 파싱 후 Playwright API를 호출한다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">액션</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">Playwright 동작</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">비고</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>CLICK [N]</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">XPath로 요소 찾아 클릭</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">domcontentloaded 대기</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>TYPE [#N] "텍스트"</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">클릭 후 fill</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">입력창 전용</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>TYPE_ENTER [#N] "텍스트"</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">fill 후 Enter</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">검색창 submit용</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>SELECT [$N] "옵션"</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">select_option(label=)</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">드롭다운</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>SCROLL DOWN/UP</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">mouse.wheel ±600px</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"></td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>GOTO https://...</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">page.goto()</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">직접 URL 이동</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>DONE "메시지"</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">루프 종료</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">최종 결과 반환</td>
    </tr>
  </tbody>
</table>

`TYPE`과 `TYPE_ENTER`를 분리한 게 실제로 필요했다. 검색창은 Enter를 눌러야 검색이 시작되는데, 일반 입력창에 Enter를 누르면 폼이 제출되거나 다른 동작이 일어난다. Claude가 상황에 맞게 둘을 구분해서 사용한다.

<div class="ornament">· · ·</div>

## EVAL: LLM-as-Judge 평가

에이전트가 DONE을 외치면, 같은 Claude 모델이 결과를 채점한다. 평가 루브릭은 5개 항목이다:

<div style="margin: 1.5rem 0;">
<svg viewBox="0 0 600 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:600px;display:block;margin:0 auto;font-family:'Source Serif 4',serif;">
  <text x="300" y="22" text-anchor="middle" font-size="13" fill="#374151" font-weight="bold">LLM-as-Judge 루브릭 (총점 50점)</text>
  <!-- bars -->
  <rect x="30" y="40" width="130" height="34" rx="5" fill="#fef3c7" stroke="#d97706" stroke-width="1"/>
  <text x="95" y="62" text-anchor="middle" font-size="11" fill="#92400e" font-weight="bold">예산 준수</text>
  <text x="95" y="52" text-anchor="middle" font-size="9" fill="#a16207">budget_compliance</text>
  <text x="175" y="62" font-size="11" fill="#6b7280">/ 10점  예산 이하 + 90% 이상 활용</text>
  <rect x="30" y="82" width="130" height="34" rx="5" fill="#d1fae5" stroke="#10b981" stroke-width="1"/>
  <text x="95" y="104" text-anchor="middle" font-size="11" fill="#065f46" font-weight="bold">부품 완성도</text>
  <text x="95" y="94" text-anchor="middle" font-size="9" fill="#047857">parts_completeness</text>
  <text x="175" y="104" font-size="11" fill="#6b7280">/ 10점  필수 부품 전부 포함</text>
  <rect x="30" y="124" width="130" height="34" rx="5" fill="#dbeafe" stroke="#3b82f6" stroke-width="1"/>
  <text x="95" y="146" text-anchor="middle" font-size="11" fill="#1d4ed8" font-weight="bold">구체성</text>
  <text x="95" y="136" text-anchor="middle" font-size="9" fill="#1e40af">specificity</text>
  <text x="175" y="146" font-size="11" fill="#6b7280">/ 10점  모델명 + 다나와 실제 가격</text>
  <rect x="30" y="166" width="130" height="34" rx="5" fill="#fce7f3" stroke="#ec4899" stroke-width="1"/>
  <text x="95" y="188" text-anchor="middle" font-size="11" fill="#9d174d" font-weight="bold">목적 적합성</text>
  <text x="95" y="178" text-anchor="middle" font-size="9" fill="#be185d">purpose_fit</text>
  <text x="175" y="188" font-size="11" fill="#6b7280">/ 10점  사용 목적에 맞는 부품</text>
  <!-- task_completion is separate -->
  <rect x="420" y="82" width="150" height="100" rx="8" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="495" y="110" text-anchor="middle" font-size="12" fill="#4c1d95" font-weight="bold">태스크 완료</text>
  <text x="495" y="128" text-anchor="middle" font-size="10" fill="#5b21b6">task_completion</text>
  <text x="495" y="148" text-anchor="middle" font-size="20" fill="#6d28d9" font-weight="bold">/ 10</text>
  <text x="495" y="168" text-anchor="middle" font-size="10" fill="#7c3aed">DONE + 명확한 최종 답변</text>
</svg>
</div>

총점 0–50점, 40점 이상이면 A등급. 결과는 `results/<task_id>_<timestamp>.json`에 저장된다.

LLM-as-Judge가 실용적인 이유는 *평가 기준이 자연어로 표현 가능할 때* 사람 못지않게 일관된 채점을 한다는 점이다. "예산 90% 이상 활용"을 코드로 판단하려면 견적 파싱, 가격 합산, 비율 계산이 필요한데, LLM은 DONE 메시지를 읽고 바로 판단한다.

<div class="ornament">· · ·</div>

## 테스트 케이스 3개

실험은 세 가지 태스크로 진행했다:

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">ID</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">예산</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">목적</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">필수 부품</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>office_50</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">50만원</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">사무용</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">CPU + 메인보드 + RAM 8GB↑ + SSD 256GB↑</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>gaming_100</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">100만원</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">게이밍</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">위 + GPU (RTX 4060급)</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>budget_30</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">30만원</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">최저가</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">CPU(내장그래픽) + RAM 8GB + SSD 128GB↑</td>
    </tr>
  </tbody>
</table>

각 태스크 프롬프트에는 다나와 페이지 구조에 대한 힌트(`_PAGE_CONTEXT`)도 포함했다. "CPU 카테고리는 좌측 사이드바에 있음", "가격순 정렬 드롭다운 위치" 같은 정보다. 이게 없으면 에이전트가 첫 몇 스텝을 사이트 구조 파악에 낭비한다.

<div class="ornament">· · ·</div>

## 실제로 돌려보니: 한계들

### 1. 근시안적 ReAct (Flat Greedy)

가장 큰 문제다. 매 스텝이 현재 화면만 보고 결정한다. "CPU → 메인보드 → RAM → SSD 순서로 찾아야지"라는 **고수준 계획이 없다.** 운이 좋으면 자연스러운 순서로 탐색하지만, 중간에 막히면 원인 분석 없이 다른 버튼을 누른다.

20스텝 제한 안에서 모든 부품을 찾으려면 스텝 낭비가 없어야 하는데, 계획 없는 탐색은 중복 클릭과 불필요한 스크롤을 만든다.

### 2. 컨텍스트 낭비

요소 목록이 매 스텝 100개 이상 그대로 전달된다. 대부분은 현재 태스크와 무관하다. CPU를 검색하는 중인데 헤더 네비게이션 링크 30개가 포함되는 식이다.

히스토리도 선형 증가한다. 스텝이 쌓일수록 이전 스크린샷과 observation이 전부 컨텍스트에 남는다. 20스텝 후반에는 컨텍스트가 상당히 무거워진다.

### 3. 구조화된 상태 없음

<div class="pullquote">
  <strong>에이전트가 "지금까지 뭘 골랐는지"를 대화 히스토리에서 직접 기억해야 한다.</strong>
</div>

예를 들어 CPU를 15만원에 골랐다면, 이후 스텝에서 "남은 예산 35만원"이라는 사실을 Claude가 히스토리를 읽어서 계산해야 한다. 스텝이 길어지면 이 계산이 틀리거나 잊혀진다.

### 4. 에러 복구 없음

로그인 팝업, 품절 페이지, 예상치 못한 리다이렉트가 발생하면 루프가 막힌다. 현재는 복구 로직 없이 그냥 다음 액션을 시도한다.

<div class="ornament">· · ·</div>

## 개선하면 어떻게 될까

세 가지 방향이 보인다. 난이도와 기대 효과 기준으로:

**1순위 — 워킹 메모리 주입 (쉬움, 즉효)**

선택된 부품과 잔여 예산을 구조화된 객체로 관리하고, 매 스텝 시스템 프롬프트에 주입한다:

```
[현재 견적 상태]
- CPU: Intel i5-13400 — 189,000원
- 메인보드: 미선택
- 남은 예산: 311,000원
```

이것만 해도 Claude가 예산 계산 실수를 줄일 수 있다.

**2순위 — SoM 필터링 (쉬움)**

100개 요소 전부가 아니라, 태스크와 관련된 top-20만 전달한다. 현재 페이지 URL과 태스크 키워드를 기반으로 관련도를 계산하면 된다.

**3순위 — Planner-Executor 분리 (중간 난이도)**

```
Task
  └→ Planner: "1. CPU 검색 2. 가격순 정렬 3. 선택 4. 메인보드 검색 ..."
  └→ Executor: 각 서브태스크를 미니 ReAct로
  └→ Re-planner: 실패 감지 → 계획 수정
```

Planner가 고수준 계획을 1회 생성하고, Executor가 각 단계를 독립된 미니 루프로 실행하는 구조다. 실패 시 Re-planner가 개입한다. 구조 재설계가 필요하지만 가장 근본적인 개선이다.

<div class="ornament">· · ·</div>

## 배운 것들

직접 구현해보니 논문에서 당연하게 쓰인 표현들이 실제로 어떤 문제를 해결하는지 체감됐다.

SoM의 핵심은 **좌표 예측 문제를 분류 문제로 바꾸는 것**이다. LLM은 연속적인 픽셀 좌표를 예측하는 데 약하지만, "이 중에 뭐를 클릭할지" 고르는 건 잘한다.

ReAct 루프의 실제 병목은 LLM 추론 능력보다 **상태 관리**에 있다. 스텝이 늘어날수록 히스토리 길이가 선형 증가하고, LLM이 오래된 정보를 제대로 참조하지 못하는 문제가 생긴다. 이걸 해결하려면 구조화된 메모리가 필수다.

LLM-as-Judge는 **평가 기준을 자연어로 쓸 수 있을 때** 놀랍도록 잘 작동한다. 하드코딩된 평가 함수보다 유연하고, 예상치 못한 DONE 형식에도 잘 대응한다.

<div class="footnote">
  코드: <a href="https://github.com/jung0228/weg-agent">github.com/jung0228/weg-agent</a>
</div>
