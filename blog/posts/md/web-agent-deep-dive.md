---
title: "웹 에이전트 심층 분석: 동작 원리, 벤치마크, 핵심 시스템"
dek: Playwright에서 Browser Use까지 — 웹 에이전트가 어떻게 브라우저를 조작하고, 어떻게 평가되고, 어떻게 설계되는지를 완전히 해부한다.
desc: 웹 에이전트의 동작 원리(Playwright/CDP/AX Tree/SoM)부터 4대 벤치마크(Mind2Web·WebArena·WebVoyager·VisualWebArena), Mind2Web·WebVoyager·Agent-E·Browser Use 시스템 분석까지 한 편에.
tags: [Agent, LLM, Multimodal]
date: Mar 2026
readtime: 35 min read
slug: web-agent-deep-dive
highlight: true
katex: false
---

## 이 포스트에 대해

웹 에이전트 논문들을 읽다 보면 세 가지 층이 뒤섞인다. **어떻게 브라우저를 조작하는가**(구현 레이어), **어떻게 평가하는가**(벤치마크 레이어), **어떤 아키텍처가 좋은가**(시스템 설계 레이어). 논문마다 자기 맥락에서만 설명하기 때문에 전체 그림이 잘 보이지 않는다.

이 포스트는 세 레이어를 순서대로 쌓는다.

<div class="ornament">· · ·</div>

## Part 1 — 동작 방식: 브라우저와 LLM 사이

### LLM의 결정을 물리적 클릭으로

LLM이 "검색 버튼을 클릭한다"는 결정을 내려도 실제 클릭이 일어나려면 뭔가가 브라우저를 직접 조작해야 한다. 웹 에이전트가 쓰는 도구는 **Playwright** 또는 **Puppeteer** 같은 브라우저 자동화 라이브러리다.

이 라이브러리들은 **Chrome DevTools Protocol(CDP)** 을 통해 Chrome 브라우저와 WebSocket으로 통신한다. CDP는 Chrome이 노출하는 저수준 API로, 마우스 이벤트 발송·DOM 조회·JavaScript 실행·스크린샷 캡처 등을 명령한다.

```python
import asyncio
from playwright.async_api import async_playwright

async def run_agent_step():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto("https://example-shop.com")

        # LLM 입력: Accessibility Tree 추출
        ax_tree = await page.accessibility.snapshot()

        # VLM 입력: 스크린샷 캡처
        screenshot = await page.screenshot()

        # LLM 결정 → 실행 (액션 타입별)
        await page.click('[aria-label="Search"]')         # click
        await page.fill('#search-input', "red shoes")     # type
        await page.select_option('#size-select', "10")    # select_option
        await page.mouse.wheel(0, 500)                    # scroll
        await page.hover(".nav-category")                 # hover
        await page.keyboard.press("Enter")                # key_press
        await page.go_back()                              # go_back
```

에이전트 관점에서 **액션 공간은 7~10개 기본 연산**으로 구성된다: `click`, `type`, `select`, `scroll`, `hover`, `goto`, `go_back`, `key_press`, 그리고 태스크 완료를 선언하는 `stop`.

<div class="ornament">· · ·</div>

### Observation: 에이전트는 뭘 보는가

브라우저에서 LLM으로 "현재 상태"를 전달하는 방법이 몇 가지 있다. 어떤 방법을 쓰느냐가 시스템 성능과 토큰 비용을 크게 좌우한다.

<figure>
  <img src="img/web-agent-deep-dive/obs-methods.png" alt="Raw HTML ~50K tokens vs Filtered DOM ~1.5K vs AX Tree ~3K vs Screenshot vs SoM 비교">
  <figcaption>5가지 Observation 방법과 토큰 비용 비교. Raw HTML은 거의 불가능한 수준이고, AX Tree가 가성비가 가장 좋다.</figcaption>
</figure>

**Raw HTML**은 페이지를 그대로 담지만 수천 줄이 된다.

```html
<!-- 수천 개 div 중 일부 -->
<div class="nav-search-wrapper">
  <input id="twotabsearchtextbox"
         type="text"
         placeholder="Search Amazon"
         aria-label="Search Amazon">
  <input id="nav-search-submit-button"
         type="submit" value="Go">
</div>
<!-- 이후 수천 줄 계속... -->
```

**Filtered DOM**은 Raw HTML에서 스크립트·스타일·숨겨진 요소 등 에이전트에게 불필요한 부분을 제거하고 인터랙티브 요소 중심으로 압축한 형태다. Raw HTML ~50K 토큰이 ~1.5K까지 줄어든다.

```html
<!-- Filtered DOM 예시 — 핵심 인터랙티브 요소만 남김 -->
<input id="twotabsearchtextbox" type="text"
       placeholder="Search Amazon" aria-label="Search Amazon">
<input id="nav-search-submit-button" type="submit" value="Go">
<a href="/deals">Today's Deals</a>
<a href="/gp/cart/view.html">Cart (3)</a>
```

HTML 태그와 속성이 유지되므로 에이전트가 XPath나 CSS 셀렉터로 직접 요소를 지정할 수 있다는 장점이 있다. 단, 페이지가 조금만 바뀌어도 셀렉터가 깨지는 취약점이 있고, AX Tree보다 여전히 노이즈가 많다.

**Accessibility Tree**는 브라우저가 스크린 리더를 위해 내부적으로 유지하는 구조다. 시각적 레이아웃 정보는 없지만 인터랙티브 요소와 역할은 정확히 담겨 있다.

```json
{
  "role": "WebArea",
  "name": "Amazon.com",
  "children": [
    { "role": "combobox", "name": "Search in", "value": "All" },
    { "role": "searchbox", "name": "Search Amazon", "value": "" },
    { "role": "button", "name": "Go" }
  ]
}
// ~80 토큰 (raw HTML은 ~800 토큰)
// 전체 페이지: AX tree ~3K vs HTML ~50K
```

<div class="callout">
  <strong>Set-of-Mark (SoM) — VLM용 시각적 grounding</strong><br>
  스크린샷만 있으면 "어디를 클릭해야 하는가"를 VLM이 픽셀 좌표로 특정해야 한다. SoM은 스크린샷 위에 인터랙티브 요소마다 번호 레이블을 오버레이하고, LLM이 "3번을 클릭하라"처럼 번호로 참조하게 한다.
</div>

<figure>
  <img src="img/web-agent-deep-dive/som-screenshot.png" alt="Google Flights 스크린샷에 SoM 번호 오버레이">
  <figcaption>Set-of-Mark 적용 예시. 각 인터랙티브 요소에 번호가 붙는다. LLM은 좌표 대신 "Click [12]"처럼 번호를 참조한다.</figcaption>
</figure>

<div class="ornament">· · ·</div>

### LLM과 VLM 입출력 실제 샘플

**태스크:** *"WebArena 쇼핑몰에서 'Waffle Knit Beanie'를 검색하고 리뷰가 가장 많은 제품을 장바구니에 담아라."*

**텍스트 기반 (WebArena 스타일) — 시스템 프롬프트:**

```text
You are a web browsing agent. Output your next action:
  Thought: [reasoning]
  Action: [action_type] [arguments]

Available actions: click [id], type [id] [text],
  scroll [dir], goto [url], go_back, stop [answer]
```

**LLM 출력 (3턴):**

```text
Thought: 검색창 [6]이 보인다.
Action: type [6] Waffle Knit Beanie

------- 다음 턴 -------

Thought: 리뷰 수 확인:
  [23] "2-Pack Waffle Knit Beanie" - ★4.3 (127 reviews)
  [31] "Organic Waffle Beanie"     - ★4.1  (43 reviews)
  [38] "Fleece Waffle Beanie"      - ★4.5  (89 reviews)
Product [23]이 127개로 가장 많다.
Action: click [23]

------- 다음 턴 -------

Thought: 상세 페이지. "Add to Cart" 버튼 발견.
Action: click [52]
Action: stop [Task completed]
```

텍스트 기반은 요소 ID(`[6]`)를 AX Tree에서 직접 가져와서 확실히 존재하는 요소를 참조한다. 비전 기반은 SoM 번호를 참조하는데, 이 번호는 JavaScript가 실시간으로 할당하므로 **턴마다 달라질 수 있다.**

<div class="ornament">· · ·</div>

### End-to-End 추적: GitLab 이슈 태스크

**태스크:** *"WebArena GitLab에서 'awesome-os' 프로젝트의 open 이슈 수를 알려라."*

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.86rem;">
  <thead><tr style="background:#f5f5f0;">
    <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;width:3rem;">스텝</th>
    <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">상태</th>
    <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">관찰 (AX Tree)</th>
    <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">액션</th>
  </tr></thead>
  <tbody>
    <tr style="background:#fafaf8;">
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;">0</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">초기화</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">Docker GitLab 인스턴스 시작</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">Playwright 브라우저 시작</td>
    </tr>
    <tr>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;">1</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">GitLab 홈</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>[3] link "awesome-os"</code></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>click [3]</code></td>
    </tr>
    <tr>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;">2</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">프로젝트 페이지</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>[12] link "Issues (7)"</code></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>click [12]</code></td>
    </tr>
    <tr>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;">3</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">이슈 목록</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>[1] tab "Open (5)"</code> &nbsp;<code>[2] tab "Closed (2)"</code></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>stop [5]</code></td>
    </tr>
    <tr style="background:#f0fdf4;">
      <td style="padding:0.6rem 1rem;text-align:center;">✓</td>
      <td style="padding:0.6rem 1rem;">평가</td>
      <td style="padding:0.6rem 1rem;" colspan="2"><code>string_match(answer="5", reference="5")</code> → <strong style="color:#166534;">Success</strong></td>
    </tr>
  </tbody>
</table>

이 4스텝 태스크는 단순한 편이다. 실제 WebArena 태스크들은 평균 10~15스텝, 어려운 것들은 30스텝을 넘는다.

<div class="ornament">· · ·</div>

## Part 2 — 벤치마크 해부

벤치마크마다 근본적인 설계 철학이 다르다. 어느 숫자를 믿을 수 있는지 이해하려면 각 벤치마크가 무엇을 측정하는지 알아야 한다.

<div class="ornament">· · ·</div>

### Mind2Web (NeurIPS 2023) — 오프라인 평가

Mind2Web은 Amazon Mechanical Turk를 통해 실제 사용자들이 웹을 탐색하는 과정을 녹화한 데이터셋이다. **137개 사이트, 2,350개 태스크, 태스크당 평균 7.3개 스텝**이 저장된 HTML 스냅샷으로 존재한다. 실제 브라우저는 없다.

<figure>
  <img src="img/web-agent-deep-dive/mind2web-offline.png" alt="Mind2Web Offline 평가 흐름">
  <figcaption>Mind2Web의 Offline 평가 구조. HTML 스냅샷 → 에이전트 → 정답 비교. 실제 브라우저 없이 저장된 데이터로만 평가한다.</figcaption>
</figure>

**세 가지 일반화 시나리오** — 논문에서 "우리 모델이 X%"라고 보고할 때 어느 분할인지를 반드시 확인해야 한다.

<figure>
  <img src="img/web-agent-deep-dive/mind2web-splits.png" alt="Cross-Task / Cross-Website / Cross-Domain 분할 설명">
  <figcaption>세 가지 테스트 분할. 성능은 Cross-Task > Cross-Website > Cross-Domain 순으로 떨어진다.</figcaption>
</figure>

**네 가지 지표** — 스텝 하나를 (요소, 액션, 값) 세 부분으로 분해해서 각 레벨에서 정확도를 측정한다. 수식보다 예시로 보는 게 훨씬 빠르다.

**예시 태스크:** *"우편번호 10002 근처 자동차 수리점 리뷰를 보여줘"* — 10개 스텝 중 Step 3:

<div class="callout">
  <strong>정답(Ground Truth):</strong> <code>[textbox] Near</code> → TYPE: <code>10002</code>
</div>

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.86rem;">
  <thead><tr style="background:#f5f5f0;">
    <th style="padding:0.65rem 1rem;text-align:left;border-bottom:2px solid #ddd;">에이전트 예측</th>
    <th style="padding:0.65rem 1rem;text-align:center;border-bottom:2px solid #ddd;">Ele. Acc</th>
    <th style="padding:0.65rem 1rem;text-align:center;border-bottom:2px solid #ddd;">Op. F1</th>
    <th style="padding:0.65rem 1rem;text-align:center;border-bottom:2px solid #ddd;">Step SR</th>
    <th style="padding:0.65rem 1rem;text-align:left;border-bottom:2px solid #ddd;">이유</th>
  </tr></thead>
  <tbody>
    <tr style="background:#f0fdf4;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>[textbox] Near</code> → TYPE: <code>10002</code></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="color:#166534;">✓</strong></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="color:#166534;">✓</strong></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 8px;border-radius:3px;">✓ 1점</strong></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">요소·오퍼레이션·값 전부 정답</td>
    </tr>
    <tr style="background:#fff8f8;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>[textbox] Near</code> → TYPE: <code>New York</code></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="color:#166534;">✓</strong></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="color:#991b1b;">✗</strong></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#ffd6d6;padding:2px 8px;border-radius:3px;">✗ 0점</strong></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">요소는 맞지만 값이 틀림 — Step SR은 부분 점수 없음</td>
    </tr>
    <tr style="background:#fff8f8;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>[searchbox] Find</code> → TYPE: <code>10002</code></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="color:#991b1b;">✗</strong></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="color:#991b1b;">✗</strong></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#ffd6d6;padding:2px 8px;border-radius:3px;">✗ 0점</strong></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">값은 맞지만 요소가 틀림 — Ele.Acc 실패 시 Op.F1도 0</td>
    </tr>
    <tr style="background:#fff8f8;">
      <td style="padding:0.6rem 1rem;"><code>[textbox] Near</code> → CLICK</td>
      <td style="padding:0.6rem 1rem;text-align:center;"><strong style="color:#166534;">✓</strong></td>
      <td style="padding:0.6rem 1rem;text-align:center;"><strong style="color:#991b1b;">✗</strong></td>
      <td style="padding:0.6rem 1rem;text-align:center;"><strong style="background:#ffd6d6;padding:2px 8px;border-radius:3px;">✗ 0점</strong></td>
      <td style="padding:0.6rem 1rem;">요소는 맞지만 오퍼레이션(CLICK ≠ TYPE)이 틀림</td>
    </tr>
  </tbody>
</table>

이 Step 3 하나만 봐도 지표 간 관계가 명확해진다: **Ele. Acc는 요소만, Op. F1은 오퍼레이션+값만, Step SR은 셋 다 맞아야 1점.** Task SR은 이 10개 스텝 전부가 Step SR = 1이어야 성공 — 하나라도 틀리면 0이다.

<div class="callout">
  <strong>Step SR 50%의 수학적 현실</strong><br>
  Step SR 50%라는 숫자만 보면 나쁘지 않아 보인다. 하지만 태스크당 평균 7.3스텝을 가정하면 Task SR ≈ 0.5<sup>7.3</sup> ≈ 0.6%. 실제 최고 성능도 Task SR 5~7%에 그치는 건 이 수학적 귀결이다. 두 숫자를 항상 같이 봐야 한다.
</div>

**실제 벤치마크 결과 (Cross-Website 기준):**

<figure>
  <img src="img/web-agent-deep-dive/mindact-results.png" alt="MindAct 성능 결과 표 (Cross-Website 기준)">
  <figcaption>Cross-Website 기준 성능. GPT-4 + SoM (MindAct)이 Ele. Acc 58.1%, Step SR 52.3%로 가장 높다.</figcaption>
</figure>

<div class="ornament">· · ·</div>

### WebArena (ICLR 2024) — 기능적 검증

WebArena의 근본적 혁신은 "**에이전트가 무엇을 클릭했는가**"가 아니라 "**환경이 올바른 상태에 도달했는가**"를 묻는다는 점이다.

**두 가지 평가 함수** — 어떤 태스크냐에 따라 r_info(정보 검색)와 r_prog(상태 변경) 중 하나로 평가한다.

**r_info — 정보 검색 태스크:** 에이전트가 `stop [answer]`로 텍스트 답변을 제출. 세 가지 매칭 방식 중 하나를 적용한다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.86rem;">
  <thead><tr style="background:#f5f5f0;">
    <th style="padding:0.65rem 1rem;text-align:left;border-bottom:2px solid #ddd;">태스크 예시</th>
    <th style="padding:0.65rem 1rem;text-align:left;border-bottom:2px solid #ddd;">정답</th>
    <th style="padding:0.65rem 1rem;text-align:left;border-bottom:2px solid #ddd;">에이전트 답변</th>
    <th style="padding:0.65rem 1rem;text-align:center;border-bottom:2px solid #ddd;">방식</th>
    <th style="padding:0.65rem 1rem;text-align:center;border-bottom:2px solid #ddd;">r_info</th>
  </tr></thead>
  <tbody>
    <tr style="background:#f0fdf4;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">GitLab open 이슈 수</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>"5"</code></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>"5"</code></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;">exact_match</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 8px;border-radius:3px;">1.0</strong></td>
    </tr>
    <tr style="background:#fff8f8;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">GitLab open 이슈 수</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>"5"</code></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>"7"</code></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;">exact_match</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#ffd6d6;padding:2px 8px;border-radius:3px;">0.0</strong></td>
    </tr>
    <tr style="background:#f0fdf4;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">열린 이슈 제목들 나열</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><code>["Fix login", "Add tests"]</code></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">"이슈: Fix login, Add tests, Update docs"</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;">must_include</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 8px;border-radius:3px;">1.0</strong></td>
    </tr>
    <tr style="background:#f0fdf4;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">병원 이름</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">"UPMC Mercy"</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">"Mercy Hospital (UPMC)"</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;">fuzzy_match<br><span style="font-size:0.78rem;color:#666">(GPT-4 심판)</span></td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 8px;border-radius:3px;">1.0</strong></td>
    </tr>
  </tbody>
</table>

**r_prog — 상태 변경 태스크:** 에이전트가 어떤 경로를 걸었든 **환경의 실제 상태를 직접 검증**한다. DB 쿼리, GitLab API, DOM JavaScript 평가 등을 조합한다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.86rem;">
  <thead><tr style="background:#f5f5f0;">
    <th style="padding:0.65rem 1rem;text-align:left;border-bottom:2px solid #ddd;">태스크 예시</th>
    <th style="padding:0.65rem 1rem;text-align:left;border-bottom:2px solid #ddd;">검증 방법</th>
    <th style="padding:0.65rem 1rem;text-align:center;border-bottom:2px solid #ddd;">r_prog</th>
  </tr></thead>
  <tbody>
    <tr style="background:#f0fdf4;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">"GitLab에서 'Fix login bug' 이슈 생성"</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">GitLab API → 해당 이슈 존재 여부</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 8px;border-radius:3px;">1.0 (존재)</strong></td>
    </tr>
    <tr style="background:#fff8f8;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">"GitLab에서 'Fix login bug' 이슈 생성"</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">GitLab API → 이슈 없음</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#ffd6d6;padding:2px 8px;border-radius:3px;">0.0 (없음)</strong></td>
    </tr>
    <tr style="background:#f0fdf4;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">"장바구니에 상품 1개 추가"</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">DOM: <code>.cart-count</code> textContent == "1"</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 8px;border-radius:3px;">1.0</strong></td>
    </tr>
    <tr style="background:#f0fdf4;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">"Reddit에 특정 댓글 게시"</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">DB 쿼리 → 최신 댓글 텍스트 일치 확인</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 8px;border-radius:3px;">1.0</strong></td>
    </tr>
  </tbody>
</table>

에이전트가 어떤 경로로 목표를 달성했든 **결과물이 DB에 존재하면 정답이다.** Mind2Web에서 "정답 경로"만 인정하던 것과 근본적으로 다르다.

WebArena에는 수행 불가능한 태스크도 포함된다. 이때 에이전트는 `stop [N/A]`를 출력해야 한다. 동일 관찰에서 같은 액션을 3번 반복하거나, 파싱 불가능한 액션을 3번 연속 생성하면 태스크 실패 처리된다.

**2년간 SOTA 변화:**

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead><tr style="background:#f5f5f0;">
    <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">에이전트</th>
    <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">발표</th>
    <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">Task SR</th>
  </tr></thead>
  <tbody>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">GPT-4 (최초)</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">2023.07</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">14.41%</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">SeeAct (GPT-4V + SoM)</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">2024.01</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">~23%</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">GPT-4o (experience replay)</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">2024.07</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">~36.7%</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Claude Computer Use 3.5</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">2024.10</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">~49.0%</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Claude Computer Use 3.7</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">2025.02</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">~56.3%</td></tr>
    <tr style="background:#fffbf0;"><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;"><strong>OpenAI Operator</strong></td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong>2025.01</strong></td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">~61.3%</strong></td></tr>
    <tr style="background:#f0fdf4;"><td style="padding:0.55rem 1rem;"><strong>인간 (CS 대학원생 5명)</strong></td><td style="padding:0.55rem 1rem;text-align:center;">—</td><td style="padding:0.55rem 1rem;text-align:center;"><strong>78.24%</strong></td></tr>
  </tbody>
</table>

<div class="ornament">· · ·</div>

### WebVoyager (ACL 2024) — GPT-4V 심판 방식

WebVoyager는 **실제 라이브 웹사이트에서 멀티모달 에이전트를 평가**하는 첫 번째 벤치마크다. Google·Amazon·GitHub·Booking.com 등 15개 실제 사이트, 643개 태스크.

기존 평가 방식과의 차이:

<figure>
  <img src="img/web-agent-deep-dive/evaluation-comparison.png" alt="기존 Mind2Web 방식 vs WebVoyager 방식 비교">
  <figcaption>왼쪽(기존): 정적 스냅샷 + 정해진 경로만 정답. 오른쪽(WebVoyager): 실시간 웹 + 어떤 경로로든 성공하면 GPT-4V가 판정.</figcaption>
</figure>

WebArena는 Docker 격리 환경에서 DB를 직접 검증한다. 하지만 실제 Amazon이나 GitHub에서는 그게 불가능하다. WebVoyager는 이 문제를 **GPT-4V를 심판으로 써서** 해결했다.

GPT-4V 심판에게 (1) 태스크 설명, (2) 에이전트의 최종 답변, (3) 궤적의 마지막 k장 스크린샷을 주고 "태스크가 완료됐는가?"를 Yes/No + 이유로 판정하도록 요청한다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.86rem;">
  <thead><tr style="background:#f5f5f0;">
    <th style="padding:0.65rem 1rem;text-align:left;border-bottom:2px solid #ddd;">태스크</th>
    <th style="padding:0.65rem 1rem;text-align:left;border-bottom:2px solid #ddd;">에이전트 답변</th>
    <th style="padding:0.65rem 1rem;text-align:center;border-bottom:2px solid #ddd;">GPT-4V 판정</th>
    <th style="padding:0.65rem 1rem;text-align:left;border-bottom:2px solid #ddd;">판정 이유</th>
  </tr></thead>
  <tbody>
    <tr style="background:#f0fdf4;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">뉴욕→파리 왕복 최저가 항공권 찾기</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">"$487 Air France"</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 8px;border-radius:3px;">Yes</strong></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">스크린샷에 검색 결과 $487 Air France 항목이 보임</td>
    </tr>
    <tr style="background:#fff8f8;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">뉴욕→파리 왕복 최저가 항공권 찾기</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">"검색 결과 없음"</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#ffd6d6;padding:2px 8px;border-radius:3px;">No</strong></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">스크린샷에 실제 검색 결과가 표시되어 있음에도 에이전트가 실패로 보고</td>
    </tr>
    <tr style="background:#f0fdf4;">
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">GitHub 특정 이슈의 첫 댓글 내용</td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">"Fix the CI pipeline"</td>
      <td style="padding:0.6rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 8px;border-radius:3px;">Yes</strong></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">스크린샷의 첫 댓글 텍스트와 일치</td>
    </tr>
    <tr style="background:#fff8f8;">
      <td style="padding:0.6rem 1rem;">Booking.com 2박 숙박 예약 확인</td>
      <td style="padding:0.6rem 1rem;">"예약 완료 확인했습니다"</td>
      <td style="padding:0.6rem 1rem;text-align:center;"><strong style="background:#ffd6d6;padding:2px 8px;border-radius:3px;">No</strong></td>
      <td style="padding:0.6rem 1rem;">마지막 스크린샷이 결제 페이지가 아닌 검색 결과 페이지 — 예약 완료 화면이 없음</td>
    </tr>
  </tbody>
</table>

k를 얼마로 설정하냐가 평가 신뢰도를 크게 좌우한다. 많은 후속 연구들이 비용 절감을 위해 마지막 1장만 쓰는데, 이 경우 κ = 0.51로 **동전 뒤집기보다 약간 나은** 수준이다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead><tr style="background:#f5f5f0;">
    <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">제공 스크린샷</th>
    <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">인간 일치율</th>
    <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">Cohen's κ</th>
    <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">실용성</th>
  </tr></thead>
  <tbody>
    <tr style="background:#fff8f8;"><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">마지막 1장만</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">~73%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><span style="background:#ffd6d6;padding:2px 6px;border-radius:3px;">0.51</span></td><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">비용 저렴, 신뢰도 낮음</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">마지막 3장</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">~80%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">0.63</td><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">균형점</td></tr>
    <tr style="background:#f0fdf4;"><td style="padding:0.55rem 1rem;"><strong>전체 궤적</strong></td><td style="padding:0.55rem 1rem;text-align:center;"><strong>85.3%</strong></td><td style="padding:0.55rem 1rem;text-align:center;"><strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">0.70</strong></td><td style="padding:0.55rem 1rem;">비용 높음, 가장 신뢰</td></tr>
  </tbody>
</table>

<details>
<summary><strong>643개 태스크는 어떻게 만들었나 — 3단계 Self-Instruct</strong></summary>

**Stage 1 — 시드 태스크 수집:** Google Flights, Google Maps, Google Search, Booking, Wolfram Alpha 5개 사이트에 Mind2Web 태스크를 샘플링하고 수동 편집해 시드 풀을 만든다.

**Stage 2 — GPT-4 Turbo 생성 + 인간 검수:** 시드 태스크를 in-context 예시로 주고 GPT-4 Turbo에게 사이트당 약 100개 태스크를 20회 반복 생성. 사람이 직접 검토·수정 후 Task Pool에 추가.

**Stage 3 — 다양성 확보:** Stage 2 풀에서 다양한 예시를 샘플링해 GPT-4 Turbo로 추가 생성. 직접 검증 후 추가 → 사이트당 40개 이상, **총 643개**.

이 과정에서 "단순 검색 한 번으로 해결 가능한 태스크"가 많이 포함된 것이 "Illusion of Progress" 논문의 비판 대상이 됐다.
</details>

**사이트별 텍스트 vs 멀티모달 비교:**

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead><tr style="background:#f5f5f0;">
    <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">사이트</th>
    <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">텍스트 전용</th>
    <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">멀티모달</th>
    <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">차이</th>
  </tr></thead>
  <tbody>
    <tr style="background:#fff8f8;"><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Booking.com</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><span style="background:#ffd6d6;padding:2px 6px;border-radius:3px;">2.3%</span></td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">43.2%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong>+40.9%p</strong></td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Google Flights</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">41.5%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">70.7%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong>+29.2%p</strong></td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Amazon</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">36.6%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">58.5%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">+21.9%p</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">BBC News</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">52.4%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">61.9%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">+9.5%p</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Cambridge Dict.</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">55.0%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">65.1%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">+10.1%p</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Google Search</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">68.3%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">76.7%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">+8.4%p</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">ArXiv</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">48.8%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">51.2%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">+2.4%p</td></tr>
    <tr style="background:#fffbf0;"><td style="padding:0.55rem 1rem;"><strong>전체 평균</strong></td><td style="padding:0.55rem 1rem;text-align:center;"><strong>40.1%</strong></td><td style="padding:0.55rem 1rem;text-align:center;"><strong>59.1%</strong></td><td style="padding:0.55rem 1rem;text-align:center;"><strong>+19.0%p</strong></td></tr>
  </tbody>
</table>

Booking.com에서 텍스트 전용이 2.3%인 것이 눈에 띈다. 달력 UI와 가격 비교 레이아웃은 DOM만으로 파악하기 극히 어렵다. 반면 ArXiv(+2.4%p)나 Google Search(+8.4%p)처럼 텍스트 정보가 대부분인 사이트는 멀티모달의 이점이 거의 없다.

<div class="ornament">· · ·</div>

### VisualWebArena (ACL 2024) — 시각 이해 없이는 못 푸는 태스크

WebArena 태스크들은 이론적으로 DOM/AX Tree만으로 전부 해결 가능하다. VisualWebArena는 이 한계를 직접 공략한다: **시각 이해 없이는 원천적으로 풀 수 없는 태스크.**

태스크의 25.2%는 이미지를 태스크 설명에 직접 포함한다:

> *[이미지: 파란 스트라이프 셔츠 사진] "이 셔츠와 동일한 스타일인데 빨간색인 제품을 찾아서 장바구니에 담아라."*

결과: 인간 **88.70%** vs 최고 모델(GPT-4o + SoM) **19.78%**. WebArena(78% vs 14%)보다 격차가 더 크다. 이미지를 텍스트로 변환하는 Caption 방식은 12.75%에 그쳤다 — 텍스트 설명으로 요약하면 시각적 세부 정보가 손실된다.

<div class="ornament">· · ·</div>

### "Illusion of Progress" — 2025년의 메타 비판

**핵심 발견:** *Google에서 검색하고 첫 결과 페이지 정보를 답변으로 제출하는* 단순한 에이전트가 WebVoyager 태스크의 **51%**를 통과한다. 즉 WebVoyager 태스크의 상당 부분은 실제 멀티스텝 웹 탐색이 필요 없다.

대안으로 제안된 **Online-Mind2Web**: 300개 태스크, 136개 사이트, 모든 태스크를 라이브 웹에서 인간이 직접 검증.

자동 평가도 개선했다. **WebJudge**:

```python
# 1단계: 요구사항 추출
requirements = o4_mini.extract(task)
# → ["navigate to youtube.com", "search 'python tutorial'",
#    "apply sort by view count", "identify top result"]

# 2단계: 관련 스크린샷만 필터링 (관련성 점수 δ=3 이상)
key_screenshots = [s for s in trajectory if relevance(s, req) > 3]

# 3단계: o4-mini 이진 판정
verdict = o4_mini.judge(task, requirements, key_screenshots, final_answer)
# → {"success": True/False, "reason": "..."}
```

**제대로 측정하면 숫자가 어떻게 달라지나:**

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead><tr style="background:#f5f5f0;">
    <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">에이전트</th>
    <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">발표된 수치</th>
    <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">Online-Mind2Web</th>
  </tr></thead>
  <tbody>
    <tr style="background:#fffbf0;"><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;"><strong>OpenAI Operator</strong></td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">WebArena ~65%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">61.3%</strong></td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Claude CU 3.7</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">WebArena ~58%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">56.3%</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Browser Use</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">자체 벤치 ~80%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><span style="background:#ffd6d6;padding:2px 6px;border-radius:3px;">~30%</span></td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Agent-E</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">WebArena ~73%</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><span style="background:#ffd6d6;padding:2px 6px;border-radius:3px;">~28%</span></td></tr>
    <tr><td style="padding:0.55rem 1rem;">SeeAct (2024 베이스라인)</td><td style="padding:0.55rem 1rem;text-align:center;">WebVoyager 59%</td><td style="padding:0.55rem 1rem;text-align:center;">~26%</td></tr>
  </tbody>
</table>

<div class="pullquote">
  <strong>2024년에 나온 수많은 웹 에이전트들이 엄격한 평가에서 2024년 초 베이스라인과 비슷한 성능을 보인다. 발표된 높은 수치들의 상당 부분은 쉬운 벤치마크나 느슨한 자동 평가의 인플레이션이다.</strong>
</div>

<div class="ornament">· · ·</div>

## Part 3 — 핵심 시스템 분석

### Mind2Web + MindAct

<figure>
  <img src="img/web-agent-deep-dive/mind2web-stats.png" alt="Mind2Web 데이터셋 통계">
  <figcaption>Mind2Web 데이터셋 통계. 2,350개 태스크, 137개 사이트, 31개 도메인, 페이지당 평균 1,135개 DOM 요소.</figcaption>
</figure>

<figure>
  <img src="img/web-agent-deep-dive/mindact-arch.png" alt="MindAct 아키텍처 — HTML Document → Ranking LM → Prediction LLM">
  <figcaption>MindAct 전체 아키텍처. HTML Document에서 Candidate Elements를 Ranking LM으로 50개로 줄이고, Prediction LLM이 최종 Target Element와 Operation을 예측한다.</figcaption>
</figure>

**MindAct의 2단계 파이프라인** — 페이지당 평균 1,135개 DOM 요소를 LLM이 직접 처리하는 것은 불가능하다. MindAct는 이 문제를 두 단계로 나눠 해결한다.

<figure>
  <img src="img/web-agent-deep-dive/mindact-stage1.png" alt="MindAct Stage 1 — DeBERTa Ranking LM이 1,135개 DOM 요소를 50개로 압축">
  <figcaption>Stage 1: 파인튜닝된 DeBERTa Ranking LM이 태스크-DOM 요소 쌍의 관련성을 점수화해 Top-50 후보로 압축. 전체 요소의 4%만 남긴다.</figcaption>
</figure>

<figure>
  <img src="img/web-agent-deep-dive/mindact-stage2.png" alt="MindAct Stage 2 — Prediction LLM이 Top-50 후보에서 최종 요소와 오퍼레이션 선택">
  <figcaption>Stage 2: Prediction LLM(GPT-3.5/4 또는 Flan-T5)이 Top-50 후보 + 태스크 + 히스토리를 보고 최종 Target Element와 Operation(CLICK/TYPE/SELECT)을 예측한다.</figcaption>
</figure>

<div class="pullquote">
  "Training LMs for discrimination rather than generation is more generalizable." — 경로를 생성(generate)하는 것보다 요소를 식별(discriminate)하도록 학습시키는 방식이 새로운 사이트와 도메인에 더 잘 일반화된다는 논문의 핵심 주장이다.
</div>

5가지 핵심 발견:

**발견 1 — 일반화 갭은 10%p 이상.** Cross-Task 52.0% → Cross-Website 38.9%. 웹사이트별 레이아웃이 모두 다르기 때문이다.

**발견 2 — Cross-Website ≈ Cross-Domain.** 도메인 지식보다 UI 패턴 다양성이 더 큰 장벽이다. 여행 vs 쇼핑 도메인 차이보다, 에어비앤비 vs 익스피디아 UI 차이가 더 크다. "도메인 특화 에이전트"보다 "다양한 UI 패턴에 강인한 에이전트"가 더 중요하다.

**발견 3 — Task SR이 낮은 건 수학적으로 당연.** Step SR 52%, 평균 7.3스텝 → 0.52^7 ≈ 0.9%. 실제 5.2%는 스텝들이 컨텍스트를 공유해서 이보다 높다.

**발견 4 — 파인튜닝이 필수.** 제로샷 Flan-T5-XL은 Element Accuracy 10.8%. 파인튜닝 후 52.0%. 대형 모델도 이 태스크에선 파인튜닝이 필수다.

**발견 5 — GPT-4가 의외로 강하다.** 파인튜닝 없이 in-context learning만으로 Element Accuracy 41.6%. GPT-3.5-turbo(20.3%)와 비교하면 압도적이다.

**발견 6 — 스케일이 일관되게 도움이 된다.** Flan-T5-B(Step SR 41.0%) → L(50.3%) → XL(52.0%). 파라미터가 클수록 성능이 오른다. 흥미로운 점: GPT-4(Step SR 36.2%)가 파인튜닝된 Flan-T5-XL(52.0%)을 밑돈다. 이 태스크에서는 **모델 크기보다 파인튜닝이 더 중요하다.**

<details>
<summary><strong>전체 실험 결과 테이블 (모든 모델 × 분할)</strong></summary>

| 모델 | 분할 | Ele. Acc | Op. F1 | Step SR | Task SR |
|---|---|---|---|---|---|
| DeBERTa (분류 전용) | Cross-Task | 26.8% | — | — | — |
| Flan-T5-B (생성, 파인튜닝 없음) | Cross-Task | 20.2% | 52.0% | 17.5% | 0.0% |
| GPT-3.5-turbo | Cross-Task | 20.3% | 56.6% | 17.4% | 0.8% |
| **MindAct (Flan-T5-B)** | Cross-Task | 43.6% | 76.8% | 41.0% | 4.0% |
| **MindAct (Flan-T5-L)** | Cross-Task | 53.4% | 75.7% | 50.3% | 7.1% |
| **MindAct (Flan-T5-XL)** | Cross-Task | **55.1%** | **75.7%** | **52.0%** | **5.2%** |
| **MindAct (Flan-T5-XL)** | Cross-Website | **42.0%** | **65.2%** | **38.9%** | **5.1%** |
| **MindAct (Flan-T5-XL)** | Cross-Domain | **42.1%** | **66.5%** | **39.6%** | **2.9%** |
| GPT-4 (50 tasks, in-context) | Cross-Task | 41.6% | 60.6% | 36.2% | 2.0% |
| GPT-4 (50 tasks, in-context) | Cross-Domain | 37.1% | 46.5% | 26.4% | 2.0% |

</details>

<details>
<summary><strong>데이터 수집 파이프라인 — 3단계</strong></summary>

**Stage 1 — Task Proposal:** Amazon Mechanical Turk 작업자들이 태스크를 제안한다. 완전한 백지 상태가 아니라, ChatGPT로 생성한 시드 태스크 50개를 참고한다. 핵심 원칙: *"diverse types, require multiple rounds of interaction, and describe the high-level goal instead of step-by-step instructions."*

**Stage 2 — Task Demonstration:** Playwright 기반 커스텀 툴로 실제 브라우저에서 액션 시퀀스를 녹화한다. 작업자는 DOM 요소를 선택한 뒤 세 가지 오퍼레이션(CLICK / TYPE / SELECT) 중 하나를 고른다. 각 액션 시점의 스냅샷은 MHTML(raw HTML), DOM snapshot(레이아웃 포함), HAR(네트워크 트래픽), trace(재현용) 네 가지 포맷으로 저장된다.

**Stage 3 — Verification:** 저자들이 직접 리뷰. 전체 2,411개 중 **61개 거절**, **390개 태스크 설명 수정**. 최종 2,350개.

</details>

<div class="ornament">· · ·</div>

### WebVoyager 시스템

<figure>
  <img src="img/web-agent-deep-dive/webvoyager-loop.png" alt="WebVoyager 에이전트 루프">
  <figcaption>WebVoyager 에이전트 루프. 태스크 입력 → 스크린샷+번호 레이블 관측 → GPT-4V 추론(Thought/Action) → Selenium 실행 → 최대 15회 반복.</figcaption>
</figure>

**Context Clipping:** 15스텝을 돌면 스크린샷 15장이 쌓여 7,000+ 토큰이 된다.

<figure>
  <img src="img/web-agent-deep-dive/context-clipping.png" alt="Context Clipping — 스크린샷은 최근 3장, Thought+Action은 전부 유지">
  <figcaption>Context Clipping 전략. 스크린샷은 최근 3장만 유지하고 나머지는 삭제. Thought+Action 텍스트는 전체 15개 모두 유지.</figcaption>
</figure>

텍스트 전용과 멀티모달의 결정적 차이:

<figure>
  <img src="img/web-agent-deep-dive/webvoyager-vs-text.png" alt="텍스트 에이전트 vs WebVoyager(멀티모달) — Booking.com 달력 UI 예시">
  <figcaption>Booking.com 달력 UI에서의 차이. 텍스트 에이전트는 달력이 몇 월인지, 차트 수치가 얼마인지 모른다. 멀티모달은 시각적으로 파악한다.</figcaption>
</figure>

**실패 300개 분류:**
- **44.4%** — Navigation Stuck: 길을 잃고 같은 행동 반복
- **24.8%** — Visual Grounding 오류: 발음 기호 못 읽기, 달력 숫자와 SoM 번호 혼동
- **21.8%** — Hallucination: 태스크 일부 누락, 엉뚱한 입력창에 입력
- **9.0%** — Prompt Misalignment: Thought만 출력하거나 미완성인데 ANSWER 조기 종료

<div class="ornament">· · ·</div>

### Agent-E — 계층적 분리와 DOM 디노이징

Agent-E는 텍스트만 본다. 스크린샷을 안 쓴다. 사이트별 특화 프롬프트도 없다. 그런데 WebVoyager 벤치마크에서 **73.2%**를 기록했다 — 멀티모달 WebVoyager(57.1%)보다 16%p 높다.

**두 에이전트의 분업:**

<figure>
  <img src="img/web-agent-deep-dive/agent-e-arch.png" alt="Agent-E 아키텍처 — Planner Agent + Browser Nav Agent">
  <figcaption>Agent-E 아키텍처. Planner Agent가 태스크를 서브태스크로 분해하고, Browser Nav Agent에 위임한다. Browser Nav Agent는 서브태스크마다 새로 초기화된다.</figcaption>
</figure>

<figure>
  <img src="img/web-agent-deep-dive/agent-e-skills.png" alt="Agent-E 전체 시스템 — Planner + Nested Chat + Skills Library">
  <figcaption>Agent-E 전체 시스템. Planner Agent → Nested Chat 안의 Browser Navigation Agent → Skills Library(Get DOM, Click, Enter text, Open URL, Press Key).</figcaption>
</figure>

<figure>
  <img src="img/web-agent-deep-dive/agent-e-loop.png" alt="Agent-E 실행 루프 플로우차트">
  <figcaption>Agent-E 실행 루프. Plan → Task Complete? → 서브태스크별로 Next step → Identify interaction → Act/Sense/Denoise 반복.</figcaption>
</figure>

**세 가지 DOM 표현 (페이로드 디노이징):**

<figure>
  <img src="img/web-agent-deep-dive/agent-e-dom-types.png" alt="Agent-E DOM Distillation — text_only, input_fields, all_fields">
  <figcaption>DOM Distillation의 세 가지 모드. text_only는 정보 수집, input_fields는 폼 입력, all_fields는 탐색·일반 태스크에 사용한다.</figcaption>
</figure>

- **`text_only`** — 정보 수집 태스크. UI 요소 제거, 텍스트만.
- **`input_fields`** — 검색·폼 입력. `<input>`, `<button>`, `<select>`만 + `mmid` 식별자 부여. XPath나 CSS 셀렉터보다 안정적이다.
- **`all_fields`** — 탐색. 전체 요소 + 부모-자식 계층 구조 보존.

**언어적 행동 피드백:**

<figure>
  <img src="img/web-agent-deep-dive/agent-e-feedback.png" alt="Agent-E 언어적 행동 피드백 — Mutation Observer API">
  <figcaption>언어적 행동 피드백. CLICK mmid=25 실행 후 DOM 변화를 Mutation Observer API로 감지해, "팝업 열림: [A, B, C 옵션]" 같은 피드백을 즉시 생성한다.</figcaption>
</figure>

일반적인 에이전트는 행동 결과를 다음 스텝의 DOM에서 간접적으로 파악한다. Agent-E는 **Mutation Observer Web API**를 써서 DOM이 변화할 때마다 실시간 콜백을 받고, 클릭 직후 무슨 일이 일어났는지 언어로 보고한다.

> *"Clicked element mmid 25. A popup appeared with following elements: [mmid=30] Option A, [mmid=31] Option B, [mmid=32] Option C"*

**사이트별 성능 — 텍스트 전용의 강점과 한계:**

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead><tr style="background:#f5f5f0;">
    <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">사이트</th>
    <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">성공률</th>
    <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">이유</th>
  </tr></thead>
  <tbody>
    <tr style="background:#f0fdf4;"><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;"><strong>WolframAlpha</strong></td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">95.7%</strong></td><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">구조적이고 예측 가능한 인터페이스 — DOM으로 충분</td></tr>
    <tr style="background:#f0fdf4;"><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;"><strong>Google Search</strong></td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">90.7%</strong></td><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">단순하고 일관된 UI</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">GitHub</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">82.9%</td><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">텍스트 중심, DOM 구조 명확</td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">ESPN</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">77.3%</td><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;"></td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Apple</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">74.4%</td><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;"></td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Allrecipes</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">71.1%</td><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;"></td></tr>
    <tr><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Amazon</td><td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">70.7%</td><td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;"></td></tr>
    <tr style="background:#fff8f8;"><td style="padding:0.55rem 1rem;"><strong>Booking.com</strong></td><td style="padding:0.55rem 1rem;text-align:center;"><strong style="background:#ffd6d6;padding:2px 6px;border-radius:3px;">27.3%</strong></td><td style="padding:0.55rem 1rem;">달력 UI — 월 이동 버튼 반복 클릭·동적 가격 갱신</td></tr>
  </tbody>
</table>

Booking.com 27.3%가 명확히 보여준다: 텍스트 에이전트의 천장은 **달력·날짜 선택기** 같은 다단계 시각적 인터랙션이다. 클릭 한 번에 달력이 열리고, 월 이동 버튼을 여러 번 눌러야 원하는 날짜에 도달하는 플로우는 DOM 텍스트만으론 파악이 어렵다. 반면 WolframAlpha/Google Search처럼 쿼리-결과 구조가 명확한 사이트에서는 스크린샷 없이도 압도적이다.

논문에서 도출한 네 가지 보편 원칙: **도메인 특화 기본 스킬 → 계층적 아키텍처 → 페이로드 디노이징 → 언어적 행동 피드백.**

<div class="ornament">· · ·</div>

### Browser Use — 단일 루프, 오픈소스 SOTA

**"Make websites accessible for AI agents."**

Agent-E의 계층 구조와 달리 **단일 에이전트 루프**다. 하나의 Agent가 모든 결정과 실행을 담당한다.

<figure>
  <img src="img/web-agent-deep-dive/browser-use-loop.png" alt="Browser Use 5-phase 에이전트 루프">
  <figcaption>Browser Use의 5-Phase 루프. State Capture → Prompt Build → LLM Call → Action Execute → History Record. 각 Phase를 매 스텝 반복한다.</figcaption>
</figure>

Phase 3에서 LLM이 반환하는 네 가지: `thinking`(내부 추론), `evaluation_previous_goal`(이전 액션 평가), `memory`(영속 메모리), `action`(실행할 액션 목록).

**DOM 처리: 10,000+에서 200으로:**

<figure>
  <img src="img/web-agent-deep-dive/browser-use-dom.png" alt="Browser Use DOM 압축 — 10,000+ 요소에서 ~200개로">
  <figcaption>Browser Use의 DOM 압축. 전체 DOM 요소에서 인터랙티브 요소만 약 200개로 추려 숫자 인덱스를 부여한다. 토큰 약 95% 절감.</figcaption>
</figure>

**DOM 우선, Vision은 보조.** 스크린샷 한 장이 추가될 때마다 스텝당 약 0.8초가 더 걸린다. 이 설계 선택이 속도를 결정한다:

| 에이전트 | 평균 태스크 완료 시간 |
|---|---|
| **Browser Use** | **68초** |
| Gemini Computer Use | 225초 |
| Claude Sonnet 4.5 | 285초 |
| OpenAI CUM | 330초 |

**세 가지 안정화 메커니즘:**

```python
# ① ActionLoopDetector — 루프 탈출
# 액션 시퀀스 + 페이지 fingerprint를 모니터링
# 반복 패턴 감지 → 다음 프롬프트에 "nudge" 메시지 주입
if loop_detected(action_history, page_fingerprint):
    prompt += "\n[SYSTEM: You seem to be stuck. Try a different approach.]"

# ② Message Compaction — 장기 태스크 토큰 관리
# 15스텝 → 히스토리만 43,000+ 토큰
# 임계값 초과 시 보조 LLM으로 오래된 스텝 요약
if token_count(history) > THRESHOLD:
    old_steps = history[:-5]
    history = [summarize_llm(old_steps)] + history[-5:]
# → 압축 후 ~12,600 토큰으로 안정화

# ③ KV Cache 인식 구조
# 히스토리(변하지 않는 부분) → 앞에 배치
# 동적 DOM 상태 → 뒤에 배치
# 출력 토큰 최소화 (입력 대비 215배 비쌈)
prompt = system_prompt + task + history_summary + current_dom
# 액션 출력: 스텝당 10~15 토큰만 사용
```

WebVoyager 벤치마크 결과:

| 에이전트 | 성공률 |
|---|---|
| **Browser Use (GPT-4o)** | **89.1%** |
| OpenAI Operator | 87% |
| Agent-E | 73.2% |
| Anthropic Computer Use | 56% |

단, 동일한 에이전트가 엄격한 Online-Mind2Web에서는 ~30%로 내려간다. WebVoyager의 쉬운 태스크 분포가 수치를 크게 부풀린다.

**커스텀 액션 확장:**

```python
from browser_use import Agent, Browser, Tools
from langchain_openai import ChatOpenAI

tools = Tools()

@tools.action(description='검색 결과를 파일에 저장')
async def save_result(content: str) -> str:
    with open('results.txt', 'a') as f:
        f.write(content + '\n')
    return f"저장됨: {content[:50]}..."

browser = Browser(keep_alive=True)
agent = Agent(
    task="각 경쟁사 제품 가격 조사하여 저장",
    llm=ChatOpenAI(model="gpt-4o"),
    browser=browser,
    tools=tools,
    allowed_domains=["competitor1.com", "competitor2.com"],
)
result = await agent.run()
```

`@tools.action` 데코레이터로 커스텀 액션을 등록하면 에이전트가 일반 액션처럼 사용할 수 있다. 복잡한 태스크를 플래너 없이 처리하는 "인라인 플래닝" 방식이다.

<div class="ornament">· · ·</div>

## 종합: 무엇을 믿고 무엇을 조심해야 하나

**벤치마크 선택이 수치를 결정한다.** 같은 에이전트가 WebVoyager에서 89%, Online-Mind2Web에서 30%일 수 있다. 논문의 수치를 볼 때 항상 어느 벤치마크인지, 어느 분할인지, 자동 평가인지 인간 평가인지 확인해야 한다.

**텍스트 vs 멀티모달의 답은 도메인에 따라 다르다.** 웹에서는 DOM이 픽셀보다 더 정확한 "진실"인 경우가 많다. Booking.com처럼 시각적 UI가 핵심인 사이트에서는 멀티모달이 압도적이지만, ArXiv 같이 텍스트 중심인 사이트에서는 차이가 거의 없다.

**아키텍처 선택이 성능을 만든다.** Agent-E의 계층 분리·DOM 디노이징·언어적 피드백, Browser Use의 인덱스 기반 DOM 압축·ActionLoopDetector·KV Cache 인식 구조 — 이 엔지니어링 디테일들이 모델 자체 능력만큼 중요하다.

**현재 위치:** 인간 78~89%(벤치마크별)에 비해 최고 에이전트들이 60% 언저리다. 갭의 대부분은 동적이고 복잡한 UI, 여러 탭을 오가는 플로우, 장기 태스크에서의 에러 누적에서 온다.

<div class="footnote">
  참고 논문:
  <a href="https://arxiv.org/abs/2306.06070">Mind2Web (NeurIPS 2023)</a> ·
  <a href="https://arxiv.org/abs/2307.13854">WebArena (ICLR 2024)</a> ·
  <a href="https://arxiv.org/abs/2401.13919">WebVoyager (ACL 2024)</a> ·
  <a href="https://arxiv.org/abs/2401.13649">VisualWebArena (ACL 2024)</a> ·
  <a href="https://arxiv.org/abs/2407.13032">Agent-E (arXiv 2024)</a>
</div>
