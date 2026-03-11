---
title: "웹 에이전트 심층 분석: 동작 원리, 벤치마크, 핵심 시스템"
dek: Playwright에서 Browser Use까지 — 웹 에이전트가 어떻게 브라우저를 조작하고, 어떻게 평가되고, 어떻게 설계되는지를 완전히 해부한다.
desc: 웹 에이전트의 동작 원리(Playwright/CDP/AX Tree/SoM)부터 4대 벤치마크(Mind2Web·WebArena·WebVoyager·VisualWebArena), Mind2Web·WebVoyager·Agent-E·Browser Use 시스템 분석까지 한 편에.
tags: [Agent, LLM, Multimodal]
date: Mar 2026
readtime: 35 min read
slug: web-agent-deep-dive
katex: false
---

## 이 포스트에 대해

웹 에이전트 논문들을 읽다 보면 세 가지 층이 뒤섞인다. **어떻게 브라우저를 조작하는가**(구현 레이어), **어떻게 평가하는가**(벤치마크 레이어), **어떤 아키텍처가 좋은가**(시스템 설계 레이어). 논문마다 자기 맥락에서만 설명하기 때문에 전체 그림이 잘 보이지 않는다.

이 포스트는 세 레이어를 순서대로 쌓는다. 먼저 브라우저 자동화의 물리적 구조를 보고, 그 위에서 어떻게 에이전트를 평가하는지, 그리고 실제 연구들이 어떤 설계 선택을 했는지까지.

<div class="ornament">· · ·</div>

## Part 1 — 동작 방식: 브라우저와 LLM 사이

### LLM의 결정을 물리적 클릭으로

LLM이 "검색 버튼을 클릭한다"는 결정을 내려도 실제 클릭이 일어나려면 뭔가가 브라우저를 직접 조작해야 한다. 웹 에이전트가 쓰는 도구는 **Playwright** 또는 **Puppeteer** 같은 브라우저 자동화 라이브러리다.

이 라이브러리들은 **Chrome DevTools Protocol(CDP)** 을 통해 Chrome 브라우저와 WebSocket으로 통신한다. CDP는 Chrome이 노출하는 저수준 API로, 마우스 이벤트 발송·DOM 조회·JavaScript 실행·스크린샷 캡처 등을 명령한다.

```python
import asyncio
from playwright.async_api import async_playwright

async def run_agent_step(task: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # 1. 페이지 이동
        await page.goto("https://example-shop.com")

        # 2. Accessibility Tree 추출 (LLM 입력용)
        ax_tree = await page.accessibility.snapshot()

        # 3. 스크린샷 캡처 (VLM 입력용)
        screenshot = await page.screenshot(full_page=False)

        # 4. LLM 결정 → 실행
        await page.click('[aria-label="Search"]')       # click
        await page.fill('#search-input', "red canvas shoes")  # type
        await page.select_option('#size-select', "10")  # select_option
        await page.mouse.wheel(0, 500)                  # scroll
        await page.hover(".nav-category")               # hover
        await page.keyboard.press("Enter")              # key_press
        await page.go_back()                            # go_back
```

에이전트 관점에서 **액션 공간은 7~10개 기본 연산**으로 구성된다: `click`, `type`, `select`, `scroll`, `hover`, `goto`, `go_back`, `key_press`, 그리고 태스크 완료를 선언하는 `stop`. 이게 LLM이 생성해야 하는 언어다.

<div class="ornament">· · ·</div>

### Observation: 에이전트는 뭘 보는가

브라우저에서 LLM으로 "현재 상태"를 전달하는 방법이 몇 가지 있고, 어떤 방법을 쓰느냐가 시스템 전체 성능과 토큰 비용을 크게 좌우한다.

**Raw HTML**은 페이지를 그대로 담지만, 수천 줄이 된다.

```html
<!-- 수천 개 div 중 일부 -->
<div class="nav-search-wrapper" data-csa-c-type="widget">
  <div id="nav-search-bar-form">
    <input id="twotabsearchtextbox"
           type="text"
           placeholder="Search Amazon"
           aria-label="Search Amazon">
    <input id="nav-search-submit-button"
           type="submit" value="Go">
  </div>
</div>
<!-- 이후 수천 줄 계속... -->
```

**Accessibility Tree**는 브라우저가 스크린 리더 같은 보조 기술을 위해 내부적으로 유지하는 구조다. 시각적 레이아웃 정보는 없지만 "인터랙티브한 요소가 무엇이 있고 어떤 역할인가"는 정확히 담겨 있다.

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

// 결과: ~80 토큰 (raw HTML은 ~800 토큰)
// 전체 페이지: AX tree ~3K vs HTML ~50K
```

토큰 비용이 약 10배 차이난다. 텍스트 기반 에이전트에게는 AX Tree만으로도 충분한 경우가 많다.

<div class="callout">
  <strong>Set-of-Mark (SoM) — VLM용 시각적 grounding</strong><br>
  스크린샷만 있으면 "어디를 클릭해야 하는가"를 VLM이 픽셀 좌표로 특정해야 한다. Set-of-Mark(SoM)는 스크린샷 위에 인터랙티브 요소마다 번호 레이블을 오버레이하고, LLM이 "3번을 클릭하라"처럼 번호로 참조하게 한다. JavaScript 룰 기반으로 추출하기 때문에 가볍고 빠르다.
</div>

<div class="ornament">· · ·</div>

### LLM과 VLM 입출력 실제 샘플

같은 태스크를 두 방식으로 비교해보자.

**태스크:** *"WebArena 쇼핑몰에서 'Waffle Knit Beanie'를 검색하고 리뷰가 가장 많은 제품을 장바구니에 담아라."*

**텍스트 기반 (WebArena 스타일)**

```
## SYSTEM PROMPT
You are a web browsing agent. Output your next action:
  Thought: [reasoning]
  Action: [action_type] [arguments]

## USER MESSAGE
Task: Search for 'Waffle Knit Beanie' and add the product
with the most reviews to the cart.
URL: http://shop.webarena.xyz/

Accessibility Tree:
[1] banner "One Stop Shop"
[6] searchbox "Search entire store here..."
[7] button "Search"
...
```

```
Thought: 검색창 [6]이 보인다. 검색어를 입력하자.
Action: type [6] Waffle Knit Beanie

------- 다음 턴 -------

Thought: 검색 결과 4개. 리뷰 수 확인:
- [23] "2-Pack Waffle Knit Beanie" - ★4.3 (127)
- [31] "Organic Waffle Beanie" - ★4.1 (43)
Product [23]이 127개로 가장 많다.
Action: click [23]

------- 다음 턴 -------

Action: click [52]  # "Add to Cart" 버튼
Action: stop [Task completed]
```

텍스트 기반은 요소 ID(`[6]`)를 AX Tree에서 직접 가져와서 확실히 존재하는 요소를 참조한다. 비전 기반(WebVoyager/SeeAct 스타일)은 동일한 스크린샷의 SoM 번호를 참조하는데, 이 번호는 JavaScript가 실시간으로 할당하므로 **턴마다 달라질 수 있다.**

<div class="ornament">· · ·</div>

### End-to-End 추적: GitLab 이슈 태스크

**태스크:** *"WebArena GitLab에서 'awesome-os' 프로젝트의 open 이슈 수를 알려라."*

1. **초기화** — Docker GitLab 인스턴스를 초기화. Playwright 브라우저 시작. 로그인 세션 설정.

2. **첫 번째 관찰 → LLM** — AX Tree 추출: `[3] link "awesome-os"` ... LLM이 클릭.

3. **프로젝트 페이지 → 이슈 탭** — `[12] link "Issues (7)"` 발견. LLM이 클릭.

4. **이슈 목록 확인** — `[1] tab "Open (5)" [2] tab "Closed (2)"` → `stop [5]`

5. **평가** — `string_match(answer="5", reference="5")` → ✓ Success

이 4스텝 태스크는 단순한 편이다. 실제 WebArena 태스크들은 평균 10~15스텝이고, 어려운 것들은 30스텝을 넘는다. 각 스텝마다 LLM API 호출이 발생하므로 긴 태스크는 비용도 시간도 많이 든다.

<div class="ornament">· · ·</div>

## Part 2 — 벤치마크 해부

벤치마크마다 근본적인 설계 철학이 다르다. 어느 숫자를 믿을 수 있는지, 어느 숫자를 조심해서 봐야 하는지를 이해하려면 각 벤치마크가 무엇을 측정하는지 알아야 한다.

<div class="ornament">· · ·</div>

### Mind2Web (NeurIPS 2023) — 오프라인 평가의 구조

Mind2Web은 Amazon Mechanical Turk를 통해 실제 사용자들이 웹을 탐색하는 과정을 녹화한 데이터셋이다. **137개 사이트, 2,350개 태스크, 태스크당 평균 7.3개 스텝**이 저장된 HTML 스냅샷으로 존재한다. 실제 브라우저는 없다.

Mind2Web의 설계에서 종종 간과되는 부분이 **세 가지 일반화 시나리오**다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.7rem 1rem;text-align:left;border-bottom:2px solid #ddd;">설정</th>
      <th style="padding:0.7rem 1rem;text-align:left;border-bottom:2px solid #ddd;">핵심 질문</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><strong>Cross-Task</strong></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">같은 사이트, 새 태스크도 되나?</td>
    </tr>
    <tr>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;"><strong>Cross-Website</strong></td>
      <td style="padding:0.6rem 1rem;border-bottom:1px solid #eee;">같은 도메인, 새 사이트도 되나?</td>
    </tr>
    <tr>
      <td style="padding:0.6rem 1rem;"><strong>Cross-Domain</strong></td>
      <td style="padding:0.6rem 1rem;">완전히 새로운 도메인도 되나?</td>
    </tr>
  </tbody>
</table>

성능은 Cross-Task > Cross-Website > Cross-Domain 순으로 떨어진다. 논문에서 "우리 모델이 X%"라고 보고할 때 **어느 분할인지를 반드시 확인해야 한다.**

**네 가지 지표**는 스텝 하나를 (요소 선택, 액션 타입, 액션 값) 세 부분으로 분해해서 각 레벨에서 정확도를 측정한다.

- **Element Accuracy** — 1,135개 DOM 요소 중 올바른 요소를 골랐는가. 오퍼레이션·값은 무관.
- **Operation F1** — CLICK은 exact match, TYPE/SELECT는 토큰 레벨 F1.
- **Step SR** — 요소 + 오퍼레이션 + 값 **셋 다 맞아야 1점**. 부분 점수 없음.
- **Task SR** — **모든 스텝이 전부 Step SR = 1**이어야 성공. 하나라도 틀리면 0.

<div class="callout">
  <strong>Step SR 50%의 수학적 현실</strong><br>
  Step SR 50%라는 숫자만 보면 나쁘지 않아 보인다. 하지만 태스크당 평균 7.3스텝을 가정하면 Task SR ≈ 0.5<sup>7.3</sup> ≈ 0.6%. 실제 최고 성능도 Task SR 5~7%에 그치는 건 이 수학적 귀결이다. 두 숫자를 같이 봐야 한다.
</div>

**MindAct 2단계 파이프라인이 필요한 이유** — 페이지당 평균 1,135개 DOM 요소를 LLM에 그대로 넣으면 컨텍스트 한계를 초과한다. MindAct는 이를 두 단계로 해결한다.

**Stage 1 — Candidate Ranker (DeBERTa-v3-base):** 전체 DOM 요소 중 상위 50개를 추린다. Task query와 각 DOM 요소를 쌍으로 묶어 cross-encoder에 넣고 관련성 점수를 뽑는다. 정확도보다 **재현율(Recall)** 이 중요한 단계다.

**Stage 2 — Action Predictor (GPT-4):** 50개 후보를 객관식 QA 형태로 LLM에게 제시한다. LLM은 정답 요소 선택 + 오퍼레이션 예측 + 값(TYPE/SELECT의 경우) 예측을 수행한다.

논문의 핵심 주장:
<div class="pullquote">
  <strong>"Training LMs for discrimination rather than generation is more generalizable."</strong>
</div>

생성보다 판별(객관식 선택)이 더 잘 일반화된다. 실험에서도 이 직관이 확인된다.

<div class="ornament">· · ·</div>

### WebArena (ICLR 2024) — 기능적 검증

WebArena의 근본적 혁신은 "**에이전트가 무엇을 클릭했는가**"가 아니라 "**환경이 올바른 상태에 도달했는가**"를 묻는다는 점이다.

**두 가지 평가 함수**:

**r_info — 정보 검색 태스크:** `stop [answer]`의 텍스트를 세 가지 방법으로 검증한다.

- `exact_match`: 완전 일치
- `must_include`: 포함 여부 (리스트 답변 등)
- `fuzzy_match`: GPT-4가 심판 → 인간 대비 99.7% 일치

**r_prog — 상태 변경 태스크:** 에이전트 실행 후 환경 상태를 직접 검증한다.

```python
# GitLab API 호출
task: "Create issue 'Fix login bug' in project 'awesome-os'"
validator: gitlab_api.get_issues(project="awesome-os")
           .filter(title="Fix login bug")
# 존재하면 1.0, 없으면 0.0

# DOM/JavaScript 검사
task: "Add Waffle Beanie to cart"
validator: page.evaluate(
    "document.querySelector('.cart-count').textContent"
) == "1"

# 데이터베이스 직접 쿼리
task: "Post comment on r/LocalLLaMA"
validator: db.query("SELECT * FROM comments WHERE user=?",
                    [agent_user]).first().text == expected_text
```

에이전트가 GitLab 이슈를 API로 직접 만들었든, 웹 폼을 채워 제출했든, 단축키를 썼든 — **결과물이 DB에 존재하면 정답이다.** Mind2Web에서 "정답 경로"만 인정하던 것과 근본적으로 다르다.

WebArena에는 실제로 수행 불가능한 태스크도 포함된다. 이때 에이전트는 `stop [N/A]`를 출력해야 한다. 동일 관찰에서 같은 액션을 3번 반복하거나, 파싱 불가능한 액션을 3번 연속 생성하면 태스크 실패 처리된다.

**2년간 SOTA 변화:**

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">에이전트</th>
      <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">발표</th>
      <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">Task SR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">GPT-4 (최초 베이스라인)</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">2023.07</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">14.41%</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">SeeAct (GPT-4V + SoM)</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">2024.01</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">~23%</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">GPT-4o (experience replay)</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">2024.07</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">~36.7%</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Claude Computer Use 3.5</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">2024.10</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">~49.0%</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Claude Computer Use 3.7</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">2025.02</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">~56.3%</td>
    </tr>
    <tr style="background:#fffbf0;">
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;"><strong>OpenAI Operator</strong></td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong>2025.01</strong></td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">~61.3%</strong></td>
    </tr>
    <tr style="background:#f0fdf4;">
      <td style="padding:0.55rem 1rem;"><strong>인간 (CS 대학원생 5명)</strong></td>
      <td style="padding:0.55rem 1rem;text-align:center;">—</td>
      <td style="padding:0.55rem 1rem;text-align:center;"><strong>78.24%</strong></td>
    </tr>
  </tbody>
</table>

<div class="ornament">· · ·</div>

### WebVoyager (ACL 2024) — GPT-4V 심판 방식

WebVoyager는 Mind2Web도, WebArena도 커버하지 못하는 지점을 겨냥했다: **실제 라이브 웹사이트에서 멀티모달 에이전트를 평가**하는 것. Google·Amazon·GitHub·Booking.com 등 15개 실제 사이트, 643개 태스크.

WebArena는 Docker로 격리된 환경에서 DB 상태를 직접 검증한다. 하지만 실제 Amazon이나 GitHub에서는 그게 불가능하다. WebVoyager는 이 문제를 **GPT-4V를 심판으로 써서** 해결했다.

심판에게 주어지는 입력: (1) 태스크 지시문, (2) 에이전트 최종 응답 텍스트, (3) 마지막 k장의 스크린샷.

```
[Evaluation prompt]
"Based on the task, the agent's answer, and the screenshots,
did the agent successfully complete the task?
Answer: Yes / No, with brief reason."

# 인간 평가자와의 일치율: 85.3% (전체 궤적 기준)
# Cohen's kappa (전체 궤적): 0.70
# Cohen's kappa (마지막 1장만): 0.51
```

k를 얼마로 설정하냐가 평가 신뢰도를 크게 좌우한다:

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">제공 스크린샷</th>
      <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">인간과의 일치율</th>
      <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">Cohen's κ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">마지막 1장만</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><span style="background:#ffd6d6;padding:2px 6px;border-radius:3px;">~73%</span></td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><span style="background:#ffd6d6;padding:2px 6px;border-radius:3px;">0.51</span></td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">마지막 3장</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">~80%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">0.63</td>
    </tr>
    <tr style="background:#fffbf0;">
      <td style="padding:0.55rem 1rem;"><strong>전체 궤적</strong></td>
      <td style="padding:0.55rem 1rem;text-align:center;"><strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">85.3%</strong></td>
      <td style="padding:0.55rem 1rem;text-align:center;"><strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">0.70</strong></td>
    </tr>
  </tbody>
</table>

kappa 0.51은 "moderate agreement"로, 동전 뒤집기보다 약간 낫다는 의미다. 많은 후속 연구들이 비용 절감을 위해 마지막 1~3장만 쓰는데, 이 경우 평가 신뢰도가 크게 낮아진다.

**사이트별 텍스트 vs 멀티모달 비교:**

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">사이트</th>
      <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">텍스트 전용</th>
      <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">WebVoyager (멀티모달)</th>
      <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">차이</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Booking.com</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><span style="background:#ffd6d6;padding:2px 6px;border-radius:3px;">2.3%</span></td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">43.2%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong>+40.9%p</strong></td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Google Flights</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">41.5%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">70.7%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong>+29.2%p</strong></td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Amazon</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">36.6%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">58.5%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">+21.9%p</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">ArXiv</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">48.8%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">51.2%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">+2.4%p</td>
    </tr>
    <tr style="background:#fffbf0;">
      <td style="padding:0.55rem 1rem;"><strong>전체 평균</strong></td>
      <td style="padding:0.55rem 1rem;text-align:center;"><strong>40.1%</strong></td>
      <td style="padding:0.55rem 1rem;text-align:center;"><strong>59.1%</strong></td>
      <td style="padding:0.55rem 1rem;text-align:center;"><strong>+19.0%p</strong></td>
    </tr>
  </tbody>
</table>

Booking.com에서 텍스트 전용이 2.3%인 것이 눈에 띈다. 달력 UI와 가격 비교 레이아웃은 DOM만으로 파악하기 극히 어렵다. **시각적 의존도 높은 사이트에서 멀티모달의 이점이 극대화된다.**

<div class="ornament">· · ·</div>

### VisualWebArena (ACL 2024) — 시각 이해 없이는 못 푸는 태스크

WebArena 태스크들은 이론적으로 DOM/AX Tree만으로 전부 해결 가능하다. VisualWebArena는 이 한계를 직접 공략한다: **시각 이해 없이는 원천적으로 풀 수 없는 태스크.**

태스크의 25.2%는 이미지를 태스크 설명에 직접 포함한다:

> *[이미지: 파란 스트라이프 셔츠 사진] "이 셔츠와 동일한 스타일인데 빨간색인 제품을 찾아서 장바구니에 담아라."*

이 태스크는 AX Tree나 HTML에서 "파란 스트라이프 셔츠"라는 정보를 찾을 수 없다. 에이전트는 이미지를 보고 시각적 특성을 이해해야 한다.

시각적 정답 검증을 위해 새로운 검증자도 추가했다:

- **`eval_vqa`** — VQA 모델로 스크린샷에 질문 ("이 상품의 색상은 빨간색인가?")
- **`eval_fuzzy_image_match`** — SSIM으로 스크린샷과 참조 이미지 비교
- **`must_exclude`** — 특정 내용이 없어야 함

결과적으로 GPT-4o + SoM 최고 성능이 **19.78%**인데, 인간은 **88.70%**다. WebArena(78% vs 14%)보다 격차가 더 크다. 이미지를 텍스트로 변환한 뒤 텍스트 에이전트에 주는 Caption 방식은 12.75%에 그쳤다 — 이미지를 설명으로 요약하면 시각적 세부 정보가 손실된다는 것을 보여준다.

<div class="ornament">· · ·</div>

### "Illusion of Progress" — 2025년의 메타 비판

2025년에 발표된 논문 한 편이 기존 벤치마크 수치들을 전면 재검토한다.

**핵심 발견 1:** *Google에서 검색하고 첫 결과 페이지의 정보를 답변으로 제출하는* 단순한 에이전트가 WebVoyager 태스크의 **51%**를 통과한다. 즉 WebVoyager 태스크의 상당 부분은 실제 멀티스텝 웹 탐색이 필요 없다.

이에 대한 대안으로 **Online-Mind2Web** 벤치마크를 제안한다:
- 300개 태스크, **136개 사이트** (다양성 확보)
- 모든 태스크를 **라이브 웹사이트에서 인간이 직접 검증**
- 난이도 분류: 쉬움(≤5스텝) / 중간(6~10스텝) / 어려움(≥11스텝)

자동 평가도 개선했다. **WebJudge**는 기존 GPT-4V 심판 방식의 한계(인간 일치율 66~79%)를 개선한다:

```python
# 1단계: 요구사항 추출
requirements = o4_mini.extract(task)
→ ["navigate to youtube.com",
   "search 'python tutorial beginners'",
   "apply sort by view count filter",
   "identify top result title & views"]

# 2단계: 관련 스크린샷만 필터링 (관련성 점수 δ=3 이상)
# 3단계: o4-mini 이진 판정
verdict = o4_mini.judge(task, requirements, key_screenshots, final_answer)
```

**제대로 측정하면 숫자가 어떻게 달라지나:**

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">에이전트</th>
      <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">발표된 수치</th>
      <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">Online-Mind2Web</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#fffbf0;">
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;"><strong>OpenAI Operator</strong></td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">WebArena ~65%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">61.3%</strong></td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Claude CU 3.7</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">WebArena ~58%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">56.3%</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Browser Use</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">자체 벤치 ~80%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><span style="background:#ffd6d6;padding:2px 6px;border-radius:3px;">~30%</span></td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Agent-E</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">WebArena ~73%</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><span style="background:#ffd6d6;padding:2px 6px;border-radius:3px;">~28%</span></td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;">SeeAct (2024 베이스라인)</td>
      <td style="padding:0.55rem 1rem;text-align:center;">WebVoyager 59%</td>
      <td style="padding:0.55rem 1rem;text-align:center;">~26%</td>
    </tr>
  </tbody>
</table>

<div class="pullquote">
  <strong>2024년에 나온 수많은 웹 에이전트들이 엄격한 평가에서 2024년 초 베이스라인과 비슷한 성능을 보인다. 발표된 높은 수치들의 상당 부분은 쉬운 벤치마크나 느슨한 자동 평가의 인플레이션이다.</strong>
</div>

<div class="ornament">· · ·</div>

## Part 3 — 핵심 시스템 분석

### Mind2Web + MindAct

Mind2Web 데이터 수집 과정이 흥미롭다. Amazon Mechanical Turk 작업자들이 ChatGPT로 생성한 시드 태스크 50개를 참고해 실제 태스크를 제안하고, Playwright 기반 커스텀 툴로 녹화했다. 전체 2,411개 중 61개는 거절, 390개는 태스크 설명을 수정했다.

수집 원칙:

> *"Diverse types, require multiple rounds of interaction, and describe the high-level goal instead of step-by-step instructions."*

MindAct 결과에서 중요한 다섯 가지 발견:

**발견 1 — 일반화 갭은 10%p 이상이다.** Cross-Task 52.0% → Cross-Website 38.9%. 웹사이트별 레이아웃, 버튼 구조, 네이밍 컨벤션이 모두 다르기 때문이다.

**발견 2 — Cross-Website ≈ Cross-Domain.** 직관과 달리 완전히 새로운 도메인과 같은 도메인의 새 사이트 간 성능 차이가 거의 없다. "도메인 지식보다 UI 패턴 다양성이 더 큰 장벽"이라는 의미다. 도메인 특화 에이전트보다 **다양한 UI 패턴에 강인한 에이전트**가 더 중요하다.

**발견 3 — Task SR이 낮은 건 수학적으로 당연하다.** 최고 성능 Cross-Task에서도 Task SR = 5.2%. Step SR 52%, 평균 7.3스텝이면 0.52^7 ≈ 0.9%. 실제 5.2%는 스텝들이 컨텍스트를 공유하기 때문에 이보다 높다.

**발견 4 — GPT-4가 파인튜닝 없이도 강하다.** In-context learning만으로 Element Accuracy 41.6%. GPT-3.5-turbo(20.3%)와 비교하면 압도적이다.

**발견 5 — 파인튜닝이 필수다.** 제로샷 Flan-T5-XL은 Element Accuracy 10.8%. 파인튜닝 후 52.0%. 대형 모델이라도 이 태스크에서는 파인튜닝이 필수다.

<div class="ornament">· · ·</div>

### WebVoyager 시스템

WebVoyager는 Set-Label(SoM)과 GPT-4V를 결합한 첫 본격적 멀티모달 웹 에이전트다.

**Context Clipping 문제:** 15스텝을 돌면 스크린샷 15장이 쌓인다. 7,000+ 토큰이 필요하다. 해결책은 단순하다 — **스크린샷은 최근 3장만, Thought+Action 텍스트는 전부 유지.**

**7가지 액션 공간:**

- `Click [N]` — N번 요소 클릭
- `Type [N]; [텍스트]` — 입력창에 텍스트 입력 (자동으로 Enter까지)
- `Scroll [N 또는 WINDOW]; [up/down]`
- `Wait` — 페이지 로딩 대기
- `GoBack` — 이전 페이지로
- `Google` — 구글 검색으로 새로 시작 (막혔을 때 탈출용)
- `ANSWER; [내용]` — 최종 답변 제출

**실패 300개 분류:**

- **44.4% — Navigation Stuck:** 길을 잃고 같은 행동 반복 (부정확한 검색어, 잘못된 스크롤 방향, context clipping으로 이전 실수 망각)
- **24.8% — Visual Grounding 오류:** 발음 기호·수식 기호 못 읽음, 미세한 변화 감지 실패, 달력 숫자와 SoM 번호 혼동
- **21.8% — Hallucination:** 태스크 일부 누락, 엉뚱한 입력창에 텍스트 입력
- **9.0% — Prompt Misalignment:** Thought만 출력하거나, 미완성 상태에서 `ANSWER` 조기 종료

WebVoyager의 의의는 두 가지다. **첫째, 시각 정보가 결정적이라는 실증.** 텍스트 40.1% → 멀티모달 59.1%, +19%p. **둘째, GPT-4V를 심사위원으로 활용하는 자동 평가 프로토콜**이 이후 웹 에이전트 연구의 평가 기준에 직접적인 영향을 줬다.

<div class="ornament">· · ·</div>

### Agent-E — 계층적 분리와 DOM 디노이징

Agent-E는 텍스트만 본다. 스크린샷을 안 쓴다. 사이트별 특화 프롬프트도 없다. 그런데 WebVoyager 벤치마크에서 **73.2%** 를 기록했다 — 멀티모달 WebVoyager(57.1%)보다 16%p 높다.

**두 에이전트의 분업:**

<div style="margin:2rem 0;">
<svg viewBox="0 0 660 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:660px;display:block;margin:0 auto;font-family:'Source Serif 4',Georgia,serif;">
  <rect width="660" height="200" fill="#fafaf7" rx="8"/>
  <rect x="15" y="75" width="95" height="50" rx="5" fill="#e8e8e2" stroke="#bbb" stroke-width="1.5"/>
  <text x="62" y="98" text-anchor="middle" font-size="11" fill="#444" font-weight="600">사용자</text>
  <text x="62" y="114" text-anchor="middle" font-size="10" fill="#666">태스크</text>
  <line x1="112" y1="100" x2="160" y2="100" stroke="#888" stroke-width="1.5" marker-end="url(#da1)"/>
  <rect x="162" y="45" width="145" height="110" rx="7" fill="#fff8ec" stroke="#e8b84b" stroke-width="2"/>
  <text x="234" y="72" text-anchor="middle" font-size="12" fill="#8a6200" font-weight="700">Planner Agent</text>
  <text x="234" y="90" text-anchor="middle" font-size="10" fill="#a07800">태스크 분해</text>
  <text x="234" y="106" text-anchor="middle" font-size="10" fill="#a07800">서브태스크 위임</text>
  <text x="234" y="122" text-anchor="middle" font-size="9.5" fill="#c08000">DOM 노이즈 차단</text>
  <text x="234" y="140" text-anchor="middle" font-size="9.5" fill="#c08000">에러 감지·복구</text>
  <line x1="309" y1="78" x2="355" y2="68" stroke="#aaa" stroke-width="1.2" stroke-dasharray="4,2" marker-end="url(#da2)"/>
  <line x1="309" y1="100" x2="355" y2="100" stroke="#aaa" stroke-width="1.2" stroke-dasharray="4,2" marker-end="url(#da2)"/>
  <line x1="309" y1="122" x2="355" y2="132" stroke="#aaa" stroke-width="1.2" stroke-dasharray="4,2" marker-end="url(#da2)"/>
  <text x="330" y="61" text-anchor="middle" font-size="8.5" fill="#999">sub 1</text>
  <text x="330" y="96" text-anchor="middle" font-size="8.5" fill="#999">sub 2</text>
  <text x="330" y="138" text-anchor="middle" font-size="8.5" fill="#999">sub 3</text>
  <rect x="357" y="40" width="145" height="60" rx="6" fill="#edf2ff" stroke="#7c9ef5" stroke-width="1.8"/>
  <text x="430" y="64" text-anchor="middle" font-size="11.5" fill="#1a3a8f" font-weight="700">Browser Nav</text>
  <text x="430" y="81" text-anchor="middle" font-size="10" fill="#2a52bf">Agent (매번 새로 초기화)</text>
  <rect x="357" y="100" width="145" height="60" rx="6" fill="#edf2ff" stroke="#7c9ef5" stroke-width="1.5" opacity="0.5"/>
  <text x="430" y="124" text-anchor="middle" font-size="11.5" fill="#1a3a8f" font-weight="700" opacity="0.5">Browser Nav</text>
  <text x="430" y="141" text-anchor="middle" font-size="10" fill="#2a52bf" opacity="0.5">Agent</text>
  <line x1="504" y1="100" x2="554" y2="100" stroke="#888" stroke-width="1.5" marker-end="url(#da1)"/>
  <rect x="556" y="75" width="85" height="50" rx="5" fill="#f0fdf4" stroke="#4ade80" stroke-width="1.5"/>
  <text x="598" y="98" text-anchor="middle" font-size="11" fill="#166534" font-weight="600">웹</text>
  <text x="598" y="114" text-anchor="middle" font-size="10" fill="#166534">브라우저</text>
  <defs>
    <marker id="da1" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L0,5 L7,2.5 z" fill="#888"/></marker>
    <marker id="da2" markerWidth="6" markerHeight="6" refX="4" refY="2.5" orient="auto"><path d="M0,0 L0,5 L6,2.5 z" fill="#aaa"/></marker>
  </defs>
</svg>
</div>

**Planner Agent**는 큰 그림만 관리한다. DOM의 세부 노이즈를 직접 보지 않는다. **Browser Navigation Agent**는 각 서브태스크마다 새로 초기화된다 — 이전 서브태스크의 기억이 없다. 지금 당장 할 일에만 집중한다.

**세 가지 DOM 표현 (페이로드 디노이징):**

- **`text_only`** — 정보 수집 태스크. UI 요소 제거, 텍스트만 추출.
- **`input_fields`** — 검색·폼 입력 태스크. `<input>`, `<button>`, `<select>`만 추출 + `mmid`(mind map identifier) 식별자 부여. LLM이 `"mmid=142에 클릭"`이라고 지정하면 정확히 그 요소를 찾아 실행. XPath나 CSS 셀렉터보다 훨씬 안정적이다.
- **`all_fields`** — 탐색·일반 태스크. 전체 요소 + 부모-자식 계층 구조 보존. 경쟁 에이전트들의 flat 인코딩과 다르다.

**언어적 행동 피드백:** 일반적인 에이전트는 행동 결과를 다음 스텝의 DOM에서 간접적으로 파악한다. Agent-E는 **Mutation Observer Web API**를 써서 DOM이 변화할 때마다 실시간 콜백을 받고, 클릭 직후 무슨 일이 일어났는지 언어로 보고한다.

> *"Clicked element mmid 25. A popup appeared with following elements: [mmid=30] Option A, [mmid=31] Option B, [mmid=32] Option C"*

논문에서 도출한 네 가지 보편 원칙: **도메인 특화 기본 스킬 → 계층적 아키텍처 → 페이로드 디노이징 → 언어적 행동 피드백.** 이 원칙들은 웹 에이전트에만 적용되는 게 아니라 에이전틱 시스템 일반에 대한 주장이다.

<div class="ornament">· · ·</div>

### Browser Use — 단일 루프, 오픈소스 SOTA

**"Make websites accessible for AI agents."**

Browser Use는 Agent-E의 계층 구조와 달리 **단일 에이전트 루프**다. 하나의 Agent가 모든 결정과 실행을 담당한다. 성능을 만드는 건 세 가지 엔지니어링 디테일이다.

<div style="margin:2rem 0;">
<svg viewBox="0 0 660 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:660px;display:block;margin:0 auto;font-family:'Source Serif 4',Georgia,serif;">
  <rect width="660" height="130" fill="#fafaf7" rx="8"/>
  <rect x="10" y="30" width="108" height="70" rx="6" fill="#fff8ec" stroke="#e8b84b" stroke-width="1.8"/>
  <text x="64" y="58" text-anchor="middle" font-size="11" fill="#8a6200" font-weight="700">Phase 1</text>
  <text x="64" y="74" text-anchor="middle" font-size="9.5" fill="#a07800">State Capture</text>
  <text x="64" y="90" text-anchor="middle" font-size="9" fill="#bbb">DOM + 스크린샷</text>
  <line x1="120" y1="65" x2="138" y2="65" stroke="#ccc" stroke-width="1.5" marker-end="url(#bu1)"/>
  <rect x="140" y="30" width="108" height="70" rx="6" fill="#edf2ff" stroke="#7c9ef5" stroke-width="1.8"/>
  <text x="194" y="58" text-anchor="middle" font-size="11" fill="#1a3a8f" font-weight="700">Phase 2</text>
  <text x="194" y="74" text-anchor="middle" font-size="9.5" fill="#2a52bf">Prompt Build</text>
  <text x="194" y="90" text-anchor="middle" font-size="9" fill="#bbb">히스토리 + DOM</text>
  <line x1="250" y1="65" x2="268" y2="65" stroke="#ccc" stroke-width="1.5" marker-end="url(#bu1)"/>
  <rect x="270" y="30" width="108" height="70" rx="6" fill="#f0fdf4" stroke="#4ade80" stroke-width="1.8"/>
  <text x="324" y="58" text-anchor="middle" font-size="11" fill="#166534" font-weight="700">Phase 3</text>
  <text x="324" y="74" text-anchor="middle" font-size="9.5" fill="#166534">LLM Call</text>
  <text x="324" y="90" text-anchor="middle" font-size="9" fill="#bbb">structured output</text>
  <line x1="380" y1="65" x2="398" y2="65" stroke="#ccc" stroke-width="1.5" marker-end="url(#bu1)"/>
  <rect x="400" y="30" width="108" height="70" rx="6" fill="#fdf0ff" stroke="#c084fc" stroke-width="1.8"/>
  <text x="454" y="58" text-anchor="middle" font-size="11" fill="#7e22ce" font-weight="700">Phase 4</text>
  <text x="454" y="74" text-anchor="middle" font-size="9.5" fill="#7e22ce">Action Execute</text>
  <text x="454" y="90" text-anchor="middle" font-size="9" fill="#bbb">CDP → 브라우저</text>
  <line x1="510" y1="65" x2="528" y2="65" stroke="#ccc" stroke-width="1.5" marker-end="url(#bu1)"/>
  <rect x="530" y="30" width="120" height="70" rx="6" fill="#fff5f5" stroke="#f87171" stroke-width="1.8"/>
  <text x="590" y="58" text-anchor="middle" font-size="11" fill="#991b1b" font-weight="700">Phase 5</text>
  <text x="590" y="74" text-anchor="middle" font-size="9.5" fill="#991b1b">History Record</text>
  <text x="590" y="90" text-anchor="middle" font-size="9" fill="#bbb">다음 Phase 2 입력</text>
  <defs>
    <marker id="bu1" markerWidth="6" markerHeight="6" refX="4" refY="2.5" orient="auto"><path d="M0,0 L0,5 L6,2.5 z" fill="#ccc"/></marker>
  </defs>
</svg>
</div>

**Phase 3에서 LLM이 반환하는 네 가지:** `thinking`(내부 추론), `evaluation_previous_goal`(이전 액션 평가), `memory`(영속 메모리 노트), `action`(실행할 액션 목록).

**DOM 처리: 10,000+에서 200으로.** Browser Use는 페이지 전체 요소에서 **~200개의 인터랙티브 요소**만 추려 숫자 인덱스로 표현한다. 토큰 약 95% 절감. 인덱스는 스텝마다 재할당되므로 XPath 같은 복잡한 셀렉터가 필요 없다.

**DOM 우선, Vision은 보조.** 스크린샷 한 장이 LLM 파이프라인에 추가될 때마다 이미지 인코더 처리 시간이 붙는다. Browser Use 측정값으로 스텝당 약 0.8초. 이 설계 선택이 속도에서 극적인 차이를 만든다:

- Browser Use: **평균 68초**
- Gemini Computer Use: 225초
- Claude Sonnet 4.5: 285초
- OpenAI Computer-Using Model: 330초

**세 가지 안정화 메커니즘:**

**① ActionLoopDetector** — 에이전트가 같은 액션을 반복하는 것을 감지해, 다음 LLM 프롬프트에 "nudge" 메시지를 주입한다. 에이전트가 다른 접근을 시도하도록 유도한다.

**② Message Compaction** — 에이전트가 15스텝을 실행하면 히스토리만으로 43,000+ 토큰이 된다. 임계값을 넘으면 보조 LLM을 불러서 오래된 스텝들을 요약한다. 압축 후 토큰 사용량은 약 12,600으로 안정화된다.

**③ KV Cache 인식 프롬프트 구조** — 히스토리(상대적으로 변하지 않는 부분)를 프롬프트 앞에 배치하고, 동적으로 바뀌는 DOM 상태를 뒤에 배치한다. 입력 토큰(29.1ms/1K)보다 출력 토큰(62.6ms/1K)이 215배 비싸기 때문에, 액션 출력도 스텝당 10~15 토큰으로 최소화했다.

**WebVoyager 벤치마크 성능:**

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">에이전트</th>
      <th style="padding:0.6rem 1rem;text-align:center;border-bottom:2px solid #ddd;">성공률</th>
      <th style="padding:0.6rem 1rem;text-align:left;border-bottom:2px solid #ddd;">특이사항</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#fffbf0;">
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;"><strong>Browser Use (GPT-4o)</strong></td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;"><strong style="background:#d4f7d4;padding:2px 6px;border-radius:3px;">89.1%</strong></td>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">단일 루프, 오픈소스</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">OpenAI Operator</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">87%</td>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">상용 서비스</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">Agent-E</td>
      <td style="padding:0.55rem 1rem;text-align:center;border-bottom:1px solid #eee;">73.2%</td>
      <td style="padding:0.55rem 1rem;border-bottom:1px solid #eee;">계층적 구조</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 1rem;">Anthropic Computer Use</td>
      <td style="padding:0.55rem 1rem;text-align:center;">56%</td>
      <td style="padding:0.55rem 1rem;">스크린샷 기반</td>
    </tr>
  </tbody>
</table>

단, "Illusion of Progress" 논문이 지적했듯이, 동일한 에이전트가 엄격한 Online-Mind2Web에서는 ~30%로 뚝 떨어진다. **WebVoyager 벤치마크의 쉬운 태스크 분포가 수치를 크게 부풀린다.**

Browser Use는 커스텀 액션을 `@tools.action` 데코레이터로 등록해 확장할 수 있다. 복잡한 태스크를 플래너 없이 처리하려면, 커스텀 액션으로 "인라인 플래닝"을 구현하는 것이다 — 결과를 중간에 파일에 저장하고, 다음 단계에서 읽어서 판단하는 방식.

<div class="ornament">· · ·</div>

## 종합: 무엇을 믿고 무엇을 조심해야 하나

지금까지 세 레이어를 쌓았다. 마지막으로 전체 그림을 정리하자.

**벤치마크 선택이 수치를 결정한다.** 같은 에이전트가 WebVoyager에서는 89%, Online-Mind2Web에서는 30%일 수 있다. 논문의 수치를 볼 때 항상 어느 벤치마크인지, 어느 분할인지, 자동 평가인지 인간 평가인지 확인해야 한다.

**텍스트 vs 멀티모달의 답은 도메인에 따라 다르다.** 웹에서는 DOM이 픽셀보다 더 정확한 "진실"인 경우가 많다. Booking.com처럼 시각적 UI가 핵심인 사이트에서는 멀티모달이 압도적이지만, ArXiv 같이 텍스트 중심인 사이트에서는 차이가 거의 없다.

**아키텍처 선택이 성능을 만든다.** Agent-E의 계층 분리, DOM 디노이징, 언어적 피드백 — Browser Use의 인덱스 기반 DOM 압축, ActionLoopDetector, KV Cache 인식 구조 — 이 엔지니어링 디테일들이 모델 자체 능력만큼 중요하다.

**현재 위치:** 인간 78~89%(벤치마크별)에 비해 최고 에이전트들이 60% 언저리다. 갭의 대부분은 동적이고 복잡한 UI (달력, 다단계 폼), 여러 탭을 오가는 플로우, 장기 태스크에서의 에러 누적에서 온다. 개별 스텝의 정확도는 많이 올랐지만, 긴 체인을 오류 없이 완주하는 것은 여전히 어렵다.

<div class="footnote">
  참고 논문:
  <a href="https://arxiv.org/abs/2306.06070">Mind2Web (NeurIPS 2023)</a> ·
  <a href="https://arxiv.org/abs/2307.13854">WebArena (ICLR 2024)</a> ·
  <a href="https://arxiv.org/abs/2401.13919">WebVoyager (ACL 2024)</a> ·
  <a href="https://arxiv.org/abs/2401.13649">VisualWebArena (ACL 2024)</a> ·
  <a href="https://arxiv.org/abs/2407.13032">Agent-E (arXiv 2024)</a>
</div>
