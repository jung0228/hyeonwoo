---
title: "웹 에이전트 직접 만들어보기: v1→v5 삽질 기록"
dek: 다나와 PC 견적 에이전트를 flat ReAct에서 Planner-Executor + Pure Vision + WALT Tools 구조로 5번 뒤집은 과정.
desc: 버전마다 깨진 이유와 고친 방법 — 구현하면서 웹 에이전트 논문들이 왜 그런 구조를 선택했는지 비로소 이해됐다.
tags: [Agent, Multimodal, LLM]
date: Mar 2026
readtime: 18 min read
slug: weg-agent
katex: false
---

## 왜 만들었나

웹 에이전트 논문은 많이 읽었는데, 실제로 돌아가는 코드를 직접 써본 적은 없었다. WebVoyager, Mind2Web, Go-Browse, WALT 방법론들을 정리하면서 계속 드는 생각이 있었다 — *저게 실제로 어떻게 작동하는 거지?* 이론으로 아는 것과 구현이 돌아가는 것 사이에는 항상 간극이 있다.

그래서 만들어봤다. 목표는 단순하다: **다나와에서 예산 내 PC 견적을 자동으로 찾아주는 에이전트.** 예산과 사용 목적을 주면 에이전트가 직접 다나와를 탐색해 부품을 골라 담는다.

결론부터 말하면, 처음 만든 구조는 5번 전면 수정됐다. 이 글은 그 5번의 과정과, 매번 왜 깨졌고 어떻게 고쳤는지에 대한 기록이다.

<div class="ornament">· · ·</div>

## v1→v5: 뭐가 어떻게 바뀌었나

| | v1 | v2 | v3 | v4 | v5 (현재) |
|---|---|---|---|---|---|
| **구조** | 단일 flat ReAct | Planner-Executor | + WALT Tools | + Knowledge + Filter | + sort 분리 + 중고 제외 |
| **인식** | SoM (DOM+오버레이) | Adaptive SoM | Pure Vision | Pure Vision | Pure Vision |
| **액션** | `CLICK [N]` DOM ID | `CLICK [N]` DOM ID | `TOOL` + `CLICK (x,y)` | + `TOOL filter` | + `--no-step-budget` |
| **담기** | 실패 | 실패 | DOM `.click()` 해결 | 동일 | 중고 자동 제외 |
| **광고 스킵** | 없음 | 없음 | `recom_area` 필터 | 동일 | 동일 |
| **LLM 출력** | `Thought/Action` | `Eval/Memory/Goal/Action` | + `Predict:` | 동일 | + 빈 Action DONE 힌트 |

가장 큰 변화는 두 가지다. **구조**: flat ReAct → Planner-Executor. **인식**: DOM+오버레이 SoM → 순수 스크린샷. 나머지는 모두 이 두 가지 변화의 결과로 따라왔다.

<div class="ornament">· · ·</div>

## 현재 구조 (v5)

<div style="margin: 2rem 0;">
<svg viewBox="0 0 740 380" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:740px;display:block;margin:0 auto;font-family:'Source Serif 4',serif;">
  <defs>
    <marker id="arr2" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#555"/>
    </marker>
  </defs>
  <!-- Task input -->
  <rect x="10" y="60" width="120" height="54" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
  <text x="70" y="83" text-anchor="middle" font-size="12" fill="#92400e" font-weight="bold">사용자 태스크</text>
  <text x="70" y="100" text-anchor="middle" font-size="11" fill="#78350f">예산 + 목적</text>
  <!-- arrow to planner -->
  <line x1="130" y1="87" x2="162" y2="87" stroke="#555" stroke-width="1.5" marker-end="url(#arr2)"/>
  <!-- Planner -->
  <rect x="162" y="60" width="110" height="54" rx="8" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
  <text x="217" y="83" text-anchor="middle" font-size="12" fill="#065f46" font-weight="bold">Planner</text>
  <text x="217" y="99" text-anchor="middle" font-size="10" fill="#047857">PlanStep 리스트</text>
  <!-- arrow to executor -->
  <line x1="272" y1="87" x2="304" y2="87" stroke="#555" stroke-width="1.5" marker-end="url(#arr2)"/>
  <!-- Executor box -->
  <rect x="304" y="30" width="280" height="170" rx="10" fill="none" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="444" y="52" text-anchor="middle" font-size="11" fill="#6b7280">Executor (미니 ReAct × N단계)</text>
  <!-- Pure Vision -->
  <rect x="318" y="62" width="88" height="44" rx="6" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="362" y="80" text-anchor="middle" font-size="10" fill="#1d4ed8" font-weight="bold">Pure Vision</text>
  <text x="362" y="94" text-anchor="middle" font-size="9" fill="#1e40af">스크린샷만 캡처</text>
  <!-- AgentBrain -->
  <rect x="420" y="62" width="88" height="44" rx="6" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
  <text x="464" y="78" text-anchor="middle" font-size="10" fill="#065f46" font-weight="bold">AgentBrain</text>
  <text x="464" y="90" text-anchor="middle" font-size="9" fill="#047857">5-field 출력</text>
  <text x="464" y="102" text-anchor="middle" font-size="9" fill="#047857">Eval/Memory/Predict/Goal/Action</text>
  <!-- TOOL/ACT -->
  <rect x="522" y="62" width="52" height="44" rx="6" fill="#fce7f3" stroke="#ec4899" stroke-width="1.5"/>
  <text x="548" y="80" text-anchor="middle" font-size="10" fill="#9d174d" font-weight="bold">TOOL</text>
  <text x="548" y="93" text-anchor="middle" font-size="9" fill="#be185d">DOM 직접</text>
  <text x="548" y="105" text-anchor="middle" font-size="9" fill="#be185d">실행</text>
  <!-- arrows between executor stages -->
  <line x1="406" y1="84" x2="420" y2="84" stroke="#555" stroke-width="1.2" marker-end="url(#arr2)"/>
  <line x1="508" y1="84" x2="522" y2="84" stroke="#555" stroke-width="1.2" marker-end="url(#arr2)"/>
  <!-- loop back in executor -->
  <path d="M 548 106 Q 548 155 444 155 Q 340 155 340 106" fill="none" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#arr2)"/>
  <text x="444" y="172" text-anchor="middle" font-size="9" fill="#9ca3af">Observation → 다음 스텝</text>
  <!-- Memory -->
  <rect x="162" y="170" width="110" height="54" rx="8" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="217" y="193" text-anchor="middle" font-size="12" fill="#4c1d95" font-weight="bold">WorkingMemory</text>
  <text x="217" y="209" text-anchor="middle" font-size="10" fill="#5b21b6">선택 부품 + 잔여예산</text>
  <!-- arrow executor to memory -->
  <line x1="444" y1="200" x2="272" y2="200" stroke="#555" stroke-width="1.5" marker-end="url(#arr2)"/>
  <!-- replan -->
  <line x1="217" y1="224" x2="217" y2="262" stroke="#555" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#arr2)"/>
  <text x="230" y="248" font-size="9" fill="#9ca3af">실패 시 replan()</text>
  <line x1="217" y1="262" x2="217" y2="224" stroke="none"/>
  <!-- arrow memory back to planner -->
  <path d="M 217 170 Q 217 130 217 114" fill="none" stroke="#7c3aed" stroke-width="1.2" stroke-dasharray="3,3" marker-end="url(#arr2)"/>
  <!-- Eval -->
  <rect x="620" y="60" width="110" height="54" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
  <text x="675" y="83" text-anchor="middle" font-size="12" fill="#92400e" font-weight="bold">LLM-as-Judge</text>
  <text x="675" y="99" text-anchor="middle" font-size="10" fill="#78350f">results/*.json 저장</text>
  <line x1="584" y1="87" x2="620" y2="87" stroke="#555" stroke-width="1.5" marker-end="url(#arr2)"/>
  <!-- Knowledge -->
  <rect x="304" y="250" width="280" height="52" rx="8" fill="#f3f4f6" stroke="#9ca3af" stroke-width="1.5"/>
  <text x="444" y="272" text-anchor="middle" font-size="11" fill="#374151" font-weight="bold">Knowledge Base</text>
  <text x="444" y="289" text-anchor="middle" font-size="10" fill="#6b7280">카테고리 ID·셀렉터·상태전이·워크플로우 quirks</text>
  <line x1="444" y1="200" x2="444" y2="250" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3,3" marker-end="url(#arr2)"/>
</svg>
</div>

v1과 지금의 차이를 한 줄로 요약하면: **"매 스텝 화면 보고 결정"에서 "계획 세우고 → 단계별로 실행하고 → 메모리에 기록하고 → 실패하면 재계획"으로 바뀌었다.**

<div class="ornament">· · ·</div>

## v1→v2: 왜 flat ReAct를 버렸나

첫 버전은 단일 ReAct 루프였다. 에이전트가 매 스텝 화면을 보고 다음 액션을 결정한다. 구조가 단순하다는 장점이 있었는데, 문제는 **20스텝 안에 CPU → 메인보드 → RAM → SSD → 케이스 → 파워 6가지를 전부 골라야 한다**는 것이었다.

실험해보니 평균 스텝 분포가 이랬다:
- 카테고리 탐색: 2~3스텝
- 필터/정렬: 1~2스텝
- 부품 확인 후 담기: 2~3스텝

부품당 최소 5~8스텝, 6개 부품이면 30~48스텝이 필요하다. 20스텝에서는 4개 정도 담고 스텝이 끝난다.

더 큰 문제는 **고수준 계획이 없다**는 것이다. 에이전트가 CPU를 담고 나서 "다음은 메인보드다"라는 인식이 없다. 히스토리에서 스스로 추론해야 하는데, 스텝이 길어지면 이전 결정을 잊거나 이미 담은 부품을 다시 찾는다.

**해결**: Planner-Executor 분리. Planner가 태스크를 PlanStep 리스트로 1회 변환하고, Executor가 각 단계를 독립된 미니 ReAct로 실행한다.

```python
# Planner 출력 예시 (50만원 사무용)
[
    {"name": "CPU",   "budget": 80000,  "hint": "Pentium Gold G7400"},
    {"name": "메인보드", "budget": 85000,  "hint": "H610M"},
    {"name": "메모리",  "budget": 30000,  "hint": "DDR4 8GB"},
    {"name": "SSD",   "budget": 50000,  "hint": "256GB"},
    {"name": "케이스",  "budget": 40000,  "hint": "미들 타워"},
    {"name": "파워",   "budget": 50000,  "hint": "500W 80+"}
]
```

각 단계가 독립적이므로 한 단계가 실패해도 다른 단계에 영향이 없다. 실패 시 `replan()`이 남은 단계만 재수립한다.

<div class="ornament">· · ·</div>

## v2→v3: SoM을 버리고 Pure Vision으로

v2까지는 Set-of-Mark(SoM) 방식을 썼다. DOM에서 인터랙티브 요소를 수집하고, 스크린샷에 번호 뱃지를 오버레이해서 LLM이 번호로 요소를 지정하게 하는 방식이다.

문제는 **다나와처럼 복잡한 DOM에서 "담기" 버튼을 제대로 집어내지 못했다**는 것이다.

<div class="callout">
  <strong>에러 #1: 담기 버튼 좌표 클릭 실패</strong><br>
  증상: <code>CLICK (x, y)</code>로 담기 버튼을 눌러도 아무 반응 없음. 같은 좌표 3~5회 반복.<br><br>
  원인: <code>&lt;a class="btn_choice2 wishAction"&gt;</code>는 jQuery <code>.on('click', ...)</code>이 걸린 앵커 태그. Playwright의 합성 마우스 이벤트가 jQuery 리스너를 트리거하지 못하는 케이스가 있다.<br>
  또한 SoM의 <code>filter_for_goal</code>이 "담기" 버튼을 goal 키워드와 무관하다고 점수 0으로 제외 → LLM이 엉뚱한 ID로 hallucination.<br><br>
  <strong>해결: <code>TOOL add_product N</code> — JS <code>element.click()</code>으로 jQuery 리스너 직접 트리거.</strong>
</div>

SoM의 더 근본적인 문제는 DOM 수집 자체가 불안정하다는 거였다. 다나와 페이지는 동적 렌더링이 많아서 수집 시점마다 요소 목록이 달라진다. 결국 v3에서 결정을 내렸다: **DOM 수집을 완전히 제거하고 순수 스크린샷만 사용한다.**

`som.py`는 이제 `perceive(page)` 한 가지 함수만 한다:

```python
async def perceive(page) -> ScreenState:
    screenshot_bytes = await page.screenshot(type="png", full_page=False)
    screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
    url = page.url
    size = page.viewport_size
    return ScreenState(screenshot_b64=screenshot_b64, url=url,
                       width=size["width"], height=size["height"])
```

스크린샷만 찍고, DOM은 건드리지 않는다. LLM이 좌표 기반으로 클릭하거나(`CLICK (x,y)`), 아니면 **TOOL 명령으로 DOM을 직접 조작**하는 두 방식으로 나뉜다.

<div class="ornament">· · ·</div>

## v3: WALT 스타일 Tool 추상화

순수 Vision으로 바꾸면서 좌표 예측 실패 문제가 더 도드라졌다. 좌표는 viewport 크기, 스크롤 위치, 렌더링 타이밍에 모두 민감하다.

이걸 해결하기 위해 WALT(Salesforce AI, 2024)에서 영감을 받아 **사이트 전용 TOOL 추상화**를 도입했다. LLM이 좌표를 추론하는 대신 고수준 명령을 쓰면, TOOL이 실제 DOM 조작을 담당한다:

```
TOOL select_category "CPU"       → JS: category(873, 2) 전역 함수 호출
TOOL filter "인텔(소켓1700)"      → JS: 체크박스 텍스트 매칭 → .click()
TOOL sort_cheapest               → JS: onclick="...CASH_PRICE_ASC" 요소 클릭
TOOL get_products                → JS: 비광고 + 비중고 목록 수집
TOOL add_product N               → JS: N번째 담기 버튼 .click()
TOOL search "H610M"              → JS: 검색창 입력 → btn_search 클릭
TOOL get_cart                    → JS: 오른쪽 패널 선택 부품 반환
```

DOM 클릭의 핵심 이점:

```javascript
// 나쁜 방법 (좌표 추론)
await page.mouse.click(934, 521)  // 픽셀 1개 차이로 실패

// 좋은 방법 (DOM 직접)
document.querySelector('.btn_choice2.wishAction').click()
// → viewport 무관, 스크롤 위치 무관, jQuery 이벤트 모두 트리거
```

<div class="ornament">· · ·</div>

## AgentBrain: 5-field 출력

v2에서 도입한 Eval/Memory/Goal/Action 4-field에, v3에서 **Predict** 필드를 추가했다. Browser Use에서 영감받은 구조다.

```
Eval:    Success
Memory:  Pentium Gold G7400 검색 완료, 85,000원
Predict: sort 후 최저가 G7400이 1번에 표시될 것
Goal:    낮은가격순 정렬 후 담기
Action:  TOOL sort_cheapest
```

**Predict 필드의 효과**: 액션 전에 예상 결과를 명시하면, LLM이 이상한 예측을 쓰다가 스스로 수정하는 경우가 생긴다. "예측: 버튼을 클릭하면 로그인 페이지로 이동할 것"처럼 이상한 예측이 나오면 Goal도 자동으로 재검토된다.

<div class="ornament">· · ·</div>

## 실제로 만나 해결한 버그들

구조 설계보다 구현 디테일에서 더 많이 막혔다. 기억나는 것들을 정리한다.

### 광고 제품이 1번으로 잡힘

`TOOL sort_cheapest` 후 `TOOL add_product 1` → 69만원짜리 CPU가 담겼다.

원인: 다나와 PC견적 목록 상단 2~3개는 `tr.recom_area` 클래스의 광고/추천 제품. 가격순 정렬 후에도 광고 행은 고정 위치를 유지한다.

```javascript
// 해결: recom_area 필터 추가
const rows = Array.from(document.querySelectorAll('tr'))
    .filter(tr => !tr.classList.contains('recom_area'));
// 1번 = 비광고 최저가 보장
```

### 카테고리 셀렉터가 작동 안 함

`TOOL select_category "CPU"` → "카테고리 'CPU' 못 찾음".

실제 DOM을 DevTools로 확인하니:

```html
<dd class="category_873 select pd_item">
  <a class="pd_item_title" onclick="category(873,2);return false;">CPU</a>
</dd>
```

카테고리는 `category(ID, 2)` JS 전역 함수 호출 방식이었다. 초기 셀렉터가 완전히 틀렸던 것.

```javascript
const NAME_TO_ID = { 'CPU': 873, '메인보드': 875, 'SSD': 32617, '메모리': 874, ... }
if (typeof category === 'function') category(NAME_TO_ID[name], 2);
```

### 검색이 전체 사이트 검색으로 이탈

`TOOL search "H410M"` → 결과: 전체(114) / 메인보드(1) / 케이스(113).

원인: `page.keyboard.press("Enter")`가 상단 글로벌 검색창(`#gnbSearchKeyword`)을 트리거해서 전체 사이트 검색을 수행했다.

```python
# Before: Enter 키 → 글로벌 검색창 트리거
# After: btn_search DOM 직접 클릭 → 카테고리 내 검색
await page.evaluate("""
    document.querySelector('button.btn_search').click()
""")
# 탭 분리 시 현재 카테고리 라디오 버튼 자동 선택
```

### sort_cheapest가 검색 결과 뷰에서 작동 안 함

`TOOL search "DDR4"` 후 `TOOL sort_cheapest` → 검색 필터가 초기화되고 제품 목록이 사라짐.

원인: 다나와의 sort 버튼은 **카테고리 뷰 전용**이었다. 검색 결과 탭 뷰에서 sort 클릭 → 카테고리 뷰로 리셋 → 검색 필터 소멸.

```
⚠️ TOOL search 이후에는 절대 sort_cheapest 사용 금지
→ 검색 후에는 바로 get_products → add_product

sort_cheapest는 TOOL select_category 이후 카테고리 뷰에서만 사용
```

시스템 프롬프트에 명시하고, 워크플로우를 검색 방식 대신 카테고리+필터 방식 우선으로 재설계했다.

### sort_cheapest 실행 후 페이지 crash

`estimateMainProduct.sort('GOODSINFO_CASH_PRICE_ASC')` JS를 `page.evaluate()`로 직접 호출 시 30초 타임아웃 → `TargetClosedError`.

원인: sort JS 함수 내부에서 `window.location.reload()`가 트리거되면서 Playwright 컨텍스트가 닫혔다.

```javascript
// 제거: estimateMainProduct.sort('GOODSINFO_CASH_PRICE_ASC')
// 안전: onclick 속성 포함 요소 직접 클릭
const sortEl = Array.from(document.querySelectorAll('[onclick]'))
    .find(el => (el.getAttribute('onclick') || '').includes('CASH_PRICE_ASC'));
if (sortEl) sortEl.click();
```

### Action 필드 비어있는 무한 루프

```
[4] Success |  → CPU가 담겼으므로 목표 달성
[5] Success |  → CPU가 성공적으로 담겼으므로 목표 달성
⚠ 동일 액션 3회 반복
```

원인: LLM이 Predict 필드에 결론을 써버리고 `Action:` 줄을 출력하지 않음. `action_raw.strip() == ""`이면 `action_error = True`지만 `eval_prev = "Success"` 상태라 실패 카운터가 올라가지 않았다.

```python
if not action_raw.strip():
    observation = "Action: 필드가 비어있습니다. 목표 달성 시 반드시 DONE \"...\" 을 출력하세요."
```

add_product 성공 후 DONE 포맷을 몰라서 비우는 경우도 있었다. 마지막 `add_product` 관측값을 추적해서 힌트에 포함했다:

```python
if not action_raw.strip() and last_add_product_obs:
    observation = (
        f"제품을 이미 담았습니다: {last_add_product_obs}. "
        f"반드시 DONE \"목표완료 | 부품:카테고리명 | 이름:제품명 | 가격:숫자\" 형식으로 출력하세요."
    )
```

### 플래너가 예산 초과 모델 추천

50만원 사무용 빌드에서 CPU 예산 95,000원 → 플래너가 `hint: "i3-12100"` 생성. i3-12100 실제 시세 ~150,000원. 에이전트가 담으려다 예산 초과 확인 → 스스로 G6900으로 재검색 → 워크플로우 꼬임.

원인: 플래너 프롬프트에 가격 정보가 없어서 LLM이 "더 좋은 CPU"를 추천.

```
# 플래너 프롬프트에 실제 시세표 추가
| CPU | Celeron G6900       | 60,000~75,000원  | LGA1700, 내장그래픽 |
| CPU | Pentium Gold G7400  | 75,000~90,000원  | LGA1700, 내장그래픽 |
| CPU | i3-12100            | 140,000~160,000원 | → 70만원 미만 예산 초과 |
```

50만원 빌드 CPU 예산 ~80,000원이면 LLM이 자동으로 G7400/G6900을 선택한다.

### 중고 제품이 담기

메모리 검색 후 `TOOL add_product 2` → "삼성전자 DDR4-2133 **중고** (8GB) / 79,120원". 신품 DDR4 8GB 시세 15,000~30,000원의 3배.

```javascript
const realBtns = allBtns.filter(btn => {
    const row = btn.closest('tr');
    if (!row || row.classList.contains('recom_area')) return false;
    const rowText = row.textContent || '';
    if (rowText.includes('중고') || rowText.includes('리퍼') || rowText.includes('재생')) return false;
    return true;
});
// 중고 제외 후 제품 없으면 fallback으로 중고 포함 목록 사용
```

<div class="ornament">· · ·</div>

## Knowledge Base: 사이트 지식 분리

v4에서 도입한 개념이다. Web-CogReasoner(ICLR 2026)의 "사실적·절차적 지식 분리"에서 영감받았다.

에이전트가 매번 UI 구조를 추론하는 대신, 사전에 파악한 지식을 `knowledge.py`에 분리해서 저장한다:

```python
BASE_KNOWLEDGE = DanawaKnowledge(
    url="https://shop.danawa.com/virtualestimate/",
    regions=[
        UIRegion(name="카테고리 패널", selector="dd[class*='category_'].pd_item > a.pd_item_title"),
        UIRegion(name="검색창", selector="#searchProduct"),
        UIRegion(name="담기 버튼", selector=".btn_choice2.wishAction"),
    ],
    category_ids={"CPU": 873, "메인보드": 875, "메모리": 874, "SSD": 32617, ...},
    quirks=[
        "담기 버튼은 jQuery 이벤트 → JS .click() 필수, 좌표 클릭 불가",
        "sort_cheapest는 카테고리 뷰 전용, search 후 사용 금지",
        "recom_area 행은 광고, 번호 카운트에서 제외",
    ]
)
```

`ui_explorer.py`가 있으면 이 지식을 자동으로 탐색·업데이트한다. Phase 1(초기 DOM), Phase 2(카테고리 클릭 상태전이), Phase 3(스크롤 탐색), Phase 4(검색 실험) 순서로 탐색하고 `knowledge/danawa_ui.json`에 캐싱한다.

<div class="ornament">· · ·</div>

## 타 사이트에 적용하려면

구조를 계층별로 분리했기 때문에 새 사이트 적용 시 교체가 필요한 부분이 명확하다:

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">계층</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">컴포넌트</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">이식성</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">오케스트레이션</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>agent.py</code>, <code>planner.py</code>, <code>memory.py</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">✅ 그대로</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">인식</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>som.py</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">✅ 그대로 (순수 스크린샷)</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">지식</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>knowledge.py</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">🔄 사이트별 신규 작성</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">도구</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>tools.py</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">🔄 사이트별 신규 작성</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">탐색</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><code>ui_explorer.py</code></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">🔄 URL만 변경</td>
    </tr>
  </tbody>
</table>

어느 쇼핑몰이든 공통으로 적용되는 원칙들:

1. **클릭 가능한 것은 모두 DOM으로** — 좌표 클릭은 viewport 크기와 스크롤에 민감하다. `element.click()`은 그렇지 않다.
2. **광고 필터링은 필수** — 어느 사이트든 결과 상단에 광고가 있다. (`recom_area`, `data-ad-info`, `[class*="sponsored"]`)
3. **전체 검색창과 카테고리 내 검색창을 구분** — 혼용하면 페이지가 이탈한다.
4. **플래너에게 실제 시세를 주라** — LLM은 "더 좋은 것"을 선택하려는 편향이 있다. 가격 정보 없이 예산 제약만 주면 예산 초과 모델을 추천한다.

<div class="ornament">· · ·</div>

## 배운 것들

직접 구현해보니 논문에서 당연하게 쓰인 표현들이 실제로 어떤 문제를 해결하는지 체감됐다.

**Planner-Executor 분리**는 단순히 구조가 예쁜 게 아니다. flat ReAct는 컨텍스트 낭비가 크다 — 이미 담은 부품 정보가 히스토리에 묻혀서 LLM이 "CPU 아직 안 골랐나?" 하며 다시 찾는다. 각 Executor가 독립된 미니 컨텍스트로 실행되면 이 문제가 사라진다.

**Pure Vision으로 단순화**가 예상보다 잘 작동했다. DOM 수집의 불안정성이 생각보다 컸고, 현대 LLM의 Vision 능력이 스크린샷만으로도 충분히 페이지를 파악한다. SoM 오버레이 없이도 좌표 기반 클릭이 가능하다.

**TOOL 추상화**는 신뢰성을 크게 높였다. LLM이 `CLICK (934, 521)` 같은 좌표를 예측하는 대신 `TOOL select_category "CPU"` 같은 의도를 표현하면 된다. 좌표 예측 실패는 거의 사라졌다.

그리고 가장 중요한 교훈: **LLM이 예상 밖으로 행동하는 이유의 절반은 프롬프트 정보 부족이다.** `sort_cheapest` 후 검색이 안 되는 건 코드 버그가 아니었다 — LLM에게 "search 후 sort 금지" 규칙을 알려주지 않았던 것이다. 중고 제품을 담는 것도, 예산 초과 CPU를 추천하는 것도 마찬가지다. 에이전트 디버깅의 상당 부분은 결국 프롬프트 엔지니어링이다.

<div class="footnote">
  코드: <a href="https://github.com/jung0228/weg-agent">github.com/jung0228/weg-agent</a>
</div>
