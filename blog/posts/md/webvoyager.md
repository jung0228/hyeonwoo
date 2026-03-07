---
title: "WebVoyager: 진짜 웹에서 동작하는 멀티모달 에이전트"
dek: GPT-4V로 스크린샷을 직접 보고, 실제 웹사이트를 탐색하다.
desc: 텍스트만 보던 에이전트에서 스크린샷을 직접 보는 에이전트로 — WebVoyager가 해결한 두 가지 핵심 문제.
tags: [Agent, Multimodal, LLM]
date: Mar 2026
readtime: 12 min read
slug: webvoyager
katex: false
---

## 웹을 탐색하는 에이전트, 무엇이 어려운가

웹을 자동으로 탐색하는 에이전트를 만드는 건 생각보다 복잡하다. 사람은 웹페이지를 눈으로 보고 직관적으로 이해하지만, 컴퓨터에게 웹페이지는 수천 줄의 HTML 코드 덩어리다. "항공편 검색하고 최저가 찾아줘" 같은 태스크를 자동화하려면 에이전트가 버튼을 인식하고, 달력을 읽고, 검색 결과를 이해하고, 여러 페이지를 넘나들어야 한다.

2024년 발표된 **WebVoyager** (ACL 2024)는 이 문제에 멀티모달 LLM을 직접 투입한 첫 번째 end-to-end 시스템 중 하나다. 핵심 질문은 간단하다: *"HTML 텍스트 대신 스크린샷을 그대로 보여주면 어떨까?"*

## 이전 방법의 두 가지 한계

WebVoyager가 등장하기 전, 웹 에이전트 연구는 두 가지 근본적인 문제를 안고 있었다.

### 문제 1 — 에이전트가 보는 것과 사람이 보는 것이 달랐다

기존 에이전트들은 HTML/DOM 텍스트만 입력으로 받았다. 웹페이지를 렌더링된 시각적 결과물이 아닌, 원본 코드로 보는 방식이다. 문제는 HTML이 엄청나게 길고 노이즈가 많다는 점, 그리고 무엇보다 **시각적 정보를 표현할 수 없다**는 점이다.

<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;margin:1.8rem 0;border-radius:6px;border:1px solid #d8d8d2;">
  <rect width="720" height="300" fill="#f9f8f4"/>
  <rect x="0" y="0" width="340" height="300" fill="#efede6"/>
  <rect x="0" y="0" width="340" height="36" fill="#d8d5cc"/>
  <text x="170" y="23" text-anchor="middle" font-size="11" font-weight="700" fill="#6b6b6b" font-family="sans-serif" letter-spacing="1">이전 에이전트가 보는 것</text>
  <text x="16" y="58"  font-size="9" fill="#333" font-family="monospace">&lt;div class="page-wrapper" id="main"&gt;</text>
  <text x="16" y="72"  font-size="9" fill="#444" font-family="monospace">  &lt;nav role="navigation" aria-label="..."&gt;</text>
  <text x="16" y="86"  font-size="9" fill="#555" font-family="monospace">    &lt;ul class="nav-list nav-primary"&gt;</text>
  <text x="16" y="100" font-size="9" fill="#666" font-family="monospace">      &lt;li class="nav-item active"&gt;&lt;a href=...&gt;</text>
  <text x="16" y="114" font-size="9" fill="#777" font-family="monospace">      &lt;li data-id="32" class="dropdown"&gt;</text>
  <text x="16" y="128" font-size="9" fill="#888" font-family="monospace">    &lt;/ul&gt;&lt;div class="search-wrap"&gt;</text>
  <text x="16" y="142" font-size="9" fill="#999" font-family="monospace">      &lt;input type="text" placeholder="검색..."&gt;</text>
  <text x="16" y="156" font-size="9" fill="#aaa" font-family="monospace">  &lt;section class="calendar-widget"&gt;</text>
  <text x="16" y="170" font-size="9" fill="#bbb" font-family="monospace">    &lt;table class="cal" data-month="3"&gt;</text>
  <text x="16" y="184" font-size="9" fill="#ccc" font-family="monospace">      &lt;tr&gt;&lt;td&gt;1&lt;/td&gt;&lt;td class="sel"&gt;2&lt;/td&gt;...</text>
  <text x="16" y="198" font-size="9" fill="#ddd" font-family="monospace">    &lt;/table&gt;&lt;div class="chart-container"&gt;</text>
  <text x="16" y="212" font-size="9" fill="#e0e0e0" font-family="monospace">      &lt;canvas id="chart1" width="400"&gt;...</text>
  <text x="16" y="226" font-size="9" fill="#e8e8e8" font-family="monospace">    &lt;/div&gt;&lt;/section&gt;&lt;/div&gt;&lt;/nav&gt;</text>
  <rect x="16" y="244" width="308" height="42" fill="#fde8e8" rx="4" stroke="#e08080" stroke-width="1"/>
  <text x="170" y="260" text-anchor="middle" font-size="10" font-weight="700" fill="#c03030" font-family="sans-serif">달력이 몇 월인지? ❌ 모름</text>
  <text x="170" y="278" text-anchor="middle" font-size="9" fill="#c03030" font-family="sans-serif">차트 수치? ❌ 모름  ·  레이아웃 구조? ❌ 모름</text>
  <text x="360" y="158" text-anchor="middle" font-size="28" fill="#1a56c4" font-family="sans-serif">→</text>
  <rect x="380" y="0" width="340" height="300" fill="white"/>
  <rect x="380" y="0" width="340" height="36" fill="#1a56c4"/>
  <text x="550" y="23" text-anchor="middle" font-size="11" font-weight="700" fill="white" font-family="sans-serif" letter-spacing="1">WebVoyager가 보는 것</text>
  <rect x="390" y="44" width="320" height="16" fill="#e8e8e8" rx="2"/>
  <circle cx="400" cy="52" r="4" fill="#ff5f57"/>
  <circle cx="413" cy="52" r="4" fill="#febc2e"/>
  <circle cx="426" cy="52" r="4" fill="#28c840"/>
  <rect x="436" y="46" width="200" height="12" fill="white" rx="3"/>
  <text x="536" y="55" text-anchor="middle" font-size="7" fill="#999" font-family="sans-serif">booking.com/flights</text>
  <rect x="390" y="60" width="320" height="24" fill="#003580"/>
  <text x="410" y="76" font-size="10" fill="white" font-weight="700" font-family="sans-serif">Booking.com</text>
  <rect x="398" y="90" width="148" height="110" fill="white" rx="3" stroke="#d8d8d2" stroke-width="1"/>
  <rect x="398" y="90" width="148" height="20" fill="#efede6" rx="3"/>
  <text x="472" y="104" text-anchor="middle" font-size="9" font-weight="700" fill="#0f0f0f" font-family="sans-serif">◀  March 2026  ▶</text>
  <text x="408" y="120" font-size="7.5" fill="#999" font-family="sans-serif">Su Mo Tu We Th Fr Sa</text>
  <text x="408" y="133" font-size="7.5" fill="#333" font-family="sans-serif"> 1   2   3   4   5   6   7</text>
  <text x="408" y="146" font-size="7.5" fill="#333" font-family="sans-serif"> 8   9  10  11  12  13  14</text>
  <rect x="438" y="150" width="16" height="13" fill="#1a56c4" rx="2"/>
  <text x="408" y="160" font-size="7.5" fill="#333" font-family="sans-serif">15  </text>
  <text x="446" y="160" font-size="7.5" fill="white" font-weight="700" font-family="sans-serif">16</text>
  <text x="463" y="160" font-size="7.5" fill="#333" font-family="sans-serif">  17  18  19  20  21</text>
  <text x="408" y="173" font-size="7.5" fill="#333" font-family="sans-serif">22  23  24  25  26  27  28</text>
  <text x="408" y="186" font-size="7.5" fill="#333" font-family="sans-serif">29  30  31</text>
  <rect x="554" y="90" width="148" height="30" fill="white" rx="3" stroke="#d8d8d2" stroke-width="1"/>
  <text x="562" y="109" font-size="8.5" fill="#999" font-family="sans-serif">어디로 가세요?</text>
  <rect x="554" y="128" width="148" height="28" fill="#1a56c4" rx="3"/>
  <text x="628" y="146" text-anchor="middle" font-size="9" fill="white" font-weight="700" font-family="sans-serif">검색</text>
  <rect x="398" y="208" width="304" height="50" fill="#f8f8f8" rx="3" stroke="#d8d8d2" stroke-width="1"/>
  <text x="404" y="220" font-size="7" fill="#999" font-family="sans-serif">최저가 달력</text>
  <rect x="408" y="228" width="12" height="22" fill="#1a56c4" opacity="0.4"/>
  <rect x="426" y="234" width="12" height="16" fill="#1a56c4" opacity="0.5"/>
  <rect x="444" y="220" width="12" height="30" fill="#1a56c4" opacity="0.3"/>
  <rect x="462" y="232" width="12" height="18" fill="#1a56c4" opacity="0.5"/>
  <rect x="480" y="225" width="12" height="25" fill="#1a56c4" opacity="0.6"/>
  <rect x="498" y="236" width="12" height="14" fill="#1a56c4" opacity="0.4"/>
  <rect x="516" y="228" width="12" height="22" fill="#1a56c4" opacity="0.7"/>
  <text x="410" y="256" font-size="6.5" fill="#666" font-family="sans-serif">₩82k  ₩71k  ₩95k  ₩75k  ₩88k  ₩68k  ₩79k</text>
  <rect x="397" y="89" width="150" height="112" fill="none" stroke="#1a56c4" stroke-width="1.5" rx="3" stroke-dasharray="3,2"/>
  <rect x="397" y="89" width="15" height="15" fill="#1a56c4" rx="3"/>
  <text x="405" y="101" text-anchor="middle" font-size="9" fill="white" font-weight="700" font-family="sans-serif">1</text>
  <rect x="553" y="89" width="150" height="32" fill="none" stroke="#1a56c4" stroke-width="1.5" rx="3" stroke-dasharray="3,2"/>
  <rect x="553" y="89" width="15" height="15" fill="#1a56c4" rx="3"/>
  <text x="561" y="101" text-anchor="middle" font-size="9" fill="white" font-weight="700" font-family="sans-serif">2</text>
  <rect x="553" y="127" width="150" height="30" fill="none" stroke="#1a56c4" stroke-width="1.5" rx="3" stroke-dasharray="3,2"/>
  <rect x="553" y="127" width="15" height="15" fill="#1a56c4" rx="3"/>
  <text x="561" y="139" text-anchor="middle" font-size="9" fill="white" font-weight="700" font-family="sans-serif">3</text>
  <rect x="397" y="207" width="306" height="52" fill="none" stroke="#1a56c4" stroke-width="1.5" rx="3" stroke-dasharray="3,2"/>
  <rect x="397" y="207" width="15" height="15" fill="#1a56c4" rx="3"/>
  <text x="405" y="219" text-anchor="middle" font-size="9" fill="white" font-weight="700" font-family="sans-serif">4</text>
  <rect x="390" y="266" width="320" height="26" fill="#e8f4e8" rx="3" stroke="#60a860" stroke-width="1"/>
  <text x="550" y="283" text-anchor="middle" font-size="9" font-weight="700" fill="#2a7a2a" font-family="sans-serif">달력 구조 ✓  ·  차트 수치 ✓  ·  번호로 클릭 ✓</text>
</svg>

달력이 몇 월인지, 바 차트의 높이가 뭘 의미하는지, 어느 버튼이 실제로 클릭 가능한지 — 이런 정보들은 HTML 텍스트에서 읽어내기 매우 어렵거나 불가능하다. 렌더링된 화면을 보면 1초면 알 수 있는 것들이다.

### 문제 2 — 평가 자체가 현실과 달랐다

대표적인 선행 연구인 **Mind2Web**은 실제 웹사이트 HTML을 긁어 정적 스냅샷으로 저장한 뒤, 그 위에서 평가를 진행했다. 실시간 웹이 아닌 냉동된 복사본이다. 또한 평가 방식이 "정해진 정답 액션 시퀀스와 step 단위로 비교"하는 방식이었는데, 이는 다른 경로로 같은 결과에 도달해도 실패로 처리되는 문제를 만든다.

<svg viewBox="0 0 720 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;margin:1.8rem 0;border-radius:6px;border:1px solid #d8d8d2;">
  <rect width="720" height="320" fill="#f9f8f4"/>
  <rect x="0" y="0" width="340" height="320" fill="#efede6"/>
  <rect x="0" y="0" width="340" height="36" fill="#d8d5cc"/>
  <text x="170" y="23" text-anchor="middle" font-size="11" font-weight="700" fill="#6b6b6b" font-family="sans-serif" letter-spacing="1">Mind2Web 평가 방식</text>
  <rect x="100" y="48" width="140" height="72" fill="white" rx="4" stroke="#bbb" stroke-width="1.5"/>
  <rect x="100" y="48" width="140" height="18" fill="#ccc" rx="4"/>
  <text x="170" y="61" text-anchor="middle" font-size="8" fill="#666" font-family="sans-serif">webpage_snapshot.html</text>
  <rect x="110" y="72" width="120" height="7" fill="#ddd" rx="2"/>
  <rect x="110" y="83" width="80" height="7" fill="#ddd" rx="2"/>
  <rect x="110" y="94" width="100" height="7" fill="#ddd" rx="2"/>
  <text x="222" y="110" font-size="18" fill="#aaa" font-family="sans-serif">❄</text>
  <text x="170" y="134" text-anchor="middle" font-size="9" fill="#999" font-family="sans-serif">정적 스냅샷 (실시간 웹 ❌)</text>
  <text x="16" y="160" font-size="10" font-weight="700" fill="#444" font-family="sans-serif">정해진 정답 경로:</text>
  <rect x="16"  y="168" width="52" height="22" fill="#d4edda" rx="4" stroke="#28a745" stroke-width="1"/>
  <text x="42"  y="183" text-anchor="middle" font-size="8" fill="#155724" font-family="sans-serif">Step 1</text>
  <text x="74"  y="182" font-size="11" fill="#28a745" font-family="sans-serif">→</text>
  <rect x="90"  y="168" width="52" height="22" fill="#d4edda" rx="4" stroke="#28a745" stroke-width="1"/>
  <text x="116" y="183" text-anchor="middle" font-size="8" fill="#155724" font-family="sans-serif">Step 2</text>
  <text x="148" y="182" font-size="11" fill="#28a745" font-family="sans-serif">→</text>
  <rect x="164" y="168" width="52" height="22" fill="#d4edda" rx="4" stroke="#28a745" stroke-width="1"/>
  <text x="190" y="183" text-anchor="middle" font-size="8" fill="#155724" font-family="sans-serif">Step 3</text>
  <text x="222" y="182" font-size="11" fill="#28a745" font-family="sans-serif">→</text>
  <text x="252" y="183" font-size="13" fill="#28a745" font-family="sans-serif">✓</text>
  <text x="16" y="210" font-size="10" font-weight="700" fill="#444" font-family="sans-serif">다른 경로 (결과는 동일):</text>
  <rect x="16"  y="218" width="52" height="22" fill="#d4edda" rx="4" stroke="#28a745" stroke-width="1"/>
  <text x="42"  y="233" text-anchor="middle" font-size="8" fill="#155724" font-family="sans-serif">Step 1</text>
  <text x="74"  y="232" font-size="11" fill="#999" font-family="sans-serif">→</text>
  <rect x="90"  y="218" width="52" height="22" fill="#f8d7da" rx="4" stroke="#dc3545" stroke-width="1"/>
  <text x="116" y="233" text-anchor="middle" font-size="8" fill="#721c24" font-family="sans-serif">Step 2'</text>
  <text x="148" y="232" font-size="11" fill="#999" font-family="sans-serif">→</text>
  <rect x="164" y="218" width="52" height="22" fill="#d4edda" rx="4" stroke="#28a745" stroke-width="1"/>
  <text x="190" y="233" text-anchor="middle" font-size="8" fill="#155724" font-family="sans-serif">Step 3</text>
  <text x="222" y="232" font-size="11" fill="#999" font-family="sans-serif">→</text>
  <text x="252" y="233" font-size="13" fill="#dc3545" font-family="sans-serif">✗</text>
  <text x="170" y="260" text-anchor="middle" font-size="8.5" fill="#dc3545" font-family="sans-serif">결과는 같아도 경로가 다르면 실패 처리</text>
  <rect x="16" y="274" width="308" height="38" fill="#fde8e8" rx="4" stroke="#e08080" stroke-width="1"/>
  <text x="170" y="289" text-anchor="middle" font-size="9" font-weight="700" fill="#c03030" font-family="sans-serif">실시간 웹이 아닌 스냅샷 ❌</text>
  <text x="170" y="304" text-anchor="middle" font-size="9" fill="#c03030" font-family="sans-serif">정해진 경로만 정답으로 인정 ❌</text>
  <text x="360" y="168" text-anchor="middle" font-size="28" fill="#1a56c4" font-family="sans-serif">→</text>
  <rect x="380" y="0" width="340" height="320" fill="white"/>
  <rect x="380" y="0" width="340" height="36" fill="#1a56c4"/>
  <text x="550" y="23" text-anchor="middle" font-size="11" font-weight="700" fill="white" font-family="sans-serif" letter-spacing="1">WebVoyager 평가</text>
  <rect x="460" y="48" width="140" height="72" fill="white" rx="4" stroke="#1a56c4" stroke-width="1.5"/>
  <rect x="460" y="48" width="140" height="18" fill="#1a56c4" rx="4"/>
  <text x="530" y="61" text-anchor="middle" font-size="8" fill="white" font-family="sans-serif">실제 웹사이트 (실시간)</text>
  <rect x="470" y="72" width="120" height="7" fill="#1a56c4" rx="2" opacity="0.3"/>
  <rect x="470" y="83" width="80" height="7" fill="#1a56c4" rx="2" opacity="0.5"/>
  <rect x="470" y="94" width="100" height="7" fill="#1a56c4" rx="2" opacity="0.4"/>
  <circle cx="683" cy="57" r="5" fill="#28c840"/>
  <text x="692" y="61" font-size="7" fill="#28c840" font-weight="700" font-family="sans-serif">LIVE</text>
  <text x="530" y="134" text-anchor="middle" font-size="9" fill="#1a56c4" font-family="sans-serif">실시간 웹사이트 ✓</text>
  <text x="390" y="160" font-size="10" font-weight="700" fill="#444" font-family="sans-serif">어떤 경로로도 성공 가능:</text>
  <text x="393" y="180" font-size="8.5" fill="#888" font-family="sans-serif">경로 A</text>
  <rect x="430" y="168" width="38" height="18" fill="#efede6" rx="3" stroke="#999" stroke-width="1"/>
  <text x="449" y="181" text-anchor="middle" font-size="7.5" fill="#333" font-family="sans-serif">검색</text>
  <text x="473" y="180" font-size="9" fill="#999" font-family="sans-serif">→</text>
  <rect x="484" y="168" width="38" height="18" fill="#efede6" rx="3" stroke="#999" stroke-width="1"/>
  <text x="503" y="181" text-anchor="middle" font-size="7.5" fill="#333" font-family="sans-serif">필터</text>
  <text x="527" y="180" font-size="9" fill="#999" font-family="sans-serif">→</text>
  <text x="546" y="181" font-size="12" fill="#28a745" font-family="sans-serif">✓</text>
  <text x="393" y="208" font-size="8.5" fill="#888" font-family="sans-serif">경로 B</text>
  <rect x="430" y="196" width="38" height="18" fill="#efede6" rx="3" stroke="#999" stroke-width="1"/>
  <text x="449" y="209" text-anchor="middle" font-size="7.5" fill="#333" font-family="sans-serif">카테고리</text>
  <text x="473" y="208" font-size="9" fill="#999" font-family="sans-serif">→</text>
  <rect x="484" y="196" width="38" height="18" fill="#efede6" rx="3" stroke="#999" stroke-width="1"/>
  <text x="503" y="209" text-anchor="middle" font-size="7.5" fill="#333" font-family="sans-serif">정렬</text>
  <text x="527" y="208" font-size="9" fill="#999" font-family="sans-serif">→</text>
  <text x="546" y="209" font-size="12" fill="#28a745" font-family="sans-serif">✓</text>
  <text x="393" y="236" font-size="8.5" fill="#888" font-family="sans-serif">경로 C</text>
  <rect x="430" y="224" width="80" height="18" fill="#efede6" rx="3" stroke="#999" stroke-width="1"/>
  <text x="470" y="237" text-anchor="middle" font-size="7.5" fill="#333" font-family="sans-serif">검색 → 바로 선택</text>
  <text x="515" y="236" font-size="9" fill="#999" font-family="sans-serif">→</text>
  <text x="534" y="237" font-size="12" fill="#28a745" font-family="sans-serif">✓</text>
  <rect x="578" y="162" width="122" height="82" fill="#e8f0fe" rx="6" stroke="#1a56c4" stroke-width="1.5"/>
  <text x="639" y="180" text-anchor="middle" font-size="9" font-weight="700" fill="#1a56c4" font-family="sans-serif">GPT-4V 판정</text>
  <text x="639" y="196" text-anchor="middle" font-size="8" fill="#444" font-family="sans-serif">"최종 결과가</text>
  <text x="639" y="208" text-anchor="middle" font-size="8" fill="#444" font-family="sans-serif">맞았는가?"</text>
  <line x1="546" y1="177" x2="578" y2="185" stroke="#1a56c4" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="546" y1="205" x2="578" y2="204" stroke="#1a56c4" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="534" y1="233" x2="578" y2="224" stroke="#1a56c4" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="639" y="228" text-anchor="middle" font-size="18" fill="#28a745" font-family="sans-serif">✓</text>
  <text x="639" y="242" text-anchor="middle" font-size="7.5" fill="#888" font-family="sans-serif">인간 일치율 85.3%</text>
  <rect x="390" y="276" width="320" height="38" fill="#e8f4e8" rx="4" stroke="#60a860" stroke-width="1"/>
  <text x="550" y="291" text-anchor="middle" font-size="9" font-weight="700" fill="#2a7a2a" font-family="sans-serif">실제 웹에서 테스트 ✓</text>
  <text x="550" y="306" text-anchor="middle" font-size="9" fill="#2a7a2a" font-family="sans-serif">경로 무관, 최종 결과로만 판단 ✓</text>
</svg>

<div class="pullquote">
  <strong>핵심 문제:</strong> 기존 에이전트는 HTML 텍스트로 시각 정보를 잃었고, 기존 평가는 정적 스냅샷과 고정 경로로 현실을 반영하지 못했다.
</div>

## WebVoyager의 접근: GPT-4V + 실제 웹

WebVoyager의 해답은 단순하다. GPT-4V (vision 기능이 있는 GPT-4 Turbo)를 사용해서 **스크린샷을 직접 입력**으로 받고, **실제 살아있는 웹사이트**에서 Selenium으로 행동한다.

### 스크린샷에 번호를 붙인다

스크린샷을 그냥 통째로 넣으면 모델이 "어디를 클릭해야 하지?"를 정확히 지시하기 어렵다. WebVoyager는 이 문제를 **GPT-4V-ACT**라는 JavaScript 기반 도구로 해결한다. 페이지에서 인터랙티브 요소(버튼, 링크, 입력창 등)를 자동으로 감지해서 **번호가 붙은 파란 박스**를 오버레이한다.

```
스크린샷 위에:
[1] 검색창
[2] 로그인 버튼
[3] 메뉴
[4] 상품 카드
...
```

모델은 이 번호를 보고 `Click [3]`, `Type [1]; Python tutorial` 같은 명확한 명령어를 출력한다. Object detection 없이 JS 룰 기반으로 추출하기 때문에 가볍고 빠르다.

## 7가지 행동으로 웹을 탐색한다

WebVoyager의 액션 스페이스는 단순하다. 7가지 행동만으로 대부분의 웹 태스크를 커버한다.

<div class="callout">
  <strong>액션 스페이스:</strong><br><br>
  <code>Click [N]</code> — N번 요소 클릭<br>
  <code>Type [N]; [텍스트]</code> — N번 입력창에 타이핑 (기존 내용 지우고 입력)<br>
  <code>Scroll [N/WINDOW]; [up/down]</code> — 스크롤<br>
  <code>Wait</code> — 페이지 로딩 대기<br>
  <code>GoBack</code> — 이전 페이지<br>
  <code>Google</code> — 구글로 새로 시작<br>
  <code>ANSWER; [내용]</code> — 최종 답변 제출, 태스크 종료
</div>

## 생각하고 행동하는 루프

WebVoyager는 **ReAct 스타일** 프롬프팅을 사용한다. 매 step마다 먼저 현재 상황을 생각(Thought)하고, 그 다음 액션 코드를 출력한다. 최대 15 step까지 반복하고, 목표를 달성하면 `ANSWER`로 종료한다.

<svg viewBox="0 0 720 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;margin:1.8rem 0;border-radius:6px;border:1px solid #d8d8d2;">
  <rect width="720" height="220" fill="#f9f8f4"/>

  <!-- 태스크 입력 -->
  <rect x="20" y="88" width="100" height="44" fill="#efede6" rx="6" stroke="#999" stroke-width="1.2"/>
  <text x="70" y="106" text-anchor="middle" font-size="9" font-weight="700" fill="#444" font-family="sans-serif">태스크 입력</text>
  <text x="70" y="120" text-anchor="middle" font-size="8" fill="#666" font-family="sans-serif">"최저가 항공권</text>
  <text x="70" y="131" text-anchor="middle" font-size="8" fill="#666" font-family="sans-serif">찾아줘"</text>

  <!-- 화살표 -->
  <line x1="122" y1="110" x2="140" y2="110" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- 스크린샷 관측 -->
  <rect x="142" y="72" width="110" height="76" fill="white" rx="6" stroke="#1a56c4" stroke-width="1.5"/>
  <rect x="142" y="72" width="110" height="20" fill="#1a56c4" rx="6"/>
  <text x="197" y="86" text-anchor="middle" font-size="9" font-weight="700" fill="white" font-family="sans-serif">관측 (Observation)</text>
  <rect x="152" y="98" width="90" height="36" fill="#efede6" rx="3"/>
  <text x="197" y="111" text-anchor="middle" font-size="7.5" fill="#666" font-family="sans-serif">스크린샷</text>
  <text x="197" y="123" text-anchor="middle" font-size="7.5" fill="#1a56c4" font-family="sans-serif">+ 번호 레이블</text>
  <text x="197" y="158" text-anchor="middle" font-size="7.5" fill="#999" font-family="sans-serif">1024×768px</text>

  <!-- 화살표 -->
  <line x1="254" y1="110" x2="272" y2="110" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- GPT-4V -->
  <rect x="274" y="72" width="110" height="76" fill="white" rx="6" stroke="#1a56c4" stroke-width="1.5"/>
  <rect x="274" y="72" width="110" height="20" fill="#1a56c4" rx="6"/>
  <text x="329" y="86" text-anchor="middle" font-size="9" font-weight="700" fill="white" font-family="sans-serif">GPT-4V 추론</text>
  <text x="329" y="106" text-anchor="middle" font-size="8" fill="#333" font-family="sans-serif">Thought:</text>
  <text x="329" y="118" text-anchor="middle" font-size="7.5" fill="#666" font-family="sans-serif">"검색창이 [1]번에</text>
  <text x="329" y="129" text-anchor="middle" font-size="7.5" fill="#666" font-family="sans-serif">있으니 입력하자"</text>
  <text x="329" y="143" text-anchor="middle" font-size="8" fill="#333" font-family="sans-serif">Action:</text>
  <text x="329" y="155" text-anchor="middle" font-size="7.5" fill="#1a56c4" font-family="sans-serif">Type [1]; Seoul</text>

  <!-- 화살표 -->
  <line x1="386" y1="110" x2="404" y2="110" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Selenium 실행 -->
  <rect x="406" y="72" width="110" height="76" fill="white" rx="6" stroke="#1a56c4" stroke-width="1.5"/>
  <rect x="406" y="72" width="110" height="20" fill="#1a56c4" rx="6"/>
  <text x="461" y="86" text-anchor="middle" font-size="9" font-weight="700" fill="white" font-family="sans-serif">Selenium 실행</text>
  <text x="461" y="106" text-anchor="middle" font-size="8" fill="#444" font-family="sans-serif">브라우저에서</text>
  <text x="461" y="118" text-anchor="middle" font-size="8" fill="#444" font-family="sans-serif">실제 실행</text>
  <text x="461" y="138" text-anchor="middle" font-size="18" fill="#1a56c4" font-family="sans-serif">🌐</text>

  <!-- 화살표 오른쪽 -->
  <line x1="518" y1="110" x2="536" y2="110" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- ANSWER 또는 반복 -->
  <rect x="538" y="72" width="110" height="76" fill="white" rx="6" stroke="#999" stroke-width="1.2" stroke-dasharray="4,2"/>
  <text x="593" y="95" text-anchor="middle" font-size="8" font-weight="700" fill="#444" font-family="sans-serif">다음 step</text>
  <text x="593" y="109" text-anchor="middle" font-size="8" fill="#888" font-family="sans-serif">새 스크린샷</text>
  <text x="593" y="121" text-anchor="middle" font-size="8" fill="#888" font-family="sans-serif">→ 반복</text>
  <text x="593" y="139" text-anchor="middle" font-size="7.5" fill="#999" font-family="sans-serif">(최대 15회)</text>

  <!-- 반환 화살표 (아래로 돌아가는) -->
  <path d="M 593 150 L 593 190 L 197 190 L 197 150" stroke="#1a56c4" stroke-width="1.5" fill="none" stroke-dasharray="4,3" marker-end="url(#arr-blue)"/>

  <!-- 종료 -->
  <rect x="538" y="20" width="110" height="30" fill="#e8f4e8" rx="6" stroke="#28a745" stroke-width="1.2"/>
  <text x="593" y="39" text-anchor="middle" font-size="9" font-weight="700" fill="#155724" font-family="sans-serif">ANSWER 출력 → 종료</text>
  <line x1="593" y1="72" x2="593" y2="52" stroke="#28a745" stroke-width="1.5" marker-end="url(#arr-green)"/>
  <text x="600" y="66" font-size="7.5" fill="#28a745" font-family="sans-serif">완료</text>

  <!-- 화살표 마커 정의 -->
  <defs>
    <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#999"/>
    </marker>
    <marker id="arr-blue" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#1a56c4"/>
    </marker>
    <marker id="arr-green" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#28a745"/>
    </marker>
  </defs>
</svg>

## 히스토리 관리: Context Clipping

15 step을 돌면 스크린샷 15장이 쌓인다. GPT-4V에 이걸 다 넣으면 context window가 금방 터진다. WebVoyager의 해결책은 **Context Clipping**이다.

<svg viewBox="0 0 720 190" xmlns="http://www.w3.org/2000/svg" style="width:100%;margin:1.8rem 0;border-radius:6px;border:1px solid #d8d8d2;">
  <rect width="720" height="190" fill="#f9f8f4"/>

  <!-- 제목 -->
  <text x="360" y="24" text-anchor="middle" font-size="12" font-weight="700" fill="#0f0f0f" font-family="sans-serif">Context Clipping</text>

  <!-- 스크린샷 행 레이블 -->
  <text x="16" y="60" font-size="9" font-weight="700" fill="#444" font-family="sans-serif">스크린샷</text>
  <text x="16" y="73" font-size="8" fill="#888" font-family="sans-serif">(관측값)</text>

  <!-- 스크린샷 블록 1~12 (삭제됨, 흐리게) -->
  <rect x="90"  y="45" width="32" height="36" fill="#e0e0e0" rx="3" stroke="#ccc" stroke-width="1"/>
  <text x="106" y="67" text-anchor="middle" font-size="7" fill="#bbb" font-family="sans-serif">1</text>
  <rect x="128" y="45" width="32" height="36" fill="#e0e0e0" rx="3" stroke="#ccc" stroke-width="1"/>
  <text x="144" y="67" text-anchor="middle" font-size="7" fill="#bbb" font-family="sans-serif">2</text>
  <rect x="166" y="45" width="32" height="36" fill="#e0e0e0" rx="3" stroke="#ccc" stroke-width="1"/>
  <text x="182" y="67" text-anchor="middle" font-size="7" fill="#bbb" font-family="sans-serif">3</text>
  <!-- ... 생략 표시 -->
  <text x="220" y="67" text-anchor="middle" font-size="10" fill="#ccc" font-family="sans-serif">···</text>
  <rect x="244" y="45" width="32" height="36" fill="#e0e0e0" rx="3" stroke="#ccc" stroke-width="1"/>
  <text x="260" y="67" text-anchor="middle" font-size="7" fill="#bbb" font-family="sans-serif">11</text>
  <rect x="282" y="45" width="32" height="36" fill="#e0e0e0" rx="3" stroke="#ccc" stroke-width="1"/>
  <text x="298" y="67" text-anchor="middle" font-size="7" fill="#bbb" font-family="sans-serif">12</text>

  <!-- 삭제 표시 -->
  <rect x="88" y="43" width="228" height="40" fill="none" stroke="#dc3545" stroke-width="1" rx="4" stroke-dasharray="4,2"/>
  <text x="202" y="96" text-anchor="middle" font-size="8" fill="#dc3545" font-family="sans-serif">삭제 (메모리 절약)</text>

  <!-- 스크린샷 13~15 (유지, 파란색) -->
  <rect x="330" y="45" width="32" height="36" fill="#dbeafe" rx="3" stroke="#1a56c4" stroke-width="1.5"/>
  <text x="346" y="61" text-anchor="middle" font-size="7" fill="#1a56c4" font-weight="700" font-family="sans-serif">13</text>
  <text x="346" y="72" text-anchor="middle" font-size="6" fill="#1a56c4" font-family="sans-serif">유지</text>
  <rect x="368" y="45" width="32" height="36" fill="#dbeafe" rx="3" stroke="#1a56c4" stroke-width="1.5"/>
  <text x="384" y="61" text-anchor="middle" font-size="7" fill="#1a56c4" font-weight="700" font-family="sans-serif">14</text>
  <text x="384" y="72" text-anchor="middle" font-size="6" fill="#1a56c4" font-family="sans-serif">유지</text>
  <rect x="406" y="45" width="32" height="36" fill="#dbeafe" rx="3" stroke="#1a56c4" stroke-width="1.5"/>
  <text x="422" y="61" text-anchor="middle" font-size="7" fill="#1a56c4" font-weight="700" font-family="sans-serif">15</text>
  <text x="422" y="72" text-anchor="middle" font-size="6" fill="#1a56c4" font-family="sans-serif">유지</text>
  <text x="376" y="96" text-anchor="middle" font-size="8" fill="#1a56c4" font-family="sans-serif">최근 3개만 유지</text>

  <!-- 생각+액션 행 레이블 -->
  <text x="16" y="130" font-size="9" font-weight="700" fill="#444" font-family="sans-serif">Thought</text>
  <text x="16" y="143" font-size="9" font-weight="700" fill="#444" font-family="sans-serif">+ Action</text>

  <!-- 생각+액션 블록 1~15 (모두 유지) -->
  <rect x="90"  y="115" width="32" height="28" fill="#d1fae5" rx="3" stroke="#28a745" stroke-width="1"/>
  <text x="106" y="133" text-anchor="middle" font-size="7" fill="#155724" font-family="sans-serif">1</text>
  <rect x="128" y="115" width="32" height="28" fill="#d1fae5" rx="3" stroke="#28a745" stroke-width="1"/>
  <text x="144" y="133" text-anchor="middle" font-size="7" fill="#155724" font-family="sans-serif">2</text>
  <rect x="166" y="115" width="32" height="28" fill="#d1fae5" rx="3" stroke="#28a745" stroke-width="1"/>
  <text x="182" y="133" text-anchor="middle" font-size="7" fill="#155724" font-family="sans-serif">3</text>
  <text x="220" y="134" text-anchor="middle" font-size="10" fill="#28a745" font-family="sans-serif">···</text>
  <rect x="244" y="115" width="32" height="28" fill="#d1fae5" rx="3" stroke="#28a745" stroke-width="1"/>
  <text x="260" y="133" text-anchor="middle" font-size="7" fill="#155724" font-family="sans-serif">11</text>
  <rect x="282" y="115" width="32" height="28" fill="#d1fae5" rx="3" stroke="#28a745" stroke-width="1"/>
  <text x="298" y="133" text-anchor="middle" font-size="7" fill="#155724" font-family="sans-serif">12</text>
  <rect x="330" y="115" width="32" height="28" fill="#d1fae5" rx="3" stroke="#28a745" stroke-width="1"/>
  <text x="346" y="133" text-anchor="middle" font-size="7" fill="#155724" font-family="sans-serif">13</text>
  <rect x="368" y="115" width="32" height="28" fill="#d1fae5" rx="3" stroke="#28a745" stroke-width="1"/>
  <text x="384" y="133" text-anchor="middle" font-size="7" fill="#155724" font-family="sans-serif">14</text>
  <rect x="406" y="115" width="32" height="28" fill="#d1fae5" rx="3" stroke="#28a745" stroke-width="1"/>
  <text x="422" y="133" text-anchor="middle" font-size="7" fill="#155724" font-family="sans-serif">15</text>
  <text x="260" y="157" text-anchor="middle" font-size="8" fill="#28a745" font-family="sans-serif">전체 15개 모두 유지</text>

  <!-- 요약 텍스트 -->
  <text x="580" y="72" text-anchor="middle" font-size="9" fill="#444" font-family="sans-serif">"뭘 했는지"는</text>
  <text x="580" y="85" text-anchor="middle" font-size="9" fill="#444" font-family="sans-serif">전부 기억</text>
  <text x="580" y="105" text-anchor="middle" font-size="9" fill="#444" font-family="sans-serif">"화면이 어떻게</text>
  <text x="580" y="118" text-anchor="middle" font-size="9" fill="#444" font-family="sans-serif">생겼는지"는</text>
  <text x="580" y="131" text-anchor="middle" font-size="9" fill="#444" font-family="sans-serif">최근 3개만</text>
</svg>

요약하면: **행동 히스토리는 전부 유지하되, 스크린샷은 최근 3장만 남긴다.** 에이전트가 맥락을 잃지 않으면서도 context를 효율적으로 관리하는 합리적인 절충이다.

## 벤치마크와 자동 평가

WebVoyager는 **643개 태스크 × 15개 실제 웹사이트**로 구성된 자체 벤치마크를 만들었다. Amazon, ArXiv, GitHub, Booking.com, ESPN, Google Flights 등 실제 사람들이 쓰는 사이트들이다.

평가는 또 다른 혁신이다. 사람이 모든 태스크를 수동으로 채점하는 건 비현실적이라, **GPT-4V를 심사위원으로 활용**했다. 태스크 지시문, 에이전트 응답, 마지막 몇 장의 스크린샷을 주고 "성공인가 실패인가"를 판정하게 한다. 이 자동 평가와 사람 판단의 일치율이 **85.3%** (Fleiss κ=0.70)로, 실용적인 수준이다.

## 결과

| 방법 | 성공률 |
|------|--------|
| GPT-4 All Tools (텍스트만) | 30.8% |
| WebVoyager 텍스트만 | 40.1% |
| **WebVoyager (멀티모달)** | **59.1%** |

스크린샷을 추가했더니 텍스트만 쓴 WebVoyager 대비 거의 20%p가 올라갔다. 시각 정보의 중요성을 수치로 보여준 결과다.

사이트별로 편차가 컸다. Google Search(76.7%), Coursera(73.8%)에서는 잘 됐지만, Booking.com(43.2%), HuggingFace(44.2%)에서는 상대적으로 낮았다. 동적이고 복잡한 UI일수록 어렵다.

## 실패 유형 분석

실패한 케이스 300개를 분석한 결과:

<div class="callout">
  <strong>실패 원인 분포</strong><br><br>
  44.4% — <strong>Navigation Stuck</strong>: 같은 행동을 반복하거나, 스크롤이 엉뚱한 방향으로 가거나, 잘못된 검색어로 헤매는 경우<br><br>
  24.8% — <strong>Visual Grounding 오류</strong>: 번호 레이블을 잘못 읽거나, 인접한 요소를 혼동하는 경우<br><br>
  21.8% — <strong>Hallucination</strong>: 실제로 없는 정보를 있다고 답하거나, 잘못된 텍스트 입력창을 선택하는 경우<br><br>
  9.0% — <strong>Prompt 파싱 실패</strong>: 출력 형식이 맞지 않아 액션을 실행 못 하거나 너무 일찍 종료하는 경우
</div>

가장 많은 실패가 "막혀서 반복"인 게 흥미롭다. 사람도 웹에서 막히면 뒤로 가거나 검색어를 바꾸는 판단을 하는데, 에이전트는 이 탈출 전략이 아직 부족하다.

## 의의와 남은 과제

WebVoyager가 보여준 것은 두 가지다. 첫째, **HTML 텍스트 대신 스크린샷을 직접 보는 것이 웹 에이전트에게 명확히 유리하다.** 달력, 차트, 복잡한 레이아웃을 텍스트로 파싱하려는 시도보다 눈으로 보는 게 더 자연스럽고 효과적이다. 둘째, **실제 웹에서의 결과 기반 평가가 가능하다.** GPT-4V를 심사위원으로 쓰는 방식은 이후 웹 에이전트 연구의 평가 표준에 영향을 줬다.

다만 여전히 한계가 있다. 드래그 같은 복잡한 인터랙션은 지원 안 되고, 로그인이나 CAPTCHA는 처리하지 못한다. 오픈소스 모델로는 context window가 부족해서(약 7000 토큰 필요) GPT-4V 없이는 재현이 어렵다. 그리고 Navigation Stuck이 가장 큰 실패 원인인 만큼, 막혔을 때 전략을 바꾸는 메타인지 능력이 다음 과제로 남는다.

<div class="footnote">
  논문: <a href="https://arxiv.org/abs/2401.13919">WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models</a> (ACL 2024) · Hongliang He et al.
</div>
