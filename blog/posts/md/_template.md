---
title: 포스트 제목
dek: 한 줄짜리 부제목 설명문.
desc: 인덱스 카드에 표시될 짧은 설명 (선택사항 — 없으면 dek 그대로 사용)
tags: [Agent, LLM]
date: Mar 2026
readtime: 8 min read
slug: my-new-post
katex: false
---

## 서론

마크다운으로 본문을 씁니다. **볼드**, *이탤릭*, `인라인 코드` 모두 됩니다.

## 섹션 제목

단락은 빈 줄로 구분합니다.

- 불릿 리스트
- **항목**: 설명

### 소제목

일반 문단입니다.

## 커스텀 컴포넌트 (raw HTML)

pullquote, callout, SVG 등은 HTML block으로 그냥 씁니다:

<div class="pullquote">
  <strong>핵심:</strong> 중요한 문장을 한 줄로 요약합니다.
</div>

<div class="callout">
  <strong>Note:</strong> 추가 설명이나 팁.
</div>

<div class="ornament">· · ·</div>

## 수식 (katex: true 일 때만)

인라인: $E = mc^2$

블록:

$$
\mathcal{L} = -\sum_{i} y_i \log \hat{y}_i
$$

<div class="footnote">
  참고: <a href="#">논문 제목</a>
</div>
