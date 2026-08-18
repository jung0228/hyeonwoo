# Web Navigation Agent

## 핵심 아이디어
대형 언어 모델(LLM)과 멀티모달 모델(VLM)이 웹 브라우저 환경에서 DOM 트리(Accessibility Tree) 및 스크린샷 렌더링을 관찰(Observation)하고, 클릭·타이핑·스크롤 등의 조작(Action)을 계획하여 사용자의 복잡한 웹 과업을 자율적으로 수행하는 자율 에이전트 시스템입니다.

---

## 핵심 메커니즘

### 1. ReAct 루프 및 Observation-Action 사이클
$$\tau = (o_0, a_0, r_0, o_1, a_1, \dots, o_T, a_T)$$
- $o_t \in \mathcal{O}$: 간소화된 HTML/DOM 구조 및 시각적 Bounding Box 정보 (Set-of-Mark)
- $a_t \in \mathcal{A}$: 브라우저 인터랙션 명령 (`click(element_id)`, `type(element_id, text)`, `scroll(direction)`, `navigate(url)`)

### 2. Grounding & Element Selection
에이전트는 텍스트 기반 접근성 트리(AXTree) 또는 화면의 시각적 요소 태그(SoM, Set-of-Marks)를 통해 특정 UI 요소의 고유 ID를 타겟팅합니다:
$$P(a_t | o_t, \text{Task}, \text{History}) = \text{LLM/VLM}(o_t, \text{Prompt})$$

---

## 직관적 설명
사람이 비행기 표를 예매할 때 브라우저 화면을 보고 필요한 날짜를 클릭하고 정보를 입력하듯, AI가 **'화면을 보고 $\rightarrow$ 생각하고 $\rightarrow$ 마우스와 키보드를 대신 조작'**하여 다단계 웹 태스크를 스스로 완료하는 스마트 웹 비서입니다.

---

## 연결 개념
- [[llm]] : 의도 파악 및 추론, 단계별 계획(Planning) 생성
- [[agent_memory]] : 과거 탐색 경험, 성공적인 워크플로우 궤적 저장 및 재사용
- [[multimodal]] : 텍스트 DOM뿐만 아니라 시각적 렌더링 화면을 동시에 해석

---

## 참고
- WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models (ACL 2024)
- Mind2Web: Towards a Generalist Agent for the Web (NeurIPS 2023)
- Browser-Use: Open-Source Web Automation with LLMs
