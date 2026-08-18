# Agent Memory & Workflow

## 핵심 아이디어
자율 에이전트가 단기 세션의 컨텍스트 윈도우 한계를 극복하고, 과거의 성공 및 실패 궤적(Trajectories)으로부터 핵심 워크플로우, 서브루틴, 규칙을 스스로 일반화하여 장기 저장소에 보관하고 유사한 상황에서 검색·재사용하는 기억 아키텍처입니다.

---

## 메모리 계층 구조

### 1. 단기 메모리 (Working Memory)
- 현재 세션의 대화 히스토리 및 최근 행동 로그:
$$\mathcal{M}_{\text{short}} = [ (s_0, a_0, o_0), (s_1, a_1, o_1), \dots ]$$

### 2. 장기 절차적 메모리 (Procedural / Workflow Memory)
- 성공한 에피소드에서 추상화된 고수준 실행 플로우(AWM, Synapse):
$$\mathcal{W}^* = \text{Induce}(\tau_{\text{success}})$$
$$\text{Replay: } a_{t} \sim \pi(a | s_t, \mathcal{W}^*)$$

### 3. 의미론적/경험 검색 (Experience Retrieval)
- 현재 태스크 $q$와 가장 코사인 유사도가 높은 과거 성공 경험 $\mathcal{E}$ 추출:
$$\mathcal{E}_{\text{top-k}} = \arg\max_{e \in \mathcal{D}} \cos(\mathbf{e}_q, \mathbf{e}_e)$$

---

## 직관적 설명
인턴이 처음에는 선배에게 일일이 물어보며 일을 배우지만, 한 번 성공한 업무 매뉴얼(체크리스트)을 개인 업무 노트에 정리해 두었다가 다음번에 비슷한 일이 생기면 노트를 꺼내보고 숙련자처럼 막힘없이 처리하는 것과 같습니다.

---

## 연결 개념
- [[web_agent]] : 복잡한 멀티스텝 웹 조작에서 반복 행동을 줄이고 성공률을 극대화
- [[llm]] : 기억 생성(Reflection) 및 프롬프트 인젝션
- [[rlhf]] / [[rl_basic]] : 경험 기반 보상 학습 및 정책 정제

---

## 참고
- Agent Workflow Memory (AWM): Inducing Reusable Workflows from Demonstrations
- Synapse: Trajectory-as-Exemplar Prompting for Web Navigation
- ReasoningBank: Scaling Memory-Augmented LLM Reasoning
