# 🧠 정현우의 AI/ML 지식 지도

**Live Site**: [jung0228.github.io/hyeonwoo](https://jung0228.github.io/hyeonwoo)

AI/ML 개념을 인터랙티브하게 탐색하고, 학습 현황을 추적하는 개인 지식 관리 사이트.

## Features

- 🌐 **Knowledge Graph** — D3.js 기반 인터랙티브 개념 맵, 개념 간 관계 시각화
- 🗓️ **Activity Heatmap** — GitHub 잔디 스타일 학습 기록
- 📊 **Progress Tracker** — 주제별 자신감 레벨 & 약점 발견

## 로컬 실행

```bash
# Python 3
python3 -m http.server 8080
# 브라우저에서 http://localhost:8080 열기
```

## 노트 추가하기

1. `data/knowledge.json`에 노드/엣지 추가
2. `data/notes/<node_id>.md` 파일 생성

### 노트 템플릿

```markdown
# 개념 이름

**카테고리**: Generative / Architecture / Language Model / Multimodal / RL / Training  
**자신감**: ⭐⭐☆☆ (기초)  
**마지막 복습**: YYYY-MM-DD

---

## 한 문장 요약

...

## 핵심 내용

...

## 체크리스트

- [ ] 항목 1
- [x] 항목 2 (완료)
```

## GitHub Pages 배포

레포지토리 Settings → Pages → Source: `main` branch, `/ (root)` 선택
