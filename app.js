/* ============================================================
   정현우의 지식 지도 — App Logic (v2 with Cluster Layout)
   ============================================================ */

/* ── State ── */
let knowledgeData = null;
let simulation = null;
let svg = null, g = null, zoom = null;
let currentNodeId = null;
let activeCategory = null;

/* ── Confidence Labels ── */
const CONFIDENCE_LABELS = ['모름', '기초', '중급', '심화', '전문가'];
const CONFIDENCE_STARS  = ['☆☆☆☆', '★☆☆☆', '★★☆☆', '★★★☆', '★★★★'];

/* ── Relation Labels ── */
const RELATION_LABELS = {
  conceptual_root: '개념 근원', derived: '파생', comparison: '비교',
  contains: '구성 요소', applied_to: '응용', scaled_up: '스케일업',
  finetuned_with: '파인튜닝', extended_to: '확장', input_to: '입력',
  enables_generation: '생성 가능', instantiated_as: '구현체',
  key_component: '핵심 요소', is_a: '종류', enables: '가능하게 함',
  used_in: '사용됨', uses_architecture: '아키텍처 사용',
  enables_efficient: '효율화', interacts_with: '상호작용',
  managed_by: '관리', implements: '구현', generalized_to: '일반화',
  uses: '사용', related_to: '관련', derives: '유도', trains: '학습',
  basis_of: '기반', learns: '학습', combined_with: '결합',
  explains: '설명', comparison: '비교'
};

/* ── Cluster Configuration ── */
// Category → Cluster (기본 매핑)
const CLUSTER_MAP = {
  'Generative':     'AI',       // 딥러닝
  'Architecture':   'AI',       // 딥러닝
  'Language Model': 'AI',       // 딥러닝
  'Multimodal':     'AI',       // 딥러닝
  'Training':       'AI',       // 딥러닝
  'RL':             'ML',       // 기본 RL은 머신러닝
  'Math & Stats':   '선형대수', // 수학 & 확률 통계 (노란색)
  'Math':           '선형대수', // 수학 (노란색)
  'LinearAlgebra':  '선형대수', // 선형대수학 독립 클러스터!
  'Math Problems':  '선형대수', // 선형대수학 연습문제
  'Systems':        '시스템',
  'Algorithm':      '알고리즘',
  'Research':       '연구',
  'Paper':          '연구'
};

// 노드별 개별 오버라이드 (카테고리 매핑보다 우선)
const NODE_CLUSTER_OVERRIDE = {
  'rlhf': 'AI'   // RLHF는 LLM alignment → 딥러닝 쪽
};

// 헬퍼: 노드의 최종 클러스터 반환
function getNodeCluster(node) {
  if (!node) return 'AI';
  if (node.id && (node.id.startsWith('linear_algebra') || node.id.startsWith('probability_') || node.id.startsWith('mml_') || node.id.startsWith('math_'))) return '선형대수';
  if (node.id && (node.id.startsWith('rq_') || node.id.startsWith('paper_'))) return '연구';
  if (node.tags && (node.tags.includes('Paper') || node.tags.includes('ResearchAgenda') || node.tags.includes('Agenda'))) return '연구';
  if (['streamkv', 'hcx_omni', 'clip'].includes(node.id)) return '연구';
  return NODE_CLUSTER_OVERRIDE[node.id] || CLUSTER_MAP[node.category] || '선형대수';
}

// 6개 주요 클러스터 좌표 설정 (수평 배치: 시스템 ➔ 알고리즘 ➔ 선형대수 ➔ 머신러닝 ➔ 딥러닝 ➔ 연구)
const CLUSTER_CONFIG = {
  '시스템':   { color: '#ef4444', label: '💻  시스템',        cx: 0.08, cy: 0.48 },
  '알고리즘': { color: '#06b6d4', label: '🔢  알고리즘',      cx: 0.24, cy: 0.48 },
  '선형대수': { color: '#f59e0b', label: '📐  선형대수학',    cx: 0.40, cy: 0.48 },
  'ML':       { color: '#34d399', label: '📊  머신러닝',      cx: 0.56, cy: 0.48 },
  'AI':       { color: '#a78bfa', label: '🤖  딥러닝',        cx: 0.74, cy: 0.48 },
  '연구':     { color: '#f43f5e', label: '🔭  연구 & 논문',   cx: 0.92, cy: 0.48 }
};

// ==========================================
// 📅 CALENDAR SCHEDULE DETAIL MODAL FUNCTIONS
// ==========================================
const scheduleDataMap = {
  '2026-08-11': {
    title: '🧹 맥북 저장 용량 대청소 & 백업',
    content: '• 맥북 39GB 저장 공간 확보 (캐시/어플 삭제)<br>• 바탕화면 및 다운로드 150+개 파일 Google Drive로 자동 이사'
  },
  '2026-08-13': {
    title: '🎯 2027 대학원 / 채용 병행 전략',
    content: '• 2027 봄학기 대학원 서류 준비 및 채용 병행 일정 수립<br>• 포스텍/카이스트/서울대 타겟 연구실 1차 정리'
  },
  '2026-08-14': {
    title: '📐 선형대수학 개념 맵 & MML 가이드',
    content: '• CONTENT_GUIDE.md 작성 및 선형대수 전체 챕터 맵 설계<br>• 선형방정식계 및 행렬 대수 기본 노드 작성'
  },
  '2026-08-15': {
    title: '✏️ 벡터공간 & 선형독립 수식 스케치',
    content: '• 2.4 벡터공간과 2.5 선형독립 수식 노트 작성<br>• 4차원-2차원 사상 커널 스케치 이미지(sketch_4d_to_2d) 작성'
  },
  '2026-08-16': {
    title: '🧠 선형사상, 기저, 어파인공간 & 연구실 탐색',
    content: '• 2.6 기저/계수, 2.7 선형사상, 2.8 어파인공간 노트 작성<br>• 포스텍(손진희 랩 1지망) 및 서울대(도재영 랩) 타겟 연구실 조사'
  },
  '2026-08-17': {
    title: '🌟 선형대수 3.1~3.8 전수 완수 & 투데이 리포트 연동',
    content: '• <b>3.1~3.8 전수 집필</b>: Norms, Inner Products, Cauchy-Schwarz, Cosine Similarity, Gram-Schmidt, Projection 등 Eq 3.1~3.73 전수 정리<br>• hyeonwoo 사이트에 [투데이 리포트] & 캘린더 모달 배포 완료!'
  },
  '2026-08-18': {
    title: '🌟 서울대 AIDAS 랩 도재영 교수님 사전 컨택 메일 최종 작성 & 발송!',
    content: '• <b>컨택 대상</b>: 서울대학교 전기·정보공학부 / 데이터사이언스 AIDAS 연구실 (도재영 교수님)<br>• <b>핵심 어필 포인트</b>: 숭실대 AI융합학부 졸업(GPA 4.32/4.5 석차 2등), KAIST DAVIAN 랩 Video Moment Retrieval 1저자 평가 파이프라인 주도 경험, 9월 네이버 HyperCLOVA X Omni팀 인턴 입과 예정 강조<br>• <b>첨부 서류</b>: CV(이력서), 학부 성적증명서, 연구 포트폴리오 첨부하여 이메일 최종 제출 완료!'
  },
  '2026-08-19': {
    title: '🌟 [오늘 (8/19)] 수신 이메일함 점검 & D-Day 차감 업데이트',
    content: '• <b>이메일 점검</b>: Apple Mail 앱 연결 수신함 전수 점검 완료 (도재영 교수님 회신 대기 중)<br>• <b>D-Day 차감</b>: 네이버클라우드 그린팩토리 입과 (D-26), 포스텍 1지망 서류 마감 (D-20)<br>• <b>오늘 목표</b>: MML Chapter 4 Matrix Decompositions (Eigenvalues & SVD) 학습'
  },
  '2026-09-14': {
    title: '🟢 네이버클라우드 체험형 인턴 첫 출근 (입과일)',
    content: '• <b>근무 기간</b>: 2026. 09. 14(월) ~ 2026. 12. 11(금)<br>• <b>근무 시간</b>: 10:00 ~ 19:00 (주 5일 대면)<br>• <b>근무 장소</b>: 경기도 성남시 분당구 불정로 6 네이버 그린팩토리<br>• <b>제출 서류</b>: 학력, 경력, 어학 및 자격 증빙서류 지참'
  }
};

function openCalendarModal(dateStr) {
  const modal = document.getElementById('calendar-modal');
  const dateLabel = document.getElementById('modal-date-label');
  const titleLabel = document.getElementById('modal-title-label');
  const contentEl = document.getElementById('modal-detail-content');

  if (!modal) return;

  dateLabel.textContent = dateStr;
  const item = scheduleDataMap[dateStr];

  if (item) {
    titleLabel.textContent = item.title;
    contentEl.innerHTML = item.content;
  } else {
    titleLabel.textContent = '등록된 일정 항목';
    contentEl.innerHTML = `• <b>${dateStr}</b> 세부 일정 및 대화 기록이 준비되어 있습니다.<br>• 자유롭게 새로운 일정을 등록하거나 AI에 추가를 요청하실 수 있습니다.`;
  }

  modal.style.display = 'flex';
}

function closeCalendarModal() {
  const modal = document.getElementById('calendar-modal');
  if (modal) modal.style.display = 'none';
}

function toggleRecallCard(cardEl) {
  if (!cardEl) return;
  const ans = cardEl.querySelector('.recall-answer');
  if (!ans) return;
  const isHidden = ans.style.display === 'none' || !ans.style.display;
  ans.style.display = isHidden ? 'block' : 'none';
  cardEl.style.borderColor = isHidden ? 'rgba(245, 158, 11, 0.6)' : 'rgba(255, 255, 255, 0.08)';
  cardEl.style.background = isHidden ? 'rgba(245, 158, 11, 0.08)' : 'rgba(30, 41, 59, 0.6)';
}

function refreshYesterdayQuiz() {
  const cards = document.querySelectorAll('.recall-card');
  cards.forEach(card => {
    const ans = card.querySelector('.recall-answer');
    if (ans) ans.style.display = 'none';
    card.style.borderColor = 'rgba(255, 255, 255, 0.08)';
    card.style.background = 'rgba(30, 41, 59, 0.6)';
  });
}

/* ============================================================
   BOOT
   ============================================================ */
document.addEventListener('DOMContentLoaded', async () => {
  // Always fetch latest knowledge.json first to prevent stale localStorage cache issues
  try {
    const res = await fetch('data/knowledge.json');
    knowledgeData = await res.json();
    localStorage.setItem('hyeonwoo_knowledge_v1', JSON.stringify(knowledgeData));
  } catch (e) {
    console.error('Failed to fetch data/knowledge.json, checking localStorage:', e);
    const stored = localStorage.getItem('hyeonwoo_knowledge_v1');
    if (stored) {
      try {
        knowledgeData = JSON.parse(stored);
      } catch (err) {
        knowledgeData = { nodes: [], edges: [], categories: {}, sessions: [] };
      }
    } else {
      knowledgeData = { nodes: [], edges: [], categories: {}, sessions: [] };
    }
  }

  // Schema Backwards Compatibility: concepts -> nodes / edges / categories
  if (knowledgeData.concepts && !knowledgeData.nodes) {
    knowledgeData.nodes = knowledgeData.concepts.map(c => ({
      id: c.id,
      label: c.title || c.label || c.id,
      category: c.category || 'Math & Stats',
      confidence: typeof c.confidence === 'number' ? c.confidence : 0,
      studyCount: c.review_count || c.studyCount || 0,
      note: c.note || `data/notes/${c.id}.md`,
      definition: c.definition,
      purpose: c.purpose,
      tradeoff_insight: c.tradeoff_insight,
      ai_connection: c.ai_connection
    }));
  }

  if (!knowledgeData.nodes) knowledgeData.nodes = [];
  if (!knowledgeData.edges) {
    // Generate sequential basis_of links for linear algebra or Math concepts so graph has connections
    knowledgeData.edges = [];
    for (let i = 0; i < knowledgeData.nodes.length - 1; i++) {
      knowledgeData.edges.push({
        source: knowledgeData.nodes[i].id,
        target: knowledgeData.nodes[i+1].id,
        relation: 'basis_of',
        weight: 3
      });
    }
  }
  if (!knowledgeData.categories || Object.keys(knowledgeData.categories).length === 0) {
    knowledgeData.categories = {
      'Generative':     { color: '#a78bfa', icon: '🎨' },
      'Architecture':   { color: '#a78bfa', icon: '🏗️' },
      'Language Model': { color: '#a78bfa', icon: '📝' },
      'Multimodal':     { color: '#a78bfa', icon: '👁️' },
      'Training':       { color: '#a78bfa', icon: '⚙️' },
      'RL':             { color: '#34d399', icon: '🤖' },
      'Math & Stats':   { color: '#f59e0b', icon: '📊' },
      'Math':           { color: '#f59e0b', icon: '📐' },
      'Systems':        { color: '#ef4444', icon: '💻' },
      'Algorithm':      { color: '#06b6d4', icon: '🔢' }
    };
  }
  if (!knowledgeData.sessions) knowledgeData.sessions = [];

  // 사용자 커스텀 데이터(AI 어시스턴트가 추가한 개념/엣지/노트) 병합
  mergeCustomData();

  initNav();
  initSearch();
  initGraph();
  safeInit('initHeatmap', initHeatmap);
  initProgress();
  initResearch();
  initCopyButtons();
  safeInit('initReview', initReview);
  safeInit('initColumn', initColumn);
  safeInit('initChat', initChat);
  updateStatsBadge();
});

/* 하나의 init 실패가 전체 초기화를 막지 않도록 안전 실행 */
function safeInit(name, fn) {
  try { fn(); }
  catch (e) { console.warn(`[init] ${name} 실패 (무시):`, e); }
}

/* ── Copy Plain Text Helper ── */
function initCopyButtons() {
  const copyNoteBtn = document.getElementById('copy-note-btn');
  if (copyNoteBtn) {
    copyNoteBtn.addEventListener('click', () => {
      let rawText = window.currentNoteRawMd || '';
      const formatted = cleanMarkdownForBlog(rawText, 'note-panel-body');
      copyStringToClipboard(formatted, copyNoteBtn);
    });
  }

  const copyColBtn = document.getElementById('copy-column-btn');
  if (copyColBtn) {
    copyColBtn.addEventListener('click', () => {
      let rawText = window.currentColumnRawMd || '';
      const formatted = cleanMarkdownForBlog(rawText, 'reader-body');
      copyStringToClipboard(formatted, copyColBtn);
    });
  }
}

// Convert Markdown / DOM to clean, readable text without KaTeX duplicate subscripts cracking
function cleanMarkdownForBlog(md, elementId) {
  const el = elementId ? document.getElementById(elementId) : null;
  if (el) {
    // Clone element to sanitize KaTeX DOM elements safely
    const clone = el.cloneNode(true);

    // Remove KaTeX MathML hidden elements that cause duplicate text like (x1 x1)
    clone.querySelectorAll('.katex-mathml').forEach(node => node.remove());

    let text = clone.innerText || clone.textContent || '';
    
    // Normalize double linebreaks
    text = text.replace(/\n{3,}/g, '\n\n');
    return text.trim();
  }

  if (!md) return '';
  let text = md;
  text = text.replace(/^#{1,6}\s+(.+)$/gm, '\n■ $1\n');
  text = text.replace(/\\\\\\\\/g, '\n');
  text = text.replace(/\\\\/g, '\n');
  text = text.replace(/\$\$(.*?)\$\$/gs, '\n$1\n');
  text = text.replace(/\$(.*?)\$/g, '$1');
  text = text.replace(/\\begin\{[a-z]+\}/g, '');
  text = text.replace(/\\end\{[a-z]+\}/g, '');
  text = text.replace(/\\quad/g, ' ');
  return text.trim();
}

function copyElementPlainText(el, btnEl) {
  if (!el) return;
  const plainText = el.innerText || el.textContent;
  copyStringToClipboard(plainText, btnEl);
}

function copyStringToClipboard(text, btnEl) {
  if (!text) {
    alert('복사할 텍스트가 없습니다.');
    return;
  }
  
  // Use a temporary textarea element for 100% reliable cross-browser clipboard copy
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.left = '-999999px';
  textarea.style.top = '-999999px';
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();

  let success = false;
  try {
    success = document.execCommand('copy');
  } catch (err) {
    success = false;
  }
  document.body.removeChild(textarea);

  if (success || navigator.clipboard) {
    if (!success && navigator.clipboard) {
      navigator.clipboard.writeText(text);
    }
    const origText = btnEl.innerText;
    btnEl.innerText = '✅ 복사 완료!';
    btnEl.style.background = '#10b981';
    btnEl.style.color = '#ffffff';
    btnEl.style.borderColor = '#10b981';
    setTimeout(() => {
      btnEl.innerText = origText;
      btnEl.style.background = '';
      btnEl.style.color = '';
      btnEl.style.borderColor = '';
    }, 2000);
  } else {
    alert('복사에 실패했습니다. 단축키 Cmd+C / Ctrl+C를 사용해 주세요.');
  }
}

/* ============================================================
   NAVIGATION
   ============================================================ */
function initNav() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const view = btn.dataset.view;
      switchView(view);
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}

function switchView(view) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById(`view-${view}`).classList.add('active');
  if (view === 'graph' && simulation) {
    setTimeout(() => { simulation.alpha(0.1).restart(); }, 100);
  }
  if (view === 'progress') {
    initProgress(); // Re-render lists with the latest localStorage/json data
  }
  if (view === 'quiz') {
    initQuiz();
  }
  if (view === 'column') {
    initColumn();
  }
}

/* ============================================================
   SEARCH
   ============================================================ */
function initSearch() {
  const input   = document.getElementById('search-input');
  const results = document.getElementById('search-results');

  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (!q) { results.classList.remove('show'); return; }
    const matches = knowledgeData.nodes.filter(n =>
      n.label.toLowerCase().includes(q) ||
      (n.tags || []).some(t => t.toLowerCase().includes(q))
    ).slice(0, 8);
    if (!matches.length) { results.classList.remove('show'); return; }

    results.innerHTML = matches.map(n => {
      const catColor = knowledgeData.categories[n.category]?.color || '#888';
      return `<div class="search-result-item" data-id="${n.id}">
        <span style="color:${catColor};font-size:16px">${knowledgeData.categories[n.category]?.icon || '◉'}</span>
        <div>
          <div class="search-result-label">${n.label}</div>
          <div class="search-result-cat">${n.category}</div>
        </div>
      </div>`;
    }).join('');
    results.classList.add('show');

    results.querySelectorAll('.search-result-item').forEach(item => {
      item.addEventListener('click', () => {
        const id = item.dataset.id;
        results.classList.remove('show');
        input.value = '';
        switchView('graph');
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('btn-graph').classList.add('active');
        setTimeout(() => focusNode(id), 300);
      });
    });
  });

  document.addEventListener('click', e => {
    if (!e.target.closest('.search-box') && !e.target.closest('.search-results'))
      results.classList.remove('show');
  });
}

function updateStatsBadge() {
  document.getElementById('total-nodes').textContent = knowledgeData.nodes.length;
}

/* ============================================================
   KNOWLEDGE GRAPH (D3.js) — with Cluster Layout
   ============================================================ */
function initGraph() {
  const container = document.getElementById('graph-container');
  const svgEl     = document.getElementById('graph-svg');
  const W = container.clientWidth;
  const H = container.clientHeight;

  svg = d3.select(svgEl).attr('viewBox', `0 0 ${W} ${H}`);

  zoom = d3.zoom()
    .scaleExtent([0.15, 3])
    .on('zoom', e => { g.attr('transform', e.transform); });
  svg.call(zoom);

  g = svg.append('g');

  /* ---- Defs ---- */
  const defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -5 10 10').attr('refX', 22).attr('refY', 0)
    .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
    .append('path').attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', 'rgba(255,255,255,0.15)');

  /* ---- Build node / link data ---- */
  const nodeMap = {};
  knowledgeData.nodes.forEach(n => { nodeMap[n.id] = n; });

  // Assign initial position biased toward cluster center
  const nodes = knowledgeData.nodes.map(n => {
    const clusterName = getNodeCluster(n);
    const cc = CLUSTER_CONFIG[clusterName];
    const jitter = () => (Math.random() - 0.5) * 130;
    return {
      ...n,
      cluster: clusterName,
      x: W * cc.cx + jitter(),
      y: H * cc.cy + jitter()
    };
  });

  const links = knowledgeData.edges.map(e => ({
    source: e.source, target: e.target,
    relation: e.relation, weight: e.weight || 1, insight: e.insight || null
  }));

  /* ---- Cluster hull layer (drawn BEFORE nodes) ---- */
  const hullGroup = g.append('g').attr('class', 'cluster-hulls');
  const labelGroup = g.append('g').attr('class', 'cluster-labels');

  const hullPaths  = {};
  const hullLabels = {};

  Object.entries(CLUSTER_CONFIG).forEach(([name, cfg]) => {
    // Background hull path
    hullPaths[name] = hullGroup.append('path')
      .attr('class', 'cluster-hull')
      .attr('fill', cfg.color)
      .attr('fill-opacity', 0.05)
      .attr('stroke', cfg.color)
      .attr('stroke-opacity', 0.25)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4');

    // Cluster title label
    hullLabels[name] = labelGroup.append('text')
      .attr('class', 'cluster-label')
      .attr('text-anchor', 'middle')
      .attr('font-size', '15')
      .attr('font-weight', '700')
      .attr('font-family', 'Pretendard, sans-serif')
      .attr('fill', cfg.color)
      .attr('fill-opacity', 0.7)
      .attr('pointer-events', 'none')
      .attr('user-select', 'none')
      .text(cfg.label);
  });

  /* ---- Force simulation ---- */
  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links)
      .id(d => d.id)
      .distance(d => {
        // Cross-cluster links: long and weak
        const sn = typeof d.source === 'object' ? d.source : nodeMap[d.source];
        const tn = typeof d.target === 'object' ? d.target : nodeMap[d.target];
        if (sn && tn && getNodeCluster(sn) !== getNodeCluster(tn))
          return 260;
        return 100 - (d.weight || 1) * 10;
      })
      .strength(d => {
        const sn = typeof d.source === 'object' ? d.source : nodeMap[d.source];
        const tn = typeof d.target === 'object' ? d.target : nodeMap[d.target];
        if (sn && tn && getNodeCluster(sn) !== getNodeCluster(tn))
          return 0.04;
        return 0.28 + (d.weight || 1) * 0.04;
      }))
    .force('charge', d3.forceManyBody().strength(-220))
    .force('collide', d3.forceCollide().radius(d => getNodeRadius(d) + 14))
    .force('cluster', clusterForce(nodes, W, H))
    .alphaDecay(0.025);

  /* ---- Category legend ---- */
  buildLegend(knowledgeData.categories);

  /* ---- Links ---- */
  const linkGroup = g.append('g').attr('class', 'links');

  const linkLine = linkGroup.selectAll('line')
    .data(links).enter().append('line')
    .attr('class', d => {
      const sn = nodeMap[typeof d.source === 'object' ? d.source.id : d.source];
      const tn = nodeMap[typeof d.target === 'object' ? d.target.id : d.target];
      const cross = sn && tn && getNodeCluster(sn) !== getNodeCluster(tn);
      return cross ? 'link-line link-cross' : 'link-line link-inner';
    })
    .attr('stroke', d => {
      const src = nodeMap[typeof d.source === 'object' ? d.source.id : d.source];
      return knowledgeData.categories[src?.category]?.color || '#555';
    })
    .attr('stroke-width', d => 0.5 + (d.weight || 1) * 0.4)
    .attr('stroke-dasharray', d => d.relation === 'comparison' ? '4,3' : null)
    .attr('marker-end', 'url(#arrow)');

  const insightLabels = linkGroup.selectAll('.insight-label')
    .data(links.filter(d => d.insight)).enter()
    .append('text')
    .attr('class', 'link-insight-badge')
    .attr('text-anchor', 'middle').attr('dy', '-4')
    .attr('font-size', '8').attr('fill', 'rgba(56,189,248,0.7)')
    .attr('font-family', 'Pretendard, sans-serif')
    .text(d => d.insight.length > 22 ? d.insight.slice(0, 22) + '…' : d.insight)
    .style('opacity', 0).style('pointer-events', 'none');

  /* ---- Nodes ---- */
  const nodeGroup = g.append('g').attr('class', 'nodes');

  const node = nodeGroup.selectAll('g')
    .data(nodes).enter().append('g')
    .attr('class', 'node-g')
    .attr('id', d => `node-${d.id}`)
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', dragStart)
      .on('drag',  dragged)
      .on('end',   dragEnd));

  node.append('circle')
    .attr('class', 'node-circle')
    .attr('r', d => getNodeRadius(d))
    .attr('fill', d => knowledgeData.categories[d.category]?.color || '#888')
    .attr('fill-opacity', d => 0.15 + (d.confidence / 4) * 0.6)
    .attr('stroke', d => knowledgeData.categories[d.category]?.color || '#888')
    .attr('stroke-width', 1.5)
    .attr('stroke-opacity', 0.65);

  node.append('circle')
    .attr('r', 3)
    .attr('fill', d => knowledgeData.categories[d.category]?.color || '#888')
    .attr('fill-opacity', 0.9);

  node.append('text')
    .attr('class', 'node-label')
    .attr('text-anchor', 'middle')
    .attr('dy', d => getNodeRadius(d) + 12)
    .attr('font-size', '11')
    .attr('font-family', 'Pretendard, sans-serif')
    .attr('font-weight', '500')
    .attr('fill', 'rgba(232,234,240,0.85)')
    .text(d => d.label);

  /* ---- Tooltip ---- */
  const tooltip = document.getElementById('node-tooltip');
  node.on('mouseover', (event, d) => {
    const color = knowledgeData.categories[d.category]?.color || '#888';
    tooltip.innerHTML = `
      <div class="tt-label" style="color:${color}">${d.label}</div>
      <div class="tt-category">${knowledgeData.categories[d.category]?.icon || ''} ${d.category}</div>
      <div class="tt-confidence">자신감: ${CONFIDENCE_STARS[d.confidence] || '?'} ${CONFIDENCE_LABELS[d.confidence] || ''}</div>
      <div class="tt-tags">${(d.tags || []).map(t => `<span class="tt-tag">${t}</span>`).join('')}</div>
    `;
    tooltip.classList.add('visible');
    highlightNeighbors(d.id, linkLine, node);
    insightLabels.style('opacity', l => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      return (s === d.id || t === d.id) ? 1 : 0;
    });
  })
  .on('mousemove', event => {
    tooltip.style.left = (event.offsetX + 14) + 'px';
    tooltip.style.top  = (event.offsetY - 10) + 'px';
  })
  .on('mouseout', () => {
    tooltip.classList.remove('visible');
    resetHighlight(linkLine, node);
    insightLabels.style('opacity', 0);
  })
  .on('click', (event, d) => { event.stopPropagation(); openNotePanel(d); });

  /* ---- Tick ---- */
  simulation.on('tick', () => {
    linkLine
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);

    insightLabels
      .attr('x', d => (d.source.x + d.target.x) / 2)
      .attr('y', d => (d.source.y + d.target.y) / 2);

    node.attr('transform', d => `translate(${d.x},${d.y})`);

    /* ---- Update cluster hulls ---- */
    Object.entries(CLUSTER_CONFIG).forEach(([clusterName, cfg]) => {
      const clusterNodes = nodes.filter(d => d.cluster === clusterName);
      if (clusterNodes.length < 2) return;

      const pts = clusterNodes.map(d => [d.x, d.y]);
      // Need at least 3 points for hull; duplicate if fewer
      while (pts.length < 3) pts.push([...pts[0]]);
      const hull = d3.polygonHull(pts);
      if (!hull) return;

      // Expand hull outward from centroid
      const cx = d3.mean(hull, p => p[0]);
      const cy = d3.mean(hull, p => p[1]);
      const pad = 52;
      const expanded = hull.map(p => {
        const dx = p[0] - cx, dy = p[1] - cy;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        return [p[0] + dx / len * pad, p[1] + dy / len * pad];
      });

      hullPaths[clusterName]
        .attr('d', `M${expanded.map(p => p.join(',')).join('L')}Z`);

      // Position label above the hull
      const minY = d3.min(expanded, p => p[1]);
      hullLabels[clusterName]
        .attr('x', cx)
        .attr('y', minY - 14);
    });
  });

  /* ---- Zoom controls ---- */
  document.getElementById('btn-zoom-in').addEventListener('click', () =>
    svg.transition().duration(300).call(zoom.scaleBy, 1.3));
  document.getElementById('btn-zoom-out').addEventListener('click', () =>
    svg.transition().duration(300).call(zoom.scaleBy, 0.77));
  document.getElementById('btn-zoom-reset').addEventListener('click', () =>
    svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity));

  svg.on('click', () => closeNotePanel());
  document.getElementById('note-close').addEventListener('click', closeNotePanel);

  // Legend toggle logic
  const legendToggle = document.getElementById('legend-toggle-btn');
  const legendEl = document.getElementById('legend');
  if (legendToggle && legendEl) {
    legendToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      const isHidden = legendEl.style.display === 'none';
      legendEl.style.display = isHidden ? 'block' : 'none';
      legendToggle.classList.toggle('active', isHidden);
    });
  }
}

/* ---- Custom Cluster Force ---- */
function clusterForce(nodes, W, H) {
  const strength = 0.12;
  return function(alpha) {
    nodes.forEach(d => {
      const cfg = CLUSTER_CONFIG[d.cluster];
      if (!cfg) return;
      const tx = W * cfg.cx;
      const ty = H * cfg.cy;
      d.vx += (tx - d.x) * alpha * strength;
      d.vy += (ty - d.y) * alpha * strength;
    });
  };
}

function getNodeRadius(d) {
  return 12 + (d.confidence || 0) * 4 + (d.studyCount || 0) * 1.2;
}

/* ---- Legend ---- */
function buildLegend(categories) {
  const legend = document.getElementById('legend');
  Object.entries(categories).forEach(([name, val]) => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<span class="legend-dot" style="background:${val.color}"></span>${val.icon} ${name}`;
    item.addEventListener('click', () => {
      if (activeCategory === name) {
        activeCategory = null;
        item.style.background = '';
        filterByCategory(null);
      } else {
        activeCategory = name;
        legend.querySelectorAll('.legend-item').forEach(i => i.style.background = '');
        item.style.background = 'rgba(255,255,255,0.08)';
        filterByCategory(name);
      }
    });
    legend.appendChild(item);
  });
}

function filterByCategory(cat) {
  d3.selectAll('.node-g').transition().duration(300)
    .style('opacity', d => (!cat || d.category === cat) ? 1 : 0.1);
  d3.selectAll('.link-line').transition().duration(300)
    .style('opacity', d => {
      if (!cat) return null;
      const sc = typeof d.source === 'object' ? d.source.category : null;
      const tc = typeof d.target === 'object' ? d.target.category : null;
      return (sc === cat || tc === cat) ? 0.6 : 0.05;
    });
}

function highlightNeighbors(nodeId, linkLine, node) {
  const neighborIds = new Set([nodeId]);
  knowledgeData.edges.forEach(e => {
    const s = typeof e.source === 'object' ? e.source.id : e.source;
    const t = typeof e.target === 'object' ? e.target.id : e.target;
    if (s === nodeId) neighborIds.add(t);
    if (t === nodeId) neighborIds.add(s);
  });
  node.style('opacity', d => neighborIds.has(d.id) ? 1 : 0.15);
  linkLine.style('stroke-opacity', d => {
    const s = typeof d.source === 'object' ? d.source.id : d.source;
    const t = typeof d.target === 'object' ? d.target.id : d.target;
    return (s === nodeId || t === nodeId) ? 0.9 : 0.03;
  });
}

function resetHighlight(linkLine, node) {
  node.style('opacity', 1);
  linkLine.style('stroke-opacity', function() {
    return this.classList.contains('link-cross') ? 0.18 : 0.38;
  });
}

function focusNode(nodeId) {
  const nodeData = knowledgeData.nodes.find(n => n.id === nodeId);
  if (!nodeData) return;
  openNotePanel(nodeData);
  d3.select(`#node-${nodeId}`).select('circle')
    .transition().duration(300).attr('r', getNodeRadius(nodeData) * 1.4)
    .transition().duration(300).attr('r', getNodeRadius(nodeData));
}

function dragStart(event, d) {
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x; d.fy = d.y;
}
function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
function dragEnd(event, d) {
  if (!event.active) simulation.alphaTarget(0);
  d.fx = null; d.fy = null;
}

/* ============================================================
   NOTE PANEL
   ============================================================ */
/** 채팅/외부 onclick 용: nodeId 로 노트 열기 + 그래프로 이동 */
function openNote(nodeId) {
  const node = knowledgeData.nodes.find(n => n.id === nodeId);
  if (!node) { alert('노드를 찾을 수 없습니다: ' + nodeId); return; }
  switchView('graph');
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-graph').classList.add('active');
  setTimeout(() => {
    try { focusNode(nodeId); } catch (e) {}
    openNotePanel(node);
  }, 250);
}

async function openNotePanel(nodeData) {
  currentNodeId = nodeData.id;
  const panel   = document.getElementById('note-panel');
  const titleEl = document.getElementById('note-panel-title');
  const metaEl  = document.getElementById('note-panel-meta');
  const bodyEl  = document.getElementById('note-panel-body');

  const catColor = knowledgeData.categories[nodeData.category]?.color || '#888';
  const catIcon  = knowledgeData.categories[nodeData.category]?.icon  || '';
  const cluster  = getNodeCluster(nodeData);
  const clusterColor = CLUSTER_CONFIG[cluster]?.color || catColor;

  titleEl.textContent = nodeData.label;
  metaEl.innerHTML = `
    <span class="note-panel-badge" style="background:${clusterColor}22;color:${clusterColor};border:1px solid ${clusterColor}44">${CLUSTER_CONFIG[cluster]?.label || cluster}</span>
    <span class="note-panel-badge" style="background:${catColor}18;color:${catColor};border:1px solid ${catColor}33">${catIcon} ${nodeData.category}</span>
    <span style="color:var(--accent-yellow)">${CONFIDENCE_STARS[nodeData.confidence] || ''}</span>
    <span style="color:var(--text-muted)">복습 ${nodeData.studyCount || 0}회</span>
  `;

  panel.classList.add('open');
  const backdrop = document.getElementById('note-panel-backdrop');
  if (backdrop) backdrop.classList.add('visible');

  let content = '';
  try {
    // AI 어시스턴트가 만든 커스텀 노트가 있으면 우선 표시
    const customNote = getCustomNote(nodeData.id);
    if (customNote) {
      let md = customNote;
      window.currentNoteRawMd = md;
      const mathBlocks = [];
      const safeMd = md.replace(/\$\$([\s\S]*?)\$\$|\$([^\$\n]+?)\$/g, (match, displayMath, inlineMath) => {
        const isDisplay = match.startsWith('$$');
        const content = isDisplay ? displayMath : inlineMath;
        const placeholder = `MATHBLOCK${mathBlocks.length}END`;
        mathBlocks.push({ match, isDisplay, content });
        return placeholder;
      });
      let html = marked.parse(safeMd);
      mathBlocks.forEach((item, idx) => {
        const placeholder = `MATHBLOCK${idx}END`;
        html = html.replace(new RegExp(placeholder, 'g'), item.match);
      });
      content = `<div class="note-content">${html}</div>
        <div style="margin-top:10px;font-size:11px;color:var(--accent-blue);opacity:.7">📌 AI 어시스턴트가 생성한 노트 (이 브라우저에 저장됨)</div>`;
    } else {
    const notePath = nodeData.note ? (nodeData.note.startsWith('data/notes/') ? nodeData.note : `data/notes/${nodeData.note}`) : `data/notes/${nodeData.id}.md`;
    const res = await fetch(notePath);
    if (res.ok) {
      let md = await res.text();
      window.currentNoteRawMd = md;
      
      // Escape KaTeX math delims before marked processing
      // Replace inline math _ with \_ placeholder or preserve verbatim via code-like token
      const mathBlocks = [];
      const safeMd = md.replace(/\$\$([\s\S]*?)\$\$|\$([^\$\n]+?)\$/g, (match, displayMath, inlineMath) => {
        const isDisplay = match.startsWith('$$');
        const content = isDisplay ? displayMath : inlineMath;
        const placeholder = `MATHBLOCK${mathBlocks.length}END`;
        mathBlocks.push({ match, isDisplay, content });
        return placeholder;
      });

      let html = marked.parse(safeMd);

      mathBlocks.forEach((item, idx) => {
        const placeholder = `MATHBLOCK${idx}END`;
        // Replace html encoded placeholders or raw placeholders
        html = html.replace(new RegExp(placeholder, 'g'), item.match);
      });
      
      content = `<div class="note-content">${html}</div>`;
    } else {
      window.currentNoteRawMd = '';
      content = `<div class="note-content note-placeholder" style="min-height:200px">
        <div class="note-placeholder-icon">📝</div>
        <p><strong>${nodeData.label}</strong>에 대한 노트가 아직 없어요.</p>
        <p style="font-size:12px;color:var(--text-muted)">data/notes/${nodeData.id}.md 파일을 만들어 채워 주세요!</p>
      </div>`;
    }
    }
  } catch {
    content = `<div class="note-placeholder" style="min-height:200px"><p>노트를 불러올 수 없습니다.</p></div>`;
  }

  const connected = getConnectedNodes(nodeData.id);
  let connectedHtml = '';
  if (connected.length) {
    connectedHtml = `
      <div class="connected-concepts">
        <div class="connected-title">연결된 개념 (${connected.length}개)</div>
        <div class="connected-list">
          ${connected.map(c => {
            const color = knowledgeData.categories[c.node.category]?.color || '#888';
            const cCluster = getNodeCluster(c.node);
            const cClusterCfg = CLUSTER_CONFIG[cCluster];
            return `<div class="connected-item" onclick="focusNode('${c.node.id}');openNotePanel(knowledgeData.nodes.find(n=>n.id==='${c.node.id}'))">
              <span class="connected-dot" style="background:${color}"></span>
              <div class="connected-info">
                <div class="connected-label">${c.node.label}
                  <span style="font-size:10px;color:${cClusterCfg?.color || '#888'};margin-left:4px">${cClusterCfg?.label || ''}</span>
                </div>
                ${c.edge.insight ? `<div class="connected-insight">${c.edge.insight}</div>` : ''}
                <div class="connected-relation">${RELATION_LABELS[c.edge.relation] || c.edge.relation || ''}</div>
              </div>
            </div>`;
          }).join('')}
        </div>
      </div>`;
  }

  bodyEl.innerHTML = content + connectedHtml;

  if (window.renderMathInElement) {
    renderMathInElement(bodyEl, {
      output: 'html',
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$',  right: '$',  display: false }
      ]
    });
  }
}

function closeNotePanel() {
  document.getElementById('note-panel').classList.remove('open');
  const backdrop = document.getElementById('note-panel-backdrop');
  if (backdrop) backdrop.classList.remove('visible');
  currentNodeId = null;
}

function getConnectedNodes(nodeId) {
  const result = [];
  knowledgeData.edges.forEach(e => {
    const s = typeof e.source === 'object' ? e.source.id : e.source;
    const t = typeof e.target === 'object' ? e.target.id : e.target;
    if (s === nodeId) {
      const n = knowledgeData.nodes.find(n => n.id === t);
      if (n) result.push({ node: n, edge: e, direction: 'out' });
    } else if (t === nodeId) {
      const n = knowledgeData.nodes.find(n => n.id === s);
      if (n) result.push({ node: n, edge: e, direction: 'in' });
    }
  });
  return result.sort((a, b) => (b.edge.weight || 0) - (a.edge.weight || 0));
}

/* ============================================================
   HEATMAP VIEW
   ============================================================ */
function initHeatmap() {
  buildCalendarHeatmap();
  buildSessionsList();
  buildCategoryActivityBars();
}

function buildCalendarHeatmap() {
  const grid     = document.getElementById('heatmap-grid');
  const monthsEl = document.getElementById('heatmap-months');
  if (!grid || !monthsEl) return; // heatmap 컨테이너가 없는 레이아웃에서는 건너뜀
  grid.innerHTML = ''; monthsEl.innerHTML = '';

  const sessionMap = {};
  knowledgeData.sessions.forEach(s => { sessionMap[s.date] = s; });

  const today = new Date();
  const startDate = new Date(today);
  startDate.setDate(today.getDate() - 112 + 1);
  const dow = (startDate.getDay() + 6) % 7;
  startDate.setDate(startDate.getDate() - dow);

  const weeks = [];
  const cur = new Date(startDate);
  while (cur <= today || weeks.length < 1 || weeks[weeks.length - 1].length < 7) {
    if (weeks.length === 0 || weeks[weeks.length - 1].length === 7) weeks.push([]);
    weeks[weeks.length - 1].push(new Date(cur));
    cur.setDate(cur.getDate() + 1);
    if (weeks.length > 16 && weeks[weeks.length - 1].length === 7) break;
  }

  let lastMonth = null;
  weeks.forEach(week => {
    const m = week[0].getMonth();
    const span = document.createElement('span');
    span.className = 'heatmap-month-label';
    span.textContent = m !== lastMonth ? week[0].toLocaleDateString('ko-KR', { month: 'short' }) : '';
    monthsEl.appendChild(span);
    lastMonth = m;
  });

  weeks.forEach(week => {
    const weekEl = document.createElement('div');
    weekEl.className = 'heatmap-week';
    week.forEach(date => {
      const cell = document.createElement('div');
      cell.className = 'heatmap-cell';
      const dateStr = formatDate(date);
      const session = sessionMap[dateStr];
      const intensity = session ? Math.min(4, session.topics.length) : 0;
      cell.dataset.intensity = intensity;
      cell.title = session ? `${dateStr}: ${session.topics.length}개 주제 — ${session.note || ''}` : dateStr;
      if (session) cell.addEventListener('click', () => scrollToSession(dateStr));
      weekEl.appendChild(cell);
    });
    grid.appendChild(weekEl);
  });
}

function buildSessionsList() {
  const list = document.getElementById('sessions-list');
  const sorted = [...knowledgeData.sessions].sort((a, b) => b.date.localeCompare(a.date));
  list.innerHTML = sorted.slice(0, 10).map(s => {
    const topicTags = s.topics.map(tid => {
      const node = knowledgeData.nodes.find(n => n.id === tid);
      if (!node) return '';
      const color = knowledgeData.categories[node.category]?.color || '#888';
      return `<span class="session-topic-tag" style="color:${color};border:1px solid ${color}44;background:${color}18">${node.label}</span>`;
    }).join('');
    return `<div class="session-item animate-in" id="session-${s.date}" data-date="${s.date}">
      <div class="session-date">${s.date}</div>
      <div class="session-info">
        <div class="session-topics">${topicTags}</div>
        ${s.note ? `<div class="session-note">${s.note}</div>` : ''}
      </div>
    </div>`;
  }).join('');
}

function buildCategoryActivityBars() {
  const barsEl = document.getElementById('category-bars');
  const counts = {};
  knowledgeData.sessions.forEach(s => {
    s.topics.forEach(tid => {
      const node = knowledgeData.nodes.find(n => n.id === tid);
      if (node) counts[node.category] = (counts[node.category] || 0) + 1;
    });
  });
  const maxCount = Math.max(...Object.values(counts), 1);
  barsEl.innerHTML = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([cat, count]) => {
      const color = knowledgeData.categories[cat]?.color || '#888';
      const icon  = knowledgeData.categories[cat]?.icon  || '';
      const pct   = Math.round((count / maxCount) * 100);
      return `<div class="cat-bar-item">
        <div class="cat-bar-label">${icon} ${cat}</div>
        <div class="cat-bar-track"><div class="cat-bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <div class="cat-bar-count">${count}회</div>
      </div>`;
    }).join('');
}

function scrollToSession(dateStr) {
  const el = document.getElementById(`session-${dateStr}`);
  if (!el) return;
  switchView('heatmap');
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-heatmap').classList.add('active');
  setTimeout(() => {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.style.outline = '2px solid var(--accent-purple)';
    setTimeout(() => el.style.outline = '', 2000);
  }, 300);
}

function formatDate(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

/* ============================================================
   PROGRESS VIEW
   ============================================================ */
/* ============================================================
   REVIEW TRACKING (SM-2 스타일 복습 스케줄러)
   - localStorage: hyeonwoo_review_v1
   - 각 노드: { interval(일), ease, due(date), reps, lastScore }
   - confidence 0~4 를 SM-2 quality 로 매핑해 간격 갱신
   ============================================================ */
const REVIEW_STORAGE_KEY = 'hyeonwoo_review_v1';
let reviewState = {};

function loadReviewState() {
  try {
    reviewState = JSON.parse(localStorage.getItem(REVIEW_STORAGE_KEY)) || {};
  } catch (e) { reviewState = {}; }
}

function saveReviewState() {
  localStorage.setItem(REVIEW_STORAGE_KEY, JSON.stringify(reviewState));
}

function todayStr(offsetDays = 0) {
  const d = new Date();
  d.setDate(d.getDate() + offsetDays);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

/** 노드의 복습 상태 가져오기 (없으면 초기화) */
function getReviewState(node) {
  let st = reviewState[node.id];
  if (!st) {
    // 첫 등장: confidence 기반으로 첫 복습일 설정
    const conf = node.confidence ?? 0;
    // 자신감이 낮을수록 빨리 복습
    const firstInterval = [1, 2, 4, 7, 14][conf] || 1;
    st = {
      interval: firstInterval,
      ease: 2.5,
      due: todayStr(firstInterval),
      reps: node.studyCount || 0,
      lastScore: null,
      started: todayStr()
    };
    reviewState[node.id] = st;
    saveReviewState();
  }
  return st;
}

/** SM-2 스타일로 복습 후 상태 갱신. score: 0(모름)~4(완벽) */
function applyReviewResult(node, score) {
  const st = getReviewState(node);
  const q = score; // 0~4

  let newInterval;
  if (q < 2) {
    // 실패: 간격 리셋, 빠르게 다시
    st.interval = 1;
    st.ease = Math.max(1.3, st.ease - 0.2);
    newInterval = 1;
  } else {
    if (st.reps === 0) newInterval = 1;
    else if (st.reps === 1) newInterval = 3;
    else newInterval = Math.round(st.interval * st.ease);
    st.interval = newInterval;
    st.ease = Math.max(1.3, st.ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)));
  }

  st.reps = (st.reps || 0) + 1;
  st.lastScore = q;
  st.due = todayStr(newInterval);

  // confidence 도 함께 업데이트 (지식 수준 반영)
  node.confidence = q;
  node.studyCount = st.reps;
  saveReviewState();
  saveKnowledgeData();
  return st;
}

/** 복습 대상 (due <= 오늘, confidence < 4) */
function getDueNodes() {
  const today = todayStr();
  return knowledgeData.nodes
    .filter(n => {
      const st = reviewState[n.id];
      if (!st) return false;
      return st.due <= today && (n.confidence ?? 0) < 4;
    })
    .sort((a, b) => {
      const da = reviewState[a.id].due, db = reviewState[b.id].due;
      return da.localeCompare(db) || (a.confidence - b.confidence);
    });
}

function getOverdueCount() {
  const today = todayStr();
  return Object.values(reviewState).filter(st => st.due < today).length;
}

/* ============================================================
   PROGRESS VIEW — 복습 대시보드
   ============================================================ */
function initProgress() {
  loadReviewState();
  buildReviewStats();
  buildDueReviewList();
  buildKnowledgeDist();
  buildClusterGroups();
  buildWeakSpots();
  buildStrongList();
}

function buildReviewStats() {
  const today = todayStr();
  let due = 0, learning = 0, mastered = 0;
  knowledgeData.nodes.forEach(n => {
    const conf = n.confidence ?? 0;
    const st = reviewState[n.id];
    if (conf >= 4) { mastered++; return; }
    if (st && st.due <= today) { due++; return; }
    if (conf >= 2) learning++;
    else due++; // 자신감 낮고 아직 스케줄 없는 것도 복습 대상
  });
  document.getElementById('stat-due').textContent = due;
  document.getElementById('stat-overdue').textContent = getOverdueCount();
  document.getElementById('stat-learning').textContent = learning;
  document.getElementById('stat-mastered').textContent = mastered;
}

function buildDueReviewList() {
  const container = document.getElementById('due-review-list');
  if (!container) return;

  // 스케줄이 아직 없는(복습 시작 전) 개념도 자신감 낮은 순으로 포함
  const today = todayStr();
  const candidates = knowledgeData.nodes.filter(n => {
    const conf = n.confidence ?? 0;
    if (conf >= 4) return false;
    const st = reviewState[n.id];
    if (st) return st.due <= today;
    return true; // 아직 복습 안 한 개념
  }).sort((a, b) => {
    const sa = reviewState[a.id]?.due, sb = reviewState[b.id]?.due;
    if (sa && sb) return sa.localeCompare(sb);
    if (sa) return 1; if (sb) return -1;
    return (a.confidence - b.confidence);
  }).slice(0, 12);

  if (!candidates.length) {
    container.innerHTML = `<p class="review-empty">🎉 오늘 복습할 개념이 없습니다. 그래프에서 새 개념을 학습해보세요!</p>`;
    return;
  }

  container.innerHTML = candidates.map(n => {
    const color = knowledgeData.categories[n.category]?.color || '#888';
    const icon  = knowledgeData.categories[n.category]?.icon  || '◉';
    const st = reviewState[n.id];
    const dueLabel = st ? (st.due < today ? `밀림 (${st.due})` : st.due === today ? '오늘' : st.due) : '시작 전';
    const confLabel = CONFIDENCE_STARS[n.confidence] || '';
    return `
      <div class="due-review-item" data-id="${n.id}">
        <div class="due-review-left" onclick="focusNode('${n.id}');switchView('graph');document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));document.getElementById('btn-graph').classList.add('active')">
          <span class="due-review-icon" style="color:${color}">${icon}</span>
          <div>
            <div class="due-review-name">${n.label} <span class="due-review-conf">${confLabel}</span></div>
            <div class="due-review-meta" style="color:${color}">${n.category} · 복습 ${st ? st.reps : 0}회</div>
          </div>
        </div>
        <div class="due-review-right">
          <span class="due-review-date ${st && st.due < today ? 'overdue' : ''}">${dueLabel}</span>
          <button class="quiz-btn small" onclick="quickReview('${n.id}')">복습 완료</button>
        </div>
      </div>`;
  }).join('');
}

/** 복습 큐에서 빠른 복습: 자신감 중간값(3)으로 간격 갱신 */
function quickReview(nodeId) {
  const node = knowledgeData.nodes.find(n => n.id === nodeId);
  if (!node) return;
  // 노트 열고 스스로 체크 후 3점 처리
  if (confirm(`「${node.label}」 복습을 완료했나요?\n\n노트가 열립니다. 확인 후 OK를 누르면 복습 간격이 갱신됩니다.`)) {
    openNotePanel(node);
    applyReviewResult(node, 3);
    initProgress();
  }
}

function buildKnowledgeDist() {
  const container = document.getElementById('knowledge-dist-bars');
  if (!container) return;
  const dist = [0, 0, 0, 0, 0];
  knowledgeData.nodes.forEach(n => {
    dist[Math.min(4, Math.max(0, n.confidence ?? 0))]++;
  });
  const total = knowledgeData.nodes.length || 1;
  const labels = ['모름', '기초', '중급', '심화', '숙달'];
  const colors = ['#64748b', '#f59e0b', '#38bdf8', '#a78bfa', '#34d399'];
  container.innerHTML = dist.map((cnt, i) => `
    <div class="dist-bar-col">
      <div class="dist-bar-num">${cnt}</div>
      <div class="dist-bar-track">
        <div class="dist-bar-fill" style="height:${Math.round(cnt / total * 100)}%;background:${colors[i]}"></div>
      </div>
      <div class="dist-bar-label" style="color:${colors[i]}">${labels[i]}</div>
    </div>
  `).join('');
}

function buildClusterGroups() {
  const container = document.getElementById('cluster-groups-container');
  if (!container) return;

  // Group nodes by their clusters
  const grouped = {};
  Object.keys(CLUSTER_CONFIG).forEach(k => { grouped[k] = []; });

  knowledgeData.nodes.forEach(n => {
    const cluster = getNodeCluster(n);
    if (!grouped[cluster]) grouped[cluster] = [];
    grouped[cluster].push(n);
  });

  container.innerHTML = Object.entries(CLUSTER_CONFIG).map(([key, cfg]) => {
    const nodes = grouped[key] || [];
    const avgConf = nodes.length > 0 ? (nodes.reduce((s, n) => s + n.confidence, 0) / nodes.length) : 0;
    const progressPct = Math.round((avgConf / 4) * 100);

    const chips = nodes.map(n => {
      const catColor = knowledgeData.categories[n.category]?.color || cfg.color;
      const stars = '★'.repeat(n.confidence) + '☆'.repeat(4 - n.confidence);
      return `<span class="cat-concept-chip"
        style="color:${catColor};border-color:${catColor}44;background:${catColor}12;cursor:pointer"
        onclick="focusNode('${n.id}');switchView('graph');document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));document.getElementById('btn-graph').classList.add('active')">
        ${n.label} <span class="confidence-star" style="color:${catColor}">${stars}</span>
      </span>`;
    }).join('');

    return `
      <div class="cluster-progress-item animate-in">
        <div class="cluster-progress-header">
          <div class="cluster-progress-name" style="color:${cfg.color}">
            ${cfg.label} <span style="font-size:12px;color:var(--text-muted);font-weight:400">(${nodes.length}개)</span>
          </div>
          <div class="cluster-progress-pct" style="color:${cfg.color}">${progressPct}%</div>
        </div>
        <div class="cluster-progress-bar-track">
          <div class="cluster-progress-bar-fill" style="width:${progressPct}%;background:linear-gradient(90deg,${cfg.color}88,${cfg.color})"></div>
        </div>
        <div class="cluster-concepts">${chips || '<p style="color:var(--text-muted);font-size:12px;margin:8px 0 0 4px">아직 개념이 등록되지 않았습니다.</p>'}</div>
      </div>
    `;
  }).join('');
}

function buildWeakSpots() {
  const weak = knowledgeData.nodes.filter(n => n.confidence <= 1)
    .sort((a, b) => a.confidence - b.confidence);
  document.getElementById('weak-spots-list').innerHTML = weak.map(n => {
    const color = knowledgeData.categories[n.category]?.color || '#888';
    return `<div class="spot-card weak animate-in" onclick="focusNode('${n.id}');switchView('graph');document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));document.getElementById('btn-graph').classList.add('active')">
      <div class="spot-name">${n.label}</div>
      <div class="spot-confidence" style="color:#f472b6">${CONFIDENCE_STARS[n.confidence]}</div>
      <div class="spot-category" style="color:${color}">${n.category}</div>
    </div>`;
  }).join('') || '<p style="color:var(--text-muted);font-size:13px;padding: 10px 0;">모든 개념이 기초 이상이에요! 👍</p>';
}

function buildStrongList() {
  const strong = knowledgeData.nodes.filter(n => n.confidence >= 3)
    .sort((a, b) => b.confidence - a.confidence || b.studyCount - a.studyCount);
  document.getElementById('strong-list').innerHTML = strong.map(n => {
    const color = knowledgeData.categories[n.category]?.color || '#888';
    return `<div class="spot-card strong animate-in" onclick="focusNode('${n.id}');switchView('graph');document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));document.getElementById('btn-graph').classList.add('active')">
      <div class="spot-name">${n.label}</div>
      <div class="spot-confidence" style="color:#34d399">${CONFIDENCE_STARS[n.confidence]}</div>
      <div class="spot-category" style="color:${color}">${n.category}</div>
    </div>`;
  }).join('') || '<p style="color:var(--text-muted);font-size:13px;padding: 10px 0;">아직 심화 자신감 개념이 등록되지 않았습니다.</p>';
}

/* ============================================================
   AI LEARNING ASSISTANT — 채팅 사이드바
   - DeepSeek API 로 대화 + 구조화된 작업(개념/엣지/노트 추가) 실행
   - 커스텀 데이터는 localStorage 에 저장 → 지식 그래프에 병합
   ============================================================ */
const CUSTOM_DATA_KEY = 'hyeonwoo_custom_v1';
const CHAT_HISTORY_KEY = 'hyeonwoo_chat_history_v1';
let chatHistory = [];
let chatBusy = false;

/* ---------- 커스텀 데이터 (AI가 추가한 개념/엣지/노트) ---------- */
function loadCustomData() {
  try {
    return JSON.parse(localStorage.getItem(CUSTOM_DATA_KEY)) || { nodes: [], edges: [], notes: {} };
  } catch (e) { return { nodes: [], edges: [], notes: {} }; }
}

function saveCustomData(cd) {
  localStorage.setItem(CUSTOM_DATA_KEY, JSON.stringify(cd));
}

/** knowledgeData 에 커스텀 데이터를 병합 (중복 id 는 커스텀이 우선) */
function mergeCustomData() {
  const cd = loadCustomData();
  if (!cd.nodes.length && !cd.edges.length) return;

  // 커스텀 노드: 이미 있으면 덮어쓰기, 없으면 추가
  const existingIds = new Set(knowledgeData.nodes.map(n => n.id));
  cd.nodes.forEach(cn => {
    if (existingIds.has(cn.id)) {
      const idx = knowledgeData.nodes.findIndex(n => n.id === cn.id);
      knowledgeData.nodes[idx] = { ...knowledgeData.nodes[idx], ...cn };
    } else {
      knowledgeData.nodes.push(cn);
      existingIds.add(cn.id);
    }
  });

  // 커스텀 엣지: 같은 source/target/relation 중복 방지
  const edgeKey = e => `${e.source}|${e.target}|${e.relation || ''}`;
  const existingEdges = new Set(knowledgeData.edges.map(edgeKey));
  cd.edges.forEach(ce => {
    if (!existingEdges.has(edgeKey(ce))) {
      knowledgeData.edges.push(ce);
      existingEdges.add(edgeKey(ce));
    }
  });

  // 커스텀 카테고리 색상 보장
  cd.nodes.forEach(cn => {
    if (cn.category && !knowledgeData.categories[cn.category]) {
      const palette = ['#38bdf8', '#a78bfa', '#34d399', '#f472b6', '#fb923c', '#fbbf24', '#06b6d4', '#818cf8'];
      const hash = [...cn.category].reduce((a, c) => a + c.charCodeAt(0), 0);
      knowledgeData.categories[cn.category] = { color: palette[hash % palette.length], icon: '📌' };
    }
  });
}

/** 커스텀 노트 가져오기 (없으면 null) */
function getCustomNote(nodeId) {
  const cd = loadCustomData();
  return cd.notes && cd.notes[nodeId] ? cd.notes[nodeId] : null;
}

/** AI가 생성한 작업 실행: {nodes:[], edges:[], notes:{}, message:""} */
function applyAIActions(actions) {
  const cd = loadCustomData();
  const results = { nodes: 0, edges: 0, notes: 0, skipped: 0 };

  const existingIds = new Set(knowledgeData.nodes.map(n => n.id));
  const firstNewNodeId = [];

  (actions.nodes || []).forEach(n => {
    if (!n.id || !n.label) return;
    // 이미 그래프에 있는 노드 id면 중복 추가하지 않음
    if (existingIds.has(n.id) || cd.nodes.some(cn => cn.id === n.id)) {
      results.skipped++;
      return;
    }
    cd.nodes.push(n);
    results.nodes++;
    firstNewNodeId.push(n.id);
  });
  (actions.edges || []).forEach(e => {
    if (!e.source || !e.target) return;
    // 양쪽 노드가 실제로 존재하는지 확인
    const srcExists = existingIds.has(e.source) || cd.nodes.some(cn => cn.id === e.source);
    const tgtExists = existingIds.has(e.target) || cd.nodes.some(cn => cn.id === e.target);
    if (!srcExists || !tgtExists) return;
    cd.edges.push(e);
    results.edges++;
  });
  (actions.notes || {}).forEach || Object.entries(actions.notes || {}).forEach(([nodeId, md]) => {
    if (!md) return;
    cd.notes[nodeId] = md;
    results.notes++;
  });

  if (results.nodes || results.edges || results.notes) {
    saveCustomData(cd);
    mergeCustomData();
    // 그래프 재렌더 + 통계 갱신
    if (typeof initGraph === 'function') {
      try { initGraph(); } catch (e) { console.warn('graph refresh:', e); }
    }
    updateStatsBadge();
    // 새 노드가 있으면 그래프 뷰로 이동 + 해당 노드 하이라이트
    if (firstNewNodeId.length) {
      setTimeout(() => {
        try { focusNode(firstNewNodeId[0]); } catch (e) { console.warn('focusNode:', e); }
      }, 600);
    }
  }
  return results;
}

/* ---------- 채팅 UI ---------- */
function initChat() {
  const fab = document.getElementById('chat-fab');
  const sendBtn = document.getElementById('chat-send-btn');
  const input = document.getElementById('chat-input');

  fab.addEventListener('click', () => toggleChatSidebar(true));
  sendBtn.addEventListener('click', () => sendChatMessage());
  input.addEventListener('keydown', e => {
    // 한글/일본어 IME 조합 중 Enter는 글자 확정이므로 전송으로 처리하지 않음
    if (e.isComposing || e.keyCode === 229) return;
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });

  updateChatApiStatus();
  loadChatHistory();
}

/** 채팅 사이드바 열기/접기 — open: true 열기, false 접기 */
function toggleChatSidebar(open) {
  const sidebar = document.getElementById('chat-sidebar');
  const fab = document.getElementById('chat-fab');
  const input = document.getElementById('chat-input');

  // body에 chat-open 클래스를 토글 → main 콘텐츠가 오른쪽으로 밀림
  if (open) {
    sidebar.classList.add('open');
    document.body.classList.add('chat-open');
    fab.classList.add('hidden');
    setTimeout(() => input && input.focus(), 250);
  } else {
    sidebar.classList.remove('open');
    document.body.classList.remove('chat-open');
    fab.classList.remove('hidden');
  }
}

function loadChatHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || '[]');
    if (saved.length && saved.length <= 40) chatHistory = saved;
  } catch (e) { chatHistory = []; }

  // 저장된 대화를 화면에 복원
  const box = document.getElementById('chat-messages');
  if (!box) return;
  // 초기 안내 메시지는 유지, 이후 메시지만 복원
  chatHistory.forEach(m => {
    const role = m.role === 'user' ? 'user' : 'assistant';
    appendChatMessage(role, escHtml(m.content));
  });
  box.scrollTop = box.scrollHeight;
}

function saveChatHistory() {
  try {
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatHistory.slice(-40)));
  } catch (e) {}
}

function updateChatApiStatus() {
  // 관리자가 코드에 넣은 기본 키를 사용 (방문자는 변경 불가)
  const el = document.getElementById('chat-api-status');
  if (!el) return;
  el.textContent = '✅ AI 어시스턴트 준비됨';
  el.style.color = '#34d399';
}

function sendChatQuick(text) {
  document.getElementById('chat-input').value = text;
  sendChatMessage();
}

function appendChatMessage(role, html) {
  const box = document.getElementById('chat-messages');
  const wrap = document.createElement('div');
  wrap.className = `chat-msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'chat-msg-bubble';
  bubble.innerHTML = html;
  wrap.appendChild(bubble);
  box.appendChild(wrap);
  // KaTeX 수식 재렌더링 ($$...$$ 및 $...$ 형태 모두 처리)
  if (typeof renderMathInElement !== 'undefined') {
    try {
      renderMathInElement(bubble, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$',  right: '$',  display: false }
        ],
        throwOnError: false
      });
    } catch (e) { /* ignore */ }
  }
  box.scrollTop = box.scrollHeight;
  return wrap;
}

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text || chatBusy) return;

  const apiKey = getQuizApiKey() || 'macbook-local-bridge';

  input.value = '';
  appendChatMessage('user', escHtml(text));
  const thinking = appendChatMessage('assistant', `<span class="chat-thinking">🤔 생각 중...</span>`);
  chatBusy = true;

  try {
    let reply = await fetchChatReply(apiKey, text);
    // 개념 추가/정리 요청인데 AI가 <ACTIONS>를 안 붙였으면 한 번 더 강제 요청
    const isAddRequest = /(추가|개념|정리|노트|만들어줘|등록|넣어)/.test(text);
    if (isAddRequest && !reply.actions) {
      const forced = await fetchChatReply(apiKey,
        `방금 요청(${text.slice(0, 60)}...)에 대한 답변에 <ACTIONS> 블록이 빠졌습니다. ` +
        `꼭 <ACTIONS> 블록으로 새 개념 노드(이미 있는 개념이면 안내만)를 반환하세요.`);
      if (forced.actions) reply = forced;
    }
    thinking.querySelector('.chat-msg-bubble').innerHTML = reply.html;
    chatHistory.push({ role: 'user', content: text });
    chatHistory.push({ role: 'assistant', content: reply.plain });
    saveChatHistory();

    // AI 작업 실행
    if (reply.actions && (reply.actions.nodes?.length || reply.actions.edges?.length || reply.actions.notes)) {
      const r = applyAIActions(reply.actions);
      const parts = [];
      if (r.nodes) parts.push(`개념 ${r.nodes}개`);
      if (r.edges) parts.push(`연결 ${r.edges}개`);
      if (r.notes) parts.push(`노트 ${r.notes}개`);

      if (r.nodes === 0 && r.edges === 0 && r.notes === 0) {
        // 중복 등으로 추가된 게 없음
        appendChatMessage('assistant',
          `ℹ️ <b>이미 그래프에 있는 개념이라 추가하지 않았어요.</b><br/>
           <span style="font-size:12px;color:var(--text-muted)">${escHtml(reply.plain.slice(0, 200))}</span>`);
      } else {
        const newNodeId = (reply.actions.nodes || []).find(n => n.id)?.id || '';
        appendChatMessage('assistant',
          `✅ <b>지식 그래프에 반영 완료!</b> (${parts.join(', ')})
           <div style="margin-top:8px;font-size:12px;color:var(--text-muted)">
             📌 이 내용은 이 브라우저(localStorage)에 저장되어 있어요.<br/>
             💾 영구 저장하려면: <button class="quiz-btn small" onclick="exportCustomData()">💾 파일로 저장 (JSON)</button>
             &nbsp;또는 <button class="quiz-btn small ghost" onclick="openNote('${newNodeId}')">📄 그래프에서 보기</button>
           </div>`);
      }
    }
  } catch (e) {
    console.error('Chat failed:', e);
    thinking.querySelector('.chat-msg-bubble').innerHTML =
      `❌ 오류가 발생했습니다: ${escHtml(e.message || e)}<br/><span style="font-size:12px;color:var(--text-muted)">네트워크나 API 키를 확인해주세요.</span>`;
  } finally {
    chatBusy = false;
  }
}

/** DeepSeek 호출 — 대화 맥락 + 지식 그래프 스키마 제공, 구조화된 작업 응답 파싱 */
async function fetchChatReply(apiKey, userText) {
  // 1. 라즈베리 파이 통합 AI 브릿지 (/api/chat) 호출
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userText }),
      signal: AbortSignal.timeout(15000)
    });
    if (res.ok) {
      const data = await res.json();
      if (data && data.response) {
        const plainText = data.response;
        const htmlText = typeof marked !== 'undefined' ? marked.parse(plainText) : escHtml(plainText);
        return { plain: plainText, html: htmlText, actions: null };
      }
    }
  } catch (err) {
    console.log('Server /api/chat error:', err);
  }

  const nodesSummary = knowledgeData.nodes
    .map(n => `${n.id} (${n.label}, ${n.category}, conf:${n.confidence})`)
    .slice(0, 200).join('\n');
  const edgesSummary = knowledgeData.edges
    .map(e => `${e.source} -[${e.relation || 'related'}]-> ${e.target}`)
    .slice(0, 120).join('\n');

  const system = `당신은 대학원 입시를 준비 중인 현우의 AI 수학/ML 튜터입니다.

## 필수 규칙
1. 서론·인사·"현우님" 호칭·칭찬 절대 금지. 바로 핵심 답변.
2. 수식은 LaTeX만 사용. 인라인: $...$, 블록: $$...$$. 텍스트로 풀어쓰기 금지.
3. 4단계 구조([1.정의] [2.이유] [3.Trade-off] [4.AI연결])는 복잡한 개념에만 사용. 간단한 질문은 2~3문장.
4. 경어체(~합니다, ~입니다) 유지. 한국어로 답변.

## 지식 그래프 스키마
- 노드: { "id": "snake_case", "label": "이름", "category": "카테고리", "confidence": 0~4, "tags": [], "definition": "1단계", "purpose": "2단계", "tradeoff_insight": "3단계", "ai_connection": "4단계" }
- 엣지: { "source": "id", "target": "id", "relation": "basis_of|uses|part_of|leads_to|applied_to", "weight": 1~5 }
- 카테고리: Math, Math & Stats, Algorithm, Systems, RL, Generative, Architecture, Language Model, Multimodal, Training

## 현재 지식 그래프
[노드]
${nodesSummary}
[엣지]
${edgesSummary}

## 개념 추가 시 규칙
- 새 개념 요청 시 응답 끝에 <ACTIONS> 블록 첨부 (기존 id 재사용 금지, 이미 있으면 안내만):
<ACTIONS>
{"nodes":[{...}],"edges":[{...}],"notes":{"<id>":"# 제목\n\n내용"}}
</ACTIONS>
- 일반 질문이면 <ACTIONS> 없이 답변만.
- 장문 원문 붙여넣기 시: 핵심 개념 1개 노드로 통합, 마크다운 노트 상세 작성.
`;

  const msgs = [
    { role: 'system', content: system },
    ...chatHistory.slice(-10),
    { role: 'user', content: userText }
  ];

  const res = await fetch('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: 'deepseek-chat',
      messages: msgs,
      temperature: 0.6,
      max_tokens: 4000
    })
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).error?.message || detail; } catch (e) {}
    throw new Error(detail);
  }

  const data = await res.json();
  const content = data.choices?.[0]?.message?.content || '';

  // <ACTIONS> 블록 파싱
  let actions = null;
  const m = content.match(/<ACTIONS>([\s\S]*?)<\/ACTIONS>/);
  if (m) {
    try {
      actions = JSON.parse(m[1].trim());
    } catch (e) {
      console.warn('ACTIONS 파싱 실패:', m[1].slice(0, 200), e);
    }
  }
  const html = marked.parse(content.replace(/<ACTIONS>[\s\S]*?<\/ACTIONS>/g, ''));

  return { html, plain: content.replace(/<ACTIONS>[\s\S]*?<\/ACTIONS>/g, '').trim(), actions };
}

/** 커스텀 데이터 내보내기 — 사용자가 knowledge.json / data/notes/ 에 반영할 수 있게 */
function exportCustomData() {
  const cd = loadCustomData();
  if (!cd.nodes.length && !cd.edges.length && Object.keys(cd.notes).length === 0) {
    alert('내보낼 커스텀 데이터가 없습니다.');
    return;
  }
  const blob = new Blob([JSON.stringify(cd, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'custom_knowledge_export.json';
  a.click();
  URL.revokeObjectURL(a.href);
  appendChatMessage('assistant',
    `💾 <b>파일로 저장되었습니다!</b> (custom_knowledge_export.json)<br/>
     <span style="font-size:12px;color:var(--text-muted)">
     영구 반영하려면 이 JSON의 nodes/edges를 <code>data/knowledge.json</code>에, notes를 <code>data/notes/&lt;id&gt;.md</code>로 옮기면 됩니다.</span>`);
}

/* ============================================================
   QUIZ VIEW — DeepSeek API 기반 개념 테스트
   ============================================================ */
const QUIZ_STORAGE_KEY = 'hyeonwoo_quiz_history_v1';
let quizTopicNode = null;

// 🔐 관리자 키: 라즈베리 파이 서버에서 이 코드로만 키를 관리.
//    방문자는 UI에서 키를 볼 수도, 바꿀 수도 없습니다.
const DEFAULT_DEEPSEEK_API_KEY = localStorage.getItem('DEEPSEEK_API_KEY') || '';

function getQuizApiKey() {
  return DEFAULT_DEEPSEEK_API_KEY;
}

function initQuiz() {
  // 개념 선택 드롭다운 채우기 (복습 필요한 것 우선)
  const select = document.getElementById('quiz-topic-select');
  if (!select || select.options.length > 1) return;

  loadReviewState();
  const today = todayStr();
  const sorted = [...knowledgeData.nodes].sort((a, b) => {
    const sa = reviewState[a.id]?.due, sb = reviewState[b.id]?.due;
    const aDue = (sa && sa <= today) || !sa ? 0 : 1;
    const bDue = (sb && sb <= today) || !sb ? 0 : 1;
    if (aDue !== bDue) return aDue - bDue;
    return (a.confidence ?? 0) - (b.confidence ?? 0);
  });

  select.innerHTML = '<option value="">-- 복습할 개념을 선택하세요 --</option>' +
    sorted.map(n => {
      const conf = CONFIDENCE_STARS[n.confidence] || '';
      const dueMark = reviewState[n.id] && reviewState[n.id].due <= today ? ' 🔴' : '';
      return `<option value="${n.id}">${n.label} (${n.category}) ${conf}${dueMark}</option>`;
    }).join('');
}

async function generateQuiz() {
  const select = document.getElementById('quiz-topic-select');
  const id = select.value;
  if (!id) { alert('먼저 테스트할 개념을 선택하세요.'); return; }

  quizTopicNode = knowledgeData.nodes.find(n => n.id === id);
  if (!quizTopicNode) return;

  const stage = document.getElementById('quiz-stage');
  const resultBox = document.getElementById('quiz-result');
  resultBox.style.display = 'none';
  stage.innerHTML = `<div class="quiz-placeholder"><p style="font-size:14px;color:var(--text)">⏳ 문제 생성 중... <span style="color:var(--text-muted)">(「${quizTopicNode.label}」 기준)</span></p></div>`;

  const apiKey = getQuizApiKey();
  if (!apiKey) {
    // API 키 없으면 로컬 모드: 노트 기반 셀프 체크
    renderLocalQuiz(stage);
    return;
  }

  try {
    const q = await fetchQuizQuestion(apiKey, quizTopicNode);
    renderQuizQuestion(stage, q);
  } catch (e) {
    console.error('Quiz generation failed:', e);
    stage.innerHTML = `
      <div class="quiz-placeholder">
        <p style="font-size:14px;font-weight:600;color:#f87171">❌ 문제 생성 실패</p>
        <p style="font-size:13px;color:var(--text-muted);margin-top:8px">${escHtml(e.message || e)}</p>
        <p style="font-size:12px;color:var(--text-muted);margin-top:6px">API 키가 올바른지, 네트워크가 정상인지 확인해보세요. API 없이도 <b>로컬 노트 모드</b>로 진행할 수 있습니다.</p>
        <button class="quiz-btn primary" style="margin-top:12px" onclick="renderLocalQuiz(document.getElementById('quiz-stage'))">📄 로컬 노트 모드로 진행</button>
      </div>`;
  }
}

/** DeepSeek API (OpenAI 호환) 로 객관식 문제 1개 생성 */
async function fetchQuizQuestion(apiKey, node) {
  // 노트 내용 로드 (문맥 제공)
  let noteText = '';
  try {
    const res = await fetch(`data/notes/${node.id}.md`);
    if (res.ok) noteText = (await res.text()).slice(0, 3000);
  } catch (e) { /* ignore */ }

  const tags = (node.tags || []).join(', ');
  const system = `당신은 ${node.category} 분야를 가르치는 튜터입니다. 사용자의 지식 수준을 정확히 테스트하는 단일 객관식 문제를 만듭니다.`;

  const prompt = `개념: ${node.label}
카테고리: ${node.category}
태그: ${tags || '없음'}
사용자 자신감: ${node.confidence}/4
${noteText ? `참고 노트:\n${noteText}` : ''}

위 개념에 대한 객관식 문제를 하나 생성하세요. 지식 수준이 낮으면(0~1) 기초 개념을, 높으면(3~4) 심화/미묘한 차이를 묻는 문제를 내세요.
반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 금지):
{"question":"문제 본문","options":["선택지 A","선택지 B","선택지 C","선택지 D"],"answer":0,"explanation":"정답 해설"}

- options는 4개, answer는 정답 인덱스(0~3)
- explanation은 2~3문장으로 개념을 설명
- 문제는 한국어로 작성`;

  const res = await fetch('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: 'deepseek-chat',
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: prompt }
      ],
      temperature: 0.7,
      max_tokens: 600
    })
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).error?.message || detail; } catch (e) {}
    throw new Error(detail);
  }

  const data = await res.json();
  const content = data.choices?.[0]?.message?.content || '';
  // JSON 코드 블록 제거 후 파싱
  const cleaned = content.replace(/```json|```/g, '').trim();
  const parsed = JSON.parse(cleaned);
  if (!parsed.question || !Array.isArray(parsed.options) || parsed.options.length < 2) {
    throw new Error('API 응답 형식이 올바르지 않습니다. 다시 시도해주세요.');
  }
  return parsed;
}

function renderQuizQuestion(stage, q) {
  const node = quizTopicNode;
  const color = knowledgeData.categories[node.category]?.color || '#888';
  const icon  = knowledgeData.categories[node.category]?.icon  || '◉';
  window.__currentQuiz = q; // answerQuiz 에서 사용

  const options = q.options.map((opt, i) =>
    `<button class="quiz-option" data-idx="${i}" onclick="answerQuiz(this, ${i})">
       <span class="quiz-option-letter">${String.fromCharCode(65 + i)}</span>
       <span>${escHtml(opt)}</span>
     </button>`
  ).join('');

  stage.innerHTML = `
    <div class="quiz-question-card">
      <div class="quiz-q-badge" style="color:${color};border-color:${color}44;background:${color}12">
        ${icon} ${node.category} · ${node.label} · 자신감 ${node.confidence}/4
      </div>
      <div class="quiz-question-text">${escHtml(q.question)}</div>
      <div class="quiz-options" id="quiz-options">${options}</div>
    </div>`;
  stage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function answerQuiz(btn, idx) {
  const node = quizTopicNode;
  if (!node) return;
  const quizData = window.__currentQuiz || {};
  const isCorrect = idx === quizData.answer;

  // 모든 옵션 비활성화
  document.querySelectorAll('.quiz-option').forEach(o => {
    o.disabled = true;
    const oi = parseInt(o.dataset.idx);
    if (oi === quizData.answer) o.classList.add('correct');
    if (oi === idx && !isCorrect) o.classList.add('wrong');
  });

  // 결과 카드 표시
  const resultBox = document.getElementById('quiz-result');
  const card = document.getElementById('quiz-result-card');
  const color = knowledgeData.categories[node.category]?.color || '#888';
  card.innerHTML = `
    <div class="quiz-result-head ${isCorrect ? 'ok' : 'no'}">
      ${isCorrect ? '✅ 정답입니다!' : '❌ 오답입니다.'}
    </div>
    <div class="quiz-result-expl">${escHtml(quizData.explanation || '')}</div>
    <div class="quiz-result-score">
      <p style="font-size:13px;color:var(--text-muted);margin-bottom:8px">이 문제를 얼마나 잘 풀었나요? (복습 간격에 반영)</p>
      <div style="display:flex;gap:6px;flex-wrap:wrap">
        ${[0,1,2,3,4].map(s => `
          <button class="quiz-score-btn" style="background:${['#64748b','#f59e0b','#38bdf8','#a78bfa','#34d399'][s]}" onclick="rateQuiz(${s})">
            ${s}점 ${['모름','힘들','애매','거의','완벽'][s]}
          </button>`).join('')}
      </div>
    </div>`;
  resultBox.style.display = 'block';

  // 자동 채점: 정답이면 3~4점, 오답이면 1~2점 기본값
  window.__pendingScore = isCorrect ? 4 : 1;
  resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function rateQuiz(score) {
  const node = quizTopicNode;
  if (!node) return;
  applyReviewResult(node, score);
  // 퀴즈 기록 저장
  const hist = JSON.parse(localStorage.getItem(QUIZ_STORAGE_KEY) || '[]');
  hist.push({ id: node.id, date: todayStr(), score, correct: score >= 3 });
  localStorage.setItem(QUIZ_STORAGE_KEY, JSON.stringify(hist.slice(-200)));

  document.getElementById('quiz-result-card').innerHTML = `
    <div class="quiz-result-head ok">✅ 반영 완료</div>
    <p style="font-size:14px;color:var(--text);margin-top:10px">
      「${node.label}」 자신감: <b>${node.confidence}/4</b> (복습 ${node.studyCount}회)<br>
      <span style="font-size:12px;color:var(--text-muted)">다음 복습: ${reviewState[node.id]?.due || '-'}</span>
    </p>`;
}

/** 로컬 노트 모드: API 없이 노트 읽고 셀프 평가 */
async function renderLocalQuiz(stage) {
  const node = quizTopicNode;
  if (!node) return;
  stage.innerHTML = `<div class="quiz-placeholder"><p style="font-size:14px;color:var(--text)">⏳ 노트 로드 중...</p></div>`;
  let md = `### ${node.label}\n\n개념 노트가 없습니다.`;
  try {
    const res = await fetch(`data/notes/${node.id}.md`);
    if (res.ok) md = await res.text();
  } catch (e) {}
  const color = knowledgeData.categories[node.category]?.color || '#888';
  stage.innerHTML = `
    <div class="quiz-question-card">
      <div class="quiz-q-badge" style="color:${color};border-color:${color}44;background:${color}12">
        ${node.category} · ${node.label} · 노트 기반 셀프 체크
      </div>
      <div class="quiz-local-note">${marked.parse(md)}</div>
      <p style="font-size:13px;color:var(--text-muted);margin:12px 0 10px">노트를 읽고, 이 개념을 얼마나 기억하고 이해했는지 스스로 평가하세요.</p>
      <div style="display:flex;gap:6px;flex-wrap:wrap">
        ${[0,1,2,3,4].map(s => `
          <button class="quiz-score-btn" style="background:${['#64748b','#f59e0b','#38bdf8','#a78bfa','#34d399'][s]}" onclick="rateLocalQuiz(${s})">
            ${s}점 ${['모름','힘들','애매','거의','완벽'][s]}
          </button>`).join('')}
      </div>
    </div>`;
}

function rateLocalQuiz(score) {
  const node = quizTopicNode;
  if (!node) return;
  applyReviewResult(node, score);
  document.getElementById('quiz-result-card').innerHTML = `
    <div class="quiz-result-head ok">✅ 반영 완료</div>
    <p style="font-size:14px;color:var(--text);margin-top:10px">
      「${node.label}」 자신감: <b>${node.confidence}/4</b> (복습 ${node.studyCount}회)<br>
      <span style="font-size:12px;color:var(--text-muted)">다음 복습: ${reviewState[node.id]?.due || '-'}</span>
    </p>`;
  document.getElementById('quiz-result').style.display = 'block';
  document.getElementById('quiz-result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetQuiz() {
  document.getElementById('quiz-stage').innerHTML = `
    <div class="quiz-placeholder">
      <p style="font-size:15px;font-weight:600;color:var(--text)">🧠 개념을 선택하고 문제를 생성해보세요</p>
      <p style="font-size:13px;color:var(--text-muted);margin-top:6px">문제를 풀면 정답 여부에 따라 해당 개념의 자신감(복습 간격)이 자동으로 갱신됩니다.</p>
    </div>`;
  document.getElementById('quiz-result').style.display = 'none';
  document.getElementById('quiz-topic-select').selectedIndex = 0;
  quizTopicNode = null;
}

function escHtml(str) {
  return String(str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* ============================================================
   RESEARCH VIEW
   ============================================================ */
const RESEARCH_STORAGE_KEY = 'hyeonwoo_research_v1';
const STATUS_CYCLE = ['idea', 'exploring', 'active', 'done'];
const STATUS_LABELS = { idea: '💭 아이디어', exploring: '🔍 탐색 중', active: '🔥 진행 중', done: '✅ 완료' };
let researchData = null;
let saveTimer = null;

async function initResearch() {
  // Load from localStorage first, fall back to research.json
  const stored = localStorage.getItem(RESEARCH_STORAGE_KEY);
  if (stored) {
    try { researchData = JSON.parse(stored); }
    catch { researchData = null; }
  }
  if (!researchData) {
    try {
      const res = await fetch('data/research.json');
      researchData = await res.json();
    } catch {
      researchData = { tagline: '', sections: {}, agenda: [], keywords: [] };
    }
  }

  renderResearch();
  bindResearchEditors();
}

function renderResearch() {
  // Initialize Research Graph Canvas
  setTimeout(initResearchGraph, 100);
}

let researchSvg, researchG, researchSimulation, researchZoom;

function initResearchGraph() {
  const container = document.getElementById('research-graph-container');
  const svgEl = document.getElementById('research-graph-svg');
  if (!container || !svgEl) return;

  const W = container.clientWidth || 1000;
  const H = container.clientHeight || 700;

  d3.select(svgEl).selectAll('*').remove();

  researchSvg = d3.select(svgEl).attr('viewBox', `0 0 ${W} ${H}`);

  researchZoom = d3.zoom()
    .scaleExtent([0.2, 3])
    .on('zoom', e => { if (researchG) researchG.attr('transform', e.transform); });
  researchSvg.call(researchZoom);

  researchG = researchSvg.append('g');

  // Defs
  const defs = researchSvg.append('defs');
  defs.append('marker')
    .attr('id', 'research-arrow')
    .attr('viewBox', '0 -5 10 10').attr('refX', 22).attr('refY', 0)
    .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
    .append('path').attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', 'rgba(244,63,94,0.3)');

  // Filter research & paper nodes
  const researchNodes = knowledgeData.nodes.filter(n =>
    n.nodeType === 'paper' || n.nodeType === 'research' || getNodeCluster(n) === '연구'
  );
  const researchNodeIds = new Set(researchNodes.map(n => n.id));

  // Define 5 Research Sub-cluster Centers
  const RESEARCH_SUB_CLUSTERS = {
    'omni': { label: '✨ Omni-modal & Unified', color: '#f43f5e', cx: 0.22, cy: 0.35, ids: ['paper_dynin_omni', 'paper_show_o', 'paper_emu3', 'paper_mini_omni2', 'paper_moshi', 'hcx_omni'] },
    'spatial': { label: '🎭 Spatiotemporal & RVOS', color: '#ec4899', cx: 0.50, cy: 0.28, ids: ['paper_virst', 'paper_lisa', 'paper_mevis', 'paper_momentseeker'] },
    'video': { label: '📹 Long Video & Memory', color: '#fb923c', cx: 0.78, cy: 0.35, ids: ['paper_qwen2_vl', 'streamkv', 'clip', 'paper_llava'] },
    'physical': { label: '🤖 Physical AI & World Model', color: '#38bdf8', cx: 0.32, cy: 0.72, ids: ['paper_cosmos', 'paper_flow_matching'] },
    'rq': { label: '🔭 현우의 핵심 연구 과제 (RQ)', color: '#a78bfa', cx: 0.68, cy: 0.72, ids: ['rq_physical_world_model', 'rq_counterfactual_video_causality', 'rq_modality_decoupled_moe', 'rq_cross_modal_alignment', 'rq_video_temporal_grounding', 'rq_data_recipe_optimization'] }
  };

  const getSubCluster = (nid) => {
    for (const [key, cfg] of Object.entries(RESEARCH_SUB_CLUSTERS)) {
      if (cfg.ids.includes(nid)) return key;
    }
    return 'rq';
  };

  const jitter = () => (Math.random() - 0.5) * 120;
  const nodes = researchNodes.map(n => {
    const sub = getSubCluster(n.id);
    const cfg = RESEARCH_SUB_CLUSTERS[sub];
    return {
      ...n,
      subCluster: sub,
      x: W * cfg.cx + jitter(),
      y: H * cfg.cy + jitter()
    };
  });

  const links = knowledgeData.edges
    .filter(e => researchNodeIds.has(e.source) && researchNodeIds.has(e.target))
    .map(e => ({
      source: e.source, target: e.target,
      relation: e.relation, weight: e.weight || 1, insight: e.insight || null
    }));

  // Hulls & Labels
  const hullGroup = researchG.append('g').attr('class', 'research-hulls');
  const labelGroup = researchG.append('g').attr('class', 'research-labels');
  const hullPaths = {};
  const hullLabels = {};

  Object.entries(RESEARCH_SUB_CLUSTERS).forEach(([key, cfg]) => {
    hullPaths[key] = hullGroup.append('path')
      .attr('fill', cfg.color)
      .attr('fill-opacity', 0.07)
      .attr('stroke', cfg.color)
      .attr('stroke-opacity', 0.35)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,4');

    hullLabels[key] = labelGroup.append('text')
      .attr('text-anchor', 'middle')
      .attr('font-size', '14')
      .attr('font-weight', '700')
      .attr('fill', cfg.color)
      .attr('pointer-events', 'none')
      .text(cfg.label);
  });

  // Force Simulation
  researchSimulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(110))
    .force('charge', d3.forceManyBody().strength(-240))
    .force('collide', d3.forceCollide(45))
    .force('x', d3.forceX(d => W * RESEARCH_SUB_CLUSTERS[d.subCluster].cx).strength(0.35))
    .force('y', d3.forceY(d => H * RESEARCH_SUB_CLUSTERS[d.subCluster].cy).strength(0.35));

  // Render Links
  const linkGroup = researchG.append('g').attr('class', 'research-links');
  const linkElements = linkGroup.selectAll('line')
    .data(links).enter().append('line')
    .attr('stroke', '#f43f5e')
    .attr('stroke-opacity', 0.3)
    .attr('stroke-width', d => Math.max(1.5, d.weight))
    .attr('marker-end', 'url(#research-arrow)');

  // Render Nodes
  const nodeGroup = researchG.append('g').attr('class', 'research-nodes');
  const nodeElements = nodeGroup.selectAll('.r-node')
    .data(nodes).enter().append('g')
    .attr('class', 'r-node')
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', (e, d) => {
        if (!e.active) researchSimulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => {
        if (!e.active) researchSimulation.alphaTarget(0);
        d.fx = null; d.fy = null;
      }))
    .on('click', (e, d) => {
      openNotePanel(d);
    });

  // Circle inside Node
  nodeElements.append('circle')
    .attr('r', d => d.nodeType === 'research' ? 22 : 18)
    .attr('fill', d => d.nodeType === 'research' ? '#ec4899' : '#f43f5e')
    .attr('fill-opacity', 0.85)
    .attr('stroke', '#fff')
    .attr('stroke-width', 2);

  // Icon in Node
  nodeElements.append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .attr('font-size', '12px')
    .text(d => d.nodeType === 'research' ? '🔭' : '📄');

  // Label under Node
  nodeElements.append('text')
    .attr('text-anchor', 'middle')
    .attr('y', 32)
    .attr('font-size', '12px')
    .attr('font-weight', '600')
    .attr('fill', '#f8fafc')
    .text(d => d.label);

  // Tick simulation
  researchSimulation.on('tick', () => {
    linkElements
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);

    nodeElements.attr('transform', d => `translate(${d.x},${d.y})`);

    // Update Hulls
    Object.entries(RESEARCH_SUB_CLUSTERS).forEach(([key, cfg]) => {
      const subNodes = nodes.filter(n => n.subCluster === key);
      if (subNodes.length >= 2) {
        const pts = subNodes.map(n => [n.x, n.y]);
        const polygon = d3.polygonHull(pts);
        if (polygon) {
          hullPaths[key].attr('d', `M${polygon.join('L')}Z`);
          const cx = d3.mean(polygon, p => p[0]);
          const cy = d3.min(polygon, p => p[1]) - 20;
          hullLabels[key].attr('x', cx).attr('y', cy);
        }
      }
    });
  });

  // Bind Zoom controls safely
  const btnIn = document.getElementById('btn-research-zoom-in');
  if (btnIn) btnIn.onclick = () => {
    researchSvg.transition().duration(300).call(researchZoom.scaleBy, 1.3);
  };
  const btnOut = document.getElementById('btn-research-zoom-out');
  if (btnOut) btnOut.onclick = () => {
    researchSvg.transition().duration(300).call(researchZoom.scaleBy, 0.7);
  };
  const btnReset = document.getElementById('btn-research-zoom-reset');
  if (btnReset) btnReset.onclick = () => {
    researchSvg.transition().duration(400).call(researchZoom.transform, d3.zoomIdentity);
  };
}

function bindResearchEditors() {
  // Tagline
  const tl = document.getElementById('research-tagline');
  if (tl) tl.addEventListener('input', () => {
    researchData.tagline = tl.textContent.trim();
    scheduleResearchSave();
  });

  // Flow sections
  ['worldview', 'need', 'value', 'research'].forEach(key => {
    const el = document.getElementById(`section-${key}`);
    if (!el) return;
    el.addEventListener('input', () => {
      if (!researchData.sections) researchData.sections = {};
      researchData.sections[key] = el.textContent;
      scheduleResearchSave();
    });
  });
}

function scheduleResearchSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    researchData.lastSaved = new Date().toISOString();
    localStorage.setItem(RESEARCH_STORAGE_KEY, JSON.stringify(researchData));
    const ind = document.getElementById('research-save-indicator');
    if (ind) {
      ind.classList.add('visible');
      setTimeout(() => ind.classList.remove('visible'), 1800);
    }
  }, 600);
}

/* ---- Agenda ---- */
function renderAgendaList() {
  const list = document.getElementById('agenda-list');
  if (!list) return;
  list.innerHTML = '';
  (researchData.agenda || []).forEach(item => {
    list.appendChild(createAgendaItem(item));
  });
}

function createAgendaItem(item) {
  const el = document.createElement('div');
  el.className = 'agenda-item';
  el.dataset.id = item.id;

  const tagsHtml = (item.tags || []).map(t =>
    `<span class="agenda-tag">${t}</span>`
  ).join('');

  el.innerHTML = `
    <div class="agenda-priority priority-${item.priority || 'medium'}"></div>
    <div class="agenda-content">
      <div class="agenda-item-title" contenteditable="true" spellcheck="false">${item.title || ''}</div>
      <div class="agenda-item-note"  contenteditable="true" spellcheck="false">${item.note  || ''}</div>
      <div class="agenda-item-tags">${tagsHtml}</div>
    </div>
    <div class="agenda-item-meta">
      <span class="agenda-status status-${item.status || 'idea'}" title="클릭해서 상태 변경">${STATUS_LABELS[item.status] || STATUS_LABELS.idea}</span>
      <button class="agenda-delete-btn" title="삭제">✕</button>
    </div>
  `;

  // Title edit
  el.querySelector('.agenda-item-title').addEventListener('input', e => {
    item.title = e.target.textContent;
    scheduleResearchSave();
  });
  // Note edit
  el.querySelector('.agenda-item-note').addEventListener('input', e => {
    item.note = e.target.textContent;
    scheduleResearchSave();
  });
  // Status cycle
  el.querySelector('.agenda-status').addEventListener('click', () => {
    const cur = STATUS_CYCLE.indexOf(item.status || 'idea');
    item.status = STATUS_CYCLE[(cur + 1) % STATUS_CYCLE.length];
    const btn = el.querySelector('.agenda-status');
    btn.className = `agenda-status status-${item.status}`;
    btn.textContent = STATUS_LABELS[item.status];
    // Sync priority dot color based on status
    scheduleResearchSave();
  });
  // Delete
  el.querySelector('.agenda-delete-btn').addEventListener('click', () => {
    researchData.agenda = researchData.agenda.filter(a => a.id !== item.id);
    el.style.transition = 'opacity 0.2s, transform 0.2s';
    el.style.opacity = '0'; el.style.transform = 'translateX(12px)';
    setTimeout(() => el.remove(), 200);
    scheduleResearchSave();
  });

  return el;
}

function addAgendaItem() {
  const newItem = {
    id: Date.now(),
    title: '',
    priority: 'medium',
    status: 'idea',
    tags: [],
    note: ''
  };
  if (!researchData.agenda) researchData.agenda = [];
  researchData.agenda.push(newItem);
  const list = document.getElementById('agenda-list');
  const el = createAgendaItem(newItem);
  list.appendChild(el);
  el.querySelector('.agenda-item-title').focus();
  scheduleResearchSave();
}

/* ---- Keywords ---- */
function renderKeywords() {
  const cloud = document.getElementById('keywords-cloud');
  if (!cloud) return;
  cloud.innerHTML = '';
  (researchData.keywords || []).forEach(kw => {
    cloud.appendChild(createKeywordChip(kw));
  });
}

function createKeywordChip(kw) {
  const chip = document.createElement('div');
  chip.className = 'keyword-chip';
  chip.innerHTML = `<span>${kw}</span><button class="keyword-delete" title="삭제">×</button>`;
  chip.querySelector('.keyword-delete').addEventListener('click', () => {
    researchData.keywords = researchData.keywords.filter(k => k !== kw);
    chip.style.transition = 'opacity 0.15s, transform 0.15s';
    chip.style.opacity = '0'; chip.style.transform = 'scale(0.8)';
    setTimeout(() => chip.remove(), 150);
    scheduleResearchSave();
  });
  return chip;
}

function addKeyword() {
  const kw = prompt('추가할 키워드를 입력하세요:');
  if (!kw || !kw.trim()) return;
  const trimmed = kw.trim();
  if (!researchData.keywords) researchData.keywords = [];
  if (researchData.keywords.includes(trimmed)) return;
  researchData.keywords.push(trimmed);
  const cloud = document.getElementById('keywords-cloud');
  cloud.appendChild(createKeywordChip(trimmed));
  scheduleResearchSave();
}

/* ============================================================
   REVIEW VIEW (플래시카드 복습 시스템)
   ============================================================ */
let reviewQueue = [];
let currentReviewIndex = 0;
let currentFilter = 'all'; // 'all', 'weak', 'random'

function initReview() {
  // Bind filters
  const filterBtns = document.querySelectorAll('.review-filter-btn');
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.dataset.filter;
      restartReview();
    });
  });

  // Bind Card Flip
  const card = document.getElementById('review-card');
  if (card) {
    card.addEventListener('click', () => {
      card.classList.toggle('flipped');
      const rating = document.getElementById('review-rating');
      if (card.classList.contains('flipped')) {
        rating.classList.add('visible');
      } else {
        rating.classList.remove('visible');
      }
    });
  }

  // Bind score buttons
  const scoreBtns = document.querySelectorAll('.rrb');
  scoreBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation(); // prevent card flip when clicking buttons
      const score = parseInt(btn.dataset.score);
      handleReviewRating(score);
    });
  });

  restartReview();
}

function restartReview() {
  const doneEl = document.getElementById('review-done');
  const cardEl = document.getElementById('review-card');
  const ratingEl = document.getElementById('review-rating');
  // Review 뷰가 없는 레이아웃에서는 건너뜀
  if (!doneEl || !cardEl || !ratingEl) return;
  doneEl.classList.remove('visible');
  cardEl.style.display = 'block';
  ratingEl.style.display = 'flex';
  cardEl.classList.remove('flipped');
  ratingEl.classList.remove('visible');

  // Prepare queue
  let list = [...knowledgeData.nodes];

  if (currentFilter === 'weak') {
    // Only nodes with confidence <= 1
    list = list.filter(n => n.confidence <= 1);
  }

  if (currentFilter === 'random') {
    // Shuffle
    list.sort(() => Math.random() - 0.5);
  } else {
    // Priority: Low confidence first, then low study count
    list.sort((a, b) => (a.confidence - b.confidence) || (a.studyCount - b.studyCount));
  }

  reviewQueue = list;
  currentReviewIndex = 0;
  showNextReviewCard();
}

function showNextReviewCard() {
  updateReviewProgress();

  if (currentReviewIndex >= reviewQueue.length || reviewQueue.length === 0) {
    showReviewFinished();
    return;
  }

  const node = reviewQueue[currentReviewIndex];
  const catColor = knowledgeData.categories[node.category]?.color || '#888';
  const catIcon  = knowledgeData.categories[node.category]?.icon  || '';

  // Set card contents
  const catBadge = document.getElementById('review-card-cat');
  const cardName = document.getElementById('review-card-name');
  const cardContent = document.getElementById('review-card-content');

  catBadge.style.background = `${catColor}22`;
  catBadge.style.color = catColor;
  catBadge.style.border = `1px solid ${catColor}44`;
  catBadge.textContent = `${catIcon} ${node.category}`;

  cardName.textContent = node.label;
  
  // Set note placeholder or content
  cardContent.innerHTML = `<div class="note-placeholder">로드 중...</div>`;
  
  fetch(`data/notes/${node.id}.md`)
    .then(res => {
      if (res.ok) return res.text();
      return `### ${node.label}\n\n개념 정보가 없습니다. 지식 지도 탭에서 정보를 추가해주세요.`;
    })
    .then(md => {
      cardContent.innerHTML = marked.parse(md);
      // MathJax/KaTeX render if active
      if (window.renderMathInElement) {
        renderMathInElement(cardContent, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$',  right: '$',  display: false }
          ]
        });
      }
    })
    .catch(() => {
      cardContent.innerHTML = `<p>노트를 가져오지 못했습니다.</p>`;
    });
}

function handleReviewRating(score) {
  const node = reviewQueue[currentReviewIndex];
  
  // Update node metrics local state
  node.confidence = score;
  if (!node.studyCount) node.studyCount = 0;
  node.studyCount++;

  // Save progress dynamically
  saveKnowledgeData();
  
  // Flip card back first
  const card = document.getElementById('review-card');
  card.classList.remove('flipped');
  document.getElementById('review-rating').classList.remove('visible');

  // Short delay for flipping animation before loading next card
  setTimeout(() => {
    currentReviewIndex++;
    showNextReviewCard();
  }, 350);
}

function updateReviewProgress() {
  const total = reviewQueue.length;
  const current = currentReviewIndex;
  
  const percent = total > 0 ? (current / total) * 100 : 0;
  document.getElementById('review-progress-bar').style.width = `${percent}%`;
  document.getElementById('review-progress-label').textContent = `${current} / ${total}`;
}

function showReviewFinished() {
  document.getElementById('review-card').style.display = 'none';
  document.getElementById('review-rating').style.display = 'none';
  
  const doneEl = document.getElementById('review-done');
  doneEl.classList.add('visible');
  
  const statsEl = document.getElementById('review-done-stats');
  statsEl.innerHTML = `
    오늘 총 <strong>${reviewQueue.length}개</strong>의 지식을 복습했습니다!<br>
    복습 결과는 지식 데이터베이스에 안전하게 자동 저장되었습니다.
  `;
}

function saveKnowledgeData() {
  // Update local storage to keep state across sessions
  localStorage.setItem('hyeonwoo_knowledge_v1', JSON.stringify(knowledgeData));
  
  // Trigger general stats updates in progress tab
  if (typeof buildOverviewStats === 'function') buildOverviewStats();
  if (typeof buildWeakSpotsList === 'function') buildWeakSpotsList();
  if (typeof buildStrongList === 'function') buildStrongList();
  if (typeof updateStatsBadge === 'function') updateStatsBadge();
}

/* ============================================================
   COLUMN VIEW (신문 아카이브 및 가판대 기능)
   ============================================================ */
let columnsData = [];

async function initColumn() {
  // Load columns metadata
  try {
    const res = await fetch('data/columns.json');
    const data = await res.json();
    columnsData = data.columns || [];
  } catch (e) {
    console.error('Failed to load columns.json:', e);
    columnsData = [];
  }

  // Set today's date in newspaper brand style
  const dateEl = document.getElementById('newspaper-date');
  if (dateEl) {
    const d = new Date();
    const options = { year: 'numeric', month: '2-digit', day: '2-digit', weekday: 'short' };
    dateEl.textContent = d.toLocaleDateString('ko-KR', options).toUpperCase();
  }

  // Render newspaper shelf cards
  renderColumnRack();

  // Close button overlay bind
  const closeBtn = document.getElementById('reader-close-btn');
  if (closeBtn) {
    closeBtn.addEventListener('click', closeColumnReader);
  }
  
  // Close reader on background click
  const overlay = document.getElementById('column-reader-overlay');
  if (overlay) {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeColumnReader();
    });
  }
}

function renderColumnRack() {
  const grid = document.getElementById('newspaper-grid');
  if (!grid) return;
  
  if (columnsData.length === 0) {
    grid.innerHTML = '<p style="color:var(--text-muted);font-size:13px;grid-column:1/-1;text-align:center">아직 발행된 칼럼이 없습니다.</p>';
    return;
  }

  grid.innerHTML = columnsData.map(col => {
    return `
      <div class="newspaper-card" onclick="openColumnReader('${col.id}')">
        <span class="npc-category">${col.category}</span>
        <h2 class="npc-title">${col.title}</h2>
        <p class="npc-summary">${col.summary}</p>
        <div class="npc-meta">
          <span>✍️ ${col.author}</span>
          <span>📅 ${col.date} • ${col.readTime}</span>
        </div>
      </div>
    `;
  }).join('');
}

async function openColumnReader(columnId) {
  const col = columnsData.find(c => c.id === columnId);
  if (!col) return;

  const overlay = document.getElementById('column-reader-overlay');
  const titleEl = document.getElementById('reader-title');
  const authorEl = document.getElementById('reader-author');
  const metaEl = document.getElementById('reader-meta');
  const bodyEl = document.getElementById('reader-body');

  titleEl.textContent = col.title;
  authorEl.textContent = `BY ${col.author}`;
  metaEl.textContent = `PUBLISHED ON ${col.date.replace(/-/g, '. ')} • ${col.category.toUpperCase()}`;
  
  bodyEl.innerHTML = `<p style="text-align:center;color:var(--text-muted)">신문을 인쇄 중입니다...</p>`;
  overlay.classList.add('open');

  try {
    const res = await fetch(col.file);
    if (!res.ok) throw new Error('File not found');
    const md = await res.text();
    window.currentColumnRawMd = md;
    
    // Parse markdown (exclude the main title since it is already rendered in header)
    const cleanMd = md.replace(/^#\s+.+$/m, '');
    bodyEl.innerHTML = marked.parse(cleanMd);

    // Apply KaTeX math rendering if equations exist
    if (window.renderMathInElement) {
      renderMathInElement(bodyEl, {
        output: 'html',
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$',  right: '$',  display: false }
        ]
      });
    }
  } catch (e) {
    bodyEl.innerHTML = `<p style="text-align:center;color:#ef4444">칼럼을 불러오지 못했습니다. 파일 경로를 확인해 주세요. (${col.file})</p>`;
  }
}

function closeColumnReader() {
  const overlay = document.getElementById('column-reader-overlay');
  if (overlay) overlay.classList.remove('open');
}


