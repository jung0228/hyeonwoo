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
  'Math & Stats':   'ML',       // 확률/통계
  'Math':           '선형대수', // 선형대수학 독립 클러스터!
  'LinearAlgebra':  '선형대수', // 선형대수학 독립 클러스터!
  'Systems':        '시스템',
  'Algorithm':      '알고리즘'
};

// 노드별 개별 오버라이드 (카테고리 매핑보다 우선)
const NODE_CLUSTER_OVERRIDE = {
  'rlhf': 'AI'   // RLHF는 LLM alignment → 딥러닝 쪽
};

// 헬퍼: 노드의 최종 클러스터 반환
function getNodeCluster(node) {
  if (node.id && node.id.startsWith('linear_algebra')) return '선형대수';
  return NODE_CLUSTER_OVERRIDE[node.id] || CLUSTER_MAP[node.category] || 'AI';
}

// 5개 클러스터, 왼쪽→오른쪽: 시스템 / 선형대수 / ML / AI / 알고리즘
const CLUSTER_CONFIG = {
  '시스템':   { color: '#ef4444', label: '💻  시스템',        cx: 0.08, cy: 0.48 },
  '선형대수': { color: '#f59e0b', label: '📐  선형대수학',    cx: 0.28, cy: 0.48 },
  'ML':       { color: '#34d399', label: '📊  머신러닝',      cx: 0.48, cy: 0.48 },
  'AI':       { color: '#a78bfa', label: '🤖  딥러닝',        cx: 0.70, cy: 0.48 },
  '알고리즘': { color: '#06b6d4', label: '🔢  알고리즘',      cx: 0.90, cy: 0.48 }
};

/* ============================================================
   BOOT
   ============================================================ */
document.addEventListener('DOMContentLoaded', async () => {
  // Load from localStorage first, then fallback to fetch
  const stored = localStorage.getItem('hyeonwoo_knowledge_v1');
  if (stored) {
    try {
      const parsed = JSON.parse(stored);
      if (parsed && Array.isArray(parsed.nodes) && parsed.nodes.length > 0) {
        knowledgeData = parsed;
      }
    } catch (e) {
      knowledgeData = null;
    }
  }

  if (!knowledgeData) {
    try {
      const res = await fetch('data/knowledge.json');
      knowledgeData = await res.json();
    } catch (e) {
      console.error('Failed to load knowledge.json:', e);
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
      'Math & Stats':   { color: '#34d399', icon: '📊' },
      'Math':           { color: '#34d399', icon: '📐' },
      'Systems':        { color: '#ef4444', icon: '💻' },
      'Algorithm':      { color: '#06b6d4', icon: '🔢' }
    };
  }
  if (!knowledgeData.sessions) knowledgeData.sessions = [];

  initNav();
  initSearch();
  initGraph();
  initHeatmap();
  initProgress();
  initResearch();
  initReview();
  initColumn();
  updateStatsBadge();
});

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
        if (sn && tn && CLUSTER_MAP[sn.category] !== CLUSTER_MAP[tn.category])
          return 260;
        return 100 - (d.weight || 1) * 10;
      })
      .strength(d => {
        const sn = typeof d.source === 'object' ? d.source : nodeMap[d.source];
        const tn = typeof d.target === 'object' ? d.target : nodeMap[d.target];
        if (sn && tn && CLUSTER_MAP[sn.category] !== CLUSTER_MAP[tn.category])
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
async function openNotePanel(nodeData) {
  currentNodeId = nodeData.id;
  const panel   = document.getElementById('note-panel');
  const titleEl = document.getElementById('note-panel-title');
  const metaEl  = document.getElementById('note-panel-meta');
  const bodyEl  = document.getElementById('note-panel-body');

  const catColor = knowledgeData.categories[nodeData.category]?.color || '#888';
  const catIcon  = knowledgeData.categories[nodeData.category]?.icon  || '';
  const cluster  = CLUSTER_MAP[nodeData.category] || '';
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
    const res = await fetch(`data/notes/${nodeData.id}.md`);
    if (res.ok) {
      const md = await res.text();
      content = `<div class="note-content">${marked.parse(md)}</div>`;
    } else {
      content = `<div class="note-content note-placeholder" style="min-height:200px">
        <div class="note-placeholder-icon">📝</div>
        <p><strong>${nodeData.label}</strong>에 대한 노트가 아직 없어요.</p>
        <p style="font-size:12px;color:var(--text-muted)">data/notes/${nodeData.id}.md 파일을 만들어 채워 주세요!</p>
      </div>`;
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
            const cCluster = CLUSTER_MAP[c.node.category];
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
function initProgress() {
  buildClusterGroups();
  buildWeakSpots();
  buildStrongList();
}

function buildClusterGroups() {
  const container = document.getElementById('cluster-groups-container');
  if (!container) return;

  // Group nodes by their clusters
  const grouped = {
    '시스템': [],
    'ML': [],
    'AI': [],
    '알고리즘': []
  };

  knowledgeData.nodes.forEach(n => {
    const cluster = getNodeCluster(n);
    if (grouped[cluster]) {
      grouped[cluster].push(n);
    }
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
  // Tagline
  const tl = document.getElementById('research-tagline');
  if (tl) tl.textContent = researchData.tagline || '';

  // Flow card sections
  const keys = ['worldview', 'need', 'value', 'research'];
  keys.forEach(key => {
    const el = document.getElementById(`section-${key}`);
    if (el) el.textContent = researchData.sections?.[key] || '';
  });

  // Agenda
  renderAgendaList();

  // Keywords
  renderKeywords();
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
  document.getElementById('review-done').classList.remove('visible');
  document.getElementById('review-card').style.display = 'block';
  document.getElementById('review-rating').style.display = 'flex';
  document.getElementById('review-card').classList.remove('flipped');
  document.getElementById('review-rating').classList.remove('visible');

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
    
    // Parse markdown (exclude the main title since it is already rendered in header)
    // We strip the first '# Title' if present in md file
    const cleanMd = md.replace(/^#\s+.+$/m, '');
    bodyEl.innerHTML = marked.parse(cleanMd);

    // Apply KaTeX math rendering if equations exist
    if (window.renderMathInElement) {
      renderMathInElement(bodyEl, {
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


