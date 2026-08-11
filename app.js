/* ============================================================
   정현우의 지식 지도 — App Logic
   ============================================================ */

/* ── State ── */
let knowledgeData = null;
let simulation = null;
let svg = null, g = null, zoom = null;
let currentNodeId = null;
let activeCategory = null;

/* ── Confidence Labels ── */
const CONFIDENCE_LABELS = ['모름', '기초', '중급', '심화', '전문가'];
const CONFIDENCE_STARS = ['☆☆☆☆', '★☆☆☆', '★★☆☆', '★★★☆', '★★★★'];

/* ── Relation Labels ── */
const RELATION_LABELS = {
  conceptual_root: '개념 근원',
  derived: '파생',
  comparison: '비교',
  contains: '구성 요소',
  applied_to: '응용',
  scaled_up: '스케일업',
  finetuned_with: '파인튜닝',
  extended_to: '확장',
  input_to: '입력',
  enables_generation: '생성 가능',
  instantiated_as: '구현체',
  key_component: '핵심 요소',
  is_a: '종류',
  enables: '가능하게 함',
  used_in: '사용됨',
  uses_architecture: '아키텍처 사용',
  enables_efficient: '효율화'
};

/* ============================================================
   BOOT
   ============================================================ */
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('data/knowledge.json');
    knowledgeData = await res.json();
  } catch (e) {
    console.error('Failed to load knowledge.json:', e);
    knowledgeData = { nodes: [], edges: [], categories: {}, sessions: [] };
  }

  initNav();
  initSearch();
  initGraph();
  initHeatmap();
  initProgress();
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
  const input = document.getElementById('search-input');
  const results = document.getElementById('search-results');

  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (!q || q.length < 1) { results.classList.remove('show'); return; }
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
    if (!e.target.closest('.search-box') && !e.target.closest('.search-results')) {
      results.classList.remove('show');
    }
  });
}

function updateStatsBadge() {
  document.getElementById('total-nodes').textContent = knowledgeData.nodes.length;
}

/* ============================================================
   KNOWLEDGE GRAPH (D3.js)
   ============================================================ */
function initGraph() {
  const container = document.getElementById('graph-container');
  const svgEl = document.getElementById('graph-svg');
  const W = container.clientWidth;
  const H = container.clientHeight;

  svg = d3.select(svgEl)
    .attr('viewBox', `0 0 ${W} ${H}`);

  // Zoom
  zoom = d3.zoom()
    .scaleExtent([0.2, 3])
    .on('zoom', e => { g.attr('transform', e.transform); });
  svg.call(zoom);

  g = svg.append('g');

  // Defs — arrowhead
  const defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 22).attr('refY', 0)
    .attr('markerWidth', 6).attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', 'rgba(255,255,255,0.15)');

  // Build link & node data
  const nodeMap = {};
  knowledgeData.nodes.forEach(n => { nodeMap[n.id] = n; });

  const links = knowledgeData.edges.map(e => ({
    source: e.source,
    target: e.target,
    relation: e.relation,
    weight: e.weight || 1,
    insight: e.insight || null
  }));

  const nodes = knowledgeData.nodes.map(n => ({ ...n }));

  // Force simulation
  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links)
      .id(d => d.id)
      .distance(d => 120 - d.weight * 15)
      .strength(d => 0.3 + d.weight * 0.05))
    .force('charge', d3.forceManyBody().strength(-280))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collide', d3.forceCollide().radius(d => getNodeRadius(d) + 12));

  // Category filter click handler
  buildLegend(knowledgeData.categories);

  // Links
  const linkGroup = g.append('g').attr('class', 'links');

  const linkLine = linkGroup.selectAll('line')
    .data(links).enter().append('line')
    .attr('class', 'link-line')
    .attr('stroke', d => {
      const src = nodeMap[typeof d.source === 'object' ? d.source.id : d.source];
      return knowledgeData.categories[src?.category]?.color || '#555';
    })
    .attr('stroke-width', d => 0.5 + d.weight * 0.4)
    .attr('stroke-dasharray', d => d.relation === 'comparison' ? '4,3' : null)
    .attr('marker-end', 'url(#arrow)');

  // Insight labels on edges
  const insightLabels = linkGroup.selectAll('.insight-label')
    .data(links.filter(d => d.insight)).enter()
    .append('text')
    .attr('class', 'link-insight-badge')
    .attr('text-anchor', 'middle')
    .attr('dy', '-4')
    .attr('font-size', '8')
    .attr('fill', 'rgba(56,189,248,0.7)')
    .attr('font-family', 'Pretendard, sans-serif')
    .text(d => {
      const text = d.insight;
      return text.length > 20 ? text.slice(0, 20) + '…' : text;
    })
    .style('opacity', 0)
    .style('pointer-events', 'none');

  // Nodes
  const nodeGroup = g.append('g').attr('class', 'nodes');

  const node = nodeGroup.selectAll('g')
    .data(nodes).enter().append('g')
    .attr('class', 'node-g')
    .attr('id', d => `node-${d.id}`)
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', dragStart)
      .on('drag', dragged)
      .on('end', dragEnd));

  // Glow filter per category
  const filterIds = {};
  Object.entries(knowledgeData.categories).forEach(([cat, val]) => {
    const fid = `glow-${cat.replace(/\s+/g, '_')}`;
    filterIds[cat] = fid;
    const filter = defs.append('filter').attr('id', fid);
    filter.append('feGaussianBlur')
      .attr('stdDeviation', '3').attr('result', 'blur');
    const merge = filter.append('feMerge');
    merge.append('feMergeNode').attr('in', 'blur');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');
  });

  // Circle
  node.append('circle')
    .attr('class', 'node-circle')
    .attr('r', d => getNodeRadius(d))
    .attr('fill', d => {
      const color = knowledgeData.categories[d.category]?.color || '#888';
      return color;
    })
    .attr('fill-opacity', d => 0.15 + (d.confidence / 4) * 0.6)
    .attr('stroke', d => knowledgeData.categories[d.category]?.color || '#888')
    .attr('stroke-width', 1.5)
    .attr('stroke-opacity', 0.6);

  // Inner dot
  node.append('circle')
    .attr('r', 3)
    .attr('fill', d => knowledgeData.categories[d.category]?.color || '#888')
    .attr('fill-opacity', 0.9);

  // Label
  node.append('text')
    .attr('class', 'node-label')
    .attr('text-anchor', 'middle')
    .attr('dy', d => getNodeRadius(d) + 12)
    .attr('font-size', '11')
    .attr('font-family', 'Pretendard, sans-serif')
    .attr('font-weight', '500')
    .attr('fill', 'rgba(232,234,240,0.85)')
    .text(d => d.label);

  // Tooltip
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

    // Highlight connected
    highlightNeighbors(d.id, linkLine, node);
    insightLabels.style('opacity', l => {
      const srcId = typeof l.source === 'object' ? l.source.id : l.source;
      const tgtId = typeof l.target === 'object' ? l.target.id : l.target;
      return (srcId === d.id || tgtId === d.id) ? 1 : 0;
    });
  })
  .on('mousemove', event => {
    tooltip.style.left = (event.offsetX + 14) + 'px';
    tooltip.style.top = (event.offsetY - 10) + 'px';
  })
  .on('mouseout', () => {
    tooltip.classList.remove('visible');
    resetHighlight(linkLine, node);
    insightLabels.style('opacity', 0);
  })
  .on('click', (event, d) => {
    event.stopPropagation();
    openNotePanel(d);
  });

  // Simulation tick
  simulation.on('tick', () => {
    linkLine
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    insightLabels
      .attr('x', d => (d.source.x + d.target.x) / 2)
      .attr('y', d => (d.source.y + d.target.y) / 2);

    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  // Zoom controls
  document.getElementById('btn-zoom-in').addEventListener('click', () => {
    svg.transition().duration(300).call(zoom.scaleBy, 1.3);
  });
  document.getElementById('btn-zoom-out').addEventListener('click', () => {
    svg.transition().duration(300).call(zoom.scaleBy, 0.77);
  });
  document.getElementById('btn-zoom-reset').addEventListener('click', () => {
    svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity);
  });

  // Close panel on svg click
  svg.on('click', () => closeNotePanel());

  // Note close button
  document.getElementById('note-close').addEventListener('click', closeNotePanel);
}

function getNodeRadius(d) {
  return 12 + (d.confidence || 0) * 4 + (d.studyCount || 0) * 1.2;
}

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
      const srcCat = typeof d.source === 'object' ? d.source.category : null;
      const tgtCat = typeof d.target === 'object' ? d.target.category : null;
      return (srcCat === cat || tgtCat === cat) ? 0.6 : 0.05;
    });
}

function highlightNeighbors(nodeId, linkLine, node) {
  const neighborIds = new Set([nodeId]);
  knowledgeData.edges.forEach(e => {
    if (e.source === nodeId || (typeof e.source === 'object' && e.source.id === nodeId)) neighborIds.add(typeof e.target === 'object' ? e.target.id : e.target);
    if (e.target === nodeId || (typeof e.target === 'object' && e.target.id === nodeId)) neighborIds.add(typeof e.source === 'object' ? e.source.id : e.source);
  });
  node.style('opacity', d => neighborIds.has(d.id) ? 1 : 0.2);
  linkLine.style('stroke-opacity', d => {
    const src = typeof d.source === 'object' ? d.source.id : d.source;
    const tgt = typeof d.target === 'object' ? d.target.id : d.target;
    return (src === nodeId || tgt === nodeId) ? 0.8 : 0.05;
  });
}

function resetHighlight(linkLine, node) {
  node.style('opacity', 1);
  linkLine.style('stroke-opacity', 0.3);
}

function focusNode(nodeId) {
  const nodeData = knowledgeData.nodes.find(n => n.id === nodeId);
  if (!nodeData) return;
  const nodeEl = d3.select(`#node-${nodeId}`);
  if (nodeEl.empty()) return;
  const transform = d3.select('#graph-container svg g').attr('transform') || '';
  openNotePanel(nodeData);
  // Pulse animation on the node
  nodeEl.select('circle')
    .transition().duration(300)
    .attr('r', getNodeRadius(nodeData) * 1.4)
    .transition().duration(300)
    .attr('r', getNodeRadius(nodeData));
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
  const panel = document.getElementById('note-panel');
  const titleEl = document.getElementById('note-panel-title');
  const metaEl = document.getElementById('note-panel-meta');
  const bodyEl = document.getElementById('note-panel-body');

  const catColor = knowledgeData.categories[nodeData.category]?.color || '#888';
  const catIcon = knowledgeData.categories[nodeData.category]?.icon || '';

  titleEl.textContent = nodeData.label;
  metaEl.innerHTML = `
    <span class="note-panel-badge" style="background:${catColor}22;color:${catColor};border:1px solid ${catColor}44">${catIcon} ${nodeData.category}</span>
    <span style="color:var(--accent-yellow)">${CONFIDENCE_STARS[nodeData.confidence] || ''}</span>
    <span style="color:var(--text-muted)">복습 ${nodeData.studyCount || 0}회</span>
  `;

  panel.classList.add('open');

  // Load note markdown
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
  } catch (e) {
    content = `<div class="note-placeholder" style="min-height:200px">
      <p>노트를 불러올 수 없습니다.</p>
    </div>`;
  }

  // Connected concepts
  const connected = getConnectedNodes(nodeData.id);
  let connectedHtml = '';
  if (connected.length) {
    connectedHtml = `
      <div class="connected-concepts">
        <div class="connected-title">연결된 개념 (${connected.length}개)</div>
        <div class="connected-list">
          ${connected.map(c => {
            const color = knowledgeData.categories[c.node.category]?.color || '#888';
            return `<div class="connected-item" onclick="focusNode('${c.node.id}');openNotePanel(knowledgeData.nodes.find(n=>n.id==='${c.node.id}'))">
              <span class="connected-dot" style="background:${color}"></span>
              <div class="connected-info">
                <div class="connected-label">${c.node.label}</div>
                ${c.edge.insight ? `<div class="connected-insight">${c.edge.insight}</div>` : ''}
                <div class="connected-relation">${RELATION_LABELS[c.edge.relation] || c.edge.relation || ''}</div>
              </div>
            </div>`;
          }).join('')}
        </div>
      </div>`;
  }

  bodyEl.innerHTML = content + connectedHtml;

  // Render KaTeX
  if (window.renderMathInElement) {
    renderMathInElement(bodyEl, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false }
      ]
    });
  }
}

function closeNotePanel() {
  document.getElementById('note-panel').classList.remove('open');
  currentNodeId = null;
}

function getConnectedNodes(nodeId) {
  const result = [];
  knowledgeData.edges.forEach(e => {
    const srcId = typeof e.source === 'object' ? e.source.id : e.source;
    const tgtId = typeof e.target === 'object' ? e.target.id : e.target;
    if (srcId === nodeId) {
      const n = knowledgeData.nodes.find(n => n.id === tgtId);
      if (n) result.push({ node: n, edge: e, direction: 'out' });
    } else if (tgtId === nodeId) {
      const n = knowledgeData.nodes.find(n => n.id === srcId);
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
  const grid = document.getElementById('heatmap-grid');
  const monthsEl = document.getElementById('heatmap-months');
  grid.innerHTML = '';
  monthsEl.innerHTML = '';

  const sessionMap = {};
  knowledgeData.sessions.forEach(s => {
    sessionMap[s.date] = s;
  });

  // Build 16 weeks (112 days) ending today
  const today = new Date();
  const startDate = new Date(today);
  startDate.setDate(today.getDate() - 112 + 1);
  // Align to Monday
  const dayOfWeek = (startDate.getDay() + 6) % 7; // 0=Mon
  startDate.setDate(startDate.getDate() - dayOfWeek);

  const weeks = [];
  const cur = new Date(startDate);
  while (cur <= today || weeks.length < 1 || weeks[weeks.length - 1].length < 7) {
    if (weeks.length === 0 || weeks[weeks.length - 1].length === 7) {
      weeks.push([]);
    }
    weeks[weeks.length - 1].push(new Date(cur));
    cur.setDate(cur.getDate() + 1);
    if (weeks.length > 16 && weeks[weeks.length - 1].length === 7) break;
  }

  // Month labels
  let lastMonth = null;
  weeks.forEach(week => {
    const firstOfWeek = week[0];
    const month = firstOfWeek.getMonth();
    if (month !== lastMonth) {
      lastMonth = month;
      const span = document.createElement('span');
      span.className = 'heatmap-month-label';
      span.textContent = firstOfWeek.toLocaleDateString('ko-KR', { month: 'short' });
      monthsEl.appendChild(span);
    } else {
      const span = document.createElement('span');
      span.className = 'heatmap-month-label';
      span.textContent = '';
      monthsEl.appendChild(span);
    }
  });

  // Cells
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
      cell.title = session
        ? `${dateStr}: ${session.topics.length}개 주제 — ${session.note || ''}`
        : dateStr;
      if (session) {
        cell.addEventListener('click', () => {
          scrollToSession(dateStr);
        });
      }
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
      const icon = knowledgeData.categories[cat]?.icon || '';
      const pct = Math.round((count / maxCount) * 100);
      return `<div class="cat-bar-item">
        <div class="cat-bar-label">${icon} ${cat}</div>
        <div class="cat-bar-track">
          <div class="cat-bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <div class="cat-bar-count">${count}회</div>
      </div>`;
    }).join('');
}

function scrollToSession(dateStr) {
  const el = document.getElementById(`session-${dateStr}`);
  if (el) {
    switchView('heatmap');
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('btn-heatmap').classList.add('active');
    setTimeout(() => {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el.style.outline = '2px solid var(--accent-purple)';
      setTimeout(() => el.style.outline = '', 2000);
    }, 300);
  }
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
  buildSummaryCards();
  buildCategoryProgress();
  buildWeakSpots();
  buildStrongList();
}

function buildSummaryCards() {
  const nodes = knowledgeData.nodes;
  const avgConf = nodes.reduce((s, n) => s + (n.confidence || 0), 0) / nodes.length;
  const totalSessions = knowledgeData.sessions.length;
  const highConf = nodes.filter(n => n.confidence >= 3).length;
  const lowConf = nodes.filter(n => n.confidence <= 1).length;

  const cards = [
    { icon: '🧠', value: nodes.length, label: '총 개념 수' },
    { icon: '📚', value: totalSessions, label: '학습 세션' },
    { icon: '⭐', value: highConf, label: '숙련 개념 (중급↑)' },
    { icon: '🔴', value: lowConf, label: '보완 필요 개념' },
    { icon: '📈', value: (avgConf / 4 * 100).toFixed(0) + '%', label: '평균 자신감' },
    { icon: '🔗', value: knowledgeData.edges.length, label: '개념 연결' }
  ];

  const gradients = [
    'linear-gradient(135deg, #8b5cf6, #6366f1)',
    'linear-gradient(135deg, #38bdf8, #0ea5e9)',
    'linear-gradient(135deg, #34d399, #10b981)',
    'linear-gradient(135deg, #f472b6, #ec4899)',
    'linear-gradient(135deg, #fbbf24, #f59e0b)',
    'linear-gradient(135deg, #fb923c, #f97316)'
  ];

  document.getElementById('summary-cards').innerHTML = cards.map((c, i) => `
    <div class="summary-card animate-in" style="animation-delay:${i * 0.08}s">
      <div class="card-icon">${c.icon}</div>
      <div class="card-value" style="background:${gradients[i]};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">${c.value}</div>
      <div class="card-label">${c.label}</div>
    </div>
  `).join('');
}

function buildCategoryProgress() {
  const el = document.getElementById('category-progress-list');
  const grouped = {};
  knowledgeData.nodes.forEach(n => {
    if (!grouped[n.category]) grouped[n.category] = [];
    grouped[n.category].push(n);
  });

  el.innerHTML = Object.entries(grouped)
    .sort((a, b) => {
      const avgA = a[1].reduce((s, n) => s + n.confidence, 0) / a[1].length;
      const avgB = b[1].reduce((s, n) => s + n.confidence, 0) / b[1].length;
      return avgB - avgA;
    })
    .map(([cat, nodes]) => {
      const color = knowledgeData.categories[cat]?.color || '#888';
      const icon = knowledgeData.categories[cat]?.icon || '';
      const avg = nodes.reduce((s, n) => s + (n.confidence || 0), 0) / nodes.length;
      const pct = Math.round((avg / 4) * 100);
      const chips = nodes.map(n => {
        const stars = '★'.repeat(n.confidence) + '☆'.repeat(4 - n.confidence);
        return `<span class="cat-concept-chip" 
          style="color:${color};border-color:${color}44;background:${color}12;cursor:pointer"
          onclick="focusNode('${n.id}');switchView('graph');document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));document.getElementById('btn-graph').classList.add('active')">
          ${n.label} <span class="confidence-star" style="color:${color}">${stars}</span>
        </span>`;
      }).join('');
      return `<div class="cat-progress-item">
        <div class="cat-progress-header">
          <div class="cat-progress-name">
            <span class="cat-progress-icon">${icon}</span>${cat}
            <span style="font-size:11px;color:var(--text-muted);font-weight:400">(${nodes.length}개)</span>
          </div>
          <div class="cat-progress-pct">${pct}%</div>
        </div>
        <div class="cat-progress-bar-track">
          <div class="cat-progress-bar-fill" style="width:${pct}%;background:linear-gradient(90deg,${color}88,${color})"></div>
        </div>
        <div class="cat-concepts">${chips}</div>
      </div>`;
    }).join('');
}

function buildWeakSpots() {
  const weak = knowledgeData.nodes
    .filter(n => n.confidence <= 1)
    .sort((a, b) => a.confidence - b.confidence);

  document.getElementById('weak-spots-list').innerHTML = weak.map(n => {
    const color = knowledgeData.categories[n.category]?.color || '#888';
    return `<div class="spot-card weak animate-in" onclick="focusNode('${n.id}');switchView('graph');document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));document.getElementById('btn-graph').classList.add('active')">
      <div class="spot-name">${n.label}</div>
      <div class="spot-confidence" style="color:#f472b6">${CONFIDENCE_STARS[n.confidence]}</div>
      <div class="spot-category" style="color:${color}">${n.category}</div>
    </div>`;
  }).join('') || '<p style="color:var(--text-muted);font-size:13px">모든 개념이 기초 이상이에요! 👍</p>';
}

function buildStrongList() {
  const strong = knowledgeData.nodes
    .filter(n => n.confidence >= 3)
    .sort((a, b) => b.confidence - a.confidence || b.studyCount - a.studyCount);

  document.getElementById('strong-list').innerHTML = strong.map(n => {
    const color = knowledgeData.categories[n.category]?.color || '#888';
    return `<div class="spot-card strong animate-in" onclick="focusNode('${n.id}');switchView('graph');document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));document.getElementById('btn-graph').classList.add('active')">
      <div class="spot-name">${n.label}</div>
      <div class="spot-confidence" style="color:#34d399">${CONFIDENCE_STARS[n.confidence]}</div>
      <div class="spot-category" style="color:${color}">${n.category}</div>
    </div>`;
  }).join('');
}
