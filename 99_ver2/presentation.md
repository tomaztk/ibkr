---
marp: true
theme: default
paginate: true
html: true
footer: "⬡ NEXUS CO"
style: |
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --accent:       #2563EB;
    --accent-light: #EFF6FF;
    --accent-mid:   #BFDBFE;
    --text:         #111827;
    --muted:        #6B7280;
    --border:       #E5E7EB;
    --surface:      #FFFFFF;
    --bg:           #F9FAFB;
  }

  /* ── Base ───────────────────────────────── */
  section {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 52px 68px 88px;
    display: flex;
    flex-direction: column;
    font-size: 18px;
  }

  /* ── Footer / logo ──────────────────────── */
  footer {
    position: absolute;
    bottom: 24px;
    left: 52px;
    right: auto;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    font-family: 'DM Sans', sans-serif;
    background: var(--accent-light);
    padding: 4px 10px;
    border-radius: 4px;
  }

  /* ── Page number ────────────────────────── */
  section::after {
    position: absolute;
    bottom: 24px;
    right: 52px;
    font-size: 10px;
    color: var(--muted);
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.05em;
  }

  /* ── Headings ───────────────────────────── */
  h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.3em;
    color: var(--text);
    margin: 0 0 0.25em;
    line-height: 1.2;
  }

  h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75em;
    color: var(--text);
    margin: 0 0 0.6em;
    padding-bottom: 0.35em;
    border-bottom: 2px solid var(--accent);
  }

  h3 { font-size: 0.95em; margin: 0 0 0.5em; }

  /* ── Code ───────────────────────────────── */
  code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82em;
    background: #F3F4F6;
    padding: 2px 7px;
    border-radius: 4px;
    color: #1D4ED8;
  }

  pre {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74em;
    background: #0F172A;
    color: #CBD5E1;
    border-radius: 10px;
    padding: 22px 26px;
    line-height: 1.75;
    margin: 0;
  }

  pre code {
    background: none;
    color: inherit;
    padding: 0;
    font-size: inherit;
  }

  /* ── Lists ──────────────────────────────── */
  ul { padding-left: 1.3em; line-height: 1.75; }
  li { margin-bottom: 0.25em; }
  li li { margin-top: 0.15em; font-size: 0.93em; color: #374151; }
  li li li { font-size: 0.9em; color: var(--muted); }

  /* ── Utility classes ────────────────────── */
  .tag {
    display: inline-block;
    background: var(--accent);
    color: white;
    font-size: 0.62em;
    padding: 3px 12px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 14px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    border-top: 4px solid var(--accent);
    padding: 22px 20px;
  }

  .card-label {
    font-size: 0.68em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent);
    margin: 0 0 10px;
  }

  .card-body {
    font-size: 0.78em;
    color: #374151;
    line-height: 1.75;
    margin: 0;
  }

  .row-card {
    display: flex;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 20px;
    gap: 18px;
  }

  .row-label {
    font-size: 1.2em;
    font-weight: 700;
    color: var(--accent);
    min-width: 32px;
  }

  .divider {
    width: 1px;
    height: 34px;
    background: var(--border);
    flex-shrink: 0;
  }

  .callout {
    background: var(--accent-light);
    border-left: 3px solid var(--accent);
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 0.74em;
    line-height: 1.5;
  }

  .callout strong { color: var(--accent); display: block; margin-bottom: 3px; }
---


<!-- ═══════════════════════════════════════ SLIDE 1 ═══ -->

<span class="tag">Typography</span>

# The Art of Clear Communication

<div style="display:flex; gap:28px; margin-top:28px;">

  <div style="flex:1; text-align:left; border-left:3px solid var(--accent); padding-left:18px;">
    <p style="font-size:0.7em; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted); margin:0 0 8px;">Left — DM Sans Light</p>
    <p style="font-weight:300; font-size:0.88em; line-height:1.65; margin:0; color:#374151;">Body copy flows naturally in a light weight. Ideal for longer passages where readability is paramount.</p>
  </div>

  <div style="flex:1; text-align:center; border-left:3px solid var(--border); border-right:3px solid var(--border); padding:0 18px;">
    <p style="font-size:0.7em; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted); margin:0 0 8px;">Center — DM Serif Display</p>
    <p style="font-family:'DM Serif Display',serif; font-size:1.35em; line-height:1.3; margin:0; color:var(--accent); font-style:italic;">Elegant display type for headlines</p>
  </div>

  <div style="flex:1; text-align:right; border-right:3px solid var(--accent); padding-right:18px;">
    <p style="font-size:0.7em; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted); margin:0 0 8px;">Right — JetBrains Mono</p>
    <code style="font-size:0.8em; display:inline-block; margin-top:4px;">const x = "monospace";</code>
  </div>

</div>

<div style="margin-top:26px; background:var(--accent-light); border-radius:8px; padding:16px 24px; display:flex; justify-content:space-between; align-items:center;">
  <span style="font-size:0.68em; letter-spacing:0.22em; text-transform:uppercase; color:var(--muted);">Small Caps</span>
  <span style="font-family:'DM Serif Display',serif; font-size:1.5em; color:var(--text);">36pt Display</span>
  <span style="font-weight:300; font-size:0.82em; color:var(--muted);">Thin 300</span>
  <strong style="font-size:0.82em; font-weight:600;">SemiBold 600</strong>
  <span style="font-family:'DM Serif Display',serif; font-style:italic; font-size:0.85em; color:var(--accent);">Italic Serif</span>
</div>

---

<!-- ═══════════════════════════════════════ SLIDE 2 ═══ -->

## Visual Gallery

<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:16px;">

  <div style="text-align:center;">
    <img src="https://picsum.photos/seed/architecture42/300/200" style="width:100%; border-radius:7px; display:block; object-fit:cover; height:150px;" />
    <p style="font-size:0.7em; color:var(--text); font-weight:600; margin:10px 0 3px;">Urban Architecture</p>
    <p style="font-size:0.63em; color:var(--muted); margin:0;">São Paulo, Brazil</p>
  </div>

  <div style="text-align:center;">
    <img src="https://picsum.photos/seed/forest99/300/200" style="width:100%; border-radius:7px; display:block; object-fit:cover; height:150px;" />
    <p style="font-size:0.7em; color:var(--text); font-weight:600; margin:10px 0 3px;">Ancient Forest</p>
    <p style="font-size:0.63em; color:var(--muted); margin:0;">Pacific Northwest</p>
  </div>

  <div style="text-align:center;">
    <img src="https://picsum.photos/seed/ocean77/300/200" style="width:100%; border-radius:7px; display:block; object-fit:cover; height:150px;" />
    <p style="font-size:0.7em; color:var(--text); font-weight:600; margin:10px 0 3px;">Ocean at Dusk</p>
    <p style="font-size:0.63em; color:var(--muted); margin:0;">Mediterranean Sea</p>
  </div>

  <div style="text-align:center;">
    <img src="https://picsum.photos/seed/alpine55/300/200" style="width:100%; border-radius:7px; display:block; object-fit:cover; height:150px;" />
    <p style="font-size:0.7em; color:var(--text); font-weight:600; margin:10px 0 3px;">Summit Ridge</p>
    <p style="font-size:0.63em; color:var(--muted); margin:0;">Swiss Alps</p>
  </div>

</div>

---

<!-- ═══════════════════════════════════════ SLIDE 3 ═══ -->

## Code Walkthrough

<div style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">

  <div style="flex:1.3;">

```python
def fetch_data(url: str, retries: int = 3):
    session = requests.Session()
    adapter = HTTPAdapter(
        max_retries=Retry(total=retries)
    )
    session.mount("https://", adapter)

    response = session.get(url, timeout=10)
    response.raise_for_status()
    return response.json()
```

  </div>

  <div style="flex:1; display:flex; flex-direction:column; justify-content:space-around; gap:11px; padding-top:6px;">

  <div class="callout">
    <strong>← Type hint: <code>url: str</code></strong>
    Enforces string input at static analysis time — caught before runtime.
  </div>

  <div class="callout">
    <strong>← HTTPAdapter + Retry</strong>
    Automatically retries on transient network failures or 5xx errors.
  </div>

  <div class="callout">
    <strong>← timeout=10</strong>
    Prevents the request from hanging indefinitely on slow servers.
  </div>

  <div class="callout">
    <strong>← raise_for_status()</strong>
    Converts HTTP 4xx / 5xx codes into Python exceptions cleanly.
  </div>

  </div>

</div>

---

<!-- ═══════════════════════════════════════ SLIDE 4 ═══ -->

## Key Principles

- **Clarity** — Design for understanding, not complexity
  - Use whitespace generously; content needs room to breathe
  - Limit each slide to a single core idea
- **Consistency** — Establish rules and follow them every time
  - Repeating visual patterns build immediate familiarity
  - Consistent spacing creates rhythm across all slides
  - Maintain a limited, purposeful colour palette
- **Hierarchy** — Guide the eye through contrast and weight
  - Size and weight signal importance at a glance
  - Colour draws attention to the single most critical point
    - Use the accent colour sparingly for maximum impact
    - Never nest more than three levels deep in a list
- **Brevity** — Respect your audience's attention
  - One slide, one message — cut everything else

---

<!-- ═══════════════════════════════════════ SLIDE 5 ═══ -->

## Vertical Perspectives

<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:18px; margin-top:18px; flex:1;">

  <div class="card">
    <p class="card-label">① Discover</p>
    <p class="card-body">Research the problem space thoroughly before proposing solutions. Gather data, interview stakeholders, and map out the competitive landscape to build shared understanding.</p>
  </div>

  <div class="card">
    <p class="card-label">② Define</p>
    <p class="card-body">Synthesise your findings into clear, testable problem statements. Align the team around a shared definition of success and documented acceptance criteria.</p>
  </div>

  <div class="card">
    <p class="card-label">③ Deliver</p>
    <p class="card-body">Build, test, and iterate in short cycles. Ship to real users early, measure impact with quantitative metrics, and continuously refine until the goal is achieved.</p>
  </div>

</div>

---

<!-- ═══════════════════════════════════════ SLIDE 6 ═══ -->

## Horizontal Roadmap

<div style="display:flex; flex-direction:column; gap:13px; margin-top:18px; flex:1; justify-content:center;">

  <div class="row-card">
    <span class="row-label">Q1</span>
    <div class="divider"></div>
    <div>
      <strong style="font-size:0.88em;">Foundation</strong><br>
      <span style="font-size:0.75em; color:var(--muted);">Set up infrastructure, define architecture, onboard the core team and establish development workflows and tooling conventions.</span>
    </div>
  </div>

  <div class="row-card">
    <span class="row-label">Q2</span>
    <div class="divider"></div>
    <div>
      <strong style="font-size:0.88em;">Alpha Launch</strong><br>
      <span style="font-size:0.75em; color:var(--muted);">Ship the MVP to a closed beta group, collect qualitative feedback, resolve critical bugs, and iterate on the core feature set.</span>
    </div>
  </div>

  <div class="row-card">
    <span class="row-label">Q3</span>
    <div class="divider"></div>
    <div>
      <strong style="font-size:0.88em;">Scale</strong><br>
      <span style="font-size:0.75em; color:var(--muted);">Optimise performance for growth, expand to new markets, and build self-serve onboarding flows with automated support systems.</span>
    </div>
  </div>

  <div class="row-card">
    <span class="row-label">Q4</span>
    <div class="divider"></div>
    <div>
      <strong style="font-size:0.88em;">Consolidate</strong><br>
      <span style="font-size:0.75em; color:var(--muted);">Harden the platform, invest in retention and monetisation levers, and plan the product roadmap for the next annual cycle.</span>
    </div>
  </div>

</div>
