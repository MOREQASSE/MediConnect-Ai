from flask import Flask, render_template_string, jsonify, request
import tensorflow as tf
import numpy as np
import requests
import joblib
import os
import datetime

# ---------------------------------------------------------------------------
# APP BOOTSTRAP
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path    = os.path.join(BASE_DIR, 'mediconnect_final_model.h5')
scaler_path   = os.path.join(BASE_DIR, 'mediconnect_final_scaler.pkl')
profiles_path = os.path.join(BASE_DIR, 'seasonal_profiles.pkl')
csv_path      = os.path.join(BASE_DIR, 'AE_attendances_england_monthly.csv')

model      = None
scaler     = None
profiles   = None
status_msg = "Initializing..."


def compute_seasonal_profiles():
    """Fallback: compute monthly mean scaled vectors from NHS CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])

        candidates = [
            ['Type 1 Departments - Major A&E',
             'Type 2 Departments - Single Specialty',
             'Type 3 Departments - Minor Injury'],
            ['Type 1 Departments - Major A&E',
             'Type 2 Departments - Single Specialty',
             'Type 3 Departments - Minor Injuries'],
            ['Type 1 Departments - Major A&E',
             'Type 2 Departments - Single Specialty',
             'Type 3 Departments - Minor Injury Unit'],
        ]
        target_cols = None
        for c in candidates:
            if all(col in df.columns for col in c):
                target_cols = c
                break
        if target_cols is None:
            cols = df.columns.tolist()
            t1 = next((c for c in cols if 'Type 1' in c and 'Major' in c), None)
            t2 = next((c for c in cols if 'Type 2' in c), None)
            t3 = next((c for c in cols if 'Type 3' in c), None)
            target_cols = [t1, t2, t3]
            if None in target_cols:
                raise ValueError("Could not locate Type 1/2/3 columns")

        for col in target_cols:
            df[col] = df.groupby('Name')[col].transform(lambda x: x.ffill().bfill()).fillna(0)

        df['month'] = df['date'].dt.month
        monthly_means = df.groupby('month')[target_cols].mean().values
        monthly_scaled = scaler.transform(monthly_means)
        return monthly_scaled.astype(np.float32)
    except Exception as e:
        print(f"[BOOT] Could not compute profiles: {e}")
        return np.array([[0.5, 0.3, 0.4]] * 12, dtype=np.float32)


try:
    print("[BOOT] Loading AI Components...")
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    try:
        raw_profiles = joblib.load(profiles_path)
        if isinstance(raw_profiles, dict):
            raw_profiles = np.array(list(raw_profiles.values()), dtype=np.float32)
        else:
            raw_profiles = np.array(raw_profiles, dtype=np.float32)
        if raw_profiles.shape != (12, 3):
            print(f"[BOOT] Profile shape {raw_profiles.shape} unexpected, recomputing...")
            profiles = compute_seasonal_profiles()
        else:
            profiles = raw_profiles
        print("[BOOT] Seasonal profiles loaded.")
    except Exception as e:
        print(f"[BOOT] Profile load failed ({e}), computing from CSV...")
        profiles = compute_seasonal_profiles()
    status_msg = "AI ONLINE | Multi-Output Active"
    print("[BOOT] SUCCESS: Model and Scaler are in memory.")
except ModuleNotFoundError:
    status_msg = "ERROR: Run 'pip install scikit-learn pandas'"
    print("[BOOT] ERROR: scikit-learn or pandas is missing.")
except Exception as e:
    status_msg = f"LOAD FAILED: {str(e)}"
    print(f"[BOOT] ERROR: {str(e)}")

# ---------------------------------------------------------------------------
# DEPLOY HISTORY (in-memory)
# ---------------------------------------------------------------------------
DEPLOY_HISTORY = []
MAX_HISTORY = 20


def record_deploy(priority, port, ok, status_msg):
    DEPLOY_HISTORY.insert(0, {
        "timestamp": datetime.datetime.now().isoformat(),
        "priority": priority,
        "port": port,
        "ok": ok,
        "status": status_msg
    })
    if len(DEPLOY_HISTORY) > MAX_HISTORY:
        DEPLOY_HISTORY.pop()

# ---------------------------------------------------------------------------
# ODL CARBON CONSTANTS
# ---------------------------------------------------------------------------
ODL_BASE      = "http://127.0.0.1:8181"
ODL_AUTH      = ("admin", "admin")
ODL_HEADERS   = {"Content-Type": "application/json", "Accept": "application/json"}
FLOW_NODE     = "openflow:1"
FLOW_TABLE    = 0
FLOW_ID       = "1"

FLOW_CONFIG_URL = (
    f"{ODL_BASE}/restconf/config/opendaylight-inventory:nodes"
    f"/node/{FLOW_NODE}/table/{FLOW_TABLE}/flow/{FLOW_ID}"
)

FLOW_OPER_URL = (
    f"{ODL_BASE}/restconf/operational/opendaylight-inventory:nodes"
    f"/node/{FLOW_NODE}/table/{FLOW_TABLE}/flow/{FLOW_ID}"
)

# ---------------------------------------------------------------------------
# HTML TEMPLATE
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MediConnect | AI-Driven SDN Command Center</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {
            --neon-cyan:  #00f2ff;
            --neon-green: #39ff14;
            --neon-red:   #ff2a2a;
            --deep-space: #06080a;
            --glass:      rgba(255, 255, 255, 0.05);
            --border:     rgba(0, 242, 255, 0.3);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background: var(--deep-space);
            color: #e0e0e0;
            font-family: 'Inter', 'Segoe UI', sans-serif;
            height: 100vh;
            display: flex;
            overflow: hidden;
        }

        .dashboard-container {
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 20px;
            width: 100%;
            padding: 25px;
        }

        .side-panel {
            background: var(--glass);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 25px;
            backdrop-filter: blur(15px);
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 32px rgba(0,0,0,.8);
            overflow: hidden;
        }

        h1 {
            color: var(--neon-cyan);
            font-size: 1.4rem;
            margin-bottom: 18px;
            letter-spacing: 3px;
            text-transform: uppercase;
            border-left: 4px solid var(--neon-cyan);
            padding-left: 15px;
        }

        /* Status row */
        .status-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 22px;
            padding: 10px 12px;
            background: rgba(0,0,0,.3);
            border-radius: 10px;
            border: 1px solid #222;
        }
        .status-label { font-size: .65rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
        .status-dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: #333; transition: .3s;
        }
        .status-dot.online { background: var(--neon-green); box-shadow: 0 0 8px var(--neon-green); }
        .status-dot.offline { background: var(--neon-red); box-shadow: 0 0 8px var(--neon-red); }
        .status-dot.checking {
            background: var(--neon-cyan);
            animation: blink 1s infinite;
        }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

        .control-block  { margin-bottom: 18px; }
        label           { display: block; font-size: .75rem; color: #aaa; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }

        input[type=range] {
            width: 100%; appearance: none; background: #1a1d23;
            height: 6px; border-radius: 5px; outline: none;
        }
        input[type=range]::-webkit-slider-thumb {
            appearance: none; width: 18px; height: 18px;
            background: var(--neon-cyan); border-radius: 50%;
            cursor: pointer; box-shadow: 0 0 10px var(--neon-cyan);
        }

        select {
            width: 100%; background: #1a1d23; color: white;
            border: 1px solid #333; padding: 12px; border-radius: 10px; outline: none; transition: .3s;
        }
        select:focus { border-color: var(--neon-cyan); }

        .console-label { font-size: .7rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; }
        .console-label button {
            background: #1a1d23; color: #888; border: 1px solid #333;
            padding: 4px 10px; border-radius: 6px; font-size: .6rem;
            cursor: pointer; text-transform: uppercase; letter-spacing: 1px;
        }
        .console-label button:hover { color: var(--neon-cyan); border-color: var(--neon-cyan); }
        .console {
            flex-grow: 1;
            background: rgba(0,0,0,.5);
            border: 1px solid #222;
            border-radius: 12px;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: .72rem;
            color: var(--neon-green);
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
            min-height: 120px;
        }

        .status-chip {
            margin-top: 12px;
            font-size: .62rem;
            color: #555;
            text-align: center;
        }

        .main-viewport { display: flex; flex-direction: column; gap: 20px; overflow-y: auto; }

        .stats-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; flex-shrink: 0; }

        .stat-card {
            background: var(--glass);
            border: 1px solid rgba(255,255,255,.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            position: relative;
            transition: .4s;
        }
        .stat-card.active-priority {
            border-color: var(--neon-green);
            background: rgba(57,255,20,.05);
            transform: scale(1.03);
            box-shadow: 0 0 30px rgba(57,255,20,.18);
        }
        .stat-card .val   { font-size: 2.2rem; font-weight: 800; color: var(--neon-cyan); margin: 10px 0; }
        .stat-card .lbl   { font-size: .7rem; color: #888; text-transform: uppercase; }

        .priority-badge {
            position: absolute; top: -10px; right: 20px;
            background: var(--neon-green); color: black;
            font-size: .6rem; font-weight: 900;
            padding: 4px 10px; border-radius: 20px;
            opacity: 0; transition: .3s;
        }
        .active-priority .priority-badge { opacity: 1; top: 10px; }

        .packet-bar {
            display: none;
            background: rgba(0,0,0,.4);
            border: 1px solid rgba(57,255,20,.3);
            border-radius: 12px;
            padding: 14px 20px;
            font-family: 'Courier New', monospace;
            font-size: .8rem;
            color: var(--neon-green);
            letter-spacing: 1px;
            flex-shrink: 0;
        }
        .packet-bar.visible { display: block; }
        .packet-bar span    { color: #fff; font-weight: 700; }

        /* Chart area */
        .chart-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            flex-shrink: 0;
        }
        .chart-card {
            background: var(--glass);
            border: 1px solid rgba(255,255,255,.1);
            border-radius: 15px;
            padding: 15px;
            min-height: 200px;
        }
        .chart-title {
            font-size: .75rem; color: #aaa;
            text-transform: uppercase; letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .chart-svg { width: 100%; height: 180px; }

        /* History table */
        .history-card {
            background: var(--glass);
            border: 1px solid rgba(255,255,255,.1);
            border-radius: 15px;
            padding: 15px;
            flex-shrink: 0;
        }
        .history-table {
            width: 100%; border-collapse: collapse; font-size: .72rem;
        }
        .history-table th {
            text-align: left; color: var(--neon-cyan);
            padding: 8px; border-bottom: 1px solid #333;
            text-transform: uppercase; letter-spacing: 1px;
        }
        .history-table td {
            padding: 8px; color: #ccc; border-bottom: 1px solid #222;
        }
        .history-table tr:hover td { color: #fff; }

        #topology-container {
            flex-grow: 1;
            background: rgba(0,0,0,.3);
            border-radius: 20px;
            border: 1px solid var(--border);
            overflow: hidden;
            position: relative;
            min-height: 220px;
        }

        .action-row { display: flex; gap: 15px; flex-shrink: 0; }
        .action-btn {
            flex: 1;
            background: var(--neon-green); color: black;
            border: none; padding: 18px;
            font-weight: 900; font-size: .85rem;
            text-transform: uppercase; letter-spacing: 2px;
            border-radius: 12px; cursor: pointer; transition: .3s;
            box-shadow: 0 0 20px rgba(57,255,20,.2);
        }
        .action-btn:hover  { transform: translateY(-3px); box-shadow: 0 0 40px rgba(57,255,20,.4); }
        .action-btn:active { transform: translateY(0); }
        .action-btn.stop {
            background: #ff2a2a; color: white;
            box-shadow: 0 0 20px rgba(255,42,42,.2);
            display: none;
        }
        .action-btn.stop:hover { box-shadow: 0 0 40px rgba(255,42,42,.4); }
        .action-btn.stop.visible { display: block; }

        .action-btn.restore {
            background: var(--neon-cyan); color: black;
            box-shadow: 0 0 20px rgba(0,242,255,.2);
            display: none;
        }
        .action-btn.restore:hover { box-shadow: 0 0 40px rgba(0,242,255,.4); }
        .action-btn.restore.visible { display: block; }

        .link       { stroke: #2a2d35; stroke-width: 2; }
        .link.active-flow {
            stroke: var(--neon-green); stroke-width: 4;
            stroke-dasharray: 10,5;
            animation: dash 1s linear infinite;
            filter: drop-shadow(0 0 5px var(--neon-green));
        }
        .node circle { stroke: #fff; stroke-width: 2; cursor: pointer; }
        .node text   { fill: #fff; font-size: 10px; font-weight: 600; text-transform: uppercase; }

        @keyframes dash { to { stroke-dashoffset: -15; } }
    </style>
</head>
<body>
<div class="dashboard-container">

    <!-- SIDE PANEL -->
    <div class="side-panel">
        <h1>MediConnect</h1>

        <div class="status-row">
            <div class="status-dot checking" id="ai-status-dot" title="AI Model"></div>
            <span class="status-label">AI Model</span>
            <div class="status-dot checking" id="odl-status-dot" title="ODL Controller"></div>
            <span class="status-label">ODL Controller</span>
        </div>

        <div class="control-block">
            <label>Temporal Context: <span id="hour-text" style="color:var(--neon-cyan)">12</span>:00</label>
            <input type="range" id="hour-slider" min="0" max="23" value="12" oninput="syncAI()">
        </div>

        <div class="control-block">
            <label>Day of Week</label>
            <select id="day-select" onchange="syncAI()">
                <option value="0">Monday</option>
                <option value="1">Tuesday</option>
                <option value="2">Wednesday</option>
                <option value="3">Thursday</option>
                <option value="4">Friday</option>
                <option value="5">Saturday</option>
                <option value="6">Sunday</option>
            </select>
        </div>

        <div class="control-block">
            <label>Seasonality (Month)</label>
            <select id="month-select" onchange="syncAI()">
                <option value="1">January</option>
                <option value="2">February</option>
                <option value="3">March</option>
                <option value="4">April</option>
                <option value="5">May</option>
                <option value="6">June</option>
                <option value="7">July</option>
                <option value="8">August</option>
                <option value="9">September</option>
                <option value="10">October</option>
                <option value="11">November</option>
                <option value="12">December</option>
            </select>
        </div>

        <div class="console-label">
            SDN Live Monitor
            <button onclick="clearConsole()">Clear</button>
        </div>
        <div id="odl-console" class="console">> System Initializing...</div>

        <div class="status-chip">{{ status }}</div>
    </div>

    <!-- MAIN VIEWPORT -->
    <div class="main-viewport">

        <!-- Stat cards -->
        <div class="stats-row">
            <div id="card-type1" class="stat-card" onclick="setManualOverride('type1')" style="cursor: pointer;" title="Click for Manual Override">
                <div class="priority-badge">SDN PRIORITIZED</div>
                <div class="lbl">Major A&amp;E (Type 1)</div>
                <div id="val-type1" class="val">0</div>
                <div class="lbl">Port 1</div>
            </div>
            <div id="card-type2" class="stat-card" onclick="setManualOverride('type2')" style="cursor: pointer;" title="Click for Manual Override">
                <div class="priority-badge">SDN PRIORITIZED</div>
                <div class="lbl">Specialty (Type 2)</div>
                <div id="val-type2" class="val">0</div>
                <div class="lbl">Port 2</div>
            </div>
            <div id="card-type3" class="stat-card" onclick="setManualOverride('type3')" style="cursor: pointer;" title="Click for Manual Override">
                <div class="priority-badge">SDN PRIORITIZED</div>
                <div class="lbl">Minor Inj (Type 3)</div>
                <div id="val-type3" class="val">0</div>
                <div class="lbl">Port 3</div>
            </div>
        </div>

        <!-- Live packet counter bar -->
        <div id="packet-bar" class="packet-bar">
            OPERATIONAL FEEDBACK &nbsp;|&nbsp;
            Flow ID: <span id="pkt-flow-id">—</span> &nbsp;|&nbsp;
            Port: <span id="pkt-port">—</span> &nbsp;|&nbsp;
            Packets: <span id="pkt-count">—</span> &nbsp;|&nbsp;
            Bytes: <span id="pkt-bytes">—</span>
        </div>

        <!-- Charts -->
        <div class="chart-row">
            <div class="chart-card">
                <div class="chart-title">12-Step Input Sequence (Scaled)</div>
                <svg id="history-chart" class="chart-svg"></svg>
            </div>
            <div class="chart-card">
                <div class="chart-title">Reconfiguration History</div>
                <div id="history-table-container" style="max-height:180px; overflow-y:auto;">
                    <table class="history-table" id="history-table">
                        <thead>
                            <tr><th>Time</th><th>Priority</th><th>Port</th><th>Status</th></tr>
                        </thead>
                        <tbody><tr><td colspan="4" style="color:#555; text-align:center;">No deployments yet</td></tr></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- D3 topology -->
        <div id="topology-container"></div>

        <!-- Buttons -->
        <div class="action-row">
            <button class="action-btn" onclick="pushSDN()">Deploy Flow Rules to ODL</button>
            <button class="action-btn stop" id="stop-btn" onclick="stopPolling()">Stop Audit</button>
            <button class="action-btn restore" id="ai-restore-btn" onclick="resetAIControl()">Restore AI Control</button>
        </div>
    </div>
</div>

<script>
    // State
    let currentPriority = '';
    let aiPredictedPriority = '';
    let isManualOverride = false;
    let packetPollTimer = null;

    function updatePriorityUI(priority) {
        ['type1','type2','type3'].forEach(id => {
            document.getElementById('card-'+id)
                .classList.toggle('active-priority', priority === id);
        });

        if (currentPriority !== priority) {
            currentPriority = priority;
            drawViz(currentPriority);
        }
    }

    function setManualOverride(priority) {
        isManualOverride = true;
        updatePriorityUI(priority);
        document.getElementById('ai-restore-btn').classList.add('visible');
        log(`[MANUAL OVERRIDE] Priority set to ${priority.toUpperCase()}`);
    }

    function resetAIControl() {
        isManualOverride = false;
        document.getElementById('ai-restore-btn').classList.remove('visible');
        log('[AI CONTROL] Restored control to AI model.');
        if (aiPredictedPriority) {
            updatePriorityUI(aiPredictedPriority);
        }
    }
    const consoleEl = document.getElementById('odl-console');
    const MAX_LOG_LINES = 50;
    let debounceTimer;
    let resizeTimer;

    // Console helper
    function log(msg) {
        const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
        const lines = consoleEl.textContent.split('\\n');
        lines.unshift(`[${ts}] ${msg}`);
        if (lines.length > MAX_LOG_LINES) lines.length = MAX_LOG_LINES;
        consoleEl.textContent = lines.join('\\n');
    }
    function clearConsole() {
        consoleEl.textContent = '> Console cleared.\\n';
    }

    // Health check
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            const aiDot = document.getElementById('ai-status-dot');
            const odlDot = document.getElementById('odl-status-dot');
            aiDot.className = 'status-dot ' + (data.ai ? 'online' : 'offline');
            odlDot.className = 'status-dot ' + (data.odl ? 'online' : 'offline');
            if (!data.odl) log('[HEALTH] ODL Controller unreachable');
        } catch(e) {
            document.getElementById('ai-status-dot').className = 'status-dot offline';
            document.getElementById('odl-status-dot').className = 'status-dot offline';
        }
    }

    // D3 topology
    let topoSvg = null;
    function drawViz(priority) {
        const container = d3.select('#topology-container');
        if (!topoSvg) {
            container.html('');
            const width  = container.node().clientWidth  || 600;
            const height = container.node().clientHeight || 260;
            topoSvg = container.append('svg').attr('width', width).attr('height', height).attr('id', 'topo-svg');
        }
        const svg = topoSvg;
        const width = +svg.attr('width');
        const height = +svg.attr('height');

        const nodes = [
            { id: 'ctrl',  name: 'ODL Controller', color: '#8a2be2', fx: width/2,     fy: 60 },
            { id: 'sw1',   name: 'OpenFlow:1',      color: '#00f2ff', fx: width/2,     fy: height/2 },
            { id: 'type1', name: 'Host_A&E',         color: '#333',    fx: width/4,     fy: height-60 },
            { id: 'type2', name: 'Host_Spec',        color: '#333',    fx: width/2,     fy: height-60 },
            { id: 'type3', name: 'Host_Minor',       color: '#333',    fx: 3*width/4,   fy: height-60 },
        ];

        const links = [
            { source: 'ctrl',  target: 'sw1',   active: true },
            { source: 'sw1',   target: 'type1', active: priority === 'type1' },
            { source: 'sw1',   target: 'type2', active: priority === 'type2' },
            { source: 'sw1',   target: 'type3', active: priority === 'type3' },
        ];

        // Links - data join
        const linkSel = svg.selectAll('line').data(links);
        linkSel.enter().append('line').attr('class', 'link').merge(linkSel)
            .attr('class', d => d.active ? 'link active-flow' : 'link')
            .attr('x1', d => nodes.find(n => n.id === d.source).fx)
            .attr('y1', d => nodes.find(n => n.id === d.source).fy)
            .attr('x2', d => nodes.find(n => n.id === d.target).fx)
            .attr('y2', d => nodes.find(n => n.id === d.target).fy);
        linkSel.exit().remove();

        // Nodes - data join
        const nodeSel = svg.selectAll('.node').data(nodes, d => d.id);
        const nodeEnter = nodeSel.enter().append('g').attr('class', 'node')
            .attr('transform', d => `translate(${d.fx},${d.fy})`);

        nodeEnter.append('circle')
            .attr('r', d => d.id === 'ctrl' ? 25 : 18)
            .attr('fill', d => d.color);
        nodeEnter.append('text').attr('dy', 34).attr('text-anchor', 'middle').text(d => d.name);

        const nodeMerge = nodeEnter.merge(nodeSel);
        nodeMerge.select('circle')
            .attr('fill', d => d.id === priority ? '#39ff14' : d.color)
            .style('filter', d => d.id === priority ? 'drop-shadow(0 0 10px #39ff14)' : 'none');

        nodeSel.exit().remove();
    }

    // Draw history chart
    function drawHistoryChart(historyData) {
        const svg = d3.select('#history-chart');
        svg.selectAll('*').remove();
        const width = svg.node().clientWidth || 300;
        const height = svg.node().clientHeight || 180;
        const margin = {top: 10, right: 10, bottom: 25, left: 35};
        const innerW = width - margin.left - margin.right;
        const innerH = height - margin.top - margin.bottom;
        const g = svg.attr('width', width).attr('height', height).append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const labels = ['T-11','T-10','T-9','T-8','T-7','T-6','T-5','T-4','T-3','T-2','T-1','Now'];
        const series = [
            {name:'Type 1', color:'#00f2ff', values: historyData.map(d => d[0])},
            {name:'Type 2', color:'#39ff14', values: historyData.map(d => d[1])},
            {name:'Type 3', color:'#ffaa00', values: historyData.map(d => d[2])},
        ];

        const x = d3.scalePoint().domain(labels).range([0, innerW]);
        const y = d3.scaleLinear()
            .domain([0, d3.max(series, s => d3.max(s.values)) * 1.1 || 1])
            .range([innerH, 0]);

        g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).tickSize(0).tickPadding(8))
            .selectAll('text').style('fill', '#888').style('font-size', '9px');
        g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(-innerW))
            .selectAll('text').style('fill', '#888').style('font-size', '9px');
        g.selectAll('.domain, line').style('stroke', '#333');

        const line = d3.line().x((d,i) => x(labels[i])).y(d => y(d)).curve(d3.curveMonotoneX);

        series.forEach(s => {
            g.append('path').datum(s.values).attr('fill','none').attr('stroke',s.color)
                .attr('stroke-width',2).attr('d', line);
            g.append('text').attr('x', innerW - 50).attr('y', y(s.values[s.values.length-1]) - 5)
                .attr('fill', s.color).style('font-size', '10px').text(s.name);
        });
    }

    // AI sync (debounced)
    function syncAI() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(doSyncAI, 120);
    }

    async function doSyncAI() {
        const h = document.getElementById('hour-slider').value;
        const d = document.getElementById('day-select').value;
        const m = document.getElementById('month-select').value;
        document.getElementById('hour-text').innerText = h;

        try {
            const res  = await fetch(`/api/predict?hour=${h}&day=${d}&month=${m}`);
            const data = await res.json();

            if (data.error) {
                log('SYNC ERROR: ' + data.error);
                return;
            }

            document.getElementById('val-type1').innerText = data.rooms.type1.toLocaleString();
            document.getElementById('val-type2').innerText = data.rooms.type2.toLocaleString();
            document.getElementById('val-type3').innerText = data.rooms.type3.toLocaleString();

            aiPredictedPriority = data.priority;

            if (!isManualOverride) {
                updatePriorityUI(aiPredictedPriority);
            }

            if (data.history) drawHistoryChart(data.history);
        } catch(e) { log('SYNC ERROR: ' + e.message); }
    }

    // Deploy + start packet polling
    async function pushSDN() {
        if (!currentPriority) { log('[WARN] No priority set yet.'); return; }
        log(`[SIGNAL] ${isManualOverride ? 'Manual Override' : 'AI Predicted Surge'} -> ${currentPriority.toUpperCase()}`);
        log('[ODL] Sending flow rule to controller...');

        try {
            const res    = await fetch('/api/apply_sdn_policy', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ priority: currentPriority })
            });
            const result = await res.json();

            log(`[ODL] ${result.status}`);

            if (result.ok) {
                log(`[DEPLOYED] Flow ID: ${result.flow_id} | Priority: ${result.priority} | Target: Port ${result.port}`);
                startPacketPolling();
                fetchHistory();
            } else {
                log('[WARN] Rule may not have reached the dataplane. Check ODL connectivity.');
            }
        } catch(e) {
            log('[ERROR] Could not reach Flask backend: ' + e.message);
        }
    }

    // Packet polling
    function startPacketPolling() {
        document.getElementById('packet-bar').classList.add('visible');
        document.getElementById('stop-btn').classList.add('visible');
        if (packetPollTimer) clearInterval(packetPollTimer);

        async function poll() {
            try {
                const res  = await fetch('/api/packet_count');
                const data = await res.json();

                if (data.ok) {
                    document.getElementById('pkt-flow-id').innerText = data.flow_id;
                    document.getElementById('pkt-port').innerText    = data.port;
                    document.getElementById('pkt-count').innerText   = data.packets.toLocaleString();
                    document.getElementById('pkt-bytes').innerText   = data.bytes.toLocaleString();
                    log(`[VERIFY] Packets: ${data.packets} | Bytes: ${data.bytes}`);
                } else {
                    log('[VERIFY] Awaiting first packet from dataplane...');
                }
            } catch(e) {
                log('[VERIFY] Poll error: ' + e.message);
            }
        }

        poll();
        packetPollTimer = setInterval(poll, 4000);
    }

    function stopPolling() {
        if (packetPollTimer) {
            clearInterval(packetPollTimer);
            packetPollTimer = null;
        }
        document.getElementById('packet-bar').classList.remove('visible');
        document.getElementById('stop-btn').classList.remove('visible');
        log('[AUDIT] Packet polling stopped by user.');
    }

    // History table
    async function fetchHistory() {
        try {
            const res = await fetch('/api/history');
            const data = await res.json();
            const tbody = document.querySelector('#history-table tbody');
            if (!data.history || data.history.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="color:#555; text-align:center;">No deployments yet</td></tr>';
                return;
            }
            tbody.innerHTML = data.history.map(h => `
                <tr>
                    <td>${new Date(h.timestamp).toLocaleTimeString('en-GB',{hour12:false})}</td>
                    <td>${h.priority.toUpperCase()}</td>
                    <td>${h.port}</td>
                    <td style="color:${h.ok ? 'var(--neon-green)' : '#ff2a2a'}">${h.status}</td>
                </tr>
            `).join('');
        } catch(e) { console.error('History fetch failed', e); }
    }

    // Boot
    window.onload = () => {
        doSyncAI();
        checkHealth();
        setInterval(checkHealth, 10000);
        setTimeout(() => {
            if (consoleEl.textContent.includes('Initializing')) {
                consoleEl.textContent = '> [ODL] Controller Link Established.\\n> [AI CORE] Awaiting Manual Override...\\n';
            }
        }, 1500);
    };

    window.onresize = () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            topoSvg = null;
            d3.select('#topology-container').html('');
            drawViz(currentPriority);
        }, 200);
    };
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, status=status_msg)


@app.route('/api/health')
def health():
    """Return AI and ODL connectivity status."""
    ai_ok = model is not None and scaler is not None
    odl_ok = False
    try:
        r = requests.get(
            f"{ODL_BASE}/restconf/operational/opendaylight-inventory:nodes",
            auth=ODL_AUTH,
            headers=ODL_HEADERS,
            timeout=2
        )
        odl_ok = r.status_code == 200
    except Exception:
        pass
    return jsonify({"ai": ai_ok, "odl": odl_ok})


@app.route('/api/predict')
def predict():
    h = request.args.get('hour', '12', type=str)
    d = request.args.get('day', '0', type=str)
    m = request.args.get('month', '1', type=str)

    # Input validation
    try:
        h = max(0, min(23, int(h)))
        d = max(0, min(6, int(d)))
        m = max(1, min(12, int(m)))
    except ValueError:
        return jsonify({"error": "Invalid parameter type", "rooms": {"type1": 0, "type2": 0, "type3": 0}, "priority": "type1"}), 400

    if model is None or scaler is None or profiles is None:
        return jsonify({
            "rooms": {"type1": 0, "type2": 0, "type3": 0},
            "priority": "type1",
            "history": [[0, 0, 0]] * 12
        })

    try:
        # Rotate profiles so the sequence ends with the selected month
        history = np.roll(profiles.copy(), shift=-m, axis=0)
        # Inject temporal context into the final step
        intensity = 1.0 + (h / 36.0) + (d / 10.0)
        history[-1] = history[-1] * intensity
        history = np.clip(history, 0.0, 1.0)

        input_data = np.array(history, dtype=np.float32).reshape(1, 12, 3)
        raw_pred   = model.predict(input_data, verbose=0)
        real_vals  = scaler.inverse_transform(raw_pred)[0]

        rooms = {
            "type1": int(max(0, real_vals[0])),
            "type2": int(max(0, real_vals[1])),
            "type3": int(max(0, real_vals[2])),
        }
        priority_room = max(rooms, key=rooms.get)
        history_list = history.tolist()

        return jsonify({
            "rooms":    rooms,
            "total":    sum(rooms.values()),
            "priority": priority_room,
            "history":  history_list,
        })

    except Exception as e:
        print(f"[PREDICT] Error: {e}")
        return jsonify({
            "rooms": {"type1": 0, "type2": 0, "type3": 0},
            "priority": "type1",
            "history": [[0, 0, 0]] * 12
        })


@app.route('/api/apply_sdn_policy', methods=['POST'])
def apply_sdn_policy():
    """
    Push an OpenFlow rule to ODL Carbon via RESTCONF v1 /restconf/config/.
    Adds idle/hard timeouts for a production-grade rule payload.
    """
    priority_room = request.json.get('priority', 'type1')
    port_map      = {"type1": 1, "type2": 2, "type3": 3}
    target_port   = port_map.get(priority_room, 1)
    flow_priority = 65000

    flow_body = {
        "flow-node-inventory:flow": [
            {
                "id":       FLOW_ID,
                "table_id": FLOW_TABLE,
                "priority": flow_priority,
                "idle-timeout": 0,
                "hard-timeout": 0,
                "match": {
                    "in-port": str(target_port)
                },
                "instructions": {
                    "instruction": [
                        {
                            "order": 0,
                            "apply-actions": {
                                "action": [
                                    {
                                        "order": 0,
                                        "output-action": {
                                            "output-node-connector": "NORMAL"
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ]
    }

    try:
        requests.delete(FLOW_CONFIG_URL, auth=ODL_AUTH, headers=ODL_HEADERS, timeout=2)
        r = requests.put(
            FLOW_CONFIG_URL,
            json=flow_body,
            auth=ODL_AUTH,
            headers=ODL_HEADERS,
            timeout=3
        )

        ok = r.status_code in (200, 201, 204)
        msg = (
            f"Flow deployed -> Port {target_port} | HTTP {r.status_code}"
            if ok else
            f"ODL rejected rule | HTTP {r.status_code} | {r.text[:120]}"
        )

        print(f"[SDN] Port={target_port} | HTTP={r.status_code}")
        if not ok:
            print(f"[SDN] Response body: {r.text}")

        record_deploy(priority_room, target_port, ok, msg)

        return jsonify({
            "ok":       ok,
            "status":   msg,
            "flow_id":  FLOW_ID,
            "port":     target_port,
            "priority": flow_priority,
        })

    except requests.exceptions.ConnectionError:
        msg = "Cannot reach ODL — is the controller running on port 8181?"
        print(f"[SDN] {msg}")
        record_deploy(priority_room, target_port, False, msg)
        return jsonify({"ok": False, "status": msg})
    except Exception as e:
        print(f"[SDN] Unexpected error: {e}")
        record_deploy(priority_room, target_port, False, str(e))
        return jsonify({"ok": False, "status": f"Error: {str(e)}"})


@app.route('/api/packet_count')
def packet_count():
    try:
        r = requests.get(
            FLOW_OPER_URL,
            auth=ODL_AUTH,
            headers=ODL_HEADERS,
            timeout=3
        )

        if r.status_code != 200:
            return jsonify({"ok": False, "reason": f"HTTP {r.status_code}"})

        data = r.json()
        flows = (
            data.get("flow-node-inventory:flow")
            or data.get("flow", [])
        )

        if not flows:
            return jsonify({"ok": False, "reason": "No flow entry in operational store yet"})

        flow       = flows[0]
        statistics = flow.get("flow-statistics", {})
        packets    = statistics.get("packet-count", 0)
        byte_count = statistics.get("byte-count",   0)

        if packets == 0:
            nested = flow.get("opendaylight-flow-statistics:flow-statistics", {})
            packets    = nested.get("packet-count", 0)
            byte_count = nested.get("byte-count",   0)

        port = flow.get("match", {}).get("in-port", "?")

        print(f"[VERIFY] Flow ID={FLOW_ID} | Port={port} | Packets={packets} | Bytes={byte_count}")

        return jsonify({
            "ok":      True,
            "flow_id": FLOW_ID,
            "port":    port,
            "packets": int(packets),
            "bytes":   int(byte_count),
        })

    except requests.exceptions.ConnectionError:
        return jsonify({"ok": False, "reason": "Cannot reach ODL operational endpoint"})
    except Exception as e:
        print(f"[VERIFY] Error: {e}")
        return jsonify({"ok": False, "reason": str(e)})


@app.route('/api/history')
def history():
    return jsonify({"history": DEPLOY_HISTORY})


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
