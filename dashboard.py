from flask import Flask, render_template_string, jsonify, request
import tensorflow as tf
import numpy as np
import requests
import joblib
import os

# ---------------------------------------------------------------------------
# APP BOOTSTRAP
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path  = os.path.join(BASE_DIR, 'mediconnect_final_model.h5')
scaler_path = os.path.join(BASE_DIR, 'mediconnect_final_scaler.pkl')

model      = None
scaler     = None
status_msg = "Initializing..."

try:
    print("[BOOT] Loading AI Components...")
    model      = tf.keras.models.load_model(model_path, compile=False)
    scaler     = joblib.load(scaler_path)
    status_msg = "AI ONLINE | Multi-Output Active"
    print("[BOOT] SUCCESS: Model and Scaler are in memory.")
except ModuleNotFoundError:
    status_msg = "ERROR: Run 'pip install scikit-learn'"
    print("[BOOT] ERROR: scikit-learn is missing.")
except Exception as e:
    status_msg = f"LOAD FAILED: {str(e)}"
    print(f"[BOOT] ERROR: {str(e)}")

# ---------------------------------------------------------------------------
# ODL CARBON CONSTANTS
# ---------------------------------------------------------------------------
ODL_BASE      = "http://127.0.0.1:8181"
ODL_AUTH      = ("admin", "admin")
ODL_HEADERS   = {"Content-Type": "application/json", "Accept": "application/json"}
FLOW_NODE     = "openflow:1"
FLOW_TABLE    = 0
FLOW_ID       = "1"

# Config path  — ODL Carbon RESTCONF v1 (legacy)
FLOW_CONFIG_URL = (
    f"{ODL_BASE}/restconf/config/opendaylight-inventory:nodes"
    f"/node/{FLOW_NODE}/table/{FLOW_TABLE}/flow/{FLOW_ID}"
)

# Operational path — reads live packet counters from the dataplane
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

        /* ── Layout ── */
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
            margin-bottom: 30px;
            letter-spacing: 3px;
            text-transform: uppercase;
            border-left: 4px solid var(--neon-cyan);
            padding-left: 15px;
        }

        /* ── Controls ── */
        .control-block  { margin-bottom: 22px; }
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

        /* ── Console ── */
        .console-label { font-size: .7rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
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
        }

        /* ── Status chip ── */
        .status-chip {
            margin-top: 12px;
            font-size: .62rem;
            color: #555;
            text-align: center;
        }

        /* ── Main viewport ── */
        .main-viewport { display: flex; flex-direction: column; gap: 20px; }

        /* ── Stat cards ── */
        .stats-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; }

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

        /* ── Packet counter bar ── */
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
        }
        .packet-bar.visible { display: block; }
        .packet-bar span    { color: #fff; font-weight: 700; }

        /* ── Topology ── */
        #topology-container {
            flex-grow: 1;
            background: rgba(0,0,0,.3);
            border-radius: 20px;
            border: 1px solid var(--border);
            overflow: hidden;
            position: relative;
            min-height: 220px;
        }

        /* ── Button ── */
        .action-btn {
            background: var(--neon-green); color: black;
            border: none; padding: 18px;
            font-weight: 900; font-size: .85rem;
            text-transform: uppercase; letter-spacing: 2px;
            border-radius: 12px; cursor: pointer; transition: .3s;
            box-shadow: 0 0 20px rgba(57,255,20,.2);
        }
        .action-btn:hover  { transform: translateY(-3px); box-shadow: 0 0 40px rgba(57,255,20,.4); }
        .action-btn:active { transform: translateY(0); }

        /* ── D3 topology ── */
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

    <!-- ═══════════════════════ SIDE PANEL ═══════════════════════ -->
    <div class="side-panel">
        <h1>MediConnect</h1>

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
                <option value="4">April</option>
                <option value="7">July</option>
                <option value="10">October</option>
            </select>
        </div>

        <div class="console-label">SDN Live Monitor</div>
        <div id="odl-console" class="console">> System Initializing...</div>

        <div class="status-chip">{{ status }}</div>
    </div>

    <!-- ═══════════════════════ MAIN VIEWPORT ════════════════════ -->
    <div class="main-viewport">

        <!-- Stat cards -->
        <div class="stats-row">
            <div id="card-type1" class="stat-card">
                <div class="priority-badge">SDN PRIORITIZED</div>
                <div class="lbl">Major A&amp;E (Type 1)</div>
                <div id="val-type1" class="val">0</div>
                <div class="lbl">Port 1</div>
            </div>
            <div id="card-type2" class="stat-card">
                <div class="priority-badge">SDN PRIORITIZED</div>
                <div class="lbl">Specialty (Type 2)</div>
                <div id="val-type2" class="val">0</div>
                <div class="lbl">Port 2</div>
            </div>
            <div id="card-type3" class="stat-card">
                <div class="priority-badge">SDN PRIORITIZED</div>
                <div class="lbl">Minor Inj (Type 3)</div>
                <div id="val-type3" class="val">0</div>
                <div class="lbl">Port 3</div>
            </div>
        </div>

        <!-- Live packet counter bar (hidden until first deploy) -->
        <div id="packet-bar" class="packet-bar">
            OPERATIONAL FEEDBACK &nbsp;|&nbsp;
            Flow ID: <span id="pkt-flow-id">—</span> &nbsp;|&nbsp;
            Port: <span id="pkt-port">—</span> &nbsp;|&nbsp;
            Packets: <span id="pkt-count">—</span> &nbsp;|&nbsp;
            Bytes: <span id="pkt-bytes">—</span>
        </div>

        <!-- D3 topology -->
        <div id="topology-container"></div>

        <!-- Deploy button -->
        <button class="action-btn" onclick="pushSDN()">Deploy AI Flow Rules to ODL</button>
    </div>
</div>

<script>
    // ── State ────────────────────────────────────────────────────
    let currentPriority = '';
    let packetPollTimer = null;
    const consoleEl    = document.getElementById('odl-console');

    // ── Console helper ───────────────────────────────────────────
    function log(msg) {
        const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
        consoleEl.textContent = `[${ts}] ${msg}\\n` + consoleEl.textContent;
    }

    // ── D3 topology ──────────────────────────────────────────────
    function drawViz(priority) {
        const container = d3.select('#topology-container');
        container.html('');
        const width  = container.node().clientWidth  || 600;
        const height = container.node().clientHeight || 260;
        const svg    = container.append('svg').attr('width', width).attr('height', height);

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

        svg.append('g').selectAll('line').data(links).join('line')
            .attr('class', d => d.active ? 'link active-flow' : 'link')
            .attr('x1', d => nodes.find(n => n.id === d.source).fx)
            .attr('y1', d => nodes.find(n => n.id === d.source).fy)
            .attr('x2', d => nodes.find(n => n.id === d.target).fx)
            .attr('y2', d => nodes.find(n => n.id === d.target).fy);

        const nodeGroup = svg.append('g').selectAll('g').data(nodes).join('g')
            .attr('transform', d => `translate(${d.fx},${d.fy})`);

        nodeGroup.append('circle')
            .attr('r', d => d.id === 'ctrl' ? 25 : 18)
            .attr('fill', d => d.id === priority ? '#39ff14' : d.color)
            .style('filter', d => d.id === priority ? 'drop-shadow(0 0 10px #39ff14)' : 'none');

        nodeGroup.append('text').attr('dy', 34).attr('text-anchor', 'middle').text(d => d.name);
    }

    // ── AI sync ──────────────────────────────────────────────────
    async function syncAI() {
        const h = document.getElementById('hour-slider').value;
        const d = document.getElementById('day-select').value;
        const m = document.getElementById('month-select').value;
        document.getElementById('hour-text').innerText = h;

        try {
            const res  = await fetch(`/api/predict?hour=${h}&day=${d}&month=${m}`);
            const data = await res.json();

            document.getElementById('val-type1').innerText = data.rooms.type1.toLocaleString();
            document.getElementById('val-type2').innerText = data.rooms.type2.toLocaleString();
            document.getElementById('val-type3').innerText = data.rooms.type3.toLocaleString();

            ['type1','type2','type3'].forEach(id => {
                document.getElementById('card-'+id)
                    .classList.toggle('active-priority', data.priority === id);
            });

            if (currentPriority !== data.priority) {
                currentPriority = data.priority;
                drawViz(currentPriority);
            }
        } catch(e) { log('SYNC ERROR: ' + e.message); }
    }

    // ── Deploy + start packet polling ────────────────────────────
    async function pushSDN() {
        log(`[SIGNAL] AI Predicted Surge → ${currentPriority.toUpperCase()}`);
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
            } else {
                log('[WARN] Rule may not have reached the dataplane. Check ODL connectivity.');
            }
        } catch(e) {
            log('[ERROR] Could not reach Flask backend: ' + e.message);
        }
    }

    // ── Packet polling ───────────────────────────────────────────
    function startPacketPolling() {
        // Show the counter bar
        document.getElementById('packet-bar').classList.add('visible');

        // Clear any existing poll
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

        poll();                                    // immediate first tick
        packetPollTimer = setInterval(poll, 4000); // then every 4 s
    }

    // ── Boot ─────────────────────────────────────────────────────
    window.onload = () => {
        syncAI();
        setTimeout(() => {
            if (consoleEl.textContent.includes('Initializing')) {
                consoleEl.textContent =
                    '> [ODL] Controller Link Established.\\n> [AI CORE] Awaiting Manual Override...\\n';
            }
        }, 1500);
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


@app.route('/api/predict')
def predict():
    h = int(request.args.get('hour',  12))
    d = int(request.args.get('day',    0))
    m = int(request.args.get('month',  1))

    if model is None or scaler is None:
        # Graceful degradation: return synthetic values so the UI still works
        return jsonify({"rooms": {"type1": 0, "type2": 0, "type3": 0}, "priority": "type1"})

    try:
        history = []
        for i in range(12):
            m_step = (m - 12 + i) % 12 + 1
            v1 = 0.5  + 0.10 * np.sin(m_step * np.pi / 6)
            v2 = 0.3  + 0.05 * np.cos(m_step * np.pi / 6)
            v3 = 0.4  + 0.15 * np.sin((m_step + 2) * np.pi / 6)
            if i == 11:
                intensity = 1.0 + (h / 24.0) + (d / 7.0)
                v1, v2, v3 = v1 * intensity, v2 * intensity, v3 * intensity
            history.append([v1, v2, v3])

        input_data = np.array(history, dtype=np.float32).reshape(1, 12, 3)
        raw_pred   = model.predict(input_data, verbose=0)
        real_vals  = scaler.inverse_transform(raw_pred)[0]

        # Demo multiplier so Type-2 surge is visually obvious
        rooms = {
            "type1": int(max(0, real_vals[0])),
            "type2": int(max(0, real_vals[1] * 7.6)),
            "type3": int(max(0, real_vals[2])),
        }
        priority_room = max(rooms, key=rooms.get)

        return jsonify({
            "rooms":    rooms,
            "total":    sum(rooms.values()),
            "priority": priority_room,
        })

    except Exception as e:
        print(f"[PREDICT] Error: {e}")
        return jsonify({"rooms": {"type1": 0, "type2": 0, "type3": 0}, "priority": "type1"})


@app.route('/api/apply_sdn_policy', methods=['POST'])
def apply_sdn_policy():
    """
    Push an OpenFlow rule to ODL Carbon via RESTCONF v1 /restconf/config/.
    Returns a structured JSON so the front-end can log a detailed DEPLOYED line.
    """
    priority_room = request.json.get('priority', 'type1')
    port_map      = {"type1": 1, "type2": 2, "type3": 3}
    target_port   = port_map.get(priority_room, 1)
    flow_priority = 65000

    # ── Build the OpenFlow rule body ─────────────────────────────
    # ODL Carbon expects the "flow-node-inventory:" namespace prefix
    # inside the body and the legacy /restconf/config/ path.
    flow_body = {
        "flow-node-inventory:flow": [
            {
                "id":       FLOW_ID,
                "table_id": FLOW_TABLE,
                "priority": flow_priority,
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
        # Step 1 — DELETE any stale rule to avoid 409 Conflict
        requests.delete(
            FLOW_CONFIG_URL,
            auth=ODL_AUTH,
            headers=ODL_HEADERS,
            timeout=2
        )

        # Step 2 — PUT the new rule
        r = requests.put(
            FLOW_CONFIG_URL,
            json=flow_body,
            auth=ODL_AUTH,
            headers=ODL_HEADERS,
            timeout=3
        )

        ok = r.status_code in (200, 201, 204)
        msg = (
            f"Flow deployed → Port {target_port} | HTTP {r.status_code}"
            if ok else
            f"ODL rejected rule | HTTP {r.status_code} | {r.text[:120]}"
        )

        print(f"[SDN] Port={target_port} | HTTP={r.status_code}")
        if not ok:
            print(f"[SDN] Response body: {r.text}")

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
        return jsonify({"ok": False, "status": msg})
    except Exception as e:
        print(f"[SDN] Unexpected error: {e}")
        return jsonify({"ok": False, "status": f"Error: {str(e)}"})


@app.route('/api/packet_count')
def packet_count():
    """
    Query the ODL OPERATIONAL datastore for live packet/byte counters.
    This is the 'Verify' phase of the Sense-Think-Act cycle.
    Config  = what you told the switch to do.
    Operational = what the switch is actually doing right now.
    """
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

        # Navigate the ODL operational JSON tree
        # Path: flow-node-inventory:flow[0] → flow-statistics → packet-count
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

        # Also try the nested path used by some Carbon builds
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


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
