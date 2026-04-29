// app.js — TOPOS Live Demo Frontend
// D3.js force-directed concept graph, affect gauges, epistemic chart, WebSocket client

(function () {
    "use strict";

    // ========== STATE ==========
    let ws = null;
    let epistemicHistory = [];
    let thresholdHistory = [];
    const MAX_HISTORY = 60;

    // ========== DOM REFS ==========
    const $ = (sel) => document.querySelector(sel);
    const statusPill = $("#status-pill");
    const statusText = $("#status-text");
    const nodeCount = $("#node-count");
    const arousalFill = $("#arousal-fill");
    const arousalValue = $("#arousal-value");
    const valenceFill = $("#valence-fill");
    const valenceValue = $("#valence-value");
    const surpriseFill = $("#surprise-fill");
    const surpriseValue = $("#surprise-value");
    const epistemicBadge = $("#epistemic-badge");
    const epistemicCanvas = $("#epistemic-canvas");
    const feedScroll = $("#feed-scroll");
    const turnBadge = $("#turn-badge");
    const progressBar = $("#progress-bar");
    const progressLabel = $("#progress-label");
    const speedSlider = $("#speed-slider");
    const speedValue = $("#speed-value");
    const gateOverlay = $("#gate-overlay");
    const gateEpistemicVal = $("#gate-epistemic-val");
    const gateExplorationText = $("#gate-exploration-text");
    const gateResponseText = $("#gate-response-text");
    const gateAnswer = $("#gate-answer");
    const gateSubmit = $("#gate-submit");
    const startOverlay = $("#start-overlay");
    const startBtn = $("#start-btn");
    const graphPanel = $("#graph-panel");

    // ========== D3 GRAPH SETUP ==========
    const svg = d3.select("#graph-svg");
    let graphWidth = 0, graphHeight = 0;
    let simulation, linkGroup, nodeGroup, labelGroup;
    let graphNodes = [], graphLinks = [];
    let nodeMap = new Map(); // id -> node data

    function initGraph() {
        const container = document.getElementById("graph-container");
        graphWidth = container.clientWidth;
        graphHeight = container.clientHeight;

        svg.attr("viewBox", [0, 0, graphWidth, graphHeight]);

        // defs for glow filter
        const defs = svg.append("defs");
        const filter = defs.append("filter").attr("id", "glow");
        filter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
        const merge = filter.append("feMerge");
        merge.append("feMergeNode").attr("in", "blur");
        merge.append("feMergeNode").attr("in", "SourceGraphic");

        linkGroup = svg.append("g").attr("class", "links");
        nodeGroup = svg.append("g").attr("class", "nodes");
        labelGroup = svg.append("g").attr("class", "labels");

        simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(80).strength(0.3))
            .force("charge", d3.forceManyBody().strength(-120))
            .force("center", d3.forceCenter(graphWidth / 2, graphHeight / 2))
            .force("collision", d3.forceCollide().radius(d => nodeRadius(d) + 8))
            .alphaDecay(0.02)
            .on("tick", ticked);
    }

    function nodeRadius(d) {
        return Math.max(4, Math.min(22, 4 + Math.sqrt(d.weight) * 3));
    }

    function nodeColor(d) {
        // color based on weight — low weight = dim, high weight = bright
        const t = Math.min(1, d.weight / 10);
        return d3.interpolateViridis(0.2 + t * 0.6);
    }

    function updateGraph(concepts, edges) {
        // merge new data with existing graph state
        const newIds = new Set(concepts.map(c => c.id));
        const prevIds = new Set(graphNodes.map(n => n.id));

        // update existing nodes
        for (const c of concepts) {
            if (nodeMap.has(c.id)) {
                const existing = nodeMap.get(c.id);
                existing.weight = c.weight;
                existing._updated = true;
            } else {
                const newNode = {
                    id: c.id,
                    weight: c.weight,
                    x: graphWidth / 2 + (Math.random() - 0.5) * 100,
                    y: graphHeight / 2 + (Math.random() - 0.5) * 100,
                    _new: true,
                    _updated: true,
                };
                graphNodes.push(newNode);
                nodeMap.set(c.id, newNode);
            }
        }

        // remove nodes no longer in the top set
        graphNodes = graphNodes.filter(n => newIds.has(n.id));
        for (const [id] of nodeMap) {
            if (!newIds.has(id)) nodeMap.delete(id);
        }

        // update links
        graphLinks = edges
            .filter(e => nodeMap.has(e.source?.id || e.source) && nodeMap.has(e.target?.id || e.target))
            .map(e => ({
                source: e.source?.id || e.source,
                target: e.target?.id || e.target,
                weight: e.weight,
            }));

        // cap links for performance
        graphLinks.sort((a, b) => b.weight - a.weight);
        graphLinks = graphLinks.slice(0, 120);

        // update D3
        const link = linkGroup.selectAll("line").data(graphLinks, d => `${d.source}-${d.target}`);
        link.exit().transition().duration(300).attr("stroke-opacity", 0).remove();
        const linkEnter = link.enter().append("line")
            .attr("class", "edge-line")
            .attr("stroke-opacity", 0);
        linkEnter.transition().duration(500).attr("stroke-opacity", d => Math.min(0.6, d.weight / 5));
        link.transition().duration(300).attr("stroke-opacity", d => Math.min(0.6, d.weight / 5));

        const node = nodeGroup.selectAll("circle").data(graphNodes, d => d.id);
        node.exit().transition().duration(300).attr("r", 0).remove();
        const nodeEnter = node.enter().append("circle")
            .attr("class", "node-circle")
            .attr("r", 0)
            .attr("fill", nodeColor)
            .attr("stroke", d => nodeColor(d))
            .attr("stroke-width", 0)
            .attr("stroke-opacity", 0.4)
            .attr("filter", "url(#glow)")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        nodeEnter.transition().duration(500).attr("r", nodeRadius);

        // update existing
        node.transition().duration(400)
            .attr("r", nodeRadius)
            .attr("fill", nodeColor);

        // pulse updated nodes
        nodeGroup.selectAll("circle")
            .filter(d => d._updated)
            .each(function(d) {
                d._updated = false;
                d._new = false;
                d3.select(this)
                    .attr("stroke-width", 6)
                    .attr("stroke-opacity", 0.8)
                    .transition().duration(600)
                    .attr("stroke-width", 0)
                    .attr("stroke-opacity", 0.4);
            });

        const label = labelGroup.selectAll("text").data(graphNodes, d => d.id);
        label.exit().transition().duration(300).attr("opacity", 0).remove();
        label.enter().append("text")
            .attr("class", "node-label")
            .attr("dy", "0.35em")
            .attr("opacity", 0)
            .text(d => d.id.length > 12 ? d.id.slice(0, 11) + "…" : d.id)
            .transition().duration(500).attr("opacity", 1);
        label.text(d => d.id.length > 12 ? d.id.slice(0, 11) + "…" : d.id);

        // restart simulation
        simulation.nodes(graphNodes);
        simulation.force("link").links(graphLinks);
        simulation.alpha(0.3).restart();

        nodeCount.textContent = `${graphNodes.length} nodes`;
    }

    function ticked() {
        linkGroup.selectAll("line")
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        nodeGroup.selectAll("circle")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

        labelGroup.selectAll("text")
            .attr("x", d => d.x)
            .attr("y", d => d.y - nodeRadius(d) - 6);
    }

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
    }
    function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
    }

    // ========== GAUGES ==========
    function updateGauges(arousal, valence, surprise) {
        // arousal: 0-1
        arousalFill.style.width = `${(arousal * 100).toFixed(1)}%`;
        arousalValue.textContent = arousal.toFixed(2);

        // valence: -1 to 1, centered bar
        const vNorm = (valence + 1) / 2; // 0–1
        const center = 50;
        if (valence >= 0) {
            valenceFill.style.left = `${center}%`;
            valenceFill.style.width = `${(valence * 50).toFixed(1)}%`;
        } else {
            const w = Math.abs(valence) * 50;
            valenceFill.style.left = `${(center - w).toFixed(1)}%`;
            valenceFill.style.width = `${w.toFixed(1)}%`;
        }
        valenceValue.textContent = (valence >= 0 ? "+" : "") + valence.toFixed(2);

        // surprise: 0-1 typically, can exceed
        const sClamp = Math.min(surprise, 1.0);
        surpriseFill.style.width = `${(sClamp * 100).toFixed(1)}%`;
        surpriseValue.textContent = surprise.toFixed(2);
    }

    // ========== EPISTEMIC CHART ==========
    function updateEpistemicChart(epistemic, threshold) {
        epistemicHistory.push(epistemic);
        thresholdHistory.push(threshold);
        if (epistemicHistory.length > MAX_HISTORY) {
            epistemicHistory.shift();
            thresholdHistory.shift();
        }

        epistemicBadge.textContent = epistemic.toFixed(6);

        const ctx = epistemicCanvas.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        const w = epistemicCanvas.clientWidth;
        const h = epistemicCanvas.clientHeight;
        epistemicCanvas.width = w * dpr;
        epistemicCanvas.height = h * dpr;
        ctx.scale(dpr, dpr);

        ctx.clearRect(0, 0, w, h);

        if (epistemicHistory.length < 2) return;

        const maxVal = Math.max(
            ...epistemicHistory,
            ...thresholdHistory,
            0.001
        ) * 1.2;

        const pad = { top: 8, bottom: 8, left: 4, right: 4 };
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        const xScale = (i) => pad.left + (i / (MAX_HISTORY - 1)) * plotW;
        const yScale = (v) => pad.top + plotH - (v / maxVal) * plotH;

        // threshold line
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "rgba(251, 191, 36, 0.4)";
        ctx.lineWidth = 1;
        for (let i = 0; i < thresholdHistory.length; i++) {
            const x = xScale(i + MAX_HISTORY - thresholdHistory.length);
            const y = yScale(thresholdHistory[i]);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // epistemic line
        ctx.beginPath();
        ctx.strokeStyle = "#638cff";
        ctx.lineWidth = 2;
        for (let i = 0; i < epistemicHistory.length; i++) {
            const x = xScale(i + MAX_HISTORY - epistemicHistory.length);
            const y = yScale(epistemicHistory[i]);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // fill under epistemic
        const lastIdx = epistemicHistory.length - 1;
        const lastX = xScale(lastIdx + MAX_HISTORY - epistemicHistory.length);
        ctx.lineTo(lastX, yScale(0));
        ctx.lineTo(xScale(MAX_HISTORY - epistemicHistory.length), yScale(0));
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
        grad.addColorStop(0, "rgba(99, 140, 255, 0.15)");
        grad.addColorStop(1, "rgba(99, 140, 255, 0.0)");
        ctx.fillStyle = grad;
        ctx.fill();

        // dot at current value
        if (epistemicHistory.length > 0) {
            const cx = xScale(MAX_HISTORY - 1);
            const cy = yScale(epistemicHistory[epistemicHistory.length - 1]);
            ctx.beginPath();
            ctx.arc(cx, cy, 4, 0, Math.PI * 2);
            ctx.fillStyle = "#638cff";
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }

    // ========== FEED ==========
    function addFeedItem(turn, domain, text, isFire) {
        const div = document.createElement("div");
        div.className = `feed-item ${isFire ? "fire" : domain}`;
        const truncated = text.length > 100 ? text.slice(0, 97) + "…" : text;
        div.innerHTML = `<span class="feed-turn">${turn}</span>${truncated}`;
        feedScroll.appendChild(div);
        feedScroll.scrollTop = feedScroll.scrollHeight;

        // keep feed manageable
        while (feedScroll.children.length > 60) {
            feedScroll.removeChild(feedScroll.firstChild);
        }
    }

    // ========== PROGRESS ==========
    function updateProgress(turn, total, phase) {
        const pct = total > 0 ? (turn / total) * 100 : 0;
        progressBar.style.width = `${pct}%`;
        progressLabel.textContent = `Turn ${turn}/${total}  •  ${phase.toUpperCase()}`;
        turnBadge.textContent = `Turn ${turn}`;
    }

    // ========== STATUS ==========
    function setStatus(text, state) {
        statusText.textContent = text;
        statusPill.className = `status-pill ${state}`;
    }

    // ========== GATE FIRE ==========
    function showGateFire(data) {
        graphPanel.classList.add("glowing");
        gateOverlay.classList.add("active");
        gateEpistemicVal.textContent = data.epistemic.toFixed(6);
        gateExplorationText.textContent = data.exploration_text || "(empty)";
        gateResponseText.textContent = data.response || "";
        gateAnswer.value = "";
        gateAnswer.focus();
    }

    function hideGateFire() {
        gateOverlay.classList.remove("active");
        graphPanel.classList.remove("glowing");
    }

    // ========== WEBSOCKET ==========
    function connect() {
        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${protocol}//${location.host}/ws`);

        ws.onopen = () => {
            setStatus("Connected", "connected");
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleMessage(msg);
        };

        ws.onclose = () => {
            setStatus("Disconnected", "");
        };

        ws.onerror = () => {
            setStatus("Error", "");
        };
    }

    function handleMessage(msg) {
        switch (msg.type) {
            case "status":
                setStatus(msg.text, "running");
                addFeedItem("•", "", msg.text, false);
                break;

            case "turn":
                handleTurn(msg.data);
                break;

            case "gate_fired":
                showGateFire(msg.data);
                break;

            case "answer_processed":
                hideGateFire();
                updateGraph(msg.data.concepts, msg.data.edges);
                updateGauges(msg.data.arousal, msg.data.valence, 0);
                if (msg.data.new_concepts.length > 0) {
                    addFeedItem("✓", "autonomous",
                        `New concepts: ${msg.data.new_concepts.join(", ")}`, true);
                }
                break;

            case "done":
                setStatus("Complete", "connected");
                updateGraph(msg.data.concepts, msg.data.edges);
                addFeedItem("★", "", `Done. ${msg.data.total_turns} turns. Top: ${msg.data.final_concepts.slice(0, 5).join(", ")}`, false);
                break;

            case "error":
                setStatus(`Error: ${msg.text}`, "");
                addFeedItem("!", "", `Error: ${msg.text}`, true);
                break;
        }
    }

    function handleTurn(data) {
        setStatus(`${data.phase} — turn ${data.turn}`, "running");

        // update graph
        updateGraph(data.concepts, data.edges);

        // update gauges
        updateGauges(data.arousal, data.valence, data.surprise);

        // update epistemic chart
        updateEpistemicChart(data.epistemic, data.threshold);

        // update feed
        addFeedItem(data.turn, data.domain, data.input_text, false);

        // update progress
        updateProgress(data.turn, data.total_turns || 0, data.phase);
    }

    // ========== EVENTS ==========
    startBtn.addEventListener("click", () => {
        const priming = parseInt($("#cfg-priming").value) || 15;
        const autonomous = parseInt($("#cfg-autonomous").value) || 20;
        const speed = parseInt($("#cfg-speed").value) || 300;

        startOverlay.classList.add("hidden");
        connect();

        // wait for connection then send start
        const checkReady = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                clearInterval(checkReady);
                ws.send(JSON.stringify({
                    type: "start",
                    priming_turns: priming,
                    autonomous_turns: autonomous,
                    delay_ms: speed,
                }));
                setStatus("Starting...", "running");
                speedSlider.value = speed;
                speedValue.textContent = `${speed}ms`;
            }
        }, 100);
    });

    speedSlider.addEventListener("input", () => {
        const val = parseInt(speedSlider.value);
        speedValue.textContent = `${val}ms`;
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "set_speed", delay_ms: val }));
        }
    });

    gateSubmit.addEventListener("click", () => {
        const text = gateAnswer.value.trim();
        if (text && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "answer", text }));
            addFeedItem("→", "autonomous", `You: ${text}`, true);
        }
    });

    gateAnswer.addEventListener("keydown", (e) => {
        if (e.key === "Enter") gateSubmit.click();
    });

    // ========== INIT ==========
    window.addEventListener("resize", () => {
        const container = document.getElementById("graph-container");
        graphWidth = container.clientWidth;
        graphHeight = container.clientHeight;
        svg.attr("viewBox", [0, 0, graphWidth, graphHeight]);
        if (simulation) {
            simulation.force("center", d3.forceCenter(graphWidth / 2, graphHeight / 2));
            simulation.alpha(0.1).restart();
        }
    });

    initGraph();
})();
