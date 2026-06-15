"""
FalconVue Graph — 3D WebGL renderer for InfraGraph AI enterprise graphs.

Two rendering modes
-------------------
scenario (default)
    Scenario Enterprise Graph — Alert Propagation.
    Used in GNN RCA tab.  All node sizes per spec (10–24), nodeRelSize=4,
    camera z=220, zoomToFit after layout, d3 charge -45,
    local-link distance 45, cross-link distance 90.
    No auto-orbit.  Alert nodes orange, root-cause red, step-node white,
    traversal cyan.  Links visible (width 1.2 / 2.2 / 4.0, opacity 0.85).

global
    InfraGraph AI — Enterprise Graph Galaxy.
    Original galaxy behaviour: z=580, nodeRelSize=3, charge -120,
    distances 90/240, auto-orbit after 3 s.  Thinner/dimmer links.

Uses 3d-force-graph + three-spritetext via CDN + streamlit.components.v1.html.
Raises on Python-level failure so callers can handle gracefully.
"""
from __future__ import annotations

import json
import streamlit as st
import streamlit.components.v1 as components

# ── Node colour palette ───────────────────────────────────────────────────────
_DEVICE_COLORS: dict[str, str] = {
    "router":        "#60a5fa",
    "switch":        "#4ade80",
    "firewall":      "#fb923c",
    "server":        "#cbd5e1",
    "database":      "#c084fc",
    "load_balancer": "#fbbf24",
    "cloud_or_wan":  "#67e8f9",
    "service":       "#34d399",
}
_DEFAULT_COLOR = "#94a3b8"

_LEGEND_ROWS: list[tuple[str, str]] = [
    ("#ef4444", "Root Cause"),
    ("#ffffff", "Current Step"),
    ("#f97316", "Alert Node"),
    ("#facc15", "Impacted"),
    ("#22d3ee", "Absorbed / Path"),
    ("#a855f7", "Shared Entity"),
    ("#60a5fa", "Router"),
    ("#4ade80", "Switch"),
    ("#cbd5e1", "Server"),
    ("#c084fc", "Database"),
    ("#67e8f9", "Cloud / WAN"),
]

# ── CSS template ──────────────────────────────────────────────────────────────
_CSS_TMPL = """\
*{margin:0;padding:0;box-sizing:border-box}
html,body{
  width:100%;height:__HEIGHT__px;overflow:hidden;
  background:
    radial-gradient(ellipse 120% 60% at 35% 55%,rgba(16,28,58,0.85) 0%,transparent 70%),
    radial-gradient(ellipse 80% 40% at 70% 40%,rgba(10,20,50,0.60) 0%,transparent 60%),
    radial-gradient(ellipse 200% 100% at 50% 50%,#050e20 0%,#030810 60%,#010508 100%);
  font-family:system-ui,-apple-system,sans-serif;
}
#sf{position:fixed;top:0;left:0;width:100%;height:__HEIGHT__px;
  pointer-events:none;z-index:0}
#gc{position:relative;width:100%;height:__HEIGHT__px;z-index:1}
#to{
  position:absolute;top:14px;left:50%;transform:translateX(-50%);
  z-index:10;pointer-events:none;text-align:center;white-space:nowrap
}
.mt{
  font-size:1.05rem;font-weight:800;letter-spacing:.06em;
  background:linear-gradient(130deg,#f8fafc 0%,#93c5fd 50%,#67e8f9 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text
}
.st{font-size:.6rem;letter-spacing:.16em;text-transform:uppercase;color:#334155;margin-top:3px}
#lo{
  position:absolute;top:14px;left:14px;z-index:10;
  background:rgba(3,8,20,0.80);border:1px solid rgba(255,255,255,0.07);
  border-radius:10px;padding:11px 14px;min-width:158px;backdrop-filter:blur(8px)
}
.ll{font-size:.57rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;
  color:#334155;margin-bottom:8px}
.li{display:flex;align-items:center;gap:7px;margin:3px 0;font-size:.7rem;color:#94a3b8}
.ld{width:9px;height:9px;border-radius:50%;flex-shrink:0;
  box-shadow:0 0 5px currentColor}
.rb{margin-top:9px;padding-top:9px;border-top:1px solid rgba(255,255,255,0.06)}
.rl{font-size:.57rem;letter-spacing:.12em;text-transform:uppercase;color:#475569;margin-bottom:3px}
.rv{font-family:monospace;font-size:.84rem;color:#ef4444;font-weight:700;word-break:break-all}
#mo{
  position:absolute;top:14px;right:14px;z-index:10;
  background:rgba(3,8,20,0.80);border:1px solid rgba(255,255,255,0.07);
  border-radius:10px;padding:11px 16px;min-width:142px;backdrop-filter:blur(8px)
}
.mt2{font-size:.57rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;
  color:#334155;margin-bottom:8px}
.mr{display:flex;justify-content:space-between;align-items:center;margin:3px 0;font-size:.72rem}
.mlb{color:#475569}.mv{color:#e2e8f0;font-weight:700;font-family:monospace}
.cyan{color:#22d3ee}.orange{color:#f97316}.white{color:#ffffff}
#tt{
  position:absolute;z-index:20;pointer-events:none;display:none;
  background:rgba(3,8,20,0.95);border:1px solid rgba(255,255,255,0.09);
  border-radius:10px;padding:11px 14px;font-size:.73rem;max-width:230px;
  backdrop-filter:blur(10px)
}
.tid{font-family:monospace;font-size:.9rem;font-weight:700;color:#f1f5f9;margin-bottom:6px}
.tr{display:flex;gap:7px;margin:2px 0}
.tk{color:#334155;min-width:50px;flex-shrink:0}
#gm>div>div:last-child{display:none!important}
"""

# ── JS template ───────────────────────────────────────────────────────────────
# Plain string — { } are literal JS.  Placeholders:
#   __MODE__        → "scenario" | "global"
#   __STEP_NODE__   → JSON string of current-step node id
#   __TRAV_PATH__   → JSON array of traversal path node ids
#   __GDATA__       → JSON graph payload
#   __HEIGHT__      → pixel height integer
_JS_TMPL = """\
(function(){
  /* ── Mode flags ── */
  var _IS_SCEN=("__MODE__"==="scenario");
  var _STEP_NODE=__STEP_NODE__;
  var _TRAV_PATH=__TRAV_PATH__;

  /* ── Galaxy starfield ── */
  var sf=document.getElementById('sf');
  var ctx=sf.getContext('2d');
  var W=window.innerWidth||900, H=__HEIGHT__;
  sf.width=W; sf.height=H;

  var grad=ctx.createRadialGradient(W*0.38,H*0.52,0,W*0.38,H*0.52,W*0.55);
  grad.addColorStop(0,'rgba(20,50,100,0.22)');
  grad.addColorStop(0.4,'rgba(10,25,60,0.12)');
  grad.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=grad; ctx.fillRect(0,0,W,H);

  var stars=[];
  for(var i=0;i<420;i++){
    var bright=(i<18);
    stars.push({
      x:Math.random()*W, y:Math.random()*H,
      r:bright?(Math.random()*1.6+0.7):(Math.random()*0.9+0.1),
      o:bright?(Math.random()*0.55+0.45):(Math.random()*0.38+0.07),
      d:Math.random()*.004+.0008,
      ph:Math.random()*6.283,
      bright:bright
    });
  }
  var t0=null;
  function drawStars(ts){
    if(t0===null) t0=ts;
    var t=ts-t0;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle=grad; ctx.fillRect(0,0,W,H);
    for(var i=0;i<stars.length;i++){
      var s=stars[i];
      var op=s.o+Math.sin(t*s.d+s.ph)*0.15;
      op=Math.max(0,Math.min(1,op));
      ctx.beginPath(); ctx.arc(s.x,s.y,s.r,0,6.283);
      if(s.bright){
        ctx.fillStyle='rgba(220,240,255,'+op+')'; ctx.fill();
        ctx.beginPath(); ctx.arc(s.x,s.y,s.r*2.8,0,6.283);
        ctx.fillStyle='rgba(180,210,255,'+(op*0.12)+')'; ctx.fill();
      } else {
        ctx.fillStyle='rgba(200,220,255,'+op+')'; ctx.fill();
      }
    }
    requestAnimationFrame(drawStars);
  }
  requestAnimationFrame(drawStars);

  /* ── Graph data ── */
  var gData=__GDATA__;
  var tt=document.getElementById('tt');
  var gm=document.getElementById('gm');
  var orbitOn=false;

  /* ── 3D Force Graph ── */
  var G=ForceGraph3D()(gm)
    .width(gm.clientWidth||W)
    .height(H)
    .backgroundColor('rgba(0,0,0,0)')
    .graphData(gData)
    .nodeId('id')
    .nodeLabel('')
    .nodeColor(function(n){ return n.color; })
    .nodeRelSize(_IS_SCEN?4:3)
    .nodeVal(function(n){ return n.size*n.size; })
    .nodeOpacity(0.95)
    .nodeThreeObject(function(node){
      var sprite=new SpriteText(node.label);
      var isStep=(node.highlight==='step');
      var isTrav=(node.highlight==='trav');
      sprite.color=isStep?'#ffffff':(isTrav?'#67e8f9':'#e8edf4');
      sprite.textHeight=isStep?6.0:(isTrav?4.5:(_IS_SCEN?3.8:2.2));
      sprite.fontFace='system-ui,-apple-system,sans-serif';
      sprite.fontWeight=isStep?'900':'600';
      sprite.material.depthWrite=false;
      var r=Math.cbrt(node.size*node.size)*(_IS_SCEN?4:3);
      sprite.position.x=r+5;
      return sprite;
    })
    .nodeThreeObjectExtend(true)
    .linkSource('source')
    .linkTarget('target')
    .linkColor(function(l){ return l.color; })
    .linkWidth(function(l){ return l.width; })
    .linkOpacity(_IS_SCEN?0.88:0.75)
    .linkCurvature(0.05)
    .linkDirectionalParticles(function(l){ return l.particles; })
    .linkDirectionalParticleSpeed(function(l){ return l.p_speed; })
    .linkDirectionalParticleWidth(function(l){ return l.p_width; })
    .linkDirectionalParticleColor(function(l){ return l.p_color; })
    .onNodeHover(function(node){
      if(!node){ tt.style.display='none'; return; }
      tt.innerHTML=
        '<div class="tid">'+node.id+'</div>'
        +'<div class="tr"><span class="tk">type</span><span>'+(node.type||'')+'</span></div>'
        +'<div class="tr"><span class="tk">status</span><span>'+(node.status||'')+'</span></div>'
        +'<div class="tr"><span class="tk">diagram</span><span>'+(node.diagram||'—')+'</span></div>'
        +'<div class="tr"><span class="tk">zone</span><span>'+(node.zone||'—')+'</span></div>'
        +'<div class="tr"><span class="tk">ip</span><span>'+(node.ip||'—')+'</span></div>';
      tt.style.display='block';
    })
    .onNodeClick(function(node){
      var dist=80;
      var nx=node.x||0,ny=node.y||0,nz=node.z||0;
      var hyp=Math.sqrt(nx*nx+ny*ny+nz*nz)||1;
      var r=1+dist/hyp;
      G.cameraPosition({x:nx*r,y:ny*r,z:nz*r},node,1200);
      var c=G.controls(); if(c) c.autoRotate=false;
    });

  /* ── Mode-specific force tuning + camera ── */
  try{
    if(_IS_SCEN){
      G.d3Force('link').distance(function(l){ return l.is_cross?90:45; });
      G.d3Force('charge').strength(-45);
      G.cameraPosition({x:0,y:0,z:220});
      setTimeout(function(){ try{ G.zoomToFit(1200,80); }catch(e){} },1200);
    }else{
      G.d3Force('link').distance(function(l){ return l.is_cross?240:90; });
      G.d3Force('charge').strength(-120);
      G.cameraPosition({x:0,y:0,z:580});
      setTimeout(function(){
        var c=G.controls();
        if(c&&!orbitOn){ c.autoRotate=true; c.autoRotateSpeed=0.45; orbitOn=true; }
      },3000);
    }
  }catch(e){}

  /* ── Tooltip follows mouse ── */
  gm.addEventListener('mousemove',function(e){
    if(tt.style.display==='none') return;
    var r=gm.getBoundingClientRect();
    var x=e.clientX-r.left+16, y=e.clientY-r.top+16;
    if(x+238>r.width)  x=e.clientX-r.left-248;
    if(y+210>r.height) y=e.clientY-r.top-214;
    tt.style.left=x+'px'; tt.style.top=y+'px';
  });

  /* ── Kill orbit on interaction ── */
  ['mousedown','touchstart','wheel'].forEach(function(ev){
    gm.addEventListener(ev,function(){
      var c=G.controls(); if(c) c.autoRotate=false;
    },{passive:true});
  });
})();
"""


# ── Public API ────────────────────────────────────────────────────────────────

def render_falconvue_graph(
    enterprise_graph: dict,
    absorbed_ids: "set[str] | None" = None,
    rca: "dict | None" = None,
    incident: "dict | None" = None,
    height: int = 800,
    mode: str = "scenario",
    current_step_node: "str | None" = None,
    traversal_path: "list[str] | None" = None,
    alert_timeline: "list[dict] | None" = None,
) -> None:
    """Render the enterprise graph as a 3D WebGL FalconVue visualisation.

    Parameters
    ----------
    enterprise_graph  : Full enterprise graph (nodes/edges/cross_diagram_edges).
    absorbed_ids      : Node IDs just absorbed (cyan highlight).
    rca               : RCA result with root_cause/alert_nodes/impact_path.
    incident          : Enterprise incident dict (alert_timeline used if alert_timeline is None).
    height            : Iframe height in pixels.
    mode              : "scenario" (default) or "global".
    current_step_node : Node ID of the current propagation step (white/bright).
    traversal_path    : Ordered path so far (cyan in graph).
    alert_timeline    : Alert event list; nodes highlighted orange before RCA exists.
    """
    absorbed_ids   = absorbed_ids or set()
    rca            = rca or {}
    incident       = incident or {}
    traversal_path = traversal_path or []

    root_cause   = rca.get("root_cause", "")
    rca_alert_set   = set(rca.get("alert_nodes", []))
    impacted_set    = set(rca.get("impacted_nodes", []))
    impact_path     = rca.get("impact_path", [])
    path_edges: set[tuple[str, str]] = set(zip(impact_path, impact_path[1:]))

    # Alert nodes: explicit parameter wins, then incident dict
    incident_alert_set: set[str] = set()
    for ev in (alert_timeline or incident.get("alert_timeline", [])):
        n = ev.get("node", "")
        if n:
            incident_alert_set.add(n)

    alert_set = rca_alert_set or incident_alert_set

    step_set = {current_step_node} if current_step_node else set()
    trav_set = set(traversal_path) - step_set

    # ── Mode-specific size and link constants ─────────────────────────────────
    if mode == "scenario":
        SZ_ROOT  = 24.0; SZ_STEP  = 22.0; SZ_ALERT = 18.0
        SZ_IMPA  = 16.0; SZ_ABS   = 18.0; SZ_TRAV  = 14.0; SZ_SHARE = 18.0
        SZ_HUB   = 14.0; SZ_CONN  = 14.0; SZ_DEF   = 10.0
        LW_PATH  = 4.0;  LW_CROSS = 2.2;  LW_LOCAL = 1.2
        LC_LOCAL = "rgba(255,255,255,0.60)"
        LC_CROSS = "rgba(255,255,255,0.90)"
    else:
        SZ_ROOT  = 12.0; SZ_STEP  = 11.0; SZ_ALERT  = 9.0
        SZ_IMPA  =  8.0; SZ_ABS   =  9.0; SZ_TRAV   = 7.0; SZ_SHARE  = 9.0
        SZ_HUB   =  7.0; SZ_CONN  =  6.0; SZ_DEF    = 5.0
        LW_PATH  =  2.5; LW_CROSS =  1.2; LW_LOCAL  = 0.7
        LC_LOCAL = "rgba(255,255,255,0.42)"
        LC_CROSS = "rgba(255,255,255,0.65)"

    # node → diagram cluster
    node_diag: dict[str, str] = {}
    clusters = enterprise_graph.get("diagram_clusters", [])
    if isinstance(clusters, list):
        for c in clusters:
            for nid in c.get("node_ids", []):
                node_diag[nid] = c.get("diagram_id", "")
    elif isinstance(clusters, dict):
        for did, c in clusters.items():
            nids = c if isinstance(c, list) else c.get("node_ids", [])
            for nid in nids:
                node_diag[nid] = did

    # ── Nodes ─────────────────────────────────────────────────────────────────
    nodes_out: list[dict] = []
    for n in enterprise_graph.get("nodes", []):
        nid    = n.get("id", "")
        ntype  = (n.get("type") or n.get("class_name") or "server").lower()
        shared = n.get("is_shared_entity", False)

        if nid == root_cause and root_cause:
            color, sz, status, hl = "#ef4444", SZ_ROOT, "root_cause", ""
        elif nid in step_set:
            color, sz, status, hl = "#ffffff", SZ_STEP, "step_node", "step"
        elif nid in alert_set:
            color, sz, status, hl = "#f97316", SZ_ALERT, "alert", ""
        elif nid in impacted_set:
            color, sz, status, hl = "#facc15", SZ_IMPA, "impacted", ""
        elif nid in absorbed_ids:
            color, sz, status, hl = "#22d3ee", SZ_ABS, "absorbed", ""
        elif nid in trav_set:
            color, sz, status, hl = "#22d3ee", SZ_TRAV, "traversal", "trav"
        elif shared:
            color, sz, status, hl = "#a855f7", SZ_SHARE, "shared", ""
        else:
            color = _DEVICE_COLORS.get(ntype, _DEFAULT_COLOR)
            if ntype in ("router", "firewall", "cloud_or_wan", "load_balancer"):
                sz = SZ_HUB
            elif ntype in ("switch",):
                sz = SZ_CONN
            else:
                sz = SZ_DEF
            status, hl = "normal", ""

        nodes_out.append({
            "id":        nid,
            "label":     nid,
            "type":      ntype,
            "status":    status,
            "highlight": hl,
            "color":     color,
            "size":      sz,
            "ip":        n.get("ip_address", ""),
            "zone":      n.get("zone", ""),
            "diagram":   node_diag.get(nid) or n.get("diagram_id", ""),
        })

    # ── Safety check ─────────────────────────────────────────────────────────
    if not nodes_out:
        st.warning(
            "Scenario graph has no renderable nodes — "
            "ensure the enterprise graph has been built from a scenario."
        )
        return

    # ── Links ─────────────────────────────────────────────────────────────────
    seen_pairs: set[tuple[str, str]] = set()
    links_out: list[dict]            = []
    valid_ids  = {n["id"] for n in nodes_out}

    def _push(e: dict, force_cross: bool = False) -> None:
        src, tgt = e.get("source", ""), e.get("target", "")
        if not src or not tgt or src not in valid_ids or tgt not in valid_ids:
            return
        k = (src, tgt)
        if k in seen_pairs:
            return
        seen_pairs.add(k)

        is_cross = (
            force_cross
            or e.get("edge_scope") == "cross_diagram"
            or e.get("edge_type")  == "cross_diagram"
        )
        is_path = k in path_edges

        if is_path:
            color = "#22d3ee"
            w     = LW_PATH
            parts, spd, pw, pc = 6, 0.007, 2.5, "#22d3ee"
        elif is_cross:
            color = LC_CROSS
            w     = LW_CROSS
            # Ambient particles on cross-links in scenario mode so the graph feels alive
            parts = 2 if mode == "scenario" else 0
            spd   = 0.003 if mode == "scenario" else 0.0
            pw    = 1.5   if mode == "scenario" else 0.0
            pc    = "#67e8f9"
        else:
            color = LC_LOCAL
            w     = LW_LOCAL
            parts, spd, pw, pc = 0, 0.0, 0.0, "#fff"

        links_out.append({
            "source":    src,
            "target":    tgt,
            "label":     e.get("label", ""),
            "is_cross":  is_cross,
            "is_path":   is_path,
            "color":     color,
            "width":     w,
            "particles": parts,
            "p_speed":   spd,
            "p_width":   pw,
            "p_color":   pc,
        })

    for e in enterprise_graph.get("edges", []):
        _push(e)
    for e in enterprise_graph.get("cross_diagram_edges", []):
        _push(e, force_cross=True)

    if not links_out and mode == "scenario":
        st.info("Scenario graph has nodes but no renderable links.")

    # ── Metrics ──────────────────────────────────────────────────────────────
    graph_json = json.dumps({"nodes": nodes_out, "links": links_out})

    n_nodes  = len(nodes_out)
    n_links  = len(links_out)
    n_cross  = sum(1 for lk in links_out if lk["is_cross"])
    n_path   = sum(1 for lk in links_out if lk["is_path"])
    n_alerts = len(alert_set)
    has_rca  = bool(root_cause)
    has_step = bool(current_step_node)

    if mode == "scenario":
        main_title = "Scenario Enterprise Graph &#8212; Alert Propagation"
        sub_title  = "Cross-diagram incident propagation across selected scenario"
    else:
        main_title = "InfraGraph AI &#8212; Global Galaxy"
        sub_title  = "Global graph-memory index across V3 scenarios"

    html = _assemble_html(
        graph_json, height, mode, main_title, sub_title,
        n_nodes, n_links, n_cross, n_path, n_alerts,
        has_rca, root_cause, has_step, current_step_node or "",
        json.dumps(current_step_node or ""),
        json.dumps(traversal_path),
    )
    components.html(html, height=height + 10, scrolling=False)


# ── HTML assembly ─────────────────────────────────────────────────────────────

def _assemble_html(
    graph_json: str,
    height: int,
    mode: str,
    main_title: str,
    sub_title: str,
    n_nodes: int, n_links: int, n_cross: int, n_path: int,
    n_alerts: int, has_rca: bool, root_cause: str,
    has_step: bool, step_node_label: str,
    step_node_json: str,
    trav_path_json: str,
) -> str:
    h   = str(height)
    css = _CSS_TMPL.replace("__HEIGHT__", h)
    js  = (
        _JS_TMPL
        .replace("__MODE__", mode)
        .replace("__STEP_NODE__", step_node_json)
        .replace("__TRAV_PATH__", trav_path_json)
        .replace("__GDATA__", graph_json)
        .replace("__HEIGHT__", h)
    )

    legend_rows = "".join(
        f'<div class="li">'
        f'<div class="ld" style="background:{c};box-shadow:0 0 5px {c}"></div>'
        f'{lb}</div>'
        for c, lb in _LEGEND_ROWS
    )

    rca_block = ""
    if has_rca and root_cause:
        rca_block = (
            '<div class="rb">'
            '<div class="rl">Root Cause</div>'
            f'<div class="rv">{root_cause}</div>'
            '</div>'
        )

    metrics_extra = ""
    if has_rca:
        metrics_extra += (
            f'<div class="mr"><span class="mlb">Path links</span>'
            f'<span class="mv cyan">{n_path}</span></div>'
            f'<div class="mr"><span class="mlb">Alerts</span>'
            f'<span class="mv orange">{n_alerts}</span></div>'
        )
    elif n_alerts:
        metrics_extra += (
            f'<div class="mr"><span class="mlb">Alert nodes</span>'
            f'<span class="mv orange">{n_alerts}</span></div>'
        )
    if has_step and step_node_label:
        metrics_extra += (
            f'<div class="mr"><span class="mlb">Step node</span>'
            f'<span class="mv white" style="font-size:.68rem;word-break:break-all">'
            f'{step_node_label}</span></div>'
        )

    error_overlay = (
        "<div id='fv-err' style='"
        "display:none;position:fixed;top:0;left:0;width:100%;height:100%;"
        "background:rgba(11,18,32,0.95);z-index:9999;"
        "align-items:center;justify-content:center;flex-direction:column;"
        "text-align:center;padding:40px;box-sizing:border-box'>"
        "<div style='font-size:1.3rem;font-weight:800;color:#f97316;margin-bottom:14px'>"
        "&#9888; FalconVue 3D renderer failed to load</div>"
        "<div style='font-size:0.88rem;color:#94a3b8;max-width:480px;line-height:1.6'>"
        "CDN scripts (<code>ForceGraph3D</code> / <code>SpriteText</code>) did not load — "
        "likely blocked by a network policy or offline environment.<br><br>"
        "<strong style='color:#22d3ee'>Switch to Stable 2D propagation graph (PyVis)</strong> "
        "in the render mode selector above for a reliable view."
        "</div></div>"
        # JS: reveal overlay after 4 s if either library is still undefined
        "<script>"
        "(function(){"
        "  setTimeout(function(){"
        "    if(typeof ForceGraph3D==='undefined'||typeof SpriteText==='undefined'){"
        "      var el=document.getElementById('fv-err');"
        "      if(el){el.style.display='flex';}"
        "    }"
        "  },4000);"
        "})();"
        "</script>"
    )

    return "".join([
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<style>", css, "</style>",
        "</head><body>",
        "<canvas id='sf'></canvas>",
        "<div id='gc'>",
        "<div id='to'>",
        f"<div class='mt'>{main_title}</div>",
        f"<div class='st'>{sub_title}</div>",
        "</div>",
        "<div id='lo'>",
        "<div class='ll'>Node Legend</div>",
        legend_rows,
        rca_block,
        "</div>",
        "<div id='mo'>",
        "<div class='mt2'>Graph Metrics</div>",
        f"<div class='mr'><span class='mlb'>Nodes</span><span class='mv'>{n_nodes}</span></div>",
        f"<div class='mr'><span class='mlb'>Links</span><span class='mv'>{n_links}</span></div>",
        f"<div class='mr'><span class='mlb'>Cross-diag</span><span class='mv cyan'>{n_cross}</span></div>",
        metrics_extra,
        "</div>",
        "<div id='tt'></div>",
        f"<div id='gm' style='width:100%;height:{height}px'></div>",
        "</div>",
        error_overlay,
        "<script src='https://cdn.jsdelivr.net/npm/three-spritetext@1.9.0/dist/three-spritetext.min.js'></script>",
        "<script src='https://cdn.jsdelivr.net/npm/3d-force-graph@1.73.0/dist/3d-force-graph.min.js'></script>",
        "<script>", js, "</script>",
        "</body></html>",
    ])
