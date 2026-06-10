"""
Simulate realistic alert sequences from a topology graph for RCA training/Presentation.
"""

import random
from collections import defaultdict

import networkx as nx


_ALERT_TEMPLATES = {
    "router":        [("BGP flap detected",           "critical"),
                      ("WAN packet loss > 5%",         "major"),
                      ("Interface utilisation > 90%",  "warning")],
    "switch":        [("Interface down",               "critical"),
                      ("VLAN unreachable",              "major"),
                      ("STP topology change",          "warning")],
    "firewall":      [("Packet drops elevated",        "critical"),
                      ("Policy deny spike",             "major"),
                      ("VPN tunnel instability",       "warning")],
    "server":        [("CPU utilisation > 95%",        "critical"),
                      ("Service unavailable",           "major"),
                      ("High memory pressure",         "warning")],
    "database":      [("High DB latency",              "critical"),
                      ("SQL connection timeout",        "major"),
                      ("Replication lag > 30s",        "warning")],
    "load_balancer": [("Backend pool unhealthy",       "critical"),
                      ("5xx spike detected",            "major"),
                      ("Health check failed",          "warning")],
    "cloud_or_wan":  [("BGP session down",             "critical"),
                      ("WAN packet loss",               "major"),
                      ("ISP latency spike",            "warning")],
}

_PROPAGATION_DELAY = 2  # minutes between root-cause alert and downstream alerts


def simulate_incident(
    G: nx.Graph,
    root_node: str | None = None,
    seed: int = 42,
) -> dict:
    """Generate a synthetic alert sequence for a random (or specified) root cause.

    Parameters
    ----------
    G:         NetworkX topology graph with ``class_name`` node attributes.
    root_node: Force a specific root-cause node; if None, one is sampled
               proportional to device criticality.
    seed:      Random seed for reproducibility.

    Returns
    -------
    Incident dict with ``root_cause``, ``alerts``, and ``impacted_nodes``.
    """
    rng = random.Random(seed)

    _WEIGHT = {"database": 4, "firewall": 3, "router": 3,
               "load_balancer": 2, "switch": 2, "server": 1, "cloud_or_wan": 1}

    if root_node is None:
        pool = []
        for n, data in G.nodes(data=True):
            w = _WEIGHT.get(data.get("class_name", "server"), 1)
            pool.extend([n] * w)
        root_node = rng.choice(pool)

    root_class = G.nodes[root_node].get("class_name", "server")
    templates  = _ALERT_TEMPLATES.get(root_class, _ALERT_TEMPLATES["server"])

    # Build alerts: root first, then neighbours
    alerts = []
    t = 0
    for alert_type, severity in templates[:rng.randint(2, len(templates))]:
        alerts.append({
            "node":          root_node,
            "class":         root_class,
            "alert_type":    alert_type,
            "severity":      severity,
            "time_offset_min": t,
        })
        t += rng.randint(0, 2)

    impacted = list(G.neighbors(root_node))
    rng.shuffle(impacted)
    for nb in impacted[:3]:
        nb_class = G.nodes[nb].get("class_name", "server")
        nb_tpl   = _ALERT_TEMPLATES.get(nb_class, _ALERT_TEMPLATES["server"])
        atype, sev = rng.choice(nb_tpl)
        alerts.append({
            "node":          nb,
            "class":         nb_class,
            "alert_type":    atype,
            "severity":      sev,
            "time_offset_min": t + _PROPAGATION_DELAY,
        })

    alerts.sort(key=lambda a: a["time_offset_min"])

    return {
        "root_cause":      root_node,
        "root_cause_class": root_class,
        "alerts":          alerts,
        "impacted_nodes":  impacted[:5],
    }

