#!/usr/bin/env python
"""Generate SUMO XML assets from the Chapter 4/5 thesis diagrams and tables.

This writes nodes, edges, routes, and SUMO configs without requiring SUMO. In a
SUMO-enabled environment, run netconvert on the generated .nod.xml/.edg.xml files or
use scripts/run_thesis_sumo.py (added later) to build/run them.
"""
from __future__ import annotations
from pathlib import Path
import html

OUT = Path("sumo/thesis_ch4_ch5")


def xml_attrs(**kwargs) -> str:
    return " ".join(f'{k}="{html.escape(str(v))}"' for k, v in kwargs.items())


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def nodes_xml(nodes: dict[str, tuple[int, int, str]]) -> str:
    lines = ["<nodes>"]
    for node_id, (x, y, kind) in nodes.items():
        attrs = {"id": node_id, "x": x, "y": y}
        if kind:
            attrs["type"] = kind
        lines.append(f"  <node {xml_attrs(**attrs)}/>")
    lines.append("</nodes>")
    return "\n".join(lines) + "\n"


def edges_xml(edges: list[tuple[str, str, str, int]]) -> str:
    lines = ["<edges>"]
    for edge_id, src, dst, length in edges:
        lines.append(f"  <edge {xml_attrs(id=edge_id, from_=src, to=dst, priority=1, numLanes=1, speed=13.9, length=length)}/>`".replace("from_=", "from=").replace("/>`", "/>"))
    lines.append("</edges>")
    return "\n".join(lines) + "\n"


def routes_xml(routes: dict[str, list[str]], interval_sets: dict[str, list[int | None]], horizon: int, split_count: int) -> str:
    lines = ["<routes>", '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.9"/>']
    for rid, edges in routes.items():
        lines.append(f'  <route id="{rid}" edges="{" ".join(edges)}"/>')
    segment = horizon // split_count
    for rid, intervals in interval_sets.items():
        for idx, period in enumerate(intervals):
            if period is None:
                continue
            begin = idx * segment
            end = (idx + 1) * segment
            lines.append(f'  <flow id="{rid}_{idx}" type="car" route="{rid}" begin="{begin}" end="{end}" period="{period}" departLane="best" departSpeed="max"/>')
    lines.append("</routes>")
    return "\n".join(lines) + "\n"


def sumocfg(net_name: str, route_name: str, end: int) -> str:
    return f'''<configuration>
  <input>
    <net-file value="{net_name}"/>
    <route-files value="{route_name}"/>
  </input>
  <time><begin value="0"/><end value="{end}"/><step-length value="1"/></time>
</configuration>
'''


def generate_two() -> None:
    base = OUT / "two_intersection"
    nodes = {
        "L": (0, 0, "traffic_light"), "R": (500, 0, "traffic_light"),
        "W": (-3000, 0, "priority"), "N_L": (0, 3000, "priority"), "S_L": (0, -3000, "priority"),
        "E": (3500, 0, "priority"), "N_R": (500, 3000, "priority"), "S_R": (500, -3000, "priority"),
    }
    edges = [
        ("h", "W", "L", 3000), ("a", "L", "W", 3000),
        ("b", "N_L", "L", 3000), ("c", "L", "N_L", 3000),
        ("f", "S_L", "L", 3000), ("g", "L", "S_L", 3000),
        ("e", "L", "R", 500), ("d", "R", "L", 500),
        ("k", "E", "R", 3000), ("l", "R", "E", 3000),
        ("i", "N_R", "R", 3000), ("j", "R", "N_R", 3000),
        ("m", "S_R", "R", 3000), ("n", "R", "S_R", 3000),
    ]
    routes = {
        "b-g": ["b", "g"], "b-e": ["b", "e"], "i-n": ["i", "n"], "f-c": ["f", "c"],
        "h-e-l": ["h", "e", "l"], "e-l": ["e", "l"], "d-a": ["d", "a"], "k-d": ["k", "d"],
    }
    intervals = {
        "b-g": [60, 30, 40], "b-e": [80, 400, 40], "i-n": [40, 40, 40], "f-c": [60, 600, 300],
        "h-e-l": [80, 1000, 20], "e-l": [30, 600, 60], "d-a": [20, 200, 20], "k-d": [60, 10, 60],
    }
    write(base / "two_intersection.nod.xml", nodes_xml(nodes))
    write(base / "two_intersection.edg.xml", edges_xml(edges))
    write(base / "two_intersection.rou.xml", routes_xml(routes, intervals, 10800, 3))
    write(base / "two_intersection.sumocfg", sumocfg("two_intersection.net.xml", "two_intersection.rou.xml", 10800))


def generate_nine() -> None:
    base = OUT / "nine_intersection"
    coords = {
        "C1": (0, 1000), "NC1": (1000, 1000), "C2": (2000, 1000),
        "NC2": (0, 0), "C3": (1000, 0), "NC4": (2000, 0),
        "C4": (0, -1000), "NC3": (1000, -1000), "C5": (2000, -1000),
    }
    nodes = {nid: (*xy, "traffic_light") for nid, xy in coords.items()}
    externals = {
        "N1": (0, 2000), "N2": (1000, 2000), "N3": (2000, 2000),
        "S1": (0, -2000), "S2": (1000, -2000), "S3": (2000, -2000),
        "W1": (-1000, 1000), "W2": (-1000, 0), "W3": (-1000, -1000),
        "E1": (3000, 1000), "E2": (3000, 0), "E3": (3000, -1000),
    }
    nodes.update({nid: (*xy, "priority") for nid, xy in externals.items()})
    edges: list[tuple[str, str, str, int]] = []
    def add_pair(a,b,length=1000):
        edges.append((f"{a}_{b}", a, b, length)); edges.append((f"{b}_{a}", b, a, length))
    for row in [("C1","NC1","C2"),("NC2","C3","NC4"),("C4","NC3","C5")]:
        add_pair(row[0], row[1]); add_pair(row[1], row[2])
    for col in [("C1","NC2","C4"),("NC1","C3","NC3"),("C2","NC4","C5")]:
        add_pair(col[0], col[1]); add_pair(col[1], col[2])
    for ext, inner in [("N1","C1"),("N2","NC1"),("N3","C2"),("S1","C4"),("S2","NC3"),("S3","C5"),("W1","C1"),("W2","NC2"),("W3","C4"),("E1","C2"),("E2","NC4"),("E3","C5")]:
        add_pair(ext, inner)
    routes = {
        "N1_S1": ["N1_C1", "C1_NC2", "NC2_C4", "C4_S1"],
        "N2_S2": ["N2_NC1", "NC1_C3", "C3_NC3", "NC3_S2"],
        "N3_S3": ["N3_C2", "C2_NC4", "NC4_C5", "C5_S3"],
        "S1_N1": ["S1_C4", "C4_NC2", "NC2_C1", "C1_N1"],
        "S2_N2": ["S2_NC3", "NC3_C3", "C3_NC1", "NC1_N2"],
        "S3_N3": ["S3_C5", "C5_NC4", "NC4_C2", "C2_N3"],
        "W1_E1": ["W1_C1", "C1_NC1", "NC1_C2", "C2_E1"],
        "W2_E2": ["W2_NC2", "NC2_C3", "C3_NC4", "NC4_E2"],
        "W3_E3": ["W3_C4", "C4_NC3", "NC3_C5", "C5_E3"],
        "E1_W1": ["E1_C2", "C2_NC1", "NC1_C1", "C1_W1"],
        "E2_W2": ["E2_NC4", "NC4_C3", "C3_NC2", "NC2_W2"],
        "E3_W3": ["E3_C5", "C5_NC3", "NC3_C4", "C4_W3"],
    }
    s1 = {"N1_S1": [40,160], "N2_S2": [None,140], "N3_S3": [None,None], "E1_W1": [50,130], "E2_W2": [10,200], "E3_W3": [50,100], "S1_N1": [100,170], "S2_N2": [100,700], "S3_N3": [None,None], "W1_E1": [100,100], "W2_E2": [100,90], "W3_E3": [10,150]}
    s2 = {"N1_S1": [10], "N2_S2": [100], "N3_S3": [None], "E1_W1": [20], "E2_W2": [10], "E3_W3": [40], "S1_N1": [10], "S2_N2": [30], "S3_N3": [None], "W1_E1": [10], "W2_E2": [50], "W3_E3": [5]}
    write(base / "nine_intersection.nod.xml", nodes_xml(nodes))
    write(base / "nine_intersection.edg.xml", edges_xml(edges))
    write(base / "scenario1.rou.xml", routes_xml(routes, s1, 3600, 2))
    write(base / "scenario2.rou.xml", routes_xml(routes, s2, 3600, 1))
    write(base / "scenario1.sumocfg", sumocfg("nine_intersection.net.xml", "scenario1.rou.xml", 3600))
    write(base / "scenario2.sumocfg", sumocfg("nine_intersection.net.xml", "scenario2.rou.xml", 3600))


def main() -> None:
    generate_two(); generate_nine()
    print(f"Generated thesis SUMO XML inputs under {OUT}")

if __name__ == "__main__":
    main()
