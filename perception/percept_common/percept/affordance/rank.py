"""⑤-5 순위 (v3 §7.8 빈도주의). ordering 키를 후보마다 계산해 정렬, 동순위는 score, 그다음 (target_id, u, v) 로 결정론.
키: tier_top_first(타깃 단 오름차순) · valley_count_desc(타깃 골 수 내림차순) · uphill_pos_desc(타깃 중심 u 내림차순) · id_asc ·
    pattern_order(generate 가 붙인 슬롯 번호) · x_asc/y_asc(후보 u/v 오름차순).
**감점(선호 거리·clearance_short)이 끝난 최종 score 로 부른다.** risk = 1 − 후보들 사이 min-max 정규화 score — **상대값**이다(절대 위험 아님).
후보가 하나이거나 전부 동점이면 0.5(중립). Step 8 에서 되먹임 outcome_rates 로 교체."""
from typing import List

import numpy as np

from ..scene.plane import to_plane


def _target_info(hyps, scene):
    info = {}
    for i, o in enumerate(hyps.items):
        if not o.pose_valid:
            continue
        cells = scene.layers["instance_id"] == i + 1
        tier = int(scene.layers["tier"][cells].max()) if cells.any() else 99
        u = float(to_plane(scene.plane, np.asarray(o.position, float)[None])[0][0] * 1000.0)
        n_valley = len((o.extras or {}).get("grooves") or []) or sum(1 for nb in o.neighbors if nb.get("contact"))
        info[o.id] = dict(tier=tier, valleys=n_valley, u=u)
    return info


def rank(cands: List[dict], hyps, scene, rules: dict) -> List[dict]:
    if not cands:
        return []
    info = _target_info(hyps, scene) if hyps is not None else {}
    keys = list(rules.get("ordering", ["id_asc"]))

    def key(c):
        t = info.get(c.get("target_id"), {})
        k = []
        for name in keys:
            if name == "tier_top_first":
                k.append(t.get("tier", 99))
            elif name == "valley_count_desc":
                k.append(-t.get("valleys", 0))
            elif name == "uphill_pos_desc":
                k.append(-t.get("u", -1e9))
            elif name == "id_asc":
                k.append(str(c.get("target_id") or ""))
            elif name == "pattern_order":
                k.append(int(c.get("slot", 10 ** 6)))
            elif name == "x_asc":
                k.append(float(c["u_mm"]))
            elif name == "y_asc":
                k.append(float(c["v_mm"]))
        return tuple(k) + (-float(c["score"]), str(c.get("target_id") or ""), float(c["u_mm"]), float(c["v_mm"]))

    ordered = sorted(cands, key=key)
    s = np.array([c["score"] for c in ordered], float)
    lo, hi = float(s.min()), float(s.max())
    for i, c in enumerate(ordered):
        c["rank"] = i + 1
        c["risk"] = float(1.0 - (c["score"] - lo) / (hi - lo)) if hi > lo else 0.5
    return ordered
