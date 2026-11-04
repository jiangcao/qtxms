"""Plot a level-1 tile partition (msbasis.partition): atoms coloured by tile in the x-z and x-y projections, slab
boundaries dashed, and a table of tile sizes and of the +d_emb Å window (tile embedding / cap-Schwarz tile window).

  python -m qtx_ms.plot_tiles --inputs inputs/mos2_hfo2 --cut '0:n2@1,2;2:9,15.5@1,2' --out results_ms/tiles_x2z2
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import msbasis as mb


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--cut", default="", help="msbasis.parse_cut spec (empty: bisection with --tile-orbs)")
    p.add_argument("--tile-orbs", type=int, default=2500)
    p.add_argument("--d-emb", type=float, default=12.0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from .prep.inputs import read_inputs
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=a.tile_orbs, cut=mb.parse_cut(a.cut))
    xyz, lab, norb = inp["xyz"], np.asarray(inp["labels"]), inp["norb"]
    atom = np.repeat(np.arange(len(norb)), norb)
    atom_d = atom[dev.dev]
    dat = np.unique(atom_d)
    geo = mb.Geometry(xyz, inp["meta"].get("periodic"))
    tree = geo.tree(dat)
    x0 = xyz[dat, 0].min()
    sb_x = [xyz[np.where(inp["blocks"] == b)[0], 0].min() - x0 for b in dev.dev_blocks[1:]]
    rows = []
    for i, b in enumerate(blocks):
        r = xyz[b["atoms"]]
        near = dat[np.unique(np.concatenate([np.asarray(q, int) for q in
                                             tree.query_ball_point(geo.points(b["atoms"]), a.d_emb + 1e-9)]))]
        nw = int(np.isin(atom_d, np.union1d(near, b["atoms"])).sum())
        sp = {str(s): int((lab[b["atoms"]] == s).sum()) for s in sorted(set(lab[b["atoms"]]))}
        rows.append(dict(tile=i, slab=int(b["slab"]), orbitals=int(len(b["orbs"])), atoms=int(len(b["atoms"])), species=sp,
                         x=[float(r[:, 0].min() - x0), float(r[:, 0].max() - x0)], z=[float(r[:, 2].min()), float(r[:, 2].max())],
                         window=nw))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(1, 2, figsize=(18, 5.5))
    for j, c2 in enumerate((2, 1)):
        A = ax[j]
        for i, b in enumerate(blocks):
            r = xyz[b["atoms"]]
            A.scatter(r[:, 0] - x0, r[:, c2], s=6, color=cmap(i % 20), lw=0, label=f"tile {i} (slab {b['slab']})")
            A.text(np.mean(r[:, 0]) - x0, np.mean(r[:, c2]), str(i), fontsize=11, weight="bold", ha="center", va="center",
                   bbox=dict(fc="white", alpha=0.7, lw=0))
        for xb in sb_x:
            A.axvline(xb, color="k", lw=0.8, ls="--")
        A.set_xlabel("x (A)"); A.set_ylabel("z (A)" if c2 == 2 else "y (A)"); A.set_aspect("equal")
    ax[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f"{os.path.basename(os.path.normpath(a.inputs))}: tiles {a.cut or f'bisection {a.tile_orbs}'} "
                 f"({len(blocks)} tiles; window = tile + {a.d_emb:g} A, sum {sum(r['window'] for r in rows)} orbitals)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(f"{a.out}.png", dpi=110)
    json.dump(dict(cut=a.cut, tiles=rows), open(f"{a.out}.json", "w"), indent=1)
    for r in rows:
        print(f"tile {r['tile']} slab {r['slab']}: {r['orbitals']} orb, {r['atoms']} atoms {r['species']}, "
              f"x {r['x'][0]:.1f}..{r['x'][1]:.1f}, z {r['z'][0]:.1f}..{r['z'][1]:.1f}, window +{a.d_emb:g} A {r['window']}")
    print(f"-> {a.out}.png")


if __name__ == "__main__":
    main()
