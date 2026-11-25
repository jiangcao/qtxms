"""Shared parts of the single-purpose drivers on the density-matrix hierarchy (user, 2026-11-25): options, device, hierarchy,
lead injection, transmission, output. Drivers: qtx_ms.mlbasis (bases into the caches), qtx_ms.mlsolve (FGMRES, exact
deepest level), qtx_ms.mlmode (mode space), qtx_ms.mlexact (direct solve of the fine system)."""
import argparse, json, os, time
import numpy as np

from qtx_ms import devstore, dist, gpu, msbasis as mb
from qtx_ms.mlevel import Hierarchy, _xp, cache_keys, parse_up

log = mb.log


def new_parser(doc):
    return argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)


def device_args(p):
    g = p.add_argument_group("device, energies")
    g.add_argument("--inputs", required=True, help="qtx_ms input set or a release prep folder")
    g.add_argument("--dev-store", default="", help="device store of --inputs (python -m qtx_ms.devstore); folded-lead OBC")
    g.add_argument("--obc", default="folded", choices=["auto", "qttools", "folded"])
    g.add_argument("--dE", default="0.15", help="energies - Ec [eV] (comma list)")
    g.add_argument("--Ec", type=float, help="reference edge of --dE, --mu (default Ec_lead of the inputs)")
    g.add_argument("--eta", type=float, default=1e-6, help="Im z of the lead Sigma on the real axis [eV]")
    g.add_argument("--sigma-cache", help="default results_ms/sigma/<inputs name>")
    g.add_argument("--gpu", action="store_true", help="cupy backend (also QTX_MODESPACE_GPU=1)")
    g.add_argument("--out", help="json (merged across runs)")


def hierarchy_args(p):
    g = p.add_argument_group("hierarchy (as qtx_ms.mlevel; the same caches)")
    g.add_argument("--dm-cache", default="", help="window density-matrix cache")
    g.add_argument("--basis-cache", default="", help="level bases: dir, none, or empty = <dm-cache>/<tag>_levels")
    g.add_argument("--cap", default="16,2,2", help="fine windows: width [A], absorbing strength [eV], ramp [A]")
    g.add_argument("--cap-spacing", type=float)
    g.add_argument("--cap-split", default="", help="transverse window split, e.g. 2,1")
    g.add_argument("--cap-tov", type=float, default=6.0)
    g.add_argument("--l1", default="dmc:1e-5", help="level 1: dmc:CUT or dm:CUT (dm: with --no-schur)")
    g.add_argument("--up", required=True, help="further levels AXIS:CUT[:mu1:mu2[:npts]] (comma list)")
    g.add_argument("--halo", default="1", help="x halo of the rho windows per further level (comma list, shorter: the rest "
                   "all; all: every coupled block); default: 1 for the first further level")
    g.add_argument("--l1-cap", default="", help="s1,d1 of the projected CAP (default: the fine s,d)")
    g.add_argument("--l1-ramp", default="dist", choices=["dist", "plane"])
    g.add_argument("--mu", default="-0.15,0.6", help="contour window relative to Ec [eV]")
    g.add_argument("--npts", type=int, default=24)
    g.add_argument("--dm-keep", type=float, default=1e-6)
    g.add_argument("--no-schur", dest="schur", action="store_false", help="window inverse + eigh instead of Schur folding")
    g.add_argument("--schur-prec", default="double", choices=["single", "double"])
    g.add_argument("--schur-refine", type=int, default=1)
    g.add_argument("--schur-probes", type=int, default=0)
    g.add_argument("--rho-eig", default="full", choices=["full", "rand"])
    g.add_argument("--merge-eig", default="full", choices=["full", "rand", "sketch"])
    g.add_argument("--sketch-prec", default="c128", choices=["c128", "c64", "lu64"])
    g = p.add_argument_group("machine, memory")
    g.add_argument("--dist", default="", choices=["", "mpi", "nccl", "auto"])
    g.add_argument("--bottom", default="dist", choices=["dist", "repl"])
    g.add_argument("--no-share-points", action="store_true")
    g.add_argument("--host-blocks", action="store_true", help="S_k, H_k blocks in host memory")
    g.add_argument("--fine-blocks", type=int, default=0, help="levels 1 .. K without stored S_k, H_k")
    g.add_argument("--mem-frac", type=float, default=0.5)
    g.add_argument("--batch-gb", type=float, default=24.0)
    g.add_argument("--dm-batch-gb", type=float, default=16.0)
    g.add_argument("--threads", type=int, default=8, help="CPU matvec threads")


def check(a):
    assert a.up not in ("", "none"), "--up: at least one level above level 1"
    assert not a.schur or a.l1.startswith("dmc:"), "Schur folding: dmc only (dm: --no-schur)"
    return a


def read_device(a):
    """-> inp, dev, name, Ec; the sliced H, S of --dev-store are not read yet."""
    name = os.path.basename(os.path.normpath(a.inputs))
    if a.dev_store:
        inp, dev = devstore.load(a.dev_store, source=a.inputs)
    else:
        inp = devstore.read_any(a.inputs)
        dev = mb.Device(inp, obc=a.obc)
    if name == "prep" and os.path.exists(os.path.join(a.inputs, "prep.json")):           # <dev>/prep -> <dev>
        name = os.path.basename(os.path.dirname(os.path.normpath(a.inputs)))
    dev.sigma_cache = a.sigma_cache or os.path.join("results_ms", "sigma", name)
    Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    log(f"{name}: device {dev.n} orbitals, {len(dev.ns)} slabs of {min(dev.ns)}..{max(dev.ns)}; Ec {Ec:.4f} eV")
    return inp, dev, name, Ec


def drop_rows(h, dev):
    """Device store, distributed: the rows of H, S this rank holds (the CAP rows still come from the files)."""
    import gc
    dev.drop()
    gc.collect()
    h.mem("device store: rows of H, S dropped")


def energies(a):
    return [float(v) for v in a.dE.split(",")]


def build(a, comm, inp, dev, name, Ec, solver="rows64", sw_stream=None, **kw):
    """Schwarz windows, rows of H, S of this rank (--dev-store), Hierarchy(**kw). -> sw, h, levels [(axis, cut, mu1, mu2,
    npts, halo, key)] in absolute energies, basis() = (block bases of level 1)."""
    from qtx_ms.schwarz import CapSchwarz
    w_, s_, d_ = (float(v) for v in a.cap.split(","))
    sw = CapSchwarz(dev, inp, w_, s_, d_, spacing=a.cap_spacing, solver=solver,
                    split=[int(v) for v in a.cap_split.split(",")] if a.cap_split else None, tov=a.cap_tov,
                    stream_gb=sw_stream)
    if comm.size > 1:
        sw.distribute(comm)
    if a.dev_store:
        t = time.time()
        if comm.size > 1:
            comm.barrier()
        dev.hold(np.concatenate([sw.idx[k] for k in sw.own]) if comm.size > 1 else None)
        log(f"  device store: rows of H, S read ({time.time() - t:.0f} s)")
    core = dict(prec=a.schur_prec, refine=a.schur_refine, probes=a.schur_probes) if a.schur else None
    h = Hierarchy(dev, inp, sw, l1_cap=tuple(float(v) for v in a.l1_cap.split(",")) if a.l1_cap else None,
                  l1_ramp=a.l1_ramp, batch_gb=a.batch_gb, mem_frac=a.mem_frac, merge_eig=a.merge_eig,
                  sketch_prec=a.sketch_prec, core=core, rho_eig=a.rho_eig, comm=comm, bottom=a.bottom,
                  share_points=not a.no_share_points, fine_blocks=a.fine_blocks, **kw)
    mu = tuple(float(v) for v in a.mu.split(","))
    ups = parse_up(a.up, a.halo, mu)
    tag, keys = cache_keys(a, name, ups, h.s1, h.d1)
    if a.basis_cache != "none":
        h.basis_dir = a.basis_cache or (os.path.join(a.dm_cache, tag + "_levels") if a.dm_cache else None)
    levels = [(ax, cut, Ec + m1, Ec + m2, npts or a.npts, halo, key) for (ax, cut, m1, m2, npts, halo), key in zip(ups, keys)]
    kind, cut1 = a.l1.split(":")[0], float(a.l1.split(":")[1])

    def basis():
        DM = h.window_dm(a.dm_cache or None, tag, Ec + mu[0], Ec + mu[1], a.npts, a.dm_keep, a.dm_batch_gb, core=core)
        return h.basis1(DM, kind, cut1)
    return sw, h, kind, levels, basis


def lead(dev, E, eta, comm, ncol=0):
    """Lead Sigma and injection of the energy E (rank 0 fills the cache, its values on every rank): SL, SR, b (n x ncol,
    the strongest columns), number of channels. The columns ascend in the Gamma eigenvalue."""
    dd = comm.size > 1
    if dd and dev.sigma_cache:
        if comm.rank == 0:
            dev.sigma(E + 1j * eta)
        comm.barrier()
    SL, SR = dev.sigma(E + 1j * eta)
    B = dev.inject(SL)
    if dd:
        SL, SR, B = comm.bcast(SL), comm.bcast(SR), comm.bcast(B)
    nch = B.shape[1]
    nsel = min(ncol, nch) if ncol else nch
    b = np.zeros((dev.n, nsel), complex)
    b[dev.bL] = B[:, nch - nsel:]
    return SL, SR, b, nch


class Trans:
    """T = sum over the columns psi_N^H Gamma_R psi_N of the owned rows of psi (set(SR) per energy)."""

    def __init__(self, dev, part, comm):
        xp = _xp()
        self.comm, self.nb = comm, len(dev.bR)
        rR = np.where(part.gown[dev.bR] == comm.rank)[0]         # last-slab rows of this rank
        self.l, self.r = xp.asarray(part.gloc[dev.bR[rR]]), xp.asarray(rR)

    def set(self, SR):
        self.GR = _xp().asarray(1j * (SR - SR.conj().T))

    def __call__(self, x):
        xp = _xp()
        y = xp.zeros((self.nb, x.shape[1]), complex)
        y[self.r] = x[self.l]
        y = self.comm.sum(y)
        return float(xp.real(xp.einsum("ic,ij,jc->", y.conj(), self.GR, y)))


def check_out(a, Ec):
    """An existing output of another reference edge is refused before anything is computed."""
    if a.out and os.path.exists(a.out):
        old = json.load(open(a.out))
        if abs(old.get("Ec", Ec) - Ec) > 1e-12:
            raise SystemExit(f"{a.out}: written with Ec {old['Ec']}, this run uses {Ec}")


def write(a, comm, Ec, dE, rec):
    """The record of the energy dE into the json of --out (rank 0)."""
    if not a.out or comm.rank:
        return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    old["Ec"] = Ec
    old.setdefault("E", {})[f"{dE:.4f}"] = rec
    json.dump(old, open(a.out + ".tmp", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(a.out + ".tmp", a.out)


def peaks(h, comm):
    """Largest live / pool GPU memory per rank [GB] (empty on the CPU), logged."""
    if not gpu.on:
        return []
    pk = comm.allgather((max((m_["peak"] for m_ in h.memlog), default=0.0), max((m_["pool"] for m_ in h.memlog), default=0.0)))
    log("  GPU peak per rank (GB, live / pool): " + ", ".join(f"{u:.1f} / {v:.1f}" for u, v in pk))
    return [float(v) for v, _ in pk]
