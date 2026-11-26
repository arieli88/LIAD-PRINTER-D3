"""Menorah generator: 8x 5.56 cases (lights) + 1x 7.62 case (shamash, raised).
Run: python menorah.py  ->  writes STL files into ./stl/
All sizes in mm. Change the parameters below and run again.
"""
import os, struct
import numpy as np

# ---------- parameters ----------
CASE_556 = dict(rim=9.6, neck=6.43, length=44.7)
CASE_762 = dict(rim=12.0, neck=8.77, length=51.2)   # 7.62x51. For 7.62x39: rim=11.35, neck=8.6, length=38.7
FIT = 0.4            # clearance around the case head in the base socket
NECK_FIT = 0.35      # clearance between the neck and the adapter sleeve (fired necks are slightly expanded)
SPACING = 32         # distance between lights. Below ~25 mm the flames merge (safety + halacha)
SHAMASH_BACK = 30    # how far behind the row the shamash sits
SHAMASH_RAISE = 16   # how much higher the shamash socket is than the lights
CANDLE_BOTTOM, CANDLE_TOP = 9.4, 11.0   # tapered wax-candle cup diameters (fits typical Hanukkah candles)
CANDLE_DEPTH = 14
OIL_GLASS = 26       # bottom diameter of the glass oil cup. The cradle must stay narrower than SPACING
SEG = 96
BED = 256            # Bambu P1S

# ---------- mesh helpers ----------
def revolve(profile, n=SEG):
    """Closed (r, z) polygon -> watertight solid of revolution. r == 0 points sit on the axis."""
    V, F, rings = [], [], []
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for r, z in profile:
        if r == 0:
            rings.append([len(V)]); V.append((0, 0, z))
        else:
            rings.append(list(range(len(V), len(V) + n)))
            V += [(r * np.cos(t), r * np.sin(t), z) for t in a]
    m = len(profile)
    for i in range(m):
        A, B = rings[i], rings[(i + 1) % m]
        for k in range(n):
            k1 = (k + 1) % n
            if len(A) > 1 and len(B) > 1:
                F += [(A[k], A[k1], B[k1]), (A[k], B[k1], B[k])]
            elif len(A) == 1 and len(B) > 1:
                F.append((A[0], B[k1], B[k]))
            elif len(A) > 1 and len(B) == 1:
                F.append((A[k], A[k1], B[0]))
    return orient(np.array(V, float), np.array(F))

def box(x0, x1, y0, y1, z0, z1):
    V = np.array([(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    F = np.array([(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
                  (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)])
    return orient(V, F)

def volume(V, F):
    return np.einsum('ij,ij->i', V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6

def orient(V, F):
    return (V, F[:, ::-1].copy()) if volume(V, F) < 0 else (V, F)

def check(mesh, name):
    """Self-check: every directed edge appears once and its reverse once (closed, consistently oriented)."""
    V, F = mesh
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    d = set(map(tuple, e))
    assert len(d) == len(e), f"{name}: duplicate edge"
    assert all((b, a) in d for a, b in d), f"{name}: open edge"
    assert volume(V, F) > 0, f"{name}: inside-out"

def move(mesh, dx=0, dy=0, dz=0):
    return mesh[0] + (dx, dy, dz), mesh[1]

def write_stl(path, meshes):
    tris = np.concatenate([V[F] for V, F in meshes])
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    with open(path, 'wb') as f:
        f.write(b'menorah'.ljust(80, b' ') + struct.pack('<I', len(tris)))
        for nn, t in zip(n, tris):
            f.write(struct.pack('<12fH', *nn, *t.ravel(), 0))

def write_3mf(path, objects):
    """One plate: each (name, mesh-or-list-of-meshes) becomes its own object, group centred on the P1S bed."""
    import zipfile
    allv = np.concatenate([V for _, ms in objects for V, _ in (ms if isinstance(ms, list) else [ms])])
    off = np.array([BED / 2, BED / 2, 0]) - [(allv[:, 0].min() + allv[:, 0].max()) / 2,
                                            (allv[:, 1].min() + allv[:, 1].max()) / 2, allv[:, 2].min()]
    res, items = [], []
    for i, (name, ms) in enumerate(objects, 1):
        vs, ts, base_i = [], [], 0
        for V, F in (ms if isinstance(ms, list) else [ms]):
            vs += [f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in V + off]
            ts += [f'<triangle v1="{a + base_i}" v2="{b + base_i}" v3="{c + base_i}"/>' for a, b, c in F]
            base_i += len(V)
        res.append(f'<object id="{i}" name="{name}" type="model"><mesh><vertices>{"".join(vs)}</vertices>'
                   f'<triangles>{"".join(ts)}</triangles></mesh></object>')
        items.append(f'<item objectid="{i}"/>')
    model = ('<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<resources>{"".join(res)}</resources><build>{"".join(items)}</build></model>')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr('3D/3dmodel.model', model)

# ---------- parts ----------
def socket_boss(case, floor_z, foot_r_extra):
    """Cylinder with a blind hole for the case head; flared foot sinks 1 mm into the plate."""
    rh, ro = case['rim'] / 2 + FIT / 2, case['rim'] / 2 + 3.2
    top = floor_z + 12
    return revolve([(0, 4), (ro + foot_r_extra, 4), (ro, 4 + foot_r_extra + 2), (ro, top),
                    (rh + 0.8, top), (rh, top - 0.8), (rh, floor_z), (0, floor_z)])

def adapter(case, kind):
    """Sleeve slides over the case neck and rests on the case mouth; wax cup or oil-glass cradle on top."""
    si = case['neck'] / 2 + NECK_FIT / 2      # sleeve inner radius
    sd = 5 if case is CASE_556 else 7         # sleeve depth (shorter than the neck)
    ro = si + 1.6
    base = [(0, sd), (si, sd), (si, 0), (ro, 0), (ro, sd)]
    if kind == 'wax':
        R = 14                                  # drip tray radius, 45-degree underside (no supports)
        zt = sd + (R - ro); ts = zt + 1.2
        cf = sd + 2; ct = cf + CANDLE_DEPTH; co = CANDLE_TOP / 2 + 1.6
        return revolve(base + [(R, zt), (R, zt + 3), (R - 1.2, zt + 3), (R - 1.2, ts),
                               (co, ts), (co, ct), (CANDLE_TOP / 2, ct), (CANDLE_BOTTOM / 2, cf), (0, cf)])
    R = OIL_GLASS / 2 + 0.5 + 1.6               # cradle outer radius
    zt = sd + (R - ro)
    return revolve(base + [(R, zt), (R, zt + 5), (R - 1.6, zt + 5), (R - 1.6, zt + 1.6), (0, zt + 1.6)])

def base():
    xs = [(i - 3.5) * SPACING for i in range(8)]
    x0, x1 = xs[0] - 12, xs[-1] + 12
    y0, y1 = -20, SHAMASH_BACK + 18
    parts = [box(x0, x1, y0, y1, 0, 5)]
    # 2 mm lip around the edge catches wax / oil drips
    parts += [box(x0, x1, y0, y0 + 2, 4.5, 7), box(x0, x1, y1 - 2, y1, 4.5, 7),
              box(x0, x0 + 2, y0, y1, 4.5, 7), box(x1 - 2, x1, y0, y1, 4.5, 7)]
    parts += [move(socket_boss(CASE_556, 6, 3), x, 0) for x in xs]
    parts.append(move(socket_boss(CASE_762, 6 + SHAMASH_RAISE, 5), 0, SHAMASH_BACK))
    return parts, (x1 - x0, y1 - y0)

def footprint_r(kind, case):
    V, _ = adapter(case, kind)
    return np.hypot(V[:, 0], V[:, 1]).max()

# ---------- build ----------
if __name__ == '__main__':
    os.makedirs('stl', exist_ok=True)
    parts, (L, W) = base()
    for i, p in enumerate(parts): check(p, f'base part {i}')
    if L > BED - 6:
        assert (L + W) / np.sqrt(2) <= BED - 6, f'base {L:.0f}x{W:.0f} mm does not fit the P1S even diagonally'
        print(f'NOTE: base is {L:.0f} mm long - rotate it 45 degrees on the plate')
    write_stl('stl/base.stl', parts)
    os.makedirs('3mf', exist_ok=True)
    write_3mf('3mf/1_base.3mf', [('menorah base', parts)])

    for kind in ('wax', 'oil'):
        r = footprint_r(kind, CASE_556)
        assert 2 * r < SPACING - 1, f'{kind} adapters ({2*r:.1f} mm) touch each other - increase SPACING'
        # the shamash adapter must not touch the two middle lights
        assert np.hypot(SPACING / 2, SHAMASH_BACK) > r + footprint_r(kind, CASE_762) + 1, 'shamash too close'
        a556, a762 = adapter(CASE_556, kind), adapter(CASE_762, kind)
        check(a556, kind + '556'); check(a762, kind + '762')
        write_stl(f'stl/adapter_{kind}_556.stl', [a556])
        write_stl(f'stl/adapter_{kind}_762.stl', [a762])
        # full plate: 8 lights + 1 shamash, laid out with gaps
        step = 2 * r + 6
        set_ = [move(a556, (i % 3) * step, (i // 3) * step) for i in range(8)] + [move(a762, 2 * step, 2 * step)]
        write_stl(f'stl/set_{kind}_8+1.stl', set_)
        names = [f'{kind} 5.56 #{i + 1}' for i in range(8)] + [f'{kind} 7.62 shamash']
        write_3mf(f'3mf/{2 if kind == "wax" else 3}_adapters_{kind}.3mf', list(zip(names, set_)))

    lights_z = 6 + CASE_556['length']; sham_z = 6 + SHAMASH_RAISE + CASE_762['length']
    print(f'base {L:.0f} x {W:.0f} mm | case mouth: lights {lights_z:.1f} mm, shamash {sham_z:.1f} mm')
    print('all meshes closed and oriented: OK')
