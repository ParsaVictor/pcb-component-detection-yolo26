"""Contact sheets: random crops of selected classes, to review class meaning by eye."""
import csv, random, sys
from pathlib import Path
import cv2, numpy as np, yaml

A = Path("data/all_data"); names = yaml.safe_load(open(A/"data.yaml", encoding="utf8"))["names"]
by_cls = {}
for lbl in (A/"labels").glob("*.txt"):
    for ln in lbl.read_text(encoding="utf8").splitlines():
        p = ln.split()
        if len(p) >= 5: by_cls.setdefault(names[int(p[0])], []).append((lbl.stem, [float(x) for x in p[1:5]]))
random.seed(1)
def sheet(keys, out, n=10, cell=110):
    rows = []
    for k in keys:
        items = by_cls.get(k, []); random.shuffle(items)
        tiles = []
        for stem, (cx, cy, w, h) in items[:n]:
            img = next((A/"images").glob(stem + ".*"))
            im = cv2.imread(str(img)); H, W = im.shape[:2]
            pw, ph = max(w*W*0.5, 8), max(h*H*0.5, 8)
            x0, x1 = int(max(cx*W-w*W/2-pw*0.3,0)), int(min(cx*W+w*W/2+pw*0.3,W))
            y0, y1 = int(max(cy*H-h*H/2-ph*0.3,0)), int(min(cy*H+h*H/2+ph*0.3,H))
            c = im[y0:y1, x0:x1]
            if c.size == 0: continue
            s = cell / max(c.shape[:2]); c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            t = np.zeros((cell, cell, 3), np.uint8); t[:c.shape[0], :c.shape[1]] = c; tiles.append(t)
        while len(tiles) < n: tiles.append(np.zeros((cell, cell, 3), np.uint8))
        row = np.hstack(tiles); lab = np.zeros((22, row.shape[1], 3), np.uint8)
        cv2.putText(lab, f"{k} (n={len(items)})", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
        rows += [lab, row]
    cv2.imwrite(out, np.vstack(rows))
groups = {
 "g8": ["carddata:Multilayer Ceramic Capacitor","carddata:capacitor","carddata:Processor","carddata:Oscillator","carddata:Resistor Network","carddata:iC"],
 "g7": ["dhbw:Oscillator","tesis:oscilador","tesis:procesador","carddata:IC","carddata:Inductor","dhbw:Inductor"],
 "g6": ["parsa:capacitor","parsa:Capacitor","parsa:Electrolytic Capacitor","carddata:Capacitor","carddata:Resistor","dhbw:Resistor"],
 "g1": ["parsa:Resistor","parsa:Resistor Jumper","parsa:Capacitor","parsa:Capacitor Jumper","parsa:Jumper","carddata:Jumper"],
 "g2": ["parsa:Ferrite Bead","carddata:Ferrite Bead","parsa:Inductor","parsa:Resistor Network","parsa:Capacitor Network","parsa:RP"],
 "g3": ["carddata:Processor","carddata:Oscillator","carddata:Clock","parsa:Clock","carddata:EM","parsa:EM"],
 "g4": ["parsa:Unknown Unlabeled","parsa:Fuse","parsa:Pins","parsa:Pads","parsa:Test Point","parsa:Fiducial"],
 "g5": ["parsa:Led","parsa:Diode","parsa:Zener Diode","parsa:Switch","parsa:Button","parsa:Connector"],
}
for g, ks in groups.items(): sheet(ks, str(Path(sys.argv[1])/f"{g}.jpg"))
