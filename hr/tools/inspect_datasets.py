import re, sys, yaml
from collections import Counter
from pathlib import Path
for d in sorted(Path("data/raw").iterdir()):
    y = yaml.safe_load(open(d/"data.yaml", encoding="utf8"))
    names = y["names"]; names = list(names.values()) if isinstance(names, dict) else names
    rf = y.get("roboflow", {})
    print(f"\n=== {d.name}  license={rf.get('license')}  version={rf.get('version')}  nc={len(names)}")
    cnt = Counter(); bases = {}
    for s in ("train","valid","test"):
        imgs = [p for p in (d/s/"images").glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")] if (d/s/"images").exists() else []
        b = {re.sub(r"\.rf\.[0-9a-f]+$","",p.stem) for p in imgs}
        bases[s] = b
        for l in (d/s/"labels").glob("*.txt") if (d/s/"labels").exists() else []:
            for ln in l.read_text(encoding="utf8").splitlines():
                p = ln.split()
                if p: cnt[int(float(p[0]))] += 1
        print(f"  {s}: {len(imgs)} images, {len(b)} unique source names")
    print("  leak train/valid:", len(bases['train']&bases['valid']), " train/test:", len(bases['train']&bases['test']))
    print("  classes:", ", ".join(f"{names[i]}={n}" for i,n in cnt.most_common()))
    print("  unused classes:", [names[i] for i in range(len(names)) if i not in cnt])
