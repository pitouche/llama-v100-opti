#!/usr/bin/env python3
"""
make_pdf.py — rapport PDF de la campagne sweep_q6_170k (06/11/2026).

Génère sweep_q6_report.pdf :
  p1  titre, configuration, méthode, états
  p2-9 graphiques (1/page) : decode OFF / ON brut / ON plateau / ON indexé,
      prefill, gain S0→S6, acceptance, multiplicateur ON/OFF
  p10-11 tableaux chiffrés (dont table ON indexée plateau × 3.4/(1+3A))
  p12 lecture synthétique + réserves

Méthode « plateau » (ajout 06/11 19:00) : la médiane des TOUTES fenêtres
tg_3s de R2 incluait le warm-up MTP (les ~1200 premiers tok générés sont
lents, l'acceptance monte). À 112k ce warm-up couvrait 11–14 des 15–26
fenêtres → creux artificiel. Le plateau = médiane des 6 dernières fenêtres.
Indexé = plateau × 3.4/(1+3A) ≈ t_step (t/s = (1+3A)/t_step validé à ±0.007
sur les 77 runs).

Données : bench/data/sweep_q6_170k.jsonl (dernier record valide par
(state, ctx, mtp)) ; sorties dans bench/charts/ (png + pdf).
USAGE : python3 bench/bench/plot_q6_170k.py && python3 bench/bench/make_pdf.py
(matplotlib pour plot, fpdf2+PIL pour make_pdf)
"""
import json
import os

from PIL import Image
from fpdf import FPDF
from fpdf.fonts import FontFace

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "charts", "sweep_q6_report.pdf")
CHARTS = os.path.join(HERE, "..", "charts")

FONT_DIR = "/usr/share/fonts/truetype/dejavu"

STATES = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
STATE_DESC = {
    "S0": "référence « 0 opti » (pre-T3)",
    "S1": "+batch 2048",
    "S2": "+T3",
    "S3": "+lba4 (launch_bounds)",
    "S4": "+cvec K 128-bit",
    "S5": "+vldg V double-buffer",
    "S6": "+FA16 compact (prod actuel)",
}
STATE_SO = {
    "S0": "t3 d2df07b8",
    "S1": "t3 d2df07b8",
    "S2": "t3 d2df07b8",
    "S3": "dual b014d026",
    "S4": "cvec 723c0016",
    "S5": "vldg 87f85442",
    "S6": "dual d4954502",
}
STATE_ENV = {
    "S0": "FA_NO_VOLTA_SMALL=1",
    "S1": "FA_NO_VOLTA_SMALL=1",
    "S2": "—",
    "S3": "—",
    "S4": "—",
    "S5": "—",
    "S6": "FA_COMPACT_TMAX=8192",
}
STATE_BATCH = {"S0": 512, "S1": 2048, "S2": 2048, "S3": 2048,
               "S4": 2048, "S5": 2048, "S6": 2048}

CTX_LABELS = {16384: "16k", 32768: "32k", 49152: "48k", 65536: "64k",
              81920: "80k", 98304: "96k", 114688: "112k", 131072: "128k",
              147456: "144k", 163840: "160k", 167568: "164k*"}


def load():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "data",
                                                    "sweep_q6_170k.jsonl"))]
    last = {}
    for r in rows:
        k = (r["state"], r["ctx_target"], r["mtp"])
        if r.get("steady_tps") is not None:
            last[k] = r
    ctxs = sorted(set(k[1] for k in last))
    return last, ctxs


def plateau_tps(r):
    """Médiane des 6 dernières fenêtres tg_3s de R2 (warm-up MTP retiré)."""
    import statistics
    tg = r["r2"]["log"]["tg"]
    v = [w[2] for w in (tg[-6:] if len(tg) >= 10 else tg)]
    return statistics.median(v)


def indexed_tps(r, a_ref=0.8):
    """Plateau indexé à acceptance a_ref (≈ t_step)."""
    if r.get("mtp_acceptance") is None:
        return None
    return plateau_tps(r) * (1 + 3 * a_ref) / (1 + 3 * r["mtp_acceptance"])


def fmt(v, n=1):
    return "—" if v is None else f"{v:.{n}f}"


def gain(a, b):
    if a and b:
        return (b / a - 1) * 100.0
    return None


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("DejaVu", "", 7.5)
            self.set_text_color(120)
            self.cell(0, 5, "Sweep Q6_K 27B — S0→S6, 11 ctx — 06/11/2026",
                      new_x="LMARGIN", new_y="NEXT", align="L")
            self.set_draw_color(200)
            self.line(self.l_margin, self.get_y(),
                      self.w - self.r_margin, self.get_y())
            self.ln(3)

    def footer(self):
        self.set_y(-14)
        self.set_font("DejaVu", "", 7.5)
        self.set_text_color(120)
        self.cell(0, 5, f"page {self.page_no()}", align="C")

    def h1(self, text):
        self.set_font("DejaVu", "B", 14)
        self.set_text_color(20)
        self.multi_cell(0, 8, text)
        self.ln(1)

    def h2(self, text):
        self.set_font("DejaVu", "B", 11)
        self.set_text_color(40, 90, 160)
        self.multi_cell(0, 6.5, text)
        self.ln(0.5)

    def para(self, text, size=9, gap=1.5):
        self.set_font("DejaVu", "", size)
        self.set_text_color(30)
        self.multi_cell(0, 4.8, text)
        self.ln(gap)

    def bullets(self, items, size=9):
        self.set_font("DejaVu", "", size)
        self.set_text_color(30)
        for it in items:
            x = self.get_x()
            self.cell(4.5, 4.8, "·")
            self.multi_cell(self.w - self.l_margin - self.r_margin - 4.5, 4.8, it)
            self.set_x(x)
        self.ln(1)

    def chart_page(self, name, title, caption):
        """1 graphique = 1 page : titre en haut, image au centre, légende en bas."""
        self.add_page()
        self.h2(title)
        path = os.path.join(CHARTS, name)
        with Image.open(path) as im:
            pw, ph = im.size
        w = self.w - self.l_margin - self.r_margin
        h = w * ph / pw
        # réduire si ça déborde avec légende
        max_h = self.h - self.t_margin - 60
        if h > max_h:
            h = max_h
            w = h * pw / ph
            x = (self.w - w) / 2
        else:
            x = self.l_margin
        self.image(path, x=x, y=self.get_y() + 2, w=w, h=h)
        self.ln(h + 6)
        self.set_font("DejaVu", "", 9)
        self.set_text_color(60)
        self.multi_cell(0, 5, caption)

    def grid_table(self, header, rows, first_w=0.10, bold_rows=(),
                   align_first="CENTER"):
        n = len(header)
        widths = (first_w,) + ((1 - first_w) / (n - 1),) * (n - 1)
        with self.table(
            borders_layout="ALL",
            col_widths=widths,
            text_align="CENTER",
            line_height=self.font_size * 1.35,
            first_row_as_headings=True,
            headings_style=FontFace(emphasis="BOLD"),
        ) as t:
            for ri, row in enumerate([header, *rows]):
                tr = t.row()
                for ci, cell in enumerate(row):
                    bold = ri in bold_rows and ci > 0
                    face = FontFace(emphasis="BOLD") if bold else FontFace()
                    al = align_first if ci == 0 else "RIGHT"
                    tr.cell(str(cell), style=face, align=al)
        self.ln(2)


def build():
    data, ctxs = load()
    assert len(data) == 154, f"attendu 154 points, vu {len(data)}"
    hdr = [CTX_LABELS[c] for c in ctxs]

    pdf = Report(format="A4")
    pdf.set_margins(12, 12, 12)
    pdf.add_font("DejaVu", "", f"{FONT_DIR}/DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", f"{FONT_DIR}/DejaVuSans-Bold.ttf")
    pdf.add_page()

    # ---------------- page 1 : titre + config + méthode + états
    pdf.h1("Apport des optimisations V100 — sweep S0→S6")
    pdf.para("Qwen3.8-27B, Q6_K (23 Go, blob identique à la prod) · MTP n_max=3 · "
             "V100 seule (split none, main-gpu 0) · KV q8_0 · FA auto · "
             "llama-server binaire prod, -c 170000 · sans mmproj · 06/11/2026",
             size=9.5, gap=3)

    pdf.h2("Objectif")
    pdf.para("Mesurer l'apport de chaque optimisation (S1…S6) à 11 tailles de "
             "contexte (16k → point haut 167 568 tok), en decode et en prefill, "
             "MTP ON et OFF. Campagne purement mesuratoire — pas de recherche "
             "d'optimisation (S7/H2 exclus du périmètre).")

    pdf.h2("Méthode de mesure")
    pdf.bullets([
        "R1 : 32 tok post-prefill froid (comparabilité rebench) ; "
        "R2 : 2048 tok → régime établi ; R3 : 32 tok chaud (diagnostique).",
        "Régime établi : OFF = médiane des fenêtres tg_3s ; ON = médiane des "
        "6 DERNIÈRES fenêtres (plateau) — le warm-up MTP des ~1200 premiers "
        "tok (acceptance qui monte) est retiré, il creusait la médiane brute "
        "(surtout à 112k : 11–14 fenêtres lentes sur 15–26).",
        "Gates par run : SM ≥ 1150 MHz, température < 93 °C, "
        "prompt_tokens croisé (log serveur, ±0,5 %), md5 du .so vérifié.",
        "Anchor anti-drift S6/32k/MTP ON tous les ~20 runs, porte 5 % — "
        "aucun drift détecté sur les 154 runs (04:06 → 17:35).",
        "4 re-mesures sur S6/off en haut de contexte (faux positifs du gate SM) "
        "— validées : le decode S6/off suit S5/off à −0,7…−1,0 % partout.",
    ])

    pdf.h2("États mesurés (delta cumulé)")
    st_rows = [
        [st, STATE_SO[st], STATE_ENV[st], STATE_BATCH[st], STATE_DESC[st]]
        for st in STATES
    ]
    pdf.set_font("DejaVu", "", 8)
    pdf.grid_table(["État", ".so (hash)", "env particulier", "batch", "Δ apporté"],
                   st_rows, first_w=0.07, bold_rows=(7,))
    pdf.para("Env commun : Q6_W4R4=1, Q8_FATTN_TC=0, FIT_TARGET=1024. "
             "MTP ON : --spec-type draft-mtp --spec-draft-n-max 3 "
             "--spec-draft-backend-sampling + SPEC_MTP_DEFER_PROMPT=1 + shortlist.",
             size=8, gap=3)

    # ---------------- pages graphiques : 1 graphique = 1 page
    pdf.chart_page("decode_off.png", "Decode — MTP OFF (courbe de référence, propre)",
                   "Ordre stable S0=S1 < S2 < S3 < S4 ≈ S5 ≈ S6 sur tout l'axe. "
                   "Le saut S1→S2 (T3) est le plus gros bloc ; les paliers "
                   "lba4/cvec/vldg prolongent la montée ; FA16 est neutre (−0,7…−1,0 %).")
    pdf.chart_page("decode_on.png", "Decode — MTP ON (brut, médiane toutes fenêtres)",
                   "À titre de référence seulement. La médiane toutes fenêtres inclut le "
                   "warm-up MTP (les ~1200 premiers tok générés sont lents, l'acceptance "
                   "monte) — c'est lui qui creuse 112k (11–14 fenêtres lentes sur 15–26). "
                   "Voir les 2 pages suivantes : plateau, puis indexé.")
    pdf.chart_page("decode_on_plateau.png",
                   "Decode — MTP ON en plateau (médiane 6 dernières fenêtres de R2)",
                   "Même signal que le brut mais sans le warm-up MTP : le creux de 112k "
                   "retombe (S6 : 42,9 → 52,5 t/s). Reste une ondulation d'acceptance "
                   "(0,65 à 112k vs 0,84 à 128k) qui égalise la comparaison sur la page "
                   "suivante.")
    pdf.chart_page("decode_on_norm.png",
                   "Decode — MTP ON INDEXÉ à acceptance 0,8 (courbe dérivée)",
                   "t/s_indexé = plateau × 3,4/(1+3A). Avec t/s = (1+3A)/t_step (validé "
                   "à ±0,007 sur les 77 runs), c'est ≈ t_step : la vitesse du pas de "
                   "vérification, indépendante de l'acceptance. À 112k le creux disparaît : "
                   "le gain S0→S6 passe de +18 % brut à +44 % indexé et l'axe 96k→144k "
                   "devient lisse (+44…+56 %). Les états S2→S6 se resserrent à ±3 %. "
                   "Courbe dérivée (égalisation) — elle ne remplace pas les mesures brutes.")
    pdf.chart_page("prefill_off.png", "Prefill — MTP OFF (froid, 1er passage)",
                   "Deux leviers seulement : batch 512→2048 (+9…+19 %, plus fort en bas "
                   "d'axe) et FA16 compact (+4…+24 %, croît avec le contexte). "
                   "S2→S5 se superposent (T3/lba4/cvec/vldg n'y touchent pas).")
    pdf.chart_page("gain_s0_s6.png", "Apport total des optimisations S0→S6",
                   "Gain en decode : +4,6 % à 16k → +43,1 % à 160k (OFF). En ON brut le "
                   "112k creuse à +18 % (artefact warm-up) ; en ON indexé la courbe suit "
                   "la tendance et le 112k remonte à +44 %. Le gain s'ouvre avec le "
                   "contexte — c'est le signal principal de la campagne.")
    pdf.chart_page("acceptance.png", "MTP — draft acceptance (n_max=3)",
                   "Acceptance stable 0,65–0,93 dès S2 ; bruitée sur S0/S1 "
                   "(0,58–0,85). Le creux commun à tous les états à 112k (0,65) "
                   "suit le contenu du prompt, pas une opti.")
    pdf.chart_page("mult_on_off.png", "MTP — multiplicateur ON/OFF par état (ON en plateau)",
                   "Multiplicateur ON(plateau)/OFF : 2,6–2,8× stable sur S2→S6 ; le "
                   "multiplicateur ON brut était creusé à 112k par le warm-up, pas par "
                   "l'acceptance. Creux résiduel de 112k sur S6 (×2,33) — effet contenu "
                   "de l'acceptance (0,65), commun à tous les états.")

    # ---------------- page 5 : tables decode + prefill
    pdf.add_page()
    pdf.h2("Chiffrage — decode MTP OFF (t/s, régime établi)")
    rows = []
    for st in STATES:
        rows.append([st] + [fmt(data.get((st, c, "off"), {}).get("steady_tps"))
                            for c in ctxs])
    pdf.set_font("DejaVu", "", 8)
    pdf.grid_table(["état"] + hdr, rows, first_w=0.08, bold_rows=(7,))
    pdf.para("* point haut 167 568 tok (163,6k). S6 = configuration prod actuelle.",
             size=8, gap=3)

    pdf.h2("Chiffrage — prefill MTP OFF (t/s, froid)")
    rows = []
    for st in STATES:
        rows.append([st] + [fmt(data.get((st, c, "off"), {}).get("prefill_tps"), 0)
                            for c in ctxs])
    pdf.grid_table(["état"] + hdr, rows, first_w=0.08, bold_rows=(7,))

    # ---------------- page 6 : gains par palier + ON + acceptance + mult
    pdf.h2("Gains par palier — decode MTP OFF (%)")
    steps = [("S0", "S1", "batch"), ("S1", "S2", "T3"), ("S2", "S3", "lba4"),
             ("S3", "S4", "cvec"), ("S4", "S5", "vldg"), ("S5", "S6", "FA16")]
    rows = []
    for a, b, nm in steps:
        rows.append([f"{a}→{b} {nm}"] +
                    [fmt(gain(data.get((a, c, "off"), {}).get("steady_tps"),
                              data.get((b, c, "off"), {}).get("steady_tps")))
                     for c in ctxs])
    rows.append(["S0→S6 total"] +
                [fmt(gain(data.get(("S0", c, "off"), {}).get("steady_tps"),
                          data.get(("S6", c, "off"), {}).get("steady_tps")))
                 for c in ctxs])
    pdf.grid_table(["palier"] + hdr, rows, first_w=0.14, bold_rows=(7,),
                   align_first="LEFT")

    pdf.h2("Chiffrage — decode MTP ON (t/s)")
    rows = []
    for st in STATES:
        rows.append([st + " brut"] +
                    [fmt(data.get((st, c, "on"), {}).get("steady_tps"))
                     for c in ctxs])
    rows.append(["—"] + ["—"] * len(ctxs))
    for st in STATES:
        rows.append([st + " plat."] +
                    [fmt(plateau_tps(data[(st, c, "on")]))
                     for c in ctxs])
    pdf.grid_table(["état"] + hdr, rows, first_w=0.08)
    pdf.para("brut = médiane toutes fenêtres R2 (inclut le warm-up MTP) · plat. = "
             "médiane des 6 dernières fenêtres (warm-up retiré). S0/S1 : le brut est "
             "de plus à prendre avec prudence (acceptance instable pré-T3).", size=8, gap=3)

    pdf.h2("Chiffrage — decode MTP ON INDEXÉ à acceptance 0,8 (t/s, dérivé)")
    rows = []
    for st in STATES:
        vals = []
        for c in ctxs:
            r = data.get((st, c, "on"), {})
            v = indexed_tps(r)
            vals.append(fmt(v) if v is not None else None)
        rows.append([st] + vals)
    pdf.grid_table(["état"] + hdr, rows, first_w=0.08, bold_rows=(7,))
    pdf.para("indexé = plateau × 3,4/(1+3A) ≈ t_step : vitesse du pas de vérification "
             "si tous les états avaient l'acceptance 0,8. Courbe dérivée (égalisation "
             "des comparaisons), pas une mesure.", size=8, gap=3)

    pdf.h2("Gain S0→S6 en decode MTP ON (%) — brut vs indexé")
    rows = []
    for c in ctxs:
        r0 = data[("S0", c, "on")]
        r6 = data[("S6", c, "on")]
        raw = 100 * (r6["steady_tps"] / r0["steady_tps"] - 1)
        i0, i6 = indexed_tps(r0), indexed_tps(r6)
        idx = 100 * (i6 / i0 - 1) if i0 and i6 else None
        rows.append([CTX_LABELS[c], fmt(raw), fmt(idx)])
    pdf.grid_table(["ctx", "gain brut", "gain indexé"], rows, first_w=0.25)
    pdf.para("Le creux brut de 112k (+18 %) est un artefact du warm-up MTP : indexé, "
             "le 112k remonte à +44 % et l'axe 96k→144k devient lisse.", size=8, gap=3)

    pdf.h2("Acceptance MTP (ON) et multiplicateur ON/OFF (×, ON en plateau)")
    rows = []
    for st in STATES:
        rows.append([st + " acc"] +
                    [fmt(data.get((st, c, "on"), {}).get("mtp_acceptance"), 2)
                     for c in ctxs])
    pdf.grid_table(["état"] + hdr, rows, first_w=0.08)
    rows = []
    for st in STATES:
        vals = []
        for c in ctxs:
            ro = data.get((st, c, "off"), {}).get("steady_tps")
            rn = data.get((st, c, "on"), {})
            vp = plateau_tps(rn) if rn else None
            vals.append(fmt(vp / ro, 2) if ro and vp else None)
        rows.append([st + " ×"] + vals)
    pdf.grid_table(["état"] + hdr, rows, first_w=0.08)

    # ---------------- page 7 : lecture synthétique + réserves
    pdf.add_page()
    pdf.h2("Lecture synthétique")
    pdf.bullets([
        "Batch 512→2048 : 0 % en decode — c'est un levier purement prefill "
        "(+9…+19 %).",
        "T3 : le cliff dominant en decode, +3,6 % (16k) → +25,9 % (160k). "
        "À long contexte T3 représente ~60 % du delta total S0→S6. "
        "Le cliff est réel mais continué : lba4 (+0,3…+6,7 %), cvec "
        "(+1,3…+5,4 %) et vldg (+0,3…+2,0 %) prolongent la montée après T3 — "
        "le saut est compensé par la suite, la courbe S0→S6 ne plate jamais.",
        "FA16 compact : neutre en decode (−0,7…−1,0 %), +4…+24 % en prefill "
        "(croît avec le contexte). En synthèse : prefill = batch + FA16.",
        "MTP : multiplicateur ON(plateau)/OFF 2,6–2,8× stable sur S2→S6. Le "
        "« creux de 112k » de la courbe ON brute est un artefact de mesure : le "
        "warm-up MTP (les ~1200 premiers tok générés sont lents, l'acceptance "
        "monte) couvrait 11–14 des 15–26 fenêtres R2 à 112k → la médiane toutes "
        "fenêtres était tirée vers le bas. En plateau (6 dernières fenêtres) il "
        "retombe (S6 : 42,9 → 52,5 t/s) ; indexé à acceptance 0,8 le gain S0→S6 "
        "y remonte à +44 % (vs +18 % brut) et l'axe 96k→144k est lisse. Reste "
        "un vrai creux d'acceptance à 112k (0,65 vs 0,84 à 128k, commun à tous "
        "les états) → effet contenu du prompt.",
        "Total S0→S6 en decode : +4,6 % à 16k → +43,1 % à 160k ; en prefill "
        "+24 % à 160k. T3 ≈ 2/3 du gain à long contexte.",
    ], size=9)

    pdf.h2("Réserves")
    pdf.bullets([
        "Méthode ON en plateau = médiane des 6 dernières fenêtres R2 — "
        "justification : séquences de fenêtres 3s montrant un warm-up MTP "
        "(~1200 tok lents) sur presque tous les runs ON ; à 112k le warm-up "
        "couvrait 11–14 fenêtres sur 15–26, d'où le creux de la médiane brute.",
        "MTP ON sur S0/S1 (7 points × 2 états) : le plateau est déjà bien "
        "aligné avec S2+, mais les 112k S0/S1 n'ont que 20 fenêtres et "
        "l'acceptance y reste instable — à re-mesurer si on veut des chiffres "
        "ON fiables avant T3. Les courbes OFF restent la base solide.",
        "Le point haut 164k* est un point unique sans répétition (limite dure "
        "-c 170000 avec les 2048 tok générés).",
        "Prefill mesuré à froid (1er passage) — sensible à la page cache ; "
        "les écarts inter-états y sont de tout façon < 1 % hors batch/FA16.",
    ], size=9)

    pdf.h2("Artefacts")
    pdf.bullets([
        "bench/data/sweep_q6_170k.jsonl — 158 records (154 points + 4 re-mesures S6/off)",
        "plot_q6_170k.py — régénère les 8 graphiques de charts/ (dont "
        "decode_on_plateau.png et decode_on_norm.png, courbes dérivées "
        "plateau / indexé à acc 0,8).",
        "make_pdf.py — régénère ce PDF depuis sweep.jsonl.",
        "server.log — log serveur complet de la campagne (source des "
        "prompt_tokens et de l'acceptance).",
    ], size=8)

    pdf.output(OUT)
    print("PDF écrit :", OUT, f"({os.path.getsize(OUT)/1024:.0f} Ko)")


if __name__ == "__main__":
    build()
