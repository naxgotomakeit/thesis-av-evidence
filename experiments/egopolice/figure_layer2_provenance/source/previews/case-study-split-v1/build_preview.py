"""Non-generative, preview-only re-layout derived from case_study_code.zip.

Reads the supplied repair.py constants without executing that script. Uses
unaltered formal JPEGs, chronological display order, and vector text/geometry.
Does not write main.tex, pics/, the existing thesis PDFs, or the source bundle.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import shutil
import zipfile
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parent
ZIP = Path('/Users/nax/Downloads/case_study_code.zip')
BUNDLE = Path('/Users/nax/Downloads/226_case_study_review_bundle_v1')
THESIS = ROOT.parents[2]
W = 150 * mm
BLUE = '#086BFF'
ORANGE = '#C94C00'
INK = '#17212D'
MUTED = '#435062'
RULE = '#BCC6D0'
FONT = 9.5
LEADING = 11.7


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


protected = [THESIS / 'main.tex', THESIS / 'pics/case_study_corrected.png',
             THESIS / 'output/pdf/large-figures.pdf']
before = {str(p): sha(p) for p in protected}
with zipfile.ZipFile(ZIP) as z:
    source = z.read('repair.py').decode()
    verification = json.loads(z.read('verification.json'))
    constants = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'V', 'AV', 'vlabels', 'avlabels'}:
                    constants[target.id] = ast.literal_eval(node.value)
    assert constants['V'] == verification['V_boundaries']
    assert constants['AV'] == verification['AV_boundaries']

for variant, key in [('V', 'V'), ('AV_SPEECH', 'AV')]:
    regions = json.loads((BUNDLE / 'maps_v2_2' / variant / 'parsed_map.json').read_text())['coarse_regions']
    edges = [r['start_sec'] for r in regions] + [regions[-1]['end_sec']]
    assert edges == constants[key], (variant, edges, constants[key])

panels = [
    dict(letter='b', title='Weapon visible - control', question='q_weapon_visible',
         V=[547, 562, 577], AV=[547, 562, 577],
         answer_v='Yes - firearm visible.', answer_av='Yes - firearm visible.',
         note='Speech cue: armed-individual coordination. Both conditions inspect the same frames; '
              'speech does not alter this visually grounded check.'),
    dict(letter='c', title='Visible injury - local navigation shift', question='q_visible_injury',
         V=[547, 652, 727], AV=[547, 637, 727],
         answer_v='Yes - visible blood / injury.', answer_av='Yes - visible blood / injury.',
         note='Speech cue: casualty / gunshot-related radio traffic. AV inspects 637 s rather than '
              '652 s; both reach the same injury region at 727 s.'),
    dict(letter='d', title='Medical assistance - speech context and ambiguity', question='q_medical_assistance',
         V=[630, 652, 667], AV=[585, 630, 720, 1050, 1125],
         answer_v='Yes - inferred from visual checking / contact.',
         answer_av='Recovered answer: Yes - cites speech-derived medical context.',
         note='Speech cues: "just for a tourniquet"; "we need EMS". Speech adds medical-response '
              'context, but the images do not visually establish the claimed treatment.',
         recovery='Format-only recovery; no new inspection or frames. Primary route remains a failure.'),
]

csv_rows = {int(float(r['timestamp_sec'])): r for r in csv.DictReader((BUNDLE / 'inspected_frames_all.csv').open())}
times = sorted({t for panel in panels for key in ['V', 'AV'] for t in panel[key]})
assert times == sorted([547, 562, 577, 652, 637, 727, 630, 667, 585, 720, 1050, 1125])
frame_hashes = {}
pixel_hashes = {}
for t in times:
    path = BUNDLE / 'inspected_frames_all' / f'frame_{t:05d}.jpg'
    frame_hashes[t] = sha(path)
    assert frame_hashes[t] == csv_rows[t]['sha256']
    with Image.open(path) as im:
        assert im.size == (1280, 720)
        pixel_hashes[t] = hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()

for panel in panels:
    for variant, key in [('V', 'V'), ('AV_SPEECH', 'AV')]:
        route = json.loads((BUNDLE / 'direct_sixq/routes' / variant / panel['question'] / 'route_result.json').read_text())
        assert sorted(f['frame_index'] for f in route['inspected_frames']) == panel[key]
        for f in route['inspected_frames']:
            assert f['observed_sha256'] == frame_hashes[f['frame_index']]
        if variant == 'V':
            panel['question_text'] = route['question_text']

recovery = json.loads((BUNDLE / 'format_only_recovery/per_route_recovery/AV_SPEECH/q_medical_assistance/recovery_result.json').read_text())
assert recovery['new_frames'] == 0 and recovery['new_inspection_calls'] == 0
assert recovery['primary_completion_unchanged']

ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)
for t in times:
    p = BUNDLE / 'inspected_frames_all' / f'frame_{t:05d}.jpg'
    target = ASSETS / p.name
    if target.exists():
        assert sha(target) == frame_hashes[t]
    else:
        shutil.copy2(p, target)


class Draw:
    def __init__(self, c=None, h=1000):
        self.c, self.h = c, h
        self.boxes = []

    def paragraph(self, text, x, y, width, size=FONT, color=INK, bold=False, center=False):
        style = ParagraphStyle('p', fontName='Helvetica-Bold' if bold else 'Helvetica',
                               fontSize=size, leading=LEADING if size == FONT else size * 1.22,
                               textColor=HexColor(color), alignment=TA_CENTER if center else TA_LEFT,
                               spaceBefore=0, spaceAfter=0, splitLongWords=False)
        p = Paragraph(escape(text).replace('\n', '<br/>'), style)
        _, height = p.wrap(width, 1000)
        for word in text.replace('\n', ' ').split():
            assert stringWidth(word, style.fontName, size) <= width + .5, (word, width)
        if self.c:
            p.drawOn(self.c, x, self.h - y - height)
        self.boxes.append((x, y, width, height, text))
        return height

    def text(self, text, x, y, size=FONT, color=INK, bold=False, anchor='left'):
        font = 'Helvetica-Bold' if bold else 'Helvetica'
        width = stringWidth(text, font, size)
        left = x - width / 2 if anchor == 'center' else x - width if anchor == 'right' else x
        if self.c:
            self.c.setFont(font, size)
            self.c.setFillColor(HexColor(color))
            self.c.drawString(left, self.h - y - size, text)
        self.boxes.append((left, y, width, size * 1.2, text))

    def rect(self, x, y, w, h, fill, stroke=RULE, radius=0):
        if self.c:
            self.c.setFillColor(HexColor(fill))
            self.c.setStrokeColor(HexColor(stroke))
            self.c.setLineWidth(.6)
            if radius:
                self.c.roundRect(x, self.h - y - h, w, h, radius, stroke=1, fill=1)
            else:
                self.c.rect(x, self.h - y - h, w, h, stroke=1, fill=1)

    def line(self, x1, y1, x2, y2, color=RULE):
        if self.c:
            self.c.setStrokeColor(HexColor(color))
            self.c.setLineWidth(.55)
            self.c.line(x1, self.h - y1, x2, self.h - y2)

    def image(self, t, x, y, width, color):
        height = width * 720 / 1280
        if self.c:
            # JPEG passthrough in the vector PDF; no crop, enhancement or redraw.
            self.c.drawImage(str(ASSETS / f'frame_{t:05d}.jpg'), x, self.h - y - height,
                             width=width, height=height, preserveAspectRatio=True)
            self.c.setStrokeColor(HexColor(color))
            self.c.setLineWidth(.7)
            self.c.rect(x, self.h - y - height, width, height, stroke=1, fill=0)
        self.text(f'{t} s', x + width / 2, y + height + 3, color=color, anchor='center')


def draw_maps(d):
    y = 3
    y += d.paragraph('(a) Map organisation and temporal boundaries', 3, y, W-6, 11.5, bold=True) + 6
    y += d.paragraph('V and AV-Speech share the same frozen visual representation. Speech changes '
                     'the semantic map supplied to Direct C.', 3, y, W-6, color=MUTED) + 10
    y += d.paragraph('Human reference - schematic, not time-scaled', 3, y, W-6, bold=True) + 5
    labels = ['In-car', 'Exit / pursuit', 'Armed confrontation', 'Arrest / restraint',
              'Checking / aid prep.', 'Care / continued restraint']
    gap = 4
    cell = (W-6-gap*5)/6
    for i, label in enumerate(labels):
        x = 3+i*(cell+gap)
        d.rect(x, y, cell, 43, '#F4F6F8', radius=2)
        h = d.paragraph(label, x+3, y+4, cell-6, center=True)
        assert h <= 35.1
    y += 53
    y += d.paragraph('Model maps - a shared linear time axis (seconds)', 3, y, W-6, bold=True) + 6
    start, end = 57, W-13
    scale = lambda t: start+(end-start)*t/1235
    d.line(start, y+13, end, y+13, MUTED)
    for t in [0, 300, 600, 900, 1235]:
        x = scale(t)
        d.line(x, y+9, x, y+16, MUTED)
        d.text(str(t), x, y-2, anchor='center', color=MUTED)
    y += 27
    for key, name, color in [('V', 'V', BLUE), ('AV', 'AV-Speech', ORANGE)]:
        d.paragraph(name, 3, y+3, 50, color=color, bold=True)
        edges = constants[key]
        for i, (a, b) in enumerate(zip(edges, edges[1:])):
            left, right = scale(a), scale(b)
            d.rect(left, y, right-left, 21, '#F2F7FD' if key == 'V' else '#FFF6EE', color)
            label = f'C{i+1}'
            if stringWidth(label, 'Helvetica', FONT) + 4 > right-left:
                d.line((left+right)/2, y+21, (left+right)/2, y+26, color)
                d.text(label, end if i == 9 else (left+right)/2, y+27,
                       anchor='right' if i == 9 else 'center', color=color)
            else:
                assert stringWidth(label, 'Helvetica', FONT) < right-left
                d.text(label, (left+right)/2, y+4, anchor='center', color=color)
        y += 45 if key == 'AV' else 33
    y += 5
    col_gap = 12
    col = (W-6-2*col_gap)/3
    for key, name, labelkey, color, column, first, last in [
        ('V', 'V - 6 regions', 'vlabels', BLUE, 0, 0, 6),
        ('AV', 'AV-Speech: C1-C5', 'avlabels', ORANGE, 1, 0, 5),
        ('AV', 'AV-Speech: C6-C10', 'avlabels', ORANGE, 2, 5, 10),
    ]:
        x = 3+column*(col+col_gap)
        d.paragraph(name, x, y, col, bold=True, color=color)
        edges = constants[key]
        for i, (a, b, label) in enumerate(zip(edges, edges[1:], constants[labelkey])):
            if not first <= i < last:
                continue
            yy = y+21+(i-first)*39
            d.text(f'C{i+1}', x, yy, color=color, bold=True)
            d.text(f'{a}-{b} s', x+25, yy, color=MUTED)
            h = d.paragraph(label.replace('\n', ' '), x, yy+12, col, color=INK)
            assert h <= 24, (label, h)
    y += 21+6*39+4
    d.line(3, y, W-3, y)
    y += 7
    y += d.paragraph('Region labels are condensed, not verbatim model outputs. Full Coarse summaries '
                     'and uncertainty notes remain in the thesis appendix. The human reference is '
                     'post-hoc and did not guide navigation or frame selection.', 3, y, W-6, color=MUTED)
    return y+4


def draw_panel(d, panel, y):
    x = 3
    y += d.paragraph(f"({panel['letter']}) {panel['title']}", x, y, W-6, 10.5, bold=True) + 4
    y += d.paragraph('Q: '+panel['question_text'], x, y, W-6) + 6
    image_x, image_width, gap = 44, 71, 5
    row_height = image_width*720/1280+17
    for key, label, color, answer in [
        ('V', 'V', BLUE, panel['answer_v']),
        ('AV', 'AV-\nSpeech', ORANGE, panel['answer_av']),
    ]:
        d.paragraph(label, x, y+8, 37, color=color, bold=True)
        for i, t in enumerate(panel[key]):
            d.image(t, image_x+i*(image_width+gap), y, image_width, color)
        if len(panel[key]) == 3:
            tx = image_x+3*(image_width+gap)+7
            h = d.paragraph('Answer: '+answer, tx, y+3, W-tx-3, color=color)
            assert h <= row_height-3, (panel['letter'], key, h)
        y += row_height+3
    if len(panel['AV']) > 3:
        y += d.paragraph(panel['answer_av'], x, y+1, W-6, color=ORANGE) + 4
        y += d.paragraph(panel['recovery'], x, y, W-6, color=MUTED) + 4
    y += d.paragraph(panel['note'], x, y+2, W-6, color=MUTED) + 6
    return y


def draw_cases(d):
    y = 3
    y += d.paragraph('Direct inspection: V / AV-Speech comparisons', 3, y, W-6, 11.5, bold=True) + 3
    y += d.paragraph('Original frames; timestamps in seconds. Model answers are condensed.',
                     3, y, W-6, color=MUTED) + 9
    for i, panel in enumerate(panels):
        if i:
            d.line(3, y+1, W-3, y+1)
            y += 10
        y = draw_panel(d, panel, y)
    return y+3


figures = []
for name, draw in [('figure-1-maps', draw_maps), ('figure-2-cases', draw_cases)]:
    measure = Draw()
    height = math.ceil(draw(measure))
    assert height <= A4[1]-55*mm-24, (name, 'too tall at readable font size', height)
    for x, y, width, h, text in measure.boxes:
        assert x >= -.1 and x+width <= W+.1 and y >= 0 and y+h <= height+.1, (name, text, x, y, width, h)
    path = ROOT / (name+'.pdf')
    c = canvas.Canvas(str(path), pagesize=(W, height), pageCompression=1)
    c.setTitle(name+' - portrait-width preview')
    c.setAuthor('Xi Nan - layout preview')
    draw(Draw(c, height))
    c.showPage(); c.save()
    figures.append((name, draw, height))
    print(name, f'{W/mm:.1f} x {height/mm:.1f} mm; smallest text {FONT} pt')

page_path = ROOT / 'portrait-pages.pdf'
c = canvas.Canvas(str(page_path), pagesize=A4, pageCompression=1)
c.setTitle('Case study split-layout preview at actual thesis text width')
for n, (name, draw, height) in enumerate(figures, 1):
    left, top = 30*mm, A4[1]-30*mm
    c.setFont('Helvetica', 9.5)
    c.setFillColor(HexColor(MUTED))
    c.drawString(left, top-9.5, 'Layout preview | A4 portrait | 150 mm text width')
    c.saveState()
    c.translate(left, top-20-height)
    draw(Draw(c, height))
    c.restoreState()
    c.setFont('Helvetica', 10)
    c.drawCentredString(A4[0]/2, 15*mm, str(n))
    c.showPage()
c.save()

assert {str(p): sha(p) for p in protected} == before
assert all(sha(BUNDLE/'inspected_frames_all'/f'frame_{t:05d}.jpg') == frame_hashes[t] for t in times)
report = {
    'scope': 'Preview only. Thesis and original assets unchanged.',
    'code_source': str(ZIP), 'code_source_sha256': sha(ZIP),
    'boundary_source': 'repair.py constants; matched to bundled maps_v2_2 parsed maps',
    'human_reference': 'Schematic; exact human boundaries unavailable; not time-scaled',
    'image_presentations': sum(len(p[k]) for p in panels for k in ['V','AV']),
    'distinct_timestamps': len(times), 'distinct_file_hashes': len(set(frame_hashes.values())),
    'distinct_decoded_pixel_hashes': len(set(pixel_hashes.values())),
    'frame_sha256': frame_hashes, 'thesis_before_and_after_sha256': before,
    'font': 'Helvetica, 9.5 pt minimum; headings 10.5-11.5 pt',
    'display_order': 'chronological within each condition; not asserted to be inspection-request order',
    'evidence_images': 'Original complete 1280x720 JPEGs; no crop, redraw, enhancement or resampling in PDF',
}
(ROOT/'preview-checks.json').write_text(json.dumps(report, indent=2)+'\n')
print('Checked:', report['image_presentations'], 'presentations;', len(times), 'timestamps;',
      report['distinct_decoded_pixel_hashes'], 'distinct decoded frames')
print('Protected thesis/source assets: unchanged')
