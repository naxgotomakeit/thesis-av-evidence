"""Compact vector layout, using the established split-figure content and JPEGs.

Only pure drawing definitions are loaded from the original layout script;
its asset auditing and output code are not executed.
"""
from pathlib import Path
import ast
import json
import math
import zipfile
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Paragraph

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'output/previews/case-study-split-v1/build_preview.py'
ASSETS = SOURCE.parent / 'assets'
OUT = PROJECT / 'output/figures/case-compact'
W = 160 * mm
FONT, LEADING = 8.5, 10.2
BLUE, ORANGE, INK, MUTED, RULE = '#086BFF', '#C94C00', '#17212D', '#435062', '#BCC6D0'

tree = ast.parse(SOURCE.read_text())
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'Draw':
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'))
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'panels' for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'))

with zipfile.ZipFile('/Users/nax/Downloads/case_study_code.zip') as archive:
    constants = {}
    for node in ast.parse(archive.read('repair.py').decode()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'V', 'AV', 'vlabels', 'avlabels'}:
                    constants[target.id] = ast.literal_eval(node.value)

for letter, panel in zip('abc', panels):
    panel['letter'] = letter
    path = Path('/Users/nax/Downloads/226_case_study_review_bundle_v1/direct_sixq/routes/V') / panel['question'] / 'route_result.json'
    panel['question_text'] = json.loads(path.read_text())['question_text']


def draw_maps(d):
    y = 2
    y += d.paragraph('Map organisation and temporal boundaries', 3, y, W-6, 10.5, bold=True) + 3
    y += d.paragraph('V and AV-Speech share the same frozen visual representation. Speech changes '
                     'the semantic map supplied to Direct C.', 3, y, W-6, color=MUTED) + 5
    y += d.paragraph('Human reference - schematic, not time-scaled', 3, y, W-6, bold=True) + 3
    labels = ['In-car', 'Exit / pursuit', 'Armed confrontation', 'Arrest / restraint',
              'Checking / aid prep.', 'Care / continued restraint']
    gap = 4
    cell = (W-6-gap*5)/6
    heights = [Draw().paragraph(label, 0, 0, cell-6, center=True) for label in labels]
    box_h = max(heights)+6
    for i, (label, height) in enumerate(zip(labels, heights)):
        x = 3+i*(cell+gap)
        d.rect(x, y, cell, box_h, '#F4F6F8', radius=2)
        d.paragraph(label, x+3, y+(box_h-height)/2, cell-6, center=True)
    y += box_h+6
    y += d.paragraph('Model maps - a shared linear time axis (seconds)', 3, y, W-6, bold=True) + 3
    start, end = 56, W-12
    scale = lambda t: start+(end-start)*t/1235
    d.line(start, y+12, end, y+12, MUTED)
    for t in [0, 300, 600, 900, 1235]:
        x = scale(t)
        d.line(x, y+9, x, y+15, MUTED)
        d.text(str(t), x, y-1, anchor='center', color=MUTED)
    y += 21
    for key, name, color in [('V', 'V', BLUE), ('AV', 'AV-Speech', ORANGE)]:
        d.paragraph(name, 3, y+3, 50, color=color, bold=True)
        for i, (a, b) in enumerate(zip(constants[key], constants[key][1:])):
            left, right = scale(a), scale(b)
            d.rect(left, y, right-left, 17, '#F2F7FD' if key == 'V' else '#FFF6EE', color)
            label = f'C{i+1}'
            if stringWidth(label, 'Helvetica', FONT)+4 > right-left:
                d.line((left+right)/2, y+17, (left+right)/2, y+20, color)
                d.text(label, end if i == 9 else (left+right)/2, y+21,
                       anchor='right' if i == 9 else 'center', color=color)
            else:
                d.text(label, (left+right)/2, y+3, anchor='center', color=color)
        y += 34 if key == 'AV' else 23
    y += 3
    gap = 10
    col = (W-6-2*gap)/3
    columns = [
        ('V', 'V - 6 regions', 'vlabels', BLUE, 0, 6),
        ('AV', 'AV-Speech: C1-C5', 'avlabels', ORANGE, 0, 5),
        ('AV', 'AV-Speech: C6-C10', 'avlabels', ORANGE, 5, 10),
    ]
    ends = []
    for column, (key, name, labelkey, color, first, last) in enumerate(columns):
        x = 3+column*(col+gap)
        yy = y+d.paragraph(name, x, y, col, bold=True, color=color)+4
        for i in range(first, last):
            a, b = constants[key][i:i+2]
            d.text(f'C{i+1}', x, yy, color=color, bold=True)
            d.text(f'{a}-{b} s', x+24, yy, color=MUTED)
            yy += 11
            yy += d.paragraph(constants[labelkey][i].replace('\n', ' '), x, yy, col)+4
        ends.append(yy)
    y = max(ends)
    d.line(3, y, W-3, y)
    y += 4
    y += d.paragraph('Region labels are condensed, not verbatim model outputs. Full Coarse summaries '
                     'and uncertainty notes remain in the thesis appendix. The human reference is '
                     'post-hoc and did not guide navigation or frame selection.', 3, y, W-6, color=MUTED)
    return y+2


def draw_panel(d, panel, y):
    x = 3
    y += d.paragraph(f"({panel['letter']}) {panel['title']}", x, y, W-6, 9.5, bold=True)+2
    y += d.paragraph('Q: '+panel['question_text'], x, y, W-6)+3
    image_x, image_width, gap = 42, 71, 5
    row_h = image_width*720/1280+15
    for key, label, color, answer in [
        ('V', 'V', BLUE, panel['answer_v']),
        ('AV', 'AV-\nSpeech', ORANGE, panel['answer_av']),
    ]:
        d.paragraph(label, x, y+6, 35, color=color, bold=True)
        for i, t in enumerate(panel[key]):
            d.image(t, image_x+i*(image_width+gap), y, image_width, color)
        if len(panel[key]) == 3:
            tx = image_x+3*(image_width+gap)+6
            h = d.paragraph('Answer: '+answer, tx, y+2, W-tx-3, color=color)
            assert h <= row_h-2
        y += row_h+1
    if len(panel['AV']) > 3:
        y += d.paragraph(panel['answer_av'], x, y, W-6, color=ORANGE)+2
        y += d.paragraph(panel['recovery'], x, y, W-6, color=MUTED)+2
    y += d.paragraph(panel['note'], x, y+1, W-6, color=MUTED)+4
    return y


def draw_cases(d):
    y = 2
    y += d.paragraph('Direct inspection: V / AV-Speech comparisons', 3, y, W-6, 10.5, bold=True)+2
    y += d.paragraph('Original frames; timestamps in seconds. Model answers are condensed.',
                     3, y, W-6, color=MUTED)+4
    for i, panel in enumerate(panels):
        if i:
            d.line(3, y, W-3, y)
            y += 5
        y = draw_panel(d, panel, y)
    return y+2


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in [('maps', draw_maps), ('evidence', draw_cases)]:
        measure = Draw()
        height = math.ceil(draw(measure))
        for x, y, width, h, text in measure.boxes:
            assert x >= -.1 and x+width <= W+.1 and y >= 0 and y+h <= height+.1, (name, text)
        c = canvas.Canvas(str(OUT / (name+'.pdf')), pagesize=(W, height), pageCompression=1)
        c.setTitle('EgoPolice '+name+' - compact vector layout')
        draw(Draw(c, height))
        c.showPage()
        c.save()
        print(f'{name}: {W/mm:.1f} x {height/mm:.1f} mm; body font {FONT} pt')
