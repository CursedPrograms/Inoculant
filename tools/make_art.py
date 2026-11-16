# Inoculant - development tool: draws the logo (assets/logo.svg, logo.png, icon.ico, icon.bmp and the site's copies)
# from the same shapes as the splash: a shield, grey on the left and Atelier blue on the right with a grain of noise,
# and a face drawn in one line across both halves. Needs Pillow:  python make_art.py
import os
import random
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DARK, BLUE, DEEP, LIGHT, FACE = (0x3a, 0x3a, 0x3a), (0x00, 0x7b, 0xb7), (0x00, 0x51, 0x84), (0x34, 0xa9, 0xe3), (0xe6, 0xe6, 0xe6)


def bez(p0, p1, p2, p3, n=40):
    out = []
    for i in range(1, n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


def shield(cx, cy, r):
    top, side = cy - r * 1.05, cy + r * 0.15
    pts = [(cx - r, top), (cx + r, top), (cx + r, side)]
    pts += bez((cx + r, side), (cx + r, cy + r * 0.7), (cx + r * 0.45, cy + r * 1.0), (cx, cy + r * 1.2))
    pts += bez((cx, cy + r * 1.2), (cx - r * 0.45, cy + r * 1.0), (cx - r, cy + r * 0.7), (cx - r, side))
    return pts


def draw(size):
    s = size * 4
    im = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = s * 0.40
    cx, cy = s / 2, s * 0.47
    pts = shield(cx, cy, r)
    d.polygon(pts, fill=DARK)
    blue = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    bd = ImageDraw.Draw(blue)
    bd.polygon(pts, fill=BLUE)
    rng = random.Random(12345)
    if size >= 48:
        for i in range(260):
            x, y, q = cx + rng.random() * r, cy - r + rng.random() * r * 2.1, r * 0.035
            bd.rectangle([x, y, x + q, y + q], fill=LIGHT if i & 1 else DEEP)
    mask = Image.new('L', (s, s), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    half = Image.new('L', (s, s), 0)
    ImageDraw.Draw(half).rectangle([cx, 0, s, s], fill=255)
    from PIL import ImageChops
    im.paste(blue, (0, 0), ImageChops.multiply(mask, half))
    fr, fx, fy, lw = r * 0.48, cx, cy - r * 0.12, max(2, r * 0.06)
    d = ImageDraw.Draw(im)
    d.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], outline=FACE, width=int(lw))
    e = fr * 0.12
    for ex in (fx - fr * 0.38, fx + fr * 0.38):
        d.ellipse([ex - e, fy - fr * 0.15 - e, ex + e, fy - fr * 0.15 + e], fill=FACE)
    d.line([fx - fr * 0.35, fy + fr * 0.42, fx + fr * 0.35, fy + fr * 0.42], fill=FACE, width=int(lw))
    return im.resize((size, size), Image.LANCZOS)


def svg():
    s, r = 512, 512 * 0.40
    cx, cy = 256, 512 * 0.47
    pts = shield(cx, cy, r)
    path = 'M' + ' L'.join('%.1f %.1f' % p for p in pts) + ' Z'
    rng = random.Random(12345)
    grain = []
    for i in range(260):
        x, y, q = cx + rng.random() * r, cy - r + rng.random() * r * 2.1, r * 0.035
        grain.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"/>' % (x, y, q, q, '#34a9e3' if i & 1 else '#005184'))
    fr, fx, fy = r * 0.48, cx, cy - r * 0.12
    lw = r * 0.06
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs><clipPath id="s"><path d="{path}"/></clipPath><clipPath id="r"><rect x="{cx}" y="0" width="{s - cx}" height="{s}"/></clipPath></defs>
  <path d="{path}" fill="#3a3a3a"/>
  <g clip-path="url(#s)"><g clip-path="url(#r)"><rect x="{cx}" y="0" width="{s - cx}" height="{s}" fill="#007bb7"/>{''.join(grain)}</g></g>
  <circle cx="{fx:.1f}" cy="{fy:.1f}" r="{fr:.1f}" fill="none" stroke="#e6e6e6" stroke-width="{lw:.1f}"/>
  <circle cx="{fx - fr * 0.38:.1f}" cy="{fy - fr * 0.15:.1f}" r="{fr * 0.12:.1f}" fill="#e6e6e6"/>
  <circle cx="{fx + fr * 0.38:.1f}" cy="{fy - fr * 0.15:.1f}" r="{fr * 0.12:.1f}" fill="#e6e6e6"/>
  <line x1="{fx - fr * 0.35:.1f}" y1="{fy + fr * 0.42:.1f}" x2="{fx + fr * 0.35:.1f}" y2="{fy + fr * 0.42:.1f}" stroke="#e6e6e6" stroke-width="{lw:.1f}"/>
</svg>
'''


if __name__ == '__main__':
    for d in ('assets', 'site'):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    big = draw(512)
    big.save(os.path.join(ROOT, 'assets', 'logo.png'))
    draw(256).save(os.path.join(ROOT, 'site', 'logo.png'))
    sizes = [16, 24, 32, 48, 64, 128, 256]
    icons = [draw(n) for n in sizes]
    icons[-1].save(os.path.join(ROOT, 'assets', 'icon.ico'), sizes=[(n, n) for n in sizes], append_images=icons[:-1])
    icons[2].save(os.path.join(ROOT, 'site', 'favicon.ico'), sizes=[(16, 16), (32, 32), (48, 48)], append_images=[icons[0], icons[3]])
    bmp = Image.new('RGB', (64, 64), (0x1a, 0x1a, 0x1a))
    bmp.paste(icons[4], (0, 0), icons[4])
    bmp.save(os.path.join(ROOT, 'assets', 'icon.bmp'))
    for d in ('assets', 'site'):
        with open(os.path.join(ROOT, d, 'logo.svg'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(svg())
    print('drew the logo')
