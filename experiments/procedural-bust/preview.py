"""Create review contact sheets using the project venv (Pillow)."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def font(size):
    return ImageFont.truetype('DejaVuSans.ttf', size)


def sheet(paths, labels, output, columns=2, size=600, title='PROCEDURAL BUST / SDF STUDY'):
    rows = (len(paths) + columns - 1) // columns
    image = Image.new('RGB', (size * columns, 90 + rows * (size + 44)), '#11171d')
    draw = ImageDraw.Draw(image)
    draw.text((24, 20), title, font=font(25), fill='#f1e7d6')
    draw.text((24, 55), 'NumPy fields > OpenVDB mesh > Blender Cycles | original procedural geometry', font=font(15), fill='#94a4b0')
    for i, (path, label) in enumerate(zip(paths, labels)):
        x, y = i % columns * size, 90 + i // columns * (size + 44)
        tile = Image.open(path).convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        image.paste(tile, (x, y + 44))
        draw.text((x + 22, y + 12), label, font=font(20), fill='#d6dfe5')
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    views = ['front', 'side', 'back', 'three-quarter']
    for run in ['03-final', '04-turned']:
        folder = args.root / run
        sheet([folder / f'{v}.png' for v in views],
              ['FRONT', 'PROFILE', 'BACK', 'THREE QUARTER'], folder / 'contact-sheet.jpg')
    runs = ['01-draft', '03-final', '04-turned']
    sheet([args.root / r / 'three-quarter.png' for r in runs],
          ['01 / INITIAL DRAFT', '02 / REVISED FORM', '03 / PARAMETER VARIANT'],
          args.root / 'comparison.jpg', columns=3, size=540,
          title='PROCEDURAL BUST / ITERATION & PARAMETERS')


if __name__ == '__main__':
    main()
