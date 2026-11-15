
from rembg import remove
from PIL import Image
from pathlib import Path

base_dir = Path('/mnt/wav2vec2/khoan/source_code/RB-Ortho/ChronosArt/zoo_web/assets/animals')
for anim_dir in base_dir.iterdir():
    if not anim_dir.is_dir():
        continue
    for f in anim_dir.glob('*_frame_*.png'):
        if f.name.endswith('_nobg.png'):
            continue
        nobg_png = anim_dir / f.name.replace('.png', '_nobg.png')
        if not nobg_png.is_file():
            print(f'Removing BG: {f.name}...')
            im = Image.open(f)
            out = remove(im)
            out.save(nobg_png, 'PNG')

        # Convert both to WebP
        webp_full = anim_dir / f.name.replace('.png', '.webp')
        if not webp_full.is_file():
            Image.open(f).save(webp_full, 'WEBP', quality=90, method=6)

        webp_nobg = anim_dir / f.name.replace('.png', '_nobg.webp')
        if not webp_nobg.is_file() and nobg_png.is_file():
            Image.open(nobg_png).save(webp_nobg, 'WEBP', quality=90, method=6)

print('Background removal and WebP conversion complete!')
