
from rembg import remove
from PIL import Image
from pathlib import Path

base_dir = Path('/mnt/wav2vec2/khoan/source_code/RB-Ortho/ChronosArt/zoo_web/assets/animals')
for anim_dir in base_dir.iterdir():
    if not anim_dir.is_dir():
        continue
    for f in anim_dir.glob('frame_*.png'):
        if f.name.endswith('_nobg.png'):
            continue
        out_f = anim_dir / f.name.replace('.png', '_nobg.png')
        if not out_f.is_file():
            print(f'Removing background for {f}...')
            im = Image.open(f)
            nobg = remove(im)
            nobg.save(out_f, 'PNG')
            print(f'Saved -> {out_f.name}')
print('Background removal completed successfully!')
