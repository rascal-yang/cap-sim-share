from pathlib import Path
from PIL import Image,ImageDraw
r=Path(__file__).resolve().parent
for group in ['no_reference','reference_augmented']:
 ps=sorted((r/'videos'/group).glob('*/verification.json'))
 if not ps:continue
 out=Image.new('RGB',(960,len(ps)*210),'white');d=ImageDraw.Draw(out)
 for j,p in enumerate(ps):
  d.text((5,j*210),p.parent.name,fill='black')
  for k,view in enumerate(['ego','external']):
   frames=sorted(p.parent.glob(view+'_*.png'))
   for i,f in enumerate([frames[len(frames)//2],frames[-1]]):out.paste(Image.open(f).resize((240,180)),(240*(2*k+i),j*210+25))
 out.save(r/(group+'_review.jpg'))
