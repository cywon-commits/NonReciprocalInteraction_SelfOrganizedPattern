"""Collect the nine figures with their captions (CAPTIONS.md) into out/B1_figures.pdf for review."""
import re, textwrap, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
txt = open('CAPTIONS.md').read(); parts = re.split(r'\n## ', txt)[1:]
names = ['fig1_concept', 'fig2_symmetry', 'fig3_walker', 'fig4_helix', 'fig5_skx', 'fig6_stripes', 'fig7_lambda', 'fig8_gibbs', 'fig9_sot']
with PdfPages('out/B1_figures.pdf') as pdf:
    for nm, p in zip(names, parts):
        lines = p.splitlines(); title = lines[0]; body = [l for l in lines[1:] if l.strip() and not l.startswith('Data:')]
        cap = ' '.join(body).replace('**', '')
        fig = plt.figure(figsize=(8.27, 11.69)); ax = fig.add_axes([0.2, 0.42, 0.6, 0.53]); ax.imshow(mpimg.imread(f'out/{nm}.png')); ax.axis('off')
        fig.text(0.08, 0.38, 'Fig. ' + title.split(' —')[0].replace('Fig. ', ''), fontsize=11, fontweight='bold', va='top')
        fig.text(0.08, 0.35, '\n'.join(textwrap.wrap(cap, 110)), fontsize=8.5, va='top', family='DejaVu Sans')
        pdf.savefig(fig); plt.close(fig)
print('ok')
