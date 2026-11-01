"""Results chart for the FF++ -> DFDC experiments (numbers from fuse.py output, 10,000 DFDC frames)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
BG='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'
metrics=['Video AUROC','Video accuracy','Frame AUROC','Frame accuracy']
models=[('GenD (frozen features + linear head)','#2a78d6',[82.352,74.15,79.88,72.43]),
        ('ForensicsAdapter (authors\' weights)','#eb6834',[83.02,74.80,80.75,73.24]),
        ('Hybrid: GenD + ForensicsAdapter','#1baf7a',[84.92,76.24,82.56,74.80])]
fig,ax=plt.subplots(figsize=(10.5,6.2),dpi=200,facecolor=BG); ax.set_facecolor(BG)
h=0.24; y0=np.arange(len(metrics))[::-1]
for i,(name,col,vals) in enumerate(models):
    ys=y0+(1-i)*(h+0.03)
    bars=ax.barh(ys,vals,height=h,color=col,label=name,zorder=3)
    for y,v in zip(ys,vals): ax.text(v+0.8,y,'%.1f%%'%v,va='center',ha='left',fontsize=10.5,color=INK,fontweight='bold' if i==2 else 'normal',zorder=4)
ax.set_yticks(y0); ax.set_yticklabels(metrics,fontsize=12,color=INK)
ax.set_xlim(0,100); ax.set_xticks([0,20,40,60,80,100]); ax.set_xticklabels(['0','20','40','60','80','100%'],fontsize=10,color=INK2)
ax.xaxis.grid(True,color=GRID,linewidth=0.8,zorder=0); ax.set_axisbelow(True)
for s in ('top','right','left'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(GRID); ax.tick_params(axis='y',length=0); ax.tick_params(axis='x',length=0)
leg=ax.legend(loc='lower center',bbox_to_anchor=(0.5,-0.2),ncol=3,frameon=False,fontsize=10,labelcolor=INK2,handlelength=1.2)
fig.suptitle('Trained on FaceForensics++, tested on unseen DFDC: the hybrid beats both detectors alone',x=0.02,ha='left',fontsize=13.5,color=INK,fontweight='bold',y=0.965)
ax.set_title('10,000 test frames (5,000 real / 5,000 fake) from 4,355 videos. Higher is better.',loc='left',fontsize=10.5,color=INK2,pad=12)
fig.subplots_adjust(left=0.22,right=0.97,top=0.84,bottom=0.17); fig.savefig('figures/results_summary.png',facecolor=BG); print('ok')
