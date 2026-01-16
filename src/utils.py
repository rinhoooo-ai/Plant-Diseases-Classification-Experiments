import torch
from pathlib import Path
import csv,json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def save_checkpoint(state:dict,out_dir:Path,name:str):
    out_dir.mkdir(parents=True,exist_ok=True)
    torch.save(state,out_dir/f'{name}.pth')

def write_csv(path:Path,rows:list,header=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        w=csv.writer(f)
        if header: w.writerow(header)
        for r in rows: w.writerow(r)

def save_metrics_summary(path:Path,metrics:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w') as f: json.dump(metrics,f,indent=2)

def plot_and_save_confusion(cm,labels,out_path:Path):
    out_path.parent.mkdir(parents=True,exist_ok=True)
    disp=ConfusionMatrixDisplay(cm,display_labels=labels)
    fig,ax=plt.subplots(figsize=(8,8))
    disp.plot(ax=ax,xticks_rotation=90)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
