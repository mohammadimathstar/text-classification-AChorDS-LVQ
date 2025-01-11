import matplotlib.pyplot as plt
import pandas as pd
import yaml


def plot_top_words(df, fig_path):

    df.sort_values(by='decision', inplace=True)
    imp_display2 = df.copy()

    cm = 1/2.54  # centimeters in inches

    fg_w,fg_h= 7.5,6
    fs_title, fs_others=6,5

    fg2,axis2=plt.subplots(figsize=(fg_w*cm,fg_h*cm), layout="tight")

    imp_display2.plot.barh(ax=axis2)
    axis2.set_ylabel('Words', fontsize=fs_title)
    axis2.set_xlabel('Impact score', fontsize=fs_title)
    axis2.set_yticklabels(imp_display2.index, fontsize=fs_others)
    axis2.grid(color='gray', linestyle='dashed', linewidth = 0.25)
    axis2.legend(loc='best', title='Importance',
                 fontsize=fs_others, title_fontsize=fs_others)
    axis2.tick_params(axis='both', which='major', labelsize=fs_others)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight', dpi=300, pad_inches=0)