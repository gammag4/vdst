import os
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator
from scipy.signal import savgol_filter
import torch
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv


@dataclass
class Metric:
    name: str
    id: str
    higher_is_better: bool


def get_metric_ids(metrics):
    return [m.id for m in metrics]


metrics_img = [
    Metric('PSNR', 'metrics/eval.images.psnr', True),
    Metric('SSIM', 'metrics/eval.images.ssim', True),
    Metric('LPIPS', 'metrics/eval.images.lpips', False),
]
metrics_depth1 = [
    Metric('Abs Rel', 'metrics/eval.depths.abs_rel', False),
    Metric('Log 10', 'metrics/eval.depths.mean_log10', False),
    Metric(r'$\delta_1$', 'metrics/eval.depths.delta_1_25', True),
]
metrics_depth_only1 = [
    Metric('Abs Rel', 'metrics/eval.depths.abs_rel', False),
    Metric('RMSE Log', 'metrics/eval.depths.rmse_log', False),
    Metric('Log 10', 'metrics/eval.depths.mean_log10', False),
    Metric('Sq Rel', 'metrics/eval.depths.sq_rel', False),
    Metric(r'$\delta_1$', 'metrics/eval.depths.delta_1_25', True),
    Metric(r'$\delta_3$', 'metrics/eval.depths.delta_1_25_3', True)
]
metrics_depth_only = [
    Metric('Abs Rel', 'metrics/eval.depths.abs_rel', False),
    Metric('Sq Rel', 'metrics/eval.depths.sq_rel', False),
    Metric('RMSE', 'metrics/eval.depths.rmse', False),
    Metric('RMSE Log', 'metrics/eval.depths.rmse_log', False),
    Metric('Log 10', 'metrics/eval.depths.mean_log10', False),
    Metric('SILog', 'metrics/eval.depths.silog', False),
    Metric(r'$\delta_1$', 'metrics/eval.depths.delta_1_25', True),
    Metric(r'$\delta_2$', 'metrics/eval.depths.delta_1_25_2', True),
    Metric(r'$\delta_3$', 'metrics/eval.depths.delta_1_25_3', True)
]
total_metrics = metrics_img + metrics_depth_only

global_tables = []

load_dotenv()

api = wandb.Api()


def create_table(table_label, table_caption, metrics, final_metrics, run_names):
    if len(metrics) > 6:
        m = math.ceil(len(metrics) / 2)
        metrics = [metrics[:m], metrics[m:]]
    else:
        metrics = [metrics]
    
    tbodies = []
    for i, ms in enumerate(metrics):
        isodd = i == 1 and len(metrics[0]) != len(metrics[1])
        max_best = [m.higher_is_better for m in ms]
        final_metrics2 = torch.tensor([[m for m in h[1]] for h in final_metrics[get_metric_ids(ms)].iterrows()])
        best_vals_min = final_metrics2.argmin(dim=-2)
        best_vals_max = final_metrics2.argmax(dim=-2)
        column_definition = '|c' * (len(ms) + (1 if isodd else 0) + 1) + '|'
        thead = ' & '.join([m.name + (r' $\uparrow$' if m.higher_is_better else r' $\downarrow$') for m in ms] + ([''] if isodd else []))
        tbody = '\n'.join([f'      {n} & {' & '.join([(r'$\mathbf{' + f'{m.item():.4f}' + '}$') if (i == vmax if mbest else i == vmin) else f'${m.item():.4f}$' for m, vmin, vmax, mbest in zip(h, best_vals_min, best_vals_max, max_best)] + ([''] if isodd else []))} \\\\\n      \\hline' for n, (i, h) in zip(run_names, enumerate(final_metrics2))])
        tbody = f'\\hline\n      & {thead} \\\\\n      \\hline\n{tbody}'
        tbodies.append(tbody)
    tbody = '\n      '.join(tbodies)
    
    table = r'''
\begin{quadro}[htpb]
  \centering
  \Caption{\label{qua:table_label} table_caption}
  \UFCqua{}{
    \begin{tabular}{column_definition}
      tbody
    \end{tabular}
  }{
    \Fonte{Elaborado pelo autor.}
  }
\end{quadro}
    '''.strip().replace('table_label', table_label).replace('table_caption', table_caption).replace('column_definition', column_definition).replace('tbody', tbody)
    
    return table


def create_plot(results_path, exp_name, plot_label, plot_caption, runs_histories, plot_metrics, steps, run_names):
    num_runs = len(runs_histories)
    # (metric, run, iteration)
    runs_histories = [torch.tensor(h.to_numpy(dtype=float)).T for h in runs_histories]
    runs_histories = [[h[i] for h in runs_histories] for i in range(len(runs_histories[0]))]
    # print([h.shape for h in runs_histories])
    # mlen = min([i.shape[-1] for i in runs_histories])
    # runs_histories = [i[:, :mlen] for i in runs_histories]
    # print([h.shape for h in runs_histories])
    # runs_histories = torch.stack(runs_histories).permute(1, 0, 2)
    
    ploty, plotx = math.ceil(len(plot_metrics) / 3), min(len(plot_metrics), 3)
    s = 4.5
    sx, sy = 1.4 * s, s
    fig, axes = plt.subplots(ploty, plotx, figsize=(sx * plotx, sy * ploty))
    axes = [axes] if (ploty == 1 and plotx == 1) else axes.flatten()
    lw = 0.8
    
    for ax, histories, metric in zip(axes, runs_histories, plot_metrics):
        histories2 = [h[-round(0.4 * len(s) ** 0.7):].mean(dim=-1).item() for h, s in zip(histories, steps)]
        best_index = histories2.index((max if metric.higher_is_better else min)(histories2))
        # best_index = (histories2.argmax(dim=-2) if metric.higher_is_better else histories2.argmin(dim=-2)).mode().values.item()
        
        histories_smooth = [torch.tensor(savgol_filter(h.log().numpy(), window_length=round(0.4 * len(s) ** 0.7), polyorder=3)).exp() for h, s in zip(histories, steps)]
        
        ax.set_title(metric.name + ('' if metric.higher_is_better is None else (r' $\uparrow$' if metric.higher_is_better else r' $\downarrow$')))
        ax.set_prop_cycle(color=plt.cm.tab10.colors)
        
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        
        last = round(0.5 * len(steps[0]))
        
        if num_runs > 1:
            htotal = [h.log10() for h in histories]
            hend = sorted([h[-1].log10().item() for h in histories_smooth])
            htotal_range = max([i.max().item() for i in htotal]) - min([i.min().item() for i in htotal])
            hend = (hend[-2], hend[-1]) if metric.higher_is_better else (hend[0], hend[1])
            hend_range = hend[1] - hend[0]
            should_zoom_graph = hend_range / htotal_range < 0.02
        else:
            should_zoom_graph = False
        
        if should_zoom_graph:
            axins = ax.inset_axes([0.05, 0.65 if metric.higher_is_better else 0.05, 0.3, 0.3])
            axins.get_xaxis().set_visible(False)
            axins.get_yaxis().set_visible(False)
            axins.set_xscale("log")
            axins.set_yscale("log")
            axins.set_xlim(min([s[-last:].min() for s in steps]), max([s[-last:].max() for s in steps]))
            axins.set_ylim(min([t[-last:].min() for t in histories]), max([t[-last:].max() for t in histories]))
            ax.indicate_inset_zoom(axins, edgecolor="black")
        
        plt.gca().set_prop_cycle(None)
        
        for i, (t, t_smooth, s, run_name) in enumerate(zip(histories, histories_smooth, steps, run_names)):
            t, t_smooth = [i.numpy() for i in (t, t_smooth)]
            
            # if len(steps) < 1000:
            #     ax.loglog(
            #         steps,
            #         t,
            #         linestyle='-' if i == best_index else '--',
            #         lw=lw,
            #         label=run_name,
            #         zorder=10 if i == best_index else 1,
            #     )
            #     continue
            
            color = plt.gca()._get_lines.get_next_color()
            
            plotters = (ax, axins) if should_zoom_graph else (ax,)
            
            for a, x, y in zip(plotters, (s, s[-last:]), (t, t[-last:])):
                a.loglog(
                    x,
                    y,
                    linestyle='-',
                    lw=lw,
                    zorder=10 if i == best_index else 1,
                    alpha=0.4,
                    color=color,
                )
            
            for a, x, y in zip(plotters, (s, s[-last:]), (t_smooth, t_smooth[-last:])):
                a.loglog(
                    x,
                    y,
                    linestyle='-' if i == best_index else '--',
                    lw=lw,
                    label=run_name,
                    zorder=11 if i == best_index else 2,
                    color=color,
                )
        
        ax.legend(ncols=2, fontsize='small', loc='lower right' if metric.higher_is_better else 'upper right')
        
        # fmt = mticker.ScalarFormatter()
        # fmt.set_scientific(False)
        # fmt.set_useOffset(False)
        
        # ax.yaxis.set_major_formatter(fmt)
        # ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        
        # print(histories.min().item(), histories.max().item())
        # ymin, ymax = histories.min().item() or 1e-12, histories.max().item() or -1e-12
        # ax.set_yticks([ymin, ymax])
        
        # ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        # print(ax.get_ylim())
        # print(ax.get_yticks(minor=True))
        # ax.set_yticks(ax.get_yticks(minor=True))
        
        # ax.minorticks_off()
        # minlim, maxlim = histories.min().item(), histories.max().item()
        # l = (maxlim / minlim) ** (1/3)
        # yticks = [minlim * l ** i for i in range(4)]
        # ax.set_yticks(yticks)
        # ax.set_yticklabels([f'{v:.2e}' for v in yticks])
        
        # ax.minorticks_on()
        # ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
        # ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999))
    
    plt.tight_layout()
    
    figs_path = os.path.join(results_path, 'figuras/experimentos')
    os.makedirs(figs_path, exist_ok=True)
    plot_path = os.path.join(figs_path, f'{exp_name}.png')
    plt.savefig(plot_path)
    
    plot_table = r'''
\begin{figure}[htpb]
  \centering
  \captionsetup{width=16cm}%Da mesma largura que a figura, max 16cm
  \Caption{\label{fig:plot_label} plot_caption}
  \UFCfig{}{
    \includegraphics[width=16cm]{figuras/experimentos/plot_name.png}
  }{
    \Fonte{Elaborada pelo autor.}
  }
\end{figure}
    '''.strip().replace('plot_label', plot_label).replace('plot_caption', plot_caption).replace('plot_name', exp_name)
    return plot_table


# metric_histories: (run[],), metric_names: (metric[],) run_names: (run[],)
def create_exp_results(results_path, run_api_urls, custom_metrics, has_images, has_depths, run_names, exp_name, label, plot_caption, table_1_caption=None):
    if custom_metrics is None:
        plot_metrics, table_metrics = [], []
        if has_images:
            plot_metrics += metrics_img
            table_metrics += metrics_img
        if has_depths:
            plot_metrics += metrics_depth1 if has_images else metrics_depth_only1
            table_metrics += metrics_depth_only
    else:
        plot_metrics = custom_metrics
        table_metrics = custom_metrics
    
    history_path = os.path.join(results_path, 'bak', exp_name)
    runs_histories = []
    if os.path.isdir(history_path):
        for i in sorted(os.listdir(history_path)):
            runs_histories.append(pd.read_pickle(os.path.join(history_path, i)))
    else:
        for run_urls in run_api_urls:
            if isinstance(run_urls, str):
                run_urls = [run_urls]
            
            h = []
            for run_url in run_urls:
                run = api.run(run_url)
                h.append(pd.DataFrame(run.scan_history(keys=get_metric_ids(total_metrics if custom_metrics is None else custom_metrics) + ['_step'])).rename(columns={'_step': 'step'}).set_index('step'))
            
            df = h[-1]
            for i in h[1::-1]:
                df = df.combine_first(i)
            
            runs_histories.append(df)
        
        os.makedirs(history_path, exist_ok=True)
        for i, h in enumerate(runs_histories):
            h.to_pickle(os.path.join(history_path, f'{i}.pkl'))
    
    steps = [h.index.to_numpy() for h in runs_histories]
    last_step = min([s[-1] for s in steps])
    smasks = [(s >= 0) & (s <= last_step) for s in steps]
    steps = [s[smask] for s, smask in zip(steps, smasks)]
    print('steps', len(steps[0]))
    runs_histories = [h.iloc[smask] for h, smask in zip(runs_histories, smasks)]
    for i in runs_histories:
        print('history len', len(i))
    final_metrics = [(lambda x: x.iloc[0] if len(x.shape) > 1 else x)(h.loc[last_step]) for h in runs_histories]
    
    # (run, metric)
    final_metrics = pd.DataFrame(final_metrics)
    
    tables_path = os.path.join(results_path, 'tables')
    os.makedirs(tables_path, exist_ok=True)
    tables_path = os.path.join(tables_path, f'{exp_name}.tex')
    
    plot_label = label
    table_1_label = f'{label}1'
    
    table_1_caption = plot_caption if table_1_caption is None else table_1_caption
    
    table = create_table(table_1_label, table_1_caption, table_metrics, final_metrics[get_metric_ids(table_metrics)], run_names)
    
    plot_table = create_plot(results_path, exp_name, plot_label, plot_caption, [h[get_metric_ids(plot_metrics)] for h in runs_histories], plot_metrics, steps, run_names)
    table = f'{plot_table}\n\n{table}'
    
    with open(tables_path, 'w', encoding='utf8') as f:
        f.write(table)
    
    global global_tables
    global_tables.append(table)


results_path = 'results/logs'
loss_metric1 = Metric('Loss', 'loss', False)
lr_metric = Metric('Learning Rate', 'optimizer_lrs.0', None)
loss_metric2 = Metric('Loss', 'metrics/loss', False)


# create_exp_results(
#     results_path=results_path,
#     run_api_urls=[
#         'gammag9-none/vdst/f5hbf4p7',
#     ],
#     custom_metrics=[loss_metric1],
#     has_images=True,
#     has_depths=True,
#     run_names=[
#         r'$1.0 \times 10^{-2}$ 10k passos',
#     ],
#     exp_name='lr_range_test1',
#     label='exp-lr-range-first',
#     plot_caption='Resultados do teste de faixa de taxas de aprendizado.',
# )

# create_exp_results(
#     results_path=results_path,
#     run_api_urls=[
#         'gammag9-none/vdst/f5hbf4p7',
#     ],
#     custom_metrics=[lr_metric],
#     has_images=True,
#     has_depths=True,
#     run_names=[
#         r'$1.0 \times 10^{-2}$ 10k passos',
#     ],
#     exp_name='lr_range_test2',
#     label='exp-lr-range-first',
#     plot_caption='Resultados do teste de faixa de taxas de aprendizado.',
# )

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        'gammag9-none/vdst/wgb6i3i4',
        'gammag9-none/vdst/qn12eya2',
        'gammag9-none/vdst/dfu2bu4b',
        'gammag9-none/vdst/u05676o8',
    ],
    custom_metrics=[loss_metric1],
    has_images=True,
    has_depths=True,
    run_names=[
        r'$2.0 \times 10^{-3}$ 5k aquecimento',
        r'$1.0 \times 10^{-3}$',
        r'$1.0 \times 10^{-4}$',
        r'$1.0 \times 10^{-5}$',
    ],
    exp_name='lr_first',
    label='exp-lr-first',
    plot_caption='Resultados do primeiro experimento de taxa de aprendizado.',
)

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        'gammag9-none/vdst/d11g4oc6',
        'gammag9-none/vdst/mdqbge0r',
        ['gammag9-none/vdst/rk6jj2m8', 'gammag9-none/vdst/y91xhsun'],
    ],
    custom_metrics=[loss_metric2],
    has_images=True,
    has_depths=True,
    run_names=[
        r'$2.0 \times 10^{-4}$',
        r'$1.0 \times 10^{-4}$',
        r'$5.0 \times 10^{-5}$',
    ],
    exp_name='lr_second',
    label='exp-lr-second',
    plot_caption='Resultados do segundo experimento de taxa de aprendizado.',
)

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        'gammag9-none/vdst/4ns27cba',
        ['gammag9-none/vdst/m8pf8iaq', 'gammag9-none/vdst/85l5uwan'],
    ],
    custom_metrics=None,
    has_images=True,
    has_depths=True,
    run_names=[
        'Com máscara',
        'Sem máscara',
    ],
    exp_name='d_mask_input',
    label='exp-d-mask-input',
    plot_caption='Resultados do experimento de usar máscara de profundidade como entrada.',
)

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        'gammag9-none/vdst/3n4chtm3',
        'gammag9-none/vdst/x06whrkj',
    ],
    custom_metrics=None,
    has_images=True,
    has_depths=True,
    run_names=[
        'Regular',
        'Cones de visão',
    ],
    exp_name='view_cone_data_processing',
    label='exp-view-cone-data-processing',
    plot_caption='Resultados do experimento de processar dados restringindo visões a cones de visão.',
)

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        'gammag9-none/vdst/es7wa9i7',
        'gammag9-none/vdst/onih7ya6',
        'gammag9-none/vdst/0m139a48',
    ],
    custom_metrics=None,
    has_images=True,
    has_depths=False,
    run_names=[
        'ConvNeXt',
        r'VGG \cite{PhotographicImageSynthesis}',
        'VGG (do autor)',
    ],
    exp_name='percep_arch',
    label='exp-percep-arch',
    plot_caption='Resultados do experimento de melhor arquitetura para perda perceptual.',
)

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        'gammag9-none/vdst/jctx78id',
        'gammag9-none/vdst/s14p3r0k',
    ],
    custom_metrics=None,
    has_images=False,
    has_depths=True,
    run_names=[
        'Comparação direta',
        'Comparação de diferenças',
    ],
    exp_name='d_percep_loss_type',
    label='exp-d-percep-loss-type',
    plot_caption='Resultados do experimento do melhor tipo de perda perceptual para profundidade.',
)

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        'gammag9-none/vdst/6mdvz1f5',
        'gammag9-none/vdst/2awi13b3',
    ],
    custom_metrics=None,
    has_images=True,
    has_depths=False,
    run_names=[
        'Comparação direta',
        'Comparação de diferenças',
    ],
    exp_name='i_percep_loss_type',
    label='exp-i-percep-loss-type',
    plot_caption='Resultados do experimento do melhor tipo de perda perceptual para imagens.',
)

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        'gammag9-none/vdst/4c65f0hz',
        'gammag9-none/vdst/p4si76gq',
        'gammag9-none/vdst/nnnf55uy',
        'gammag9-none/vdst/y3plnzgp',
    ],
    custom_metrics=None,
    has_images=False,
    has_depths=True,
    run_names=[
        r'$t = \mu + \sigma$',
        r'$t = 0.4 \mu + 0.6 \mathrm{max}$',
        r'$t = 0.5$',
        'Sem máscara',
    ],
    exp_name='d_percep_mask_type',
    label='exp-d-percep-mask-type',
    plot_caption='Resultados do experimento de melhor máscara para perda perceptual para profundidade.',
)

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        'gammag9-none/vdst/unz075me',
        'gammag9-none/vdst/alu6sul2',
        'gammag9-none/vdst/yz3ten0q',
        'gammag9-none/vdst/55db64ae',
        'gammag9-none/vdst/ot2g5ly4',
        'gammag9-none/vdst/tsqdjzs2',
        'gammag9-none/vdst/p0yqdozs',
        'gammag9-none/vdst/984xtgxc',
    ],
    custom_metrics=None,
    has_images=False,
    has_depths=True,
    run_names=[
        'Prof. Regular + Percep. S/ Norm',
        'P. Norm. Lin. + Percep. S/ Norm',
        'P. Norm. Sig. + Percep. S/ Norm',
        r'P. Norm. $\mu$ $\sigma$ + Percep. S/ Norm',
        'Prof. Regular + Percep. C/ Norm',
        'P. Norm. Lin. + Percep. C/ Norm',
        'P. Norm. Sig. + Percep. C/ Norm',
        r'P. Norm. $\mu$ $\sigma$ + Percep. C/ Norm',
    ],
    exp_name='d_norm1',
    label='exp-d-norm1',
    plot_caption='Resultados do primeiro experimento de melhor normalização para profundidade e para perda perceptual, apenas com perdas de profundidade.',
)

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        'gammag9-none/vdst/zgzehdnb',
        'gammag9-none/vdst/1yczpb2b',
        'gammag9-none/vdst/rhwetaad',
    ],
    custom_metrics=None,
    has_images=True,
    has_depths=True,
    run_names=[
        'Prof. Regular S/ Percep.',
        'Prof. Regular + Percep. S/ Norm',
        r'P. Norm. $\mu$ $\sigma$ + Percep. S/ Norm',
    ],
    exp_name='d_norm2',
    label='exp-d-norm2',
    plot_caption='Resultados do segundo experimento de melhor normalização para profundidade e para perda perceptual, usando também perdas de imagem, considerando também hipótese nula (sem perda perceptual).',
)

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        'gammag9-none/vdst/96euth8g',
        'gammag9-none/vdst/dezc1lwq',
        'gammag9-none/vdst/zm8vtady',
        'gammag9-none/vdst/73ggjjmv',
    ],
    custom_metrics=None,
    has_images=False,
    has_depths=True,
    run_names=[
        'SILog + Grad. + Percep. Prof.',
        'SILog + Grad.',
        'SILog + Percep. Prof.',
        'SILog',
    ],
    exp_name='d_losses',
    label='exp-d-losses',
    plot_caption='Resultados do experimento de melhores perdas para profundidade.',
)

create_exp_results(
    results_path=results_path,
    run_api_urls=[
        # 'gammag9-none/vdst/rhwetaad',
        'gammag9-none/vdst/oou18yif',
    ],
    custom_metrics=None,
    has_images=True,
    has_depths=True,
    run_names=[
        'Modelo final',
    ],
    exp_name='final_model',
    label='exp-modelo-final',
    plot_caption='Resultados do modelo final no conjunto de validação.',
)

global_tables = '\n\n'.join(global_tables)
with open(os.path.join(results_path, f'global_tables.tex'), 'w', encoding='utf8') as f:
    f.write(global_tables)
