"""
Model Evaluation Pipeline
=========================

This script processes JSON files from ML experiments. It evaluates model performance,
computes a performance index, conducts statistical analysis (ANOVA, Tukey HSD),
and visualizes uncertainty, parameter tuning, and Taylor diagrams.
"""
import os,sys
import json
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import skill_metrics as sm
from collections import defaultdict, Counter
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import mean_squared_error
from permetrics.regression import RegressionMetric
#%%

sns.set_context("paper")
sns.set_style(style="white", rc={
    #"font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})
sns.set_context("paper", font_scale=1.8, 
        rc={"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
            'xtick.labelsize':16,'ytick.labelsize':16,
            'font.family':"Times New Roman", }
        ) 
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Palatino']})
import matplotlib
matplotlib.rcParams['text.usetex'] = False

#%%
# --- CONFIGURATION ---
REFERENCE_METRIC = 'RMSE'
N_RUNS = 50
METRICS = ['R2', 'RMSE', 'MAE', 'MAPE', 
           #"KGE", 'A10', 'VAF',
           ]
FOLDER_FIG='./img'
FOLDER_TAB='./tab'

metrics_max =  ['NSE', 'VAF', 'R', 'Accuracy','R2','R$^2$', 'KGE', 'WI', 'A20', 'A10']  

PALETTE="Reds_r"
PALETTE="cividis"
#PALETTE="mako"

# --- UTILITIES ---
def calculate_taylor_metrics(y_true, y_pred):
    std_pred = np.std(y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    rms = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return std_pred, corr, rms

# --- 1. LOAD JSON FILES ---


def load_json_data(folder_path):
    results, uncertainty, times = [], [], []
    for filepath in glob.glob(os.path.join(folder_path, '*.json')):
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)[0]
                #if data.get('run')<25:
                y_true, y_pred = data.get("y_test", []), data.get("y_pred", [])
                model_name = data.get("estimator", "unknown")
                #print(model_name)
                metrics = {}
                if y_true and y_pred:
                    metric_obj = RegressionMetric(y_true, y_pred)
                    metrics = metric_obj.get_metrics_by_list_names([m for m in METRICS if m != 'Time'])
                    if 'Time' in METRICS:
                        metrics['Time'] = data.get('elapsed_time')
                    
                    metrics['Model'] = model_name
                    results.append(metrics)
                uncertainty.append({
                    'Model': model_name,
                    'MAD': data.get('mad',-1),
                    'Uncertainty': data.get('uncertainty',-1),
                    REFERENCE_METRIC: metrics.get(REFERENCE_METRIC, None)
                })
                times.append({
                    'Model': model_name,
                    'Time (s)': data.get('elapsed_time',None),
                })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    return pd.DataFrame(results), pd.DataFrame(uncertainty), pd.DataFrame(times)



def filter_models(df, models_to_remove):
    """
    Remove linhas de um DataFrame cujos modelos estão na lista de exclusão.
    A função também remove modelos com o sufixo '-FS'.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com coluna 'Model'
    - models_to_remove (list): Lista de nomes de modelos a remover (ex: ['RF', 'ANN'])

    Retorna:
    - pd.DataFrame filtrado
    """
    # Inclui versões com sufixo '-FS'
    full_models_to_remove = models_to_remove + [m + '-FS' for m in models_to_remove]
    
    # Remove modelos com ou sem sufixo
    return df[~df['Model'].str.split('-').str[0].isin(full_models_to_remove)]


# --- 2. MODEL METRICS PLOT ---

def plot_model_metrics(refname, df, METRICS, save_fig=False, output_dir='./img'):
   
    
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_metrics = len(METRICS)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        # ✅ COMPUTE STATISTICS FOR ERROR BARS
        model_stats = df.groupby('Model')[metric].agg(['mean', 'std', 'count']).reset_index()
        model_stats['sem'] = model_stats['std'] / np.sqrt(model_stats['count'])  # Standard Error
        
        # ✅ PROFESSIONAL BARPLOT WITH ERROR BARS
        bars = ax.bar(model_stats['Model'], model_stats['mean'],
                     yerr=model_stats['sem'],
                     capsize=6,
                     error_kw={
                         'elinewidth': 1.8,
                         'capthick': 1.0,
                         'ecolor': '#d62728'  # Publication red
                     },
                     #color=plt.cm.Reds_r(np.linspace(0.3, 0.9, len(model_stats))),
                     color=sns.color_palette(PALETTE, len(model_stats)),        
                     alpha=0.85,
                     edgecolor='black',
                     linewidth=1.0)
        
        # ✅ VALUE LABELS ON BARS
        for bar, mean_val, sem_val in zip(bars, model_stats['mean'], model_stats['sem']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + sem_val*1.2,
                   f'{mean_val:.3f}', 
                   ha='center', va='bottom',
                   fontweight='bold', 
                   rotation=90,
                   #fontsize=9,
                   )
        
        # ✅ SCIENTIFIC FORMATTING
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        #ax.set_title(f'{metric}', fontweight='bold', pad=12, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_ylabel(f'{metric}', fontweight='bold')
        
        # ✅ REFERENCE LINES (publication standard)
        if metric == 'MSE':
            ax.axhline(y=0.025, color='red', ls=':', lw=2, alpha=0.8, 
                      label='Baseline')
        elif metric == 'R²':
            ax.axhline(y=0.80, color='red', ls=':', lw=2, alpha=0.8, 
                      label='Baseline')
        
        # ✅ CLEAN SPINES
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    # Remove extra subplots
    for i in range(len(METRICS), len(axes)):
        fig.delaxes(axes[i])

    # ✅ MAIN TITLE & SUBTITLE
    #fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=0.98)
    #fig.supxlabel('Models', fontsize=12, fontweight='bold')
    #fig.text(0.02, 0.02, 
    #         'Mean ± SEM | 5×3 Repeated Cross-Validation | n=120 samples', 
    #         fontsize=9, style='italic',
    #         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout(pad=2.0)
    
    # ✅ MULTI-FORMAT PUBLICATION EXPORT
    if save_fig:
        os.system(f"mkdir -p {FOLDER_FIG}")
        filename = f"{BASENAME}_cmp_metrics.png".replace(" ", "_").replace("/", "_")
        
        # High-quality PNG
        plt.savefig(os.path.join(output_dir, filename), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        
        # Vector formats for papers
        plt.savefig(os.path.join(output_dir, filename.replace('.png', '.pdf')), 
                   format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, filename.replace('.png', '.svg')), 
                   format='svg', bbox_inches='tight')
        
        plt.show()
        plt.show()
        plt.close()
        print(f"✅ Publication plots saved: {filename} (PNG/PDF/SVG)")
        
    else:
        plt.show()

    
# --- 3. UNCERTAINTY SCATTER ---
def plot_uncertainty_grouped(refname, df, save_fig=False, output_dir='./img'):
    plt.figure(figsize=(7, 5))
    scatter = sns.scatterplot(
        data=df,
        x='Uncertainty', y=REFERENCE_METRIC,
        hue='Model', style='Model', s=200,
        palette='tab10', markers=True, legend=False  # Disable legend
    )
    plt.title(f"{REFERENCE_METRIC} vs Uncertainty")
    plt.grid(True)
    plt.tight_layout()

    model_names= list(df['Model'].unique())
    cmap = plt.get_cmap('tab20')
    color_map = {name: cmap(i % cmap.N) for i, name in enumerate(model_names)}

    for i in range(df.shape[0]):
        model = df['Model'].iloc[i]
        x = df['Uncertainty'].iloc[i]
        y = df[REFERENCE_METRIC].iloc[i]
        
        # Randomly choose ha and va
        ha = np.random.choice(['left', 'right'])
        va = np.random.choice(['top', 'bottom'])
        color = color_map[model]
        plt.text(x, y,
                 model,
                 fontsize=12,
                 ha=ha,
                 va=va,
                 alpha=1,
                 color='black',
                 )

    if save_fig:        
        filename = f"{BASENAME}_cmp_uncertainty_grouped.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()
    else:
        plt.show()
    

def plot_uncertainty(refname, df, save_fig=False, output_dir='./img'):
    plt.figure(figsize=(7, 5))
    scatter = sns.scatterplot(
        data=df,
        x='Uncertainty', y=REFERENCE_METRIC,
        hue='Model', style='Model', s=200,
        palette='tab10', markers=True, legend=True  # Disable legend
    )
    plt.title(f"{REFERENCE_METRIC} vs Uncertainty")
    plt.grid(True)
    plt.tight_layout()

    model_names= list(df['Model'].unique())
    cmap = plt.get_cmap('tab20')
    color_map = {name: cmap(i % cmap.N) for i, name in enumerate(model_names)}

   

    if save_fig:        
        filename = f"{BASENAME}_cmp_uncertainty.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()
    else:
        plt.show()
        
def get_pareto_front(df, x_col='Uncertainty', y_col=REFERENCE_METRIC):
    """Returns the Pareto front: points where lower is better for both x and y.
    Assumes we want to minimize both columns."""
    # Sort by x ascending, then by y ascending (important for staircase detection)
    df_sorted = df.sort_values([x_col, y_col]).copy()
    
    pareto_points = []
    current_min_y = float('inf')
    
    for _, row in df_sorted.iterrows():
        if row[y_col] < current_min_y:
            pareto_points.append(row)
            current_min_y = row[y_col]
    
    return pd.DataFrame(pareto_points)



def plot_uncertainty_pareto(refname, df, save_fig=False, output_dir='./img'):
    plt.figure(figsize=(7, 5))
    
    df_pareto = get_pareto_front(df, x_col='Uncertainty', y_col=REFERENCE_METRIC)
    

    
    p1 = sns.relplot(
        data=df_pareto,
        x='Uncertainty', y=REFERENCE_METRIC,
        hue='Model', style='Model', s=200,
        palette='tab10', markers=True, legend=True  # Disable legend
    )
    
    # Access the Axes object from the FacetGrid
    ax = p1.axes[0, 0]  # For a single plot
    ax.plot(
        df_pareto['Uncertainty'],
        df_pareto[REFERENCE_METRIC],
        color='gray',
        linestyle='--',
        linewidth=1.5,
        alpha=0.6,
        zorder=0
    )
    
    ax.plot(
        df['Uncertainty'],
        df[REFERENCE_METRIC],
        color='gray',
        marker='o',        # Add this to specify circle markers
        linestyle='None',  # Remove the line (or set to 'None')
        alpha=0.6, markersize=10, 
        zorder=0
    )
    
    # Optional: Add a light grid for readability
    ax.grid(True, linestyle=':', alpha=0.5)
        

    
    plt.title(f"{REFERENCE_METRIC} vs Uncertainty")
    plt.grid(True)
    plt.tight_layout()

    model_names= list(df['Model'].unique())
    cmap = plt.get_cmap('tab20')
    color_map = {name: cmap(i % cmap.N) for i, name in enumerate(model_names)}
    
    if save_fig:        
        filename = f"{BASENAME}_cmp_uncertainty_pareto.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()
    else:
        plt.show()

# --- 4. PERFORMANCE INDEX (PI) ---

def compute_performance_index(BASENAME, df_results, metrics, save_fig=False, output_dir='./img'):
    """Compute a weighted Performance Index (PI) and rank models."""
    weight_array = np.array([1/len(METRICS)] * len(METRICS))
    df_normalized = df_results[METRICS].copy()

    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Normalize and invert error metrics so higher is better
    for col in df_normalized.columns:
        df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / \
                             (df_normalized[col].max() - df_normalized[col].min())

    error_metrics = [i if i not  in metrics_max else None for i in metrics]
    for m in error_metrics:
        if m in df_normalized.columns:
            df_normalized[m] = 1 - df_normalized[m]

    # Compute PI
    df_results['PI'] = (df_normalized * weight_array).sum(axis=1)
    df_ranked = df_results.sort_values(by='PI', ascending=False).reset_index(drop=True)
    df_ranked['Rank'] = df_ranked.index + 1
    print("\n📊 Ranked Models by Performance Index (PI):")
    print(df_ranked[['Model', 'PI', 'Rank'] + METRICS])
        
    # Group by model and calculate mean PI
    model_stats = df_ranked.groupby('Model')['PI'].agg(['mean', 'std']).reset_index()
    model_stats = model_stats.sort_values(by='mean', ascending=False).reset_index(drop=True)
    model_stats['Rank'] = model_stats.index + 1
    print("\n📊 Mean PI Scores Across Runs:")
    print(model_stats[['Rank', 'Model', 'mean', 'std']])

    pi_scores_df = model_stats[['Rank', 'Model', 'mean', 'std']]
    # Format mean ± std as a single column
    pi_scores_df["PI"] = pi_scores_df.apply(
        lambda row: f"{row['mean']:.3f} (± {row['std']:.3f})", axis=1
    )
    
    # Select only desired columns
    latex_df = pi_scores_df[["Rank", "Model", "PI"]]
    
    # Generate LaTeX table
    latex_pi_path = "./pi_scores_table.tex"
    latex_pi = latex_df.to_latex(
        index=False,
        escape=False,
        caption="Mean PI scores across model runs with standard deviation.",
        label="tab:pi_scores",
        column_format="ccl"
    )
    
    with open(latex_pi_path, "w") as f:
        f.write(latex_pi)
        
    # In compute_performance_index, replace plot section with:
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(range(len(model_stats)), model_stats['mean'], 
                  yerr=model_stats['std'], capsize=6, alpha=0.85, 
                  edgecolor='black', linewidth=1.0, color=sns.color_palette(PALETTE, len(model_stats)))
                  
    # Value labels
    for i, (bar, mean_val) in enumerate(zip(bars, model_stats['mean'])):
        ax.text(bar.get_x() + bar.get_width()/2., mean_val + model_stats['std'].iloc[i]*1.1,
               f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', rotation=90)
    
    ax.set_xticks(range(len(model_stats)))
    ax.set_xticklabels(model_stats['Model'], rotation=45, ha='right')
    ax.set_ylabel('Performance Index (PI)', fontweight='bold')
    #ax.set_title('Model Ranking by Performance Index', fontweight='bold', pad=35)
    ax.grid(True, axis='y', alpha=0.3)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    
    
    
    if save_fig:
        filename = f"{BASENAME}_pi.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()
    else:
        plt.show()


    return df_ranked



# --- 5. FEATURE SELECTION FREQUENCY ---
def analyze_feature_selection(models_to_remove, folder_path, save_fig=False, output_dir='./img'):
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    freq = defaultdict(lambda: defaultdict(int))
    all_features = set()
    for file in glob.glob(os.path.join(folder_path, '*.json')):
        with open(file, 'r') as f:
            data = json.load(f)[0]
            model = data.get('estimator', '')
            if model not in models_to_remove:
                if model.endswith('-FS'):
                    for feat in data.get('selected_features', []):
                        freq[model][feat] += 1
                        all_features.add(feat)
    
    df = pd.DataFrame(freq).fillna(0).reindex(index=sorted(all_features))
    df_pct = df.apply(lambda col: (col / N_RUNS) * 100, axis=0).round(1)
    df_long = df_pct.T.reset_index().melt(id_vars='index', var_name='Feature', value_name='Percentage')
    df_long['Feature'] = df_long['Feature'].replace(rename_dict)
    
    # plt.figure(figsize=(7, 4))
    # sns.barplot(data=df_long, x='index', y='Percentage', hue='Feature', palette='tab10')
    # #plt.title("Feature Selection Frequency (\%)")
    # plt.ylabel("Feature Selection Frequency (\%)")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.xlabel(None)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
    #       ncol=5, 
    #       fancybox=True, shadow=True,
    #       )    

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=df_long, x='index', y='Percentage', hue='Feature', palette='tab10')

    # --- Add percentage values on top of bars ---
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='edge', fontsize=10, padding=3, )
    
    plt.ylabel("Feature Selection Frequency (\%)")
    plt.xticks(rotation=0)
    plt.ylim([0,110])
    plt.xlabel('')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=5, )
    plt.tight_layout()
    
    if save_fig:
        filename = f"{BASENAME}_fs.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()
    else:
        plt.show()


# --- 5B. FEATURE PARETO PER MODEL ---
def plot_feature_pareto(refname, folder_path, save_fig=False, output_dir='./img'):
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_features = defaultdict(Counter)
    for file in glob.glob(os.path.join(folder_path, '*.json')):
        with open(file, 'r') as f:
            data = json.load(f)[0]
            model = data.get('model_name', '')
            if model.endswith('-FS'):
                short_model = model.split('-')[0].upper()
                for feat in data.get('selected_features', []):
                    model_features[short_model][feat] += 1

    for model, feat_counter in model_features.items():
        df_aux = pd.DataFrame.from_dict(feat_counter, orient='index', columns=['Frequency'])
        df_aux.sort_values(by='Frequency', ascending=False, inplace=True)
        df_aux['Cumulative %'] = df_aux['Frequency'].cumsum() / df_aux['Frequency'].sum() * 100

        fig, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=df.index, y='Frequency', data=df_aux, ax=ax1, color='skyblue')
        ax1.set_ylabel('Frequency')
        ax1.set_xlabel('Features')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

        # ax2 = ax1.twinx()
        # ax2.plot(df.index, df['Cumulative %'], color='orange', marker='o', linestyle='--')
        # ax2.set_ylabel('Cumulative %')
        # ax2.set_ylim(0, 105)
        # ax2.axhline(80, linestyle='--', color='gray')

        plt.title(f"Feature Selection Pareto - {model}")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        if save_fig:
            filename = f"{BASENAME}_freq_features.png".replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
            plt.show()
            plt.close()
        else:
            plt.show()


# --- 6. ANOVA & TUKEY ---
def run_anova_and_tukey(df, metrics=METRICS):
    for metric in metrics:
        groups = df.groupby('Model')[metric].apply(list).values
        f, p = stats.f_oneway(*groups)
        print(f"\nANOVA {metric}: F={f:.2f}, p={p:.4f}")
        tukey = pairwise_tukeyhsd(endog=df[metric], groups=df['Model'], alpha=0.05)
        print(tukey.summary())

    # Prepare ANOVA summary table
    anova_results = []
    
    # Calculate ANOVA F and p-values for each metric
    for metric in metrics:
        #groups = df.groupby('Model')[metric].apply(dict).values
        groups = df.groupby('Model')[metric].apply(list).values
        f, p = stats.f_oneway(*groups)
        anova_results.append({'Metric': metric, 'F-value': round(f, 3), 
                              #'p-value': round(p, 4),
                              'p-value': p #if p>1e-4 else '$<10^{-4}$',
                              })

    # Convert to DataFrame and pivot to get metrics as columns
    anova_df = pd.DataFrame(anova_results).set_index('Metric').T

    # Reformat p-values to scientific (exponential) notation
    anova_df.loc['p-value'] = anova_df.loc['p-value'].apply(lambda x: f"{x:.3e}")
    anova_df.loc['F-value'] = anova_df.loc['F-value'].apply(lambda x: f"{x:.3f}")

    # Save ANOVA summary table to LaTeX format
    anova_latex_path = "./anova_summary_table.tex"
    
    # Format the DataFrame as LaTeX with caption and label
    latex_anova = anova_df.to_latex(
        caption="ANOVA F and p-values for each performance metric across models.",
        label="tab:anova_metrics",
        escape=False,
        index=True,
        column_format="l" + "c" * len(anova_df.columns)
    )
    
    # Save to .tex file
    with open(anova_latex_path, "w") as f:
        f.write(latex_anova)
        
# --- 6.1 COMPARISON BOOSTRAP ---

def run_bootstrap(df, metrics=METRICS, model_col="Model", n_boot=20_000,
                   ci=95, random_state=42,save_fig=False, output_dir='.',
                   out_dir=".", out_prefix="bootstrap_summary_table"):
    """
    Calcula um score composto (media normalizada das metricas) por linha,
    faz bootstrap por modelo para estimar media + intervalo de confianca,
    e salva os resultados em CSV, tabela LaTeX e figura (barras + IC).

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame contendo uma coluna `model_col` e as colunas de `metrics`.
    metrics : dict[str, bool]
        Mapa {nome_da_coluna: maior_e_melhor}. Default = METRICS.
    model_col : str
        Nome da coluna que identifica o modelo.
    n_boot : int
        Numero de reamostragens bootstrap.
    ci : float
        Nivel do intervalo de confianca (ex.: 95 -> IC 95%).
    random_state : int
        Semente para reprodutibilidade.
    out_dir : str
        Diretorio de saida para os 3 arquivos gerados.
    out_prefix : str
        Prefixo usado nos nomes dos arquivos de saida.

    Retorna
    -------
    bootstrap_df : pd.DataFrame
        Tabela resumo (indexada por modelo) com score medio, IC, std, sem,
        rank e probabilidade de ser o melhor modelo.
    paths : dict[str, str]
        Caminhos dos 3 arquivos salvos: {"csv", "latex", "figure"}.
    """
    rng = np.random.default_rng(random_state)

    # --- normaliza cada metrica para [0,1], com sinal ajustado ---
    norm = df.copy()
    for col in metrics:
        mn, mx = df[col].min(), df[col].max()
        scaled = (df[col] - mn) / (mx - mn)
        norm[col + "_n"] = scaled if (col in metrics_max) else 1 - scaled

    score_cols = [c + "_n" for c in metrics]
    df = df.copy()
    df["score"] = norm[score_cols].mean(axis=1)

    # --- bootstrap por modelo ---
    alpha = (100 - ci) / 2
    rows, boot_dists = [], {}
    for model, g in df.groupby(model_col):
        vals = g["score"].values
        n = len(vals)
        boot_means = np.array(
            [rng.choice(vals, size=n, replace=True).mean() for _ in range(n_boot)]
        )
        boot_dists[model] = boot_means
        mean = vals.mean()
        ci_low, ci_high = np.percentile(boot_means, [alpha, 100 - alpha])
        rows.append({
            model_col: model,
            "n_runs": n,
            "score_mean": mean,
            f"ci{ci}_low": ci_low,
            f"ci{ci}_high": ci_high,
            "std": vals.std(ddof=1),
            "sem": vals.std(ddof=1) / np.sqrt(n),
        })

    bootstrap_df = pd.DataFrame(rows).sort_values("score_mean", ascending=False)
    bootstrap_df = bootstrap_df.reset_index(drop=True)
    bootstrap_df["rank"] = bootstrap_df.index + 1

    # probabilidade de cada modelo ser o melhor, via comparacao das
    # distribuicoes bootstrap
    models = bootstrap_df[model_col].tolist()
    boot_matrix = np.vstack([boot_dists[m] for m in models])
    best_idx = np.argmax(boot_matrix, axis=0)
    bootstrap_df["prob_best"] = [(best_idx == i).mean() for i in range(len(models))]

    bootstrap_df = bootstrap_df.set_index(model_col)

    # --- salva CSV ---
    csv_path = f"{out_dir}/{out_prefix}.csv"
    bootstrap_df.to_csv(csv_path)

    # --- salva tabela LaTeX ---
    latex_path = f"{out_dir}/{out_prefix}.tex"
    bootstrap_latex = bootstrap_df.round(4).to_latex(
        caption="Bootstrap summary for the composite performance score across models.",
        label="tab:bootstrap_metrics",
        escape=False,
        index=True,
        column_format="l" + "c" * len(bootstrap_df.columns),
    )
    with open(latex_path, "w") as f:
        f.write(bootstrap_latex)

    # --- salva figura ---
    fig_path = f"{out_dir}/{out_prefix}.png"
    plot_df = bootstrap_df.sort_values("score_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    yerr_low = plot_df["score_mean"] - plot_df[f"ci{ci}_low"]
    yerr_high = plot_df[f"ci{ci}_high"] - plot_df["score_mean"]
    best_prob = plot_df["prob_best"].max()
    colors = ["#C44E52" if p == best_prob else "#4C72B0" for p in plot_df["prob_best"]]

    ax.barh(plot_df.index, plot_df["score_mean"], xerr=[yerr_low, yerr_high],
            color=colors, capsize=4, edgecolor="black", linewidth=0.5)
    for i, (s, p) in enumerate(zip(plot_df["score_mean"], plot_df["prob_best"])):
        ax.text(s + 0.03, i, f"P(melhor)={p:.0%}", va="center", fontsize=9)
    ax.set_xlabel("Score composto (0-1, maior = melhor)")
    ax.set_title(f"Ranking de modelos com IC {ci}% (bootstrap, n={n_boot})")
    ax.set_xlim(0, 1.15)
    plt.tight_layout()
    
    if save_fig:
        filename = f"{BASENAME}_bootstrap.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        #plt.close()
    
        #plt.savefig(fig_path, dpi=150)
        plt.close(fig)

    else:
        plt.show()
        
    
    paths = {"csv": csv_path, "latex": latex_path, "figure": fig_path}
    return bootstrap_df, paths

        
# --- 7. TAYLOR DIAGRAM ---
def plot_taylor_diagram(refname, folder_path, save_fig=False, output_dir='./img'):
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob.glob(os.path.join(folder_path, '*.json'))
    results = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)[0]
            model = data.get("estimator", "Unknown").split("-")[0].upper()
            y_true, y_pred = np.array(data['y_test']), np.array(data['y_pred'])
            std, corr, rms = calculate_taylor_metrics(y_true, y_pred)
            results.append({'model': model, 'std': std, 'corr': corr, 'y_pred':y_pred, 'ref':y_true})

    model_names = sorted(set(r['model'] for r in results))
    cmap = plt.get_cmap('tab20')
    model_colors = {name: cmap(i % cmap.N) for i, name in enumerate(model_names)}

    # ref=results[0]['ref']
    # taylor_stats=[{'sdev':np.std(ref), 'crmsd':0, 'ccoef':1,  
    #                         'label':'Observation', 'bias':1, 'rmsd':0}]
            
    # for i in range(len(results)):
    #     pred=results[i]['y_pred']
    #     ref=results[i]['ref']
    #     e = results[i]['model']
    #     ts=sm.taylor_statistics(pred,ref,'data')
    #     taylor_stats.append({'sdev':ts['sdev'][1], 'crmsd':ts['crmsd'][1], 
    #                           'ccoef':ts['ccoef'][1],  'label':e, 
    #                           'bias':sm.bias(pred, ref),
    #                           'rmsd':sm.rmsd(pred, ref),                                 
    #                           })

    # taylor_stats = pd.DataFrame(taylor_stats)

    # sm.taylor_diagram(taylor_stats['sdev'].values, 
    #                           taylor_stats['crmsd'].values, 
    #                           taylor_stats['ccoef'].values,
    #                           #markercolor =model_colors[k], 
    #                           alpha = 0.00,
    #                           markerSize = 20, rmsLabelFormat='0:.2f',
    #                           colSTD='k', colRMS='k', colCOR='k',
    #                           #overlay = overlay, 
    #                           #markerLabel = label
    #                           )
    


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(np.linspace(0, math.pi/2, 100), [1]*100, 'k--', alpha=0.5)
    for r in results:
        angle, radius = np.arccos(r['corr']), r['std']
        ax.plot(angle, radius, 'o', color=model_colors[r['model']], label=r['model'])

    handles, labels = ax.get_legend_handles_labels()
    used = dict()
    for h, l in zip(handles, labels):
        if l not in used:
            used[l] = h
    ax.legend(used.values(), used.keys(), bbox_to_anchor=(1.4, 1.05))
    ax.text(math.pi/4, max(r['std'] for r in results)*1.15, 'Correlation →', ha='center', fontsize=10)
    ax.text(-0.25, max(r['std'] for r in results)*0.25, 'Standard Deviation', ha='center', va='center', fontsize=10, rotation=90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xlim(0, math.pi/2)
    plt.title("Taylor Diagram (0°–90° Sector)")
    plt.tight_layout()
    
    if save_fig:
        filename = f"{BASENAME}_taylor.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()
    else:
        plt.show()


# --- 7. FEATURE IMPORTANCE BARPLOT ---
def plot_feature_importance(refname, folder_path, save_fig=False, output_dir='./img'):
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    model_feature_importance = defaultdict(lambda: defaultdict(list))
    all_features = set()

    # Coleta os dados
    for file in glob.glob(os.path.join(folder_path, '*.json')):
        with open(file, 'r') as f:
            data = json.load(f)[0]
            model = data.get('estimator', 'unknown')
            
            if 'shap_coefficients' in data.keys():
                print(data.keys())
                data['feature_importance'] = dict(zip(data.get('feature_names'), data.get('shap_coefficients')))
                print(data['feature_importance'])
            else:
                data['feature_importance'] = dict(zip(data.get('feature_names'), data.get('feature_importances')))
                #data['feature_importance'] = dict(zip(data.get('feature_names'), [0]*len(data.get('feature_names'))))
            
            feat_imp = data.get('feature_importance', {})
            #all_features.update(feat_imp.keys())
            #renamed_feat_imp = {rename_dict.get(k, k): v for k, v in feat_imp.items()}
            renamed_feat_imp = feat_imp
            all_features.update(renamed_feat_imp.keys())
            for feat, val in renamed_feat_imp.items():
                model_feature_importance[model][feat].append(val)

    all_features = sorted(list(all_features))

    for model, feat_dict in model_feature_importance.items():
        # Cria lista de registros por execução
        exec_data = []
        max_len = max(len(v) for v in feat_dict.values())

        for i in range(max_len):
            row = {feat: feat_dict[feat][i] if i < len(feat_dict[feat]) else 0.0 for feat in all_features}
            exec_data.append(row)

        df = pd.DataFrame(exec_data)
        
        plt.figure(figsize=(3, 4))    
        g = sns.catplot(
            data=df,
            kind='box',
            #height=5,
            #aspect=1.5,
            palette='viridis',
            linewidth=1.2,
            showfliers=False     # Hide outliers for cleaner look
        )
        
        stripplot = sns.stripplot(
            data=df,
            color=".25",
            jitter=True,
            size=6,
            alpha=0.7
        )
        g.set_axis_labels("Input variable", "Feature Importance")  # Y-axis label only
        #plt.xticks([])  # Remove x-axis ticks and labels (for cleaner look)

        if save_fig:
            filename = f"{BASENAME}_fi_bxp_{model}.png".replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
            plt.show()
            plt.close()
        else:
            plt.show()

        df_mean = df.mean()
        df_std = df.std()

        # Gráfico
        x = np.arange(len(all_features))
        plt.figure(figsize=(4, 4))
        plt.bar(x, df_mean.values, #yerr=df_std.values, 
                capsize=5, color='skyblue', edgecolor='black')
        plt.xticks(x, all_features, rotation=45, ha='right')
        plt.ylabel("Mean Importance ± Std")
        plt.title(f"Feature Importance - {model}")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        if save_fig:
            filename = f"{BASENAME}_fi_{model}.png".replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
            plt.show()
            plt.close()
        else:
            plt.show()
            

# --- 7. PARAMETRIC ANALYSIS ---
def plot_model_params_distribution(refname, folder_path='./json-files', save_fig=False, output_dir='./img'):
    """
    Gera gráficos para os parâmetros dos modelos salvos em arquivos JSON.
    Gera boxplots para variáveis contínuas e barplots para variáveis discretas com até 7 valores únicos.

    Parâmetros:
    - folder_path (str): caminho da pasta com os arquivos JSON.
    - save_fig (bool): se True, salva os gráficos em vez de exibir.
    - output_dir (str): pasta onde salvar os gráficos (usado se save_fig=True).
    """

  

    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    param_data_2 = defaultdict(lambda: defaultdict(list))

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            with open(os.path.join(folder_path, file_name), 'r') as f:
                data = json.load(f)[0]
                model_name = data.get('model_name', 'unknown')
                params = data.get('model_params', {}).get(model_name.lower(), {})
                for param, value in params.items():
                    if isinstance(value, (int, float)):
                        param_data_2[model_name][param].append(value)

    # Helper to decide plot type
    def is_discrete_with_cutoff(values):
        if all(isinstance(v, int) or isinstance(v, bool) for v in values):
            return 'boxplot' if len(set(values)) > 5 else 'barplot'
        return 'boxplot'

    # Plot per model/param
    for model, params in param_data_2.items():
        for param, values in params.items():
            values_series = pd.Series(values)
            plot_type = is_discrete_with_cutoff(values)
            nv = values_series.unique().shape[0]
            
            plt.figure(figsize=(7, 5) if plot_type == 'barplot' and nv>=3 else (2, 5) )
            if plot_type == 'barplot':
                sns.countplot(x=values_series)
                plt.xlabel(param)
                plt.ylabel("Count")
                plt.xlabel(None)
            else:
                sns.boxplot(y=values_series)
                plt.ylabel(param)
                plt.xlabel(None)

            plt.title(f"{model} \n {param} ({'Boxplot' if plot_type == 'boxplot' else 'Barplot'})")
            plt.title(f"{model} \n {param}")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            if save_fig:
                filename = f"{BASENAME}_{model}_{param}_{plot_type}.png".replace(" ", "_").replace("/", "_")
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', transparent=True)
                plt.show()
                plt.close()
            else:
                plt.show()


def format_df_table(df, ref_colum, columns, nam):
        
    # Recreate the formatted summary using string format mean (± std)
    formatted_summary = pd.DataFrame()
    
    # Format each metric as "mean (± std)"
    for metric in columns:
        formatted_summary[metric] = df.groupby(ref_colum)[metric].agg(['mean', 'std']).apply(
            lambda row: f"{row['mean']:.3f} (± {row['std']:.3f})" if pd.notnull(row['std']) else f"{row['mean']:.3f}",
            axis=1
        )

    # Convert to LaTeX table with caption and label
    latex_table = formatted_summary.to_latex(
        caption="Performance metrics by model (mean ± std).",
        label=f"tab:summary_{nam}",
        escape=False,        index=True,
        column_format="l" + "c" * len(formatted_summary.columns),    )
    
    # Save to file
    latex_file_path = f"./model_summary_{nam}.tex"
    with open(latex_file_path, "w") as f:
        f.write(latex_table)
    
    return formatted_summary




def generate_latex_figures_from_folder(compute_performance_index, folder_path="./img", output_tex_path="insert_all_figures_from_folder.tex"):
    """
    Gera comandos LaTeX para incluir todas as imagens PNG em uma pasta.
    
    Args:
        folder_path (str): Caminho para a pasta contendo arquivos .png.
        output_tex_path (str): Caminho do arquivo .tex a ser salvo.
        
    Returns:
        str: Caminho do arquivo gerado.
    """
    # Lista todos os arquivos .png
    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

    # Monta os comandos LaTeX
    latex_figures = ""
    for filename in png_files:
        base = os.path.splitext(filename)[0].replace("an__", "")
        caption = base.replace("_", " ").title()
        label = base.lower().replace("_", "-")
        latex_figures += f"""\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{folder_path}/{filename}}}
    \\caption{{{caption}}}
    \\label{{fig:{label}}}
\\end{{figure}}

"""

    # Salva em arquivo .tex
    with open(output_tex_path, "w") as f:
        f.write(latex_figures)

    return output_tex_path



def plot_best_run_per_model(refname, models_to_remove, folder_path, save_fig=False, output_dir='./img'):
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    all_results = []
    for filepath in glob.glob(os.path.join(folder_path, '*.json')):
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)[0]
                y_true, y_pred = data.get("y_test", []), data.get("y_pred", [])
                model_name = data.get("estimator", "unknown")
                print(model_name)
                metrics = {}
                if y_true and y_pred:
                    metric_obj = RegressionMetric(y_true, y_pred)
                    metrics = metric_obj.get_metrics_by_list_names(METRICS)
                    metrics['Model'] = model_name
                    metrics['y_true']=y_true
                    metrics['y_pred']=y_pred
                    all_results.append(metrics)                    
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                


    # --- Step 2: Group by model and select best (min RMSE) per model ---
    models_seen = {}
    for res in all_results:
        model = res['Model']
        if model not in models_to_remove:
            if model not in models_seen or res['R2'] > models_seen[model]['R2']:
                models_seen[model] = res

    sorted_models = dict(sorted(models_seen.items()))
    best_results = list(sorted_models.values())

    # --- Step 3: Plot best run of each model ---
    cmap = plt.get_cmap('tab10')

    for i, result in enumerate(best_results):
        model_name = result['Model']
        y_true = result['y_true']
        y_pred = result['y_pred']
        r2 = result['R2']
        mape = result['MAPE']
        rmse = result['RMSE']
        

        plt.figure(figsize=(5, 5))

        # Scatter plot
        plt.scatter(y_true, y_pred,
                    alpha=0.7,
                    edgecolor='k',
                    color=cmap(i),
                    label=model_name,
                    s=50)

        # Diagonal line
        lim_min = min(np.min(y_true), np.min(y_pred))
        lim_max = max(np.max(y_true), np.max(y_pred))
        plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Ideal')

        # Labels and title
        plt.xlabel('Observed Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.title(f"{refname} - Best {model_name}\n"
                  f"R² = {r2:.3f}, RMSE = {rmse:.2f}", fontsize=14)
        plt.title(f"{refname} - Best {model_name} - "
                  f"R² = {r2:.3f}", fontsize=14)

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        plt.axis('equal')  # Square axis with equal scaling
        plt.xlim(lim_min - 0.1 * (lim_max - lim_min), lim_max + 0.1 * (lim_max - lim_min))
        plt.ylim(lim_min - 0.1 * (lim_max - lim_min), lim_max + 0.1 * (lim_max - lim_min))
        plt.tight_layout()

        # Save or show
        if save_fig:
            filename = f"{model_name}_best_scatter_{refname}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved best scatter plot for {model_name} at {output_dir}")
        else:
            plt.show()



# Carregar arquivos e separar por tipo de alvo
def load_grouped_json_files(folder_path):
    ucs_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)            
                ucs_files.append(data)
               
    return ucs_files

# Extrair importâncias das features por grupo de arquivos
def extract_feature_importances(json_files):
    feature_importances = {}

    for file in json_files:
        if isinstance(file, list):
            for sub_file in file:
                model = sub_file.get('estimator')
                importances = sub_file.get('feature_importances', [])
                feature_names = sub_file.get('feature_names', [])

                if not model or len(importances) != len(feature_names):
                    continue

                for i, feature in enumerate(feature_names):
                    # Ignorar features chamadas "fs" ou "cs"
                    if feature in ['fs', 'cs']:
                        continue
                    if feature not in feature_importances:
                        feature_importances[feature] = {}
                    feature_importances[feature][model] = importances[i]
        elif isinstance(file, dict):
            model = file.get('estimator')
            importances = file.get('feature_importances', [])
            feature_names = file.get('feature_names', [])

            if not model or len(importances) != len(feature_names):
                continue

            for i, feature in enumerate(feature_names):
                # Ignorar features chamadas "fs" ou "cs"
                if feature in ['fs', 'cs']:
                    continue
                if feature not in feature_importances:
                    feature_importances[feature] = {}
                feature_importances[feature][model] = importances[i]

    # Converter para DataFrame por feature
    dfs = {}
    for feature, values in feature_importances.items():
        dfs[feature] = pd.Series(values).sort_index()
    return dfs



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Literal, Optional

def plotl_feature_importances(
    data: Dict[str, 'pd.Series'],
    title: str,
    orientation: Literal['horizontal', 'vertical'] = 'horizontal',
    figsize: Optional[tuple] = None,
    color_palette: str = 'Set2',
    max_features: Optional[int] = None,
    normalize: bool = True,
    show_values: bool = True,
    value_format: str = ".2f",
    sort_by_mean: bool = True,
    log_scale: bool = False   # ← NEW: log y-scale (only for vertical)
):
    """
    Updated version:
      • Bar values on top of columns are now ALWAYS rotated 90° in vertical mode
      • New parameter `log_scale=True` (only works in vertical orientation)
        → y-axis (Feature Importance) becomes logarithmic
        → automatically adds "(Log Scale)" to ylabel + nice scientific ticks
        → filename gets _log suffix
    """
    # --- 1. Preparação dos Dados (unchanged) ---
    features = list(data.keys())
    models = sorted({model for feature_values in data.values()
                     for model in feature_values.index})
   
    if max_features and len(features) > max_features:
        mean_importances = {feature: df.mean() for feature, df in data.items()}
        top_features = sorted(mean_importances, key=mean_importances.get, reverse=True)[:max_features]
        data = {f: data[f] for f in top_features}
        features = top_features
   
    if normalize:
        model_totals = {}
        for model in models:
            model_total = sum(data[feature].get(model, 0) for feature in features)
            model_totals[model] = model_total
       
        normalized_data = {}
        for feature in features:
            series = data[feature].copy()
            for model in models:
                if model in series.index and model_totals[model] > 0:
                    series[model] = series[model] / model_totals[model]
            normalized_data[feature] = series
        data = normalized_data
   
    if sort_by_mean:
        mean_importances = {feature: df.mean() for feature, df in data.items()}
        sorted_features = sorted(mean_importances,
                                key=mean_importances.get,
                                reverse=(orientation == 'horizontal'))
    else:
        sorted_features = features
   
    # --- 2. Configuração do Gráfico ---
    plt.style.use(['seaborn-v0_8-paper', 'seaborn-v0_8-whitegrid'])
    sns.set_palette(color_palette)
    colors = sns.color_palette(color_palette, n_colors=len(models))
   
    if figsize is None:
        if orientation == 'horizontal':
            figsize = (8, max(4, len(sorted_features) * 0.4))
        else:
            figsize = (max(12, len(sorted_features) * 0.5), 4)
   
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
   
    n_models = len(models)
    n_features = len(sorted_features)
   
    # Auto-fix format (already working from previous fix)
    if show_values and value_format and not value_format.startswith('{'):
        value_format = f"{{:{value_format}}}"
   
    # --- 3. Plotagem ---
    if orientation == 'horizontal':
        # (unchanged - horizontal never uses log_scale or 90° rotation)
        bar_height = 0.8 / n_models
        y_pos = np.arange(n_features)
       
        for i, model in enumerate(models):
            importances = [data[feature].get(model, 0) for feature in sorted_features]
            offset = i * bar_height
            bars = ax.barh(y_pos + offset, importances,
                          height=bar_height, label=model,
                          color=colors[i], edgecolor='white', linewidth=0.5)
            if show_values:
                ax.bar_label(bars, padding=3, fontsize=9, fmt=value_format)
       
        ax.set_yticks(y_pos + bar_height * (n_models - 1) / 2)
        ax.set_yticklabels(sorted_features, fontsize=11)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        max_x = max(data[feature].get(model, 0) for feature in sorted_features for model in models)
        ax.set_xlim(left=0, right=max_x * 1.15 if max_x > 0 else 1)
       
    else:  # vertical (the one in your image)
        bar_width = 0.8 / n_models
        x_pos = np.arange(n_features)
       
        for i, model in enumerate(models):
            importances = [data[feature].get(model, 0) for feature in sorted_features]
            offset = i * bar_width
            bars = ax.bar(x_pos + offset, importances,
                         width=bar_width, label=model,
                         color=colors[i], edgecolor='white', linewidth=0.5)
           
            if show_values:
                # ← ALWAYS rotate 90° on top of columns (your request)
                ax.bar_label(bars, padding=5, fontsize=9, fmt=value_format,
                            rotation=90)
       
        ax.set_xticks(x_pos + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(sorted_features, fontsize=11,
                          rotation=45 if len(sorted_features) > 5 else 0,
                          ha='right')
        ax.set_ylabel('Feature Importance', fontsize=12)
        ax.set_xlabel('Features', fontsize=12)
        max_y = max(data[feature].get(model, 0) for feature in sorted_features for model in models)
        ax.set_ylim(bottom=0, top=max_y * 1.15 if max_y > 0 else 1)
       
        # === NEW: Log scale on Y-axis (Feature Importance) ===
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.0005, top=max_y * 1.2)  # safe range for normalized data
            ax.yaxis.set_major_formatter(plt.LogFormatterSciNotation())
            ylabel_text = 'Feature Importance (Log Scale)'
        else:
            ylabel_text = 'Feature Importance'
        ax.set_ylabel(ylabel_text, fontsize=12)
   
    # --- 4. Ajustes Estéticos (unchanged) ---
    norm_text = " (Normalized)" if normalize else ""
    ax.set_title(f'{title}{norm_text}', fontsize=14, pad=15, weight='semibold')
   
    ax.grid(True, axis='x' if orientation == 'horizontal' else 'y',
            linestyle='--', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
   
    ax.legend(title='Models', fontsize=10, title_fontsize=11,
             frameon=True, framealpha=0.9, edgecolor='gray',
             bbox_to_anchor=(1.02, 1) if orientation == 'horizontal' else (1.02, 1),
             loc='upper left')
   
    fig.tight_layout()
   
    # --- 5. Salvando (with _log suffix when log_scale=True) ---
    orientation_code = 'horiz' if orientation == 'horizontal' else 'vert'
    log_code = '_log' if log_scale else ''
    output_filename_base = f'./img/Feature_Importance_{title.replace(" ", "_")}_{orientation_code}{log_code}'
   
    print(f"✅ Salvando gráficos em {output_filename_base}.png e .pdf")
    plt.savefig(f'{output_filename_base}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_filename_base}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
   
    plt.show()
    return fig, ax




def analyze_training_times(df_time,
                           save_fig: bool = True,
                           output_dir: str = './img',
                           basename: str = "TDS") -> pd.DataFrame:
    """
    COMPLETE ANALYSIS FUNCTION FOR TRAINING TIMES
    • Horizontal boxplot (super detailed: box + all points + mean/median/n annotations)
    • Publication-ready LaTeX table
    • Creative plot #2: Lollipop Ranking Plot (modern, clean, perfect for papers)
    """
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ── 1. FULL STATISTICS (exact from your 40 runs) ─────────────────────
    stats = (df_time
             .groupby('Model')['Time (s)']
             .agg(['count', 'mean', 'std', 'min', 'median', 'max'])
             .round(3))
    stats['sem'] = (stats['std'] / np.sqrt(stats['count'])).round(3)
    stats = stats.sort_values('mean').reset_index()
    stats.insert(0, 'Rank', range(1, len(stats) + 1))
    stats['mean ± std'] = stats.apply(lambda row: f"{row['mean']:.1f} (±{row['std']:.1f})", axis=1)

    print("\n" + "="*70)
    print("📊 TRAINING TIME FULL STATISTICS (40 runs)")
    print("="*70)
    print(stats[['Rank', 'Model', 'count', 'mean ± std', 'min', 'median', 'max']].to_string(index=False))
    print("="*70)

    # ── 2. LaTeX TABLE (ready for your paper) ─────────────────────────────
    latex_df = stats[['Rank', 'Model', 'mean ± std', 'count']].copy()
    latex_table = latex_df.to_latex(
        index=False,
        escape=False,
        caption="Training time statistics per model (mean ± std, 40 independent runs).",
        label="tab:training_times",
        column_format="rllr"
    )
    latex_path = os.path.join(output_dir, f"{basename}_training_times_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"✅ LaTeX table saved → {latex_path}")

    # ── 3. HORIZONTAL BOXPLOT (very rich - all points, annotations, outliers highlighted) ──
    if save_fig:
        plt.figure(figsize=(11, 7))
        ax = sns.boxplot(data=df_time, y='Model', x='Time (s)',
                         orient='h', palette=PALETTE, width=0.6,
                         linewidth=1.8, flierprops={'marker': 'o', 'markersize': 6})
        
        # Add ALL individual runs (jittered swarm)
        sns.stripplot(data=df_time, y='Model', x='Time (s)',
                      orient='h', color='black', alpha=0.7, size=5, jitter=0.25, ax=ax)

        # Annotate EVERY model with rich info
        # for i, model in enumerate(stats['Model']):
        #     row = stats.iloc[i]
        #     x_pos = row['mean'] * 1.02
        #     ax.text(x_pos, i, f"n={int(row['count'])}  •  mean={row['mean']:.1f}s  •  median={row['median']:.1f}s",
        #             va='center', ha='left', fontsize=11, fontweight='bold',
        #             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='gray'))
            
        #     # Highlight outliers (AutoSklearn 601s is obvious)
        #     if row['max'] > row['mean'] + 3 * row['std']:
        #         ax.text(row['max'] + 10, i, f"Outlier: {row['max']:.0f}s",
        #                 color='red', fontsize=10, fontweight='bold')

        ax.set_xlabel('Training Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('')
        ax.set_title(f'Training Time Distribution – {basename} (40 runs)', fontsize=16, pad=20)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        sns.despine(left=True)

        # Save high-quality
        filename = f"{basename}_time_boxplot_horizontal"
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches='tight')
        print(f"✅ Horizontal Boxplot saved → {output_dir}/{filename}.png + .pdf")
        plt.show()
        plt.close()

    # ── 4. CREATIVE PLOT FOR PAPER: LOLLIPOP RANKING (modern & elegant) ─────────────
    if save_fig:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Lollipop lines
        ax.hlines(y=stats['Model'], xmin=-6, xmax=stats['mean'],
                  color='lightblue', linewidth=4, alpha=0.9)
        # Big dots at mean
        scatter = ax.scatter(stats['mean'], stats['Model'],
                             s=stats['mean']*3 + 5,   # size by number of runs
                             c=stats['mean'], cmap=PALETTE+'_r',
                             edgecolor='black', linewidth=1.5, zorder=5)
        
        # Value labels + rank
        for i, row in stats.iterrows():
            ax.text(row['mean'] + 16, i, f"{row['mean']:.1f}s\n(n={int(row['count'])})",
                    va='center', fontsize=12, fontweight='bold')
            ax.text(-6, i, f"#{row['Rank']}", va='center', ha='right',
                    fontsize=14, fontweight='bold', color='darkred')

        ax.set_xlabel('Mean Training Time (seconds)', fontsize=14, fontweight='normal')
        #ax.set_title(f'Training Time Ranking – Lollipop View\n{basename} (size = #runs)', fontsize=16, pad=20)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        plt.gca().invert_yaxis()  # Rank 1 at top
        
        # Colorbar for time
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Time (s)', fontsize=12)
        
        sns.despine()
        plt.tight_layout()

        filename = f"{basename}_time_lollipop_ranking"
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches='tight')
        print(f"✅ Creative Lollipop Ranking saved → {output_dir}/{filename}.png + .pdf")
        plt.show()
        plt.close()

    return stats

#%%

# --- MAIN EXECUTION ---

save_fig = True

CONFIG = [(f"f{i}_", f'./json_automl_f{i}') for i in range(1,2)]

for BASENAME, FOLDER_PATH in CONFIG:    
    
    print(f"Processando pasta: {FOLDER_PATH}")
    
    refname = BASENAME.replace('_','').upper()
    df_results, df_uncertainty, df_time = load_json_data(FOLDER_PATH)
    models_to_remove = [
        'AutoKeras',
        #'TPOT',
        'TabPFN','TabICL',
        #'H2O',
        ]
    models_to_remove = models_to_remove + [m+'-FS' for m in models_to_remove]
    df_results = filter_models(df_results, models_to_remove)
    df_uncertainty = filter_models(df_uncertainty, models_to_remove)
    df_time = filter_models(df_time, models_to_remove)

    df_table = format_df_table(df_results, 'Model', METRICS, 'metrics')
    uncertainty_table = format_df_table(df_uncertainty, 'Model', ['MAD','Uncertainty', 'RMSE'], 'uncertainty')

    print(df_table)
    print(uncertainty_table)

    plot_model_metrics(refname, df_results, [m for m in METRICS if m != 'Time'], save_fig=save_fig, output_dir=FOLDER_FIG)
    run_anova_and_tukey(df_results)
    bootstrap_df, paths = run_bootstrap(df_results, output_dir=FOLDER_FIG)
    
    time_stats = analyze_training_times(df_time, save_fig=True, output_dir=FOLDER_FIG,basename=BASENAME)
    
    df_ranked = compute_performance_index(refname, df_results, [m for m in METRICS if m != 'Time'], save_fig=save_fig, output_dir=FOLDER_FIG)
    df_uncertainty_grouped = df_uncertainty.groupby('Model').mean().reset_index()
    plot_uncertainty(refname, df_uncertainty, save_fig=save_fig, output_dir=FOLDER_FIG)
    plot_uncertainty_pareto(refname, df_uncertainty, save_fig=save_fig, output_dir=FOLDER_FIG)
    plot_uncertainty_grouped(refname, df_uncertainty_grouped, save_fig=save_fig, output_dir=FOLDER_FIG)
    
    #df_feature_selection = analyze_feature_selection(models_to_remove,FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)
    plot_feature_pareto(refname, FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)
    plot_taylor_diagram(refname, FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)
    plot_feature_importance(refname, FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)
    plot_model_params_distribution(refname, FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)
    
    generate_latex_figures_from_folder(refname, FOLDER_FIG, "figures_output.tex")

    plot_best_run_per_model(refname, models_to_remove, FOLDER_PATH, save_fig=save_fig, output_dir=FOLDER_FIG)


    ucs_json = load_grouped_json_files(FOLDER_PATH)
    ucs_json = [[entry for entry in data if entry.get('estimator') not in models_to_remove] for data in ucs_json]

    # Extract dataaset name from JSON
    datasets = [entry['dataset'] for sublist in ucs_json for entry in sublist if 'dataset' in entry]
    if not datasets:
        print(f"Aviso: Nenhum dataset encontrado em {folder_path}. Pulando.")
        continue
    unique_dataset = list(set(datasets))[0].split('-')[0]

    # Extract importances
    feature_importance_data = extract_feature_importances(ucs_json)

    # Remove fs and cs if necessary
    feature_importance_data = {key: value for key, value in feature_importance_data.items() if key not in ['fs', 'cs']}

    # Plot
    plotl_feature_importances(
        data=feature_importance_data,
        title="Feature Importance Analysis - TDS",
        orientation='vertical',
        normalize=True,
        log_scale=False,          # ← activates log y + "(Log Scale)" label
        value_format=".2f"
    )
    
    os.system(f"mkdir -p {FOLDER_TAB}")
    os.system(f"mv *.tex {FOLDER_TAB}")
