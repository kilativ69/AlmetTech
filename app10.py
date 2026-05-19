# -*- coding: utf-8 -*-
"""
Геологический ИИ-анализатор месторождений · AWW 2026
pip install PyQt6 matplotlib pandas numpy scipy scikit-learn openpyxl rarfile
"""

import sys, os, re, zipfile, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QFileDialog, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QSizePolicy,
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QScrollArea,
    QComboBox, QMessageBox, QTextEdit, QProgressBar, QSplitter, QDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

STYLE = """
QMainWindow, QWidget { background:#1a1f2e; color:#d0d4e8; font-family:'Segoe UI',Arial,sans-serif; }
QTabWidget::pane { border:1px solid #2e3550; background:#1a1f2e; }
QTabBar::tab { background:#232940; color:#9aa0b8; padding:10px 20px; margin-right:2px;
               border-top-left-radius:6px; border-top-right-radius:6px; font-size:13px; }
QTabBar::tab:selected { background:#2e75b6; color:#fff; font-weight:bold; }
QPushButton { background:#2e75b6; color:#fff; border:none; border-radius:6px;
              padding:8px 20px; font-size:13px; font-weight:bold; }
QPushButton:hover { background:#3a8fd1; }
QPushButton:pressed { background:#235f96; }
QPushButton:disabled { background:#3e4560; color:#6e7590; }
QPushButton#btn_green { background:#27ae60; }
QPushButton#btn_green:hover { background:#2ecc71; }
QPushButton#btn_gray { background:#2e3550; color:#c0c8e0; padding:7px 16px; font-size:12px; font-weight:normal; }
QTableWidget { background:#232940; color:#d0d4e8; gridline-color:#2e3550; border:none; font-size:12px; }
QHeaderView::section { background:#2e3550; color:#c0c8e0; padding:6px; border:none; font-weight:bold; }
QLabel#metric_val { font-size:22px; font-weight:bold; color:#4d9de0; }
QLabel#metric_lbl { font-size:11px; color:#9aa0b8; }
QFrame#metric_card { background:#232940; border-radius:10px; }
QLabel#header { font-size:15px; font-weight:bold; color:#fff; padding:4px 0; }
QLabel#subheader { font-size:12px; color:#9aa0b8; }
QGroupBox { color:#c0c8e0; border:1px solid #2e3550; border-radius:6px; margin-top:8px;
            padding-top:8px; font-size:12px; font-weight:bold; }
QGroupBox::title { subcontrol-origin:margin; left:10px; color:#a0c0e0; }
QDoubleSpinBox { background:#2e3550; color:#d0d4e8; border:1px solid #3e4560; border-radius:4px; padding:4px 8px; }
QComboBox { background:#2e3550; color:#d0d4e8; border:1px solid #3e4560; border-radius:4px; padding:5px 8px; font-size:12px; min-height:28px; }
QComboBox::drop-down { border:none; }
QComboBox QAbstractItemView { background:#232940; color:#d0d4e8; selection-background-color:#2e75b6; }
QCheckBox { color:#c0c8e0; font-size:12px; }
QScrollArea { border:none; background:transparent; }
QTextEdit { background:#232940; color:#d0d4e8; border:1px solid #2e3550; border-radius:6px; font-size:12px; padding:6px; }
QProgressBar { background:#232940; border:1px solid #2e3550; border-radius:4px; text-align:center; color:#d0d4e8; }
QProgressBar::chunk { background:#2e75b6; border-radius:3px; }
QSplitter::handle { background:#2e3550; }
"""

MPLSTYLE = {
    'figure.facecolor':'#1a1f2e', 'axes.facecolor':'#232940',
    'axes.edgecolor':'#3e4560', 'axes.labelcolor':'#b0b8d0',
    'xtick.color':'#9aa0b8', 'ytick.color':'#9aa0b8',
    'text.color':'#d0d4e8', 'grid.color':'#2e3550',
    'grid.linestyle':'--', 'grid.alpha':0.5,
}
for k, v in MPLSTYLE.items():
    try: plt.rcParams[k] = v
    except: pass


class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(8, 4)):
        self.fig = Figure(figsize=figsize, facecolor='#1a1f2e')
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class InteractiveCanvas(MplCanvas):
    def __init__(self, figsize=(14, 7), allow_nav=True, is_fullscreen=False):
        super().__init__(figsize)
        self._allow_nav = allow_nav
        self._is_fullscreen = is_fullscreen
        self._panning = False
        self._pan_x = self._pan_y = None
        self._xlim0 = self._ylim0 = None
        # _legend_map: handle/text -> list of artists
        self._legend_map = {}
        self._mouse_buttons = set()
        self.fig.canvas.mpl_connect('scroll_event', self._scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._press)
        self.fig.canvas.mpl_connect('button_release_event', self._release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._motion)
        self.fig.canvas.mpl_connect('pick_event', self._on_legend_pick)
        self.fig.canvas.mpl_connect('button_press_event', self._dblclick)

    def _scroll(self, e):
        if not self._allow_nav or e.inaxes is None: return
        ax = e.inaxes
        f = 0.82 if e.button == 'up' else 1.2
        xl, yl = ax.get_xlim(), ax.get_ylim()
        cx, cy = e.xdata, e.ydata
        if cx is None or cy is None: return
        ax.set_xlim([cx+(xl[0]-cx)*f, cx+(xl[1]-cx)*f])
        ax.set_ylim([cy+(yl[0]-cy)*f, cy+(yl[1]-cy)*f])
        self.fig.canvas.draw_idle()

    def _press(self, e):
        if e.button: self._mouse_buttons.add(e.button)
        if not self._allow_nav or e.inaxes is None or e.button != 1: return
        self._panning = True
        self._pan_x, self._pan_y = e.xdata, e.ydata
        self._xlim0, self._ylim0 = e.inaxes.get_xlim(), e.inaxes.get_ylim()

    def _release(self, e):
        if e.button: self._mouse_buttons.discard(e.button)
        if e.button == 1 or 1 not in self._mouse_buttons: self._panning = False

    def _motion(self, e):
        if 1 not in self._mouse_buttons: self._panning = False
        if not self._allow_nav or not self._panning or e.inaxes is None or self._pan_x is None: return
        dx = e.xdata - self._pan_x; dy = e.ydata - self._pan_y
        e.inaxes.set_xlim([self._xlim0[0]-dx, self._xlim0[1]-dx])
        e.inaxes.set_ylim([self._ylim0[0]-dy, self._ylim0[1]-dy])
        self.fig.canvas.draw_idle()

    def _dblclick(self, e):
        if self._is_fullscreen: return
        if e.dblclick and e.inaxes is not None:
            if hasattr(e, 'artist') and e.artist is not None: return
            self._open_fullscreen(e.inaxes)

    def _open_fullscreen(self, clicked_ax):
        import matplotlib.collections as mcoll
        dlg = QDialog()
        dlg.setWindowTitle('График — полный экран')
        dlg.resize(1200, 750)
        dlg.setStyleSheet('background:#1a1f2e;')
        vl = QVBoxLayout(dlg); vl.setContentsMargins(4, 4, 4, 4)
        fs_canvas = InteractiveCanvas(figsize=(14, 8), allow_nav=True, is_fullscreen=True)
        vl.addWidget(fs_canvas)
        new_ax = fs_canvas.fig.add_subplot(111)
        new_ax.set_facecolor(clicked_ax.get_facecolor())

        for line in clicked_ax.get_lines():
            new_ax.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(), linewidth=line.get_linewidth(),
                        linestyle=line.get_linestyle(), marker=line.get_marker(),
                        markersize=line.get_markersize(), alpha=line.get_alpha() or 1.0,
                        label=line.get_label())

        for coll in clicked_ax.collections:
            try:
                if isinstance(coll, mcoll.PathCollection):
                    offsets = coll.get_offsets()
                    if len(offsets) > 0:
                        kwargs = {'alpha': coll.get_alpha() or 1.0, 'zorder': coll.get_zorder()}
                        lbl = coll.get_label() if hasattr(coll, 'get_label') else ''
                        if lbl and not lbl.startswith('_'): kwargs['label'] = lbl
                        sz = coll.get_sizes()
                        if len(sz) > 0: kwargs['s'] = sz
                        fc = coll.get_facecolors()
                        if len(fc) > 0: kwargs['c'] = fc
                        ec = coll.get_edgecolors()
                        if len(ec) > 0: kwargs['edgecolors'] = ec
                        lw = coll.get_linewidths()
                        if len(lw) > 0: kwargs['linewidths'] = lw
                        new_ax.scatter(offsets[:, 0], offsets[:, 1], **kwargs)
                elif isinstance(coll, mcoll.PolyCollection):
                    # Используем сохранённые данные fill_between
                    fd = getattr(coll, '_fill_data', None)
                    if fd is not None:
                        x_d, y1_d, y2_d = fd
                        lbl = coll.get_label() if hasattr(coll, 'get_label') else ''
                        fc = coll.get_facecolor()
                        color = fc[0] if len(fc) > 0 else '#e74c3c'
                        kw = {'alpha': coll.get_alpha() or 0.25, 'color': color, 'zorder': coll.get_zorder()}
                        if lbl and not lbl.startswith('_'): kw['label'] = lbl
                        new_coll = new_ax.fill_between(x_d, y1_d, y2_d, **kw)
                        new_coll._fill_data = fd
            except Exception as ex:
                print(f'FS copy error: {ex}')

        new_ax.set_xlim(clicked_ax.get_xlim())
        new_ax.set_ylim(clicked_ax.get_ylim())
        new_ax.set_xlabel(clicked_ax.get_xlabel(), color='#b0b8d0')
        new_ax.set_ylabel(clicked_ax.get_ylabel(), color='#b0b8d0')
        new_ax.set_title(clicked_ax.get_title(), color='#ffffff')
        new_ax.tick_params(colors='#9aa0b8')
        new_ax.grid(True, color='#2e3550', linestyle='--', alpha=0.5)
        if clicked_ax.yaxis_inverted(): new_ax.invert_yaxis()

        handles_use, labels_use = [], []
        for l in new_ax.get_lines():
            if l.get_label() and not l.get_label().startswith('_'):
                handles_use.append(l); labels_use.append(l.get_label())
        for c in new_ax.collections:
            lbl = c.get_label() if hasattr(c, 'get_label') else ''
            if lbl and not lbl.startswith('_'):
                handles_use.append(c); labels_use.append(lbl)
        if handles_use:
            leg = new_ax.legend(handles_use, labels_use, fontsize=10,
                                facecolor='#2e3550', edgecolor='#3e4560', labelcolor='#d0d4e8')
            if leg: fs_canvas.register_legend(leg)

        fs_canvas.fig.tight_layout()
        fs_canvas.draw()
        dlg.exec()

    def _on_legend_pick(self, event):
        artist = event.artist
        if artist not in self._legend_map: return
        targets = self._legend_map[artist]
        if not isinstance(targets, list): targets = [targets]
        vis = not targets[0].get_visible()
        for t in targets: t.set_visible(vis)
        for art, tgts in self._legend_map.items():
            if not isinstance(tgts, list): tgts = [tgts]
            if targets[0] in tgts:
                art.set_alpha(1.0 if vis else 0.3)
        self.fig.canvas.draw_idle()

    def register_legend(self, legend, extra_groups=None):
        """
        extra_groups: {label_str: [extra_artist, ...]}
        Все extra artists скрываются вместе с основным по клику на легенду.
        """
        self._legend_map = {}
        ax = legend.axes
        real_lines = [l for l in ax.get_lines() if l.get_label() and not l.get_label().startswith('_')]
        real_colls = [c for c in ax.collections
                      if hasattr(c, 'get_label') and c.get_label() and not c.get_label().startswith('_')]

        handles = legend.legend_handles
        texts = legend.get_texts()

        for handle, leg_text in zip(handles, texts):
            label = leg_text.get_text()
            real = None
            for rl in real_lines:
                if rl.get_label() == label: real = rl; break
            if real is None:
                for rc in real_colls:
                    if rc.get_label() == label: real = rc; break
            if real is None:
                all_real = real_lines + real_colls
                try:
                    idx = list(texts).index(leg_text)
                    if 0 <= idx < len(all_real): real = all_real[idx]
                except: pass
            if real is None: continue

            group = [real]
            if extra_groups and label in extra_groups:
                group.extend(extra_groups[label])

            handle.set_picker(True)
            try: handle.set_pickradius(10)
            except: pass
            self._legend_map[handle] = group
            leg_text.set_picker(True)
            self._legend_map[leg_text] = group


# ─────────────────────── helpers ───────────────────────────────
def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

COLUMN_MAPPING = {
    'well_id':'well_id','well':'well_id','скважина':'well_id','well_name':'well_id','name':'well_id',
    'x_m':'x_m','x':'x_m','coord_x':'x_m','x_coord':'x_m','east':'x_m',
    'y_m':'y_m','y':'y_m','coord_y':'y_m','y_coord':'y_m','north':'y_m',
    'depth_m':'depth_m','depth':'depth_m','глубина':'depth_m','depthm':'depth_m','md':'depth_m','depth_md':'depth_m',
    'porosity':'porosity','пористость':'porosity','por':'porosity','phi':'porosity','phie':'porosity',
    'permeability_md':'permeability_mD','permeability':'permeability_mD','проницаемость':'permeability_mD',
    'perm':'permeability_mD','k':'permeability_mD','perm_md':'permeability_mD','k_md':'permeability_mD',
    'density_gcc':'density_gcc','density':'density_gcc','плотность':'density_gcc','dens':'density_gcc',
    'rhob':'density_gcc','bulk_density':'density_gcc',
    'water_saturation':'water_saturation','sw':'water_saturation','кв':'water_saturation','swt':'water_saturation',
    'oil_saturation':'oil_saturation','so':'oil_saturation','кн':'oil_saturation',
    'zone':'zone','зона':'zone','геол_зона':'zone','formation':'zone','пласт':'zone',
    'data_type':'data_type','type':'data_type','тип':'data_type',
    'lab_id':'lab_id','lab':'lab_id','лаборатория':'lab_id',
    'year':'year','год':'year',
    'pz':'PZ','ps':'PS','gk':'GK','ngk':'NGK','gr':'GK','sp':'PS',
}

def read_csv_auto(path):
    from io import StringIO
    encodings = ['utf-8', 'cp1251', 'windows-1251', 'latin-1']
    raw = None
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f: raw = f.read(); break
        except UnicodeDecodeError: continue
    if raw is None:
        with open(path, 'r', encoding='utf-8', errors='replace') as f: raw = f.read()
    lines = [l for l in raw.split('\n') if l.strip()]
    if not lines: raise ValueError('Пустой файл')
    first = lines[0]
    n_tab = first.count('\t'); n_semi = first.count(';'); n_com = first.count(',')
    if n_tab >= 2 and n_tab >= n_semi: sep = '\t'
    elif n_semi >= 2 and n_semi >= n_com: sep = ';'
    elif n_com >= 2: sep = ','
    else: sep = None
    has_decimal_comma = bool(re.search(r'\d,\d', raw[:2000]))
    def try_read(s, decimal):
        if s: return pd.read_csv(StringIO(raw), sep=s, decimal=decimal, low_memory=False)
        return pd.read_csv(StringIO(raw), sep=None, engine='python', decimal=decimal, low_memory=False)
    for decimal in ([',', '.'] if has_decimal_comma else ['.', ',']):
        try:
            df = try_read(sep, decimal)
            if df.shape[1] >= 2: return df
        except Exception: pass
    return pd.read_csv(StringIO(raw), sep=r'\s+', engine='python')

def normalize_columns(df):
    new = {}
    for col in df.columns:
        cl = col.strip().lower().replace(' ', '_').replace('-', '_')
        if cl in COLUMN_MAPPING: new[col] = COLUMN_MAPPING[cl]
    return df.rename(columns=new)

def ensure_numeric(df):
    for col in ['depth_m','porosity','permeability_mD','density_gcc',
                'water_saturation','oil_saturation','x_m','y_m','PZ','PS','GK','NGK']:
        if col not in df.columns: continue
        s = df[col]
        if s.dtype == object:
            s = s.astype(str).str.replace(',', '.').str.strip()
            s = s.str.replace(r'[^\d\.\-eE]', '', regex=True)
        df[col] = pd.to_numeric(s, errors='coerce')
    return df

def try_read_rar(path):
    try:
        import rarfile; all_dfs = []
        with rarfile.RarFile(path) as rf:
            for name in rf.namelist():
                if name.lower().endswith('.csv'):
                    with rf.open(name) as f:
                        content = f.read()
                        tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
                        tmp.write(content); tmp.close()
                        try:
                            df = read_csv_auto(tmp.name)
                            df = normalize_columns(df); df = ensure_numeric(df)
                            wn = os.path.splitext(os.path.basename(name))[0]
                            if 'well_id' not in df.columns: df['well_id'] = wn
                            all_dfs.append(df)
                        except Exception: pass
                        finally: os.unlink(tmp.name)
        return all_dfs
    except ImportError: return None
    except Exception: return None

def well_level_stats(df):
    agg = {'samples': df.groupby('well_id').size()}
    for col in ['depth_m','porosity','permeability_mD','density_gcc']:
        if col in df.columns:
            agg[f'{col}_mean'] = df.groupby('well_id')[col].mean()
            agg[f'{col}_std']  = df.groupby('well_id')[col].std()
    if 'x_m' in df.columns: agg['x'] = df.groupby('well_id')['x_m'].first()
    if 'y_m' in df.columns: agg['y'] = df.groupby('well_id')['y_m'].first()
    result = pd.DataFrame(agg).reset_index(); result.columns.name = None
    return result

def geo_interpretation(prop, mean_val, std_val):
    """Геологическая интерпретация прогнозного значения."""
    if mean_val == 0 and std_val == 0: return ''
    if prop == 'porosity':
        if mean_val > 0.20:   q, verdict = 'высокая', '✅ Хороший коллектор — значимая ёмкость пор'
        elif mean_val > 0.12: q, verdict = 'средняя', '⚠️ Умеренный коллектор — ёмкость ограничена'
        elif mean_val > 0.05: q, verdict = 'низкая', '❌ Слабый коллектор — высокий риск нерентабельности'
        else:                 q, verdict = 'очень низкая', '🚫 Плотные породы — коллекторские свойства маловероятны'
        cv = std_val / mean_val if mean_val > 0 else 0
        hom = 'однородный' if cv < 0.3 else 'умеренно неоднородный' if cv < 0.6 else 'сильно неоднородный'
        return f'Пористость {q} ({mean_val:.1%}). {verdict}. Пласт {hom} (CV={cv:.2f}).'
    elif prop == 'permeability_mD':
        if mean_val > 100:  verdict = '✅ Отличная проницаемость — свободная фильтрация флюида'
        elif mean_val > 10: verdict = '✅ Хорошая проницаемость — коммерческие дебиты возможны'
        elif mean_val > 1:  verdict = '⚠️ Умеренная — может потребоваться ГРП'
        elif mean_val > 0.1:verdict = '❌ Низкая — плотный коллектор, ГРП обязателен'
        else:               verdict = '🚫 Непроницаемые породы — флюидоупор'
        return f'Проницаемость {mean_val:.2f} мД ± {std_val:.2f}. {verdict}'
    elif prop == 'density_gcc':
        if mean_val < 2.2:   rock = 'уголь / лёгкие породы'
        elif mean_val < 2.5: rock = 'песчаник / алевролит'
        elif mean_val < 2.65:rock = 'известняк / доломит'
        elif mean_val < 2.75:rock = 'плотный доломит / гранит'
        else:                rock = 'плотные кристаллические породы'
        return f'Плотность {mean_val:.3f} г/см³ ± {std_val:.3f}. Типичные породы: {rock}.'
    return ''


# ─────────────────────── ML Worker ─────────────────────────────
class MLWorker(QThread):
    finished = pyqtSignal(object, object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, df_all, well_id=None, x=None, y=None, target_props=None):
        super().__init__()
        self.df_all = df_all.copy()
        self.well_id = well_id
        self.x, self.y = x, y
        self.target_props = target_props or ['porosity']

    def _predict_one(self, prop, train_coords, train_prop_vals, target_xy, depths, p0, p1):
        df = self.df_all
        df_prop = df[df[prop].notna() & df['well_id'].isin(train_coords.index)]
        if len(df_prop) < 5: return None

        sample_n = min(3000, len(df_prop))
        samp = df_prop.sample(n=sample_n, random_state=42) if len(df_prop) > sample_n else df_prop

        X_tr, y_tr = [], []
        for _, row in samp.iterrows():
            wid = row['well_id']
            if wid not in train_coords.index: continue
            xy = train_coords.loc[wid].values
            other = train_coords.drop(wid, errors='ignore')
            if len(other) == 0: continue
            dists = np.sqrt(((other.values - xy)**2).sum(axis=1))
            nn_idx = dists.argmin()
            tp = train_prop_vals.reindex(other.index)
            nn_val = tp.iloc[nn_idx] if nn_idx < len(tp) else 0
            if pd.isna(nn_val): nn_val = train_prop_vals.median()
            depth = float(row.get('depth_m', 0) or 0)
            X_tr.append([dists.min(), nn_val, (dists < 1000).sum(), xy[0], xy[1], depth])
            y_tr.append(row[prop])

        if len(X_tr) < 5: return None
        X_tr = np.array(X_tr, dtype=float)
        y_tr = np.array(y_tr, dtype=float)
        med = np.nanmedian(X_tr, axis=0)
        for c in range(X_tr.shape[1]): X_tr[np.isnan(X_tr[:, c]), c] = med[c]

        model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        self.progress.emit(p0 + (p1 - p0) // 2)

        tx, ty = target_xy
        dists_t = np.sqrt(((train_coords.values - np.array([tx, ty]))**2).sum(axis=1))
        nn_d = dists_t.min()
        tp_all = train_prop_vals.reindex(train_coords.index)
        nn_v = tp_all.iloc[dists_t.argmin()] if dists_t.argmin() < len(tp_all) else train_prop_vals.median()
        if pd.isna(nn_v): nn_v = train_prop_vals.median()
        n1km = int((dists_t < 1000).sum())

        preds = []
        for d in depths:
            xp = np.array([[nn_d, nn_v, n1km, tx, ty, float(d)]])
            tree_p = np.array([t.predict(xp)[0] for t in model.estimators_])
            m, s = float(tree_p.mean()), float(tree_p.std())
            if prop == 'porosity':      m = float(np.clip(m, 0, 1)); s = float(np.clip(s, 0, 0.5))
            elif prop == 'permeability_mD': m = max(0.0, m); s = max(0.0, s)
            elif prop == 'density_gcc': m = float(np.clip(m, 1.5, 3.5)); s = max(0.0, s)
            preds.append({f'{prop}_pred': m, f'{prop}_lower': max(0, m-s), f'{prop}_upper': m+s})

        self.progress.emit(p1)
        return pd.DataFrame(preds), float(nn_d), n1km

    def run(self):
        try:
            df = self.df_all
            props = [p for p in self.target_props if p in df.columns]
            if not props: self.error.emit('Ни одно свойство не найдено'); return

            ref = props[0]
            well_coords = df[df[ref].notna()].groupby('well_id')[['x_m','y_m']].first()
            well_coords = well_coords[well_coords['x_m'].notna() & well_coords['y_m'].notna()]

            if self.well_id:
                if self.well_id not in df['well_id'].values:
                    self.error.emit(f'Скважина {self.well_id} не найдена'); return
                train_coords = well_coords.drop(self.well_id, errors='ignore')
            else:
                train_coords = well_coords

            if len(train_coords) < 2: self.error.emit('Нужно ≥2 скважин с координатами'); return

            if self.well_id:
                if self.well_id in well_coords.index: tx, ty = well_coords.loc[self.well_id].values
                else: self.error.emit(f'У скважины {self.well_id} нет координат'); return
                tdf = df[df['well_id'] == self.well_id]
                depths = sorted(tdf['depth_m'].dropna().unique())
                if len(depths) == 0:
                    depths = np.linspace(df['depth_m'].dropna().min(), df['depth_m'].dropna().max(), 20)
            else:
                tx, ty = float(self.x), float(self.y)
                depths = np.linspace(df['depth_m'].dropna().min(), df['depth_m'].dropna().max(), 30)

            all_preds = pd.DataFrame({'depth_m': depths})
            summary = {}
            step = 90 // max(len(props), 1)

            for i, prop in enumerate(props):
                p0, p1 = 5 + i * step, 5 + (i+1) * step
                train_prop = df[df[prop].notna()].groupby('well_id')[prop].median()
                tc = train_coords.loc[train_coords.index.intersection(train_prop.index)]
                tp = train_prop.reindex(tc.index)
                if len(tc) < 2: continue

                res = self._predict_one(prop, tc, tp, (tx, ty), depths, p0, p1)
                if res is None: continue
                pred_df, nn_d, n1km = res
                pred_df['depth_m'] = depths
                all_preds = all_preds.merge(pred_df, on='depth_m', how='left')
                summary[prop] = {
                    'mean':  float(all_preds[f'{prop}_pred'].mean()),
                    'std':   float(all_preds[f'{prop}_pred'].std()),
                    'lower': float(all_preds[f'{prop}_lower'].mean()) if f'{prop}_lower' in all_preds else 0,
                    'upper': float(all_preds[f'{prop}_upper'].mean()) if f'{prop}_upper' in all_preds else 0,
                    'nearest_distance': nn_d,
                    'n_neighbors': n1km,
                }

            self.progress.emit(100)
            first_s = list(summary.values())[0] if summary else {}
            info = {
                'target_x': tx, 'target_y': ty, 'summary': summary,
                'mean_por': summary.get('porosity', {}).get('mean', 0),
                'std_por':  summary.get('porosity', {}).get('std', 0),
                'lower':    summary.get('porosity', {}).get('lower', 0),
                'upper':    summary.get('porosity', {}).get('upper', 0),
                'nearest_distance': first_s.get('nearest_distance', 0),
                'n_neighbors':      first_s.get('n_neighbors', 0),
            }
            self.finished.emit(all_preds, info)
        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n{traceback.format_exc()[-400:]}')


# ─────────────────────── Main Window ───────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df_all = self.df_single = self.df_ml_result = self.df_registry = None
        self.well_coords = {}
        self.ml_worker = None
        self._map_selected_point = None
        self.setWindowTitle('Геологический ИИ-анализатор месторождений · AWW 2026')
        self.resize(1440, 900)
        self._build_ui()

    def _make_fill(self, ax, x, y1, y2, **kw):
        """fill_between с сохранёнными данными для fullscreen-копирования."""
        coll = ax.fill_between(x, y1, y2, **kw)
        coll._fill_data = (np.asarray(x), np.asarray(y1), np.asarray(y2))
        return coll

    def _build_header(self):
        bar = QWidget(); bar.setStyleSheet('background:#0f1420; padding:4px;')
        h = QHBoxLayout(bar); h.setContentsMargins(16,6,16,6); h.setSpacing(10)
        title = QLabel('⛽  Геологический ИИ-анализатор')
        title.setFont(QFont('Segoe UI', 13, QFont.Weight.Bold))
        title.setStyleSheet('color:#ffffff;')
        self.lbl_status = QLabel('Загрузите архив со скважинами')
        self.lbl_status.setStyleSheet('color:#9aa0b8; font-size:12px;')
        self.btn_archive = QPushButton('📂  Загрузить архив')
        self.btn_archive.clicked.connect(self.load_archive)
        self.btn_well = QPushButton('🛢  Загрузить скважину')
        self.btn_well.clicked.connect(self.load_single_well)
        self.btn_well.setEnabled(False)
        self.btn_export = QPushButton('📊  Экспорт отчёта')
        self.btn_export.setObjectName('btn_green')
        self.btn_export.clicked.connect(self.export_report)
        self.btn_export.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200); self.progress.setMaximumHeight(16)
        self.progress.setVisible(False)
        h.addWidget(title); h.addWidget(self.lbl_status); h.addWidget(self.progress)
        h.addStretch()
        h.addWidget(self.btn_archive); h.addWidget(self.btn_well); h.addWidget(self.btn_export)
        return bar

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        mv = QVBoxLayout(central); mv.setContentsMargins(0,0,0,0); mv.setSpacing(0)
        mv.addWidget(self._build_header())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_summary(),   '📊  Сводка')
        self.tabs.addTab(self._build_tab_maps(),      '🗺  Карты')
        self.tabs.addTab(self._build_tab_profiles(),  '📈  Профили')
        self.tabs.addTab(self._build_tab_variogram(), '📐  Вариограмма')
        self.tabs.addTab(self._build_tab_single(),    '🔍  Скважина')
        self.tabs.addTab(self._build_tab_restore(),   '🤖  Восстановление')
        mv.addWidget(self.tabs)

    def _metric_card(self, label, value='—'):
        card = QFrame(); card.setObjectName('metric_card')
        vl = QVBoxLayout(card); vl.setContentsMargins(14,10,14,10); vl.setSpacing(2)
        lv = QLabel(value); lv.setObjectName('metric_val')
        ll = QLabel(label); ll.setObjectName('metric_lbl')
        vl.addWidget(lv); vl.addWidget(ll)
        return card, lv

    # ══ ВК 1: СВОДКА ══════════════════════════════════════════════════════
    def _build_tab_summary(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(16,16,16,16); v.setSpacing(10)
        lbl = QLabel('Сводка по месторождению'); lbl.setObjectName('header'); v.addWidget(lbl)
        grid = QGridLayout(); self.metrics = {}
        for i, (k, t) in enumerate([('wells','Скважин'),('samples','Образцов'),('mean_por','Ср. пористость φ'),
                                     ('mean_perm','Ср. проницаемость, мД'),('depth_min','Глубина min, м'),('depth_max','Глубина max, м')]):
            card, val = self._metric_card(t); self.metrics[k] = val; grid.addWidget(card, i//3, i%3)
        v.addLayout(grid)
        hdr = QHBoxLayout()
        lbl2 = QLabel('Реестр скважин'); lbl2.setObjectName('header'); hdr.addWidget(lbl2); hdr.addStretch()
        hint = QLabel('клик по легенде графиков — скрыть/показать')
        hint.setStyleSheet('color:#5a6080; font-size:11px; font-style:italic;'); hdr.addWidget(hint)
        btn_exp = QPushButton('⛶  Развернуть'); btn_exp.setObjectName('btn_gray')
        btn_exp.clicked.connect(self._expand_registry); hdr.addWidget(btn_exp)
        v.addLayout(hdr)
        self.tbl_registry = QTableWidget()
        self.tbl_registry.setAlternatingRowColors(True)
        self.tbl_registry.setStyleSheet('alternate-background-color:#1e2438;')
        self.tbl_registry.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_registry.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl_registry, stretch=1)
        return w

    def _expand_registry(self):
        if self.df_all is None: return
        dlg = QDialog(self); dlg.setWindowTitle('Реестр скважин — полный экран')
        dlg.resize(1300, 750); dlg.setStyleSheet(STYLE)
        vl = QVBoxLayout(dlg)
        tbl = QTableWidget(); tbl.setAlternatingRowColors(True)
        tbl.setStyleSheet('alternate-background-color:#1e2438; font-size:12px;')
        src = self.tbl_registry
        tbl.setRowCount(src.rowCount()); tbl.setColumnCount(src.columnCount())
        tbl.setHorizontalHeaderLabels([src.horizontalHeaderItem(c).text() if src.horizontalHeaderItem(c) else '' for c in range(src.columnCount())])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for r in range(src.rowCount()):
            for c in range(src.columnCount()):
                it = src.item(r, c)
                if it:
                    ni = QTableWidgetItem(it.text())
                    ni.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                    tbl.setItem(r, c, ni)
        vl.addWidget(tbl)
        bc = QPushButton('✕  Закрыть'); bc.setObjectName('btn_gray'); bc.clicked.connect(dlg.close)
        hl = QHBoxLayout(); hl.addStretch(); hl.addWidget(bc); vl.addLayout(hl)
        dlg.exec()

    def update_summary(self):
        if self.df_all is None: return
        df = self.df_all
        self.metrics['wells'].setText(str(df['well_id'].nunique()))
        self.metrics['samples'].setText(str(len(df)))
        if 'porosity' in df.columns:
            v = df['porosity'].dropna()
            self.metrics['mean_por'].setText(f'{v.mean():.4f}' if len(v) else '—')
        if 'permeability_mD' in df.columns:
            v = df['permeability_mD'].dropna()
            self.metrics['mean_perm'].setText(f'{v.mean():.2f}' if len(v) else '—')
        if 'depth_m' in df.columns:
            self.metrics['depth_min'].setText(f'{df["depth_m"].min():.1f}')
            self.metrics['depth_max'].setText(f'{df["depth_m"].max():.1f}')
        ws = well_level_stats(df)
        ws['_s'] = ws['well_id'].apply(natural_sort_key)
        ws = ws.sort_values('_s').drop(columns=['_s']).reset_index(drop=True)
        show_cols = ['well_id','samples']
        col_labels = ['Скважина','Образцов']
        for col, t in [('depth_m_mean','Глубина, м'),('porosity_mean','φ сред.'),
                       ('permeability_mD_mean','k сред., мД'),('density_gcc_mean','ρ сред.'),('x','X, м'),('y','Y, м')]:
            if col in ws.columns: show_cols.append(col); col_labels.append(t)
        self.tbl_registry.setRowCount(len(ws)); self.tbl_registry.setColumnCount(len(show_cols))
        self.tbl_registry.setHorizontalHeaderLabels(col_labels)
        for r, row in ws.iterrows():
            for c, col in enumerate(show_cols):
                val = row.get(col, '')
                if col == 'well_id': txt = str(val)
                elif col == 'samples': txt = str(int(val)) if pd.notna(val) else '—'
                elif pd.isna(val): txt = '—'
                else: txt = f'{val:.3f}' if abs(val) < 1e4 else f'{val:.1f}'
                item = QTableWidgetItem(txt)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.tbl_registry.setItem(r, c, item)

    # ══ ВК 2: КАРТЫ ═══════════════════════════════════════════════════════
    def _build_tab_maps(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(16,16,16,16); v.setSpacing(6)
        h_top = QHBoxLayout()
        lbl = QLabel('Карта месторождения'); lbl.setObjectName('header')
        note = QLabel('🖱 Колёсико — зум · ЛКМ — двигать · ПКМ на карте — выбрать точку для ML')
        note.setStyleSheet('color:#5a6080; font-size:11px; font-style:italic;')
        h_top.addWidget(lbl); h_top.addStretch(); h_top.addWidget(note); v.addLayout(h_top)
        ctrl = QHBoxLayout(); ctrl.addWidget(QLabel('Цвет скважин:'))
        self.combo_map_color = QComboBox()
        self.combo_map_color.addItems(['Медианная φ (пористость)','Медианная k (проницаемость)','Кол-во образцов'])
        self.combo_map_color.currentIndexChanged.connect(self.update_maps); ctrl.addWidget(self.combo_map_color)
        self.chk_labels = QCheckBox('Подписи скважин'); self.chk_labels.setChecked(True)
        self.chk_labels.stateChanged.connect(self.update_maps); ctrl.addWidget(self.chk_labels)
        ctrl.addStretch()
        self.lbl_map_point = QLabel('ПКМ на карте — выбрать точку для ML-прогноза')
        self.lbl_map_point.setStyleSheet('color:#4a8a6a; font-size:11px;'); ctrl.addWidget(self.lbl_map_point)
        v.addLayout(ctrl)
        self.canvas_map = InteractiveCanvas(figsize=(14,8))
        self.canvas_map.fig.canvas.mpl_connect('button_press_event', self._on_map_click)
        v.addWidget(self.canvas_map)
        return w

    def _on_map_click(self, event):
        if event.button != 3 or event.inaxes is None: return
        x, y = event.xdata, event.ydata
        if x is None or y is None: return
        self._map_selected_point = (x, y)
        ax = event.inaxes
        for art in list(ax.get_children()):
            if getattr(art, '_map_selected', False): art.remove()
        sc = ax.scatter(x, y, s=400, c='#f1c40f', marker='*', edgecolors='#ffffff', linewidths=2, zorder=12)
        sc._map_selected = True
        ann = ax.annotate(f'({x:.0f}, {y:.0f})\nПерейти → Восстановление', (x, y),
                          textcoords='offset points', xytext=(12, 12), fontsize=9, color='#fff',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='#8b5e00', alpha=0.9, edgecolor='none'), zorder=13)
        ann._map_selected = True
        self.canvas_map.fig.canvas.draw_idle()
        self.spin_x.setValue(x); self.spin_y.setValue(y)
        self.lbl_map_point.setText(f'✅ Выбрано: X={x:.0f}  Y={y:.0f}  → вкладка Восстановление')
        self.lbl_map_point.setStyleSheet('color:#f1c40f; font-size:11px; font-weight:bold;')

    def update_maps(self):
        if self.df_all is None: return
        df = self.df_all; mode = self.combo_map_color.currentIndex()
        show_labels = self.chk_labels.isChecked()
        ws = df.groupby('well_id').agg(por=('porosity','median'), perm=('permeability_mD','median'),
                                        n=('depth_m','count'), x=('x_m','first'), y=('y_m','first')).reset_index()
        ws = ws[ws['x'].notna() & ws['y'].notna()]
        if len(ws) == 0: return
        if mode == 0: cc, cm, cl = 'por', 'plasma', 'Медианная пористость φ'
        elif mode == 1: cc, cm, cl = 'perm', 'viridis', 'Медианная проницаемость, мД'
        else: cc, cm, cl = 'n', 'cool', 'Кол-во образцов'
        valid = ws[ws[cc].notna()]
        if len(valid) == 0: valid = ws
        vmin, vmax = valid[cc].min(), valid[cc].max()
        if vmin == vmax: vmin -= 0.001; vmax += 0.001
        norm = Normalize(vmin=vmin, vmax=vmax); cmap = plt.get_cmap(cm)
        self.canvas_map.fig.clear(); ax = self.canvas_map.fig.add_subplot(111)
        ax.set_facecolor('#232940'); self.canvas_map.fig.patch.set_facecolor('#1a1f2e')
        if mode in (0,1) and len(valid) >= 4:
            try:
                from matplotlib.tri import Triangulation
                tri = Triangulation(valid['x'], valid['y'])
                ax.tricontourf(tri, valid[cc], levels=12, cmap=cm, alpha=0.25, zorder=1)
            except: pass
        sizes = np.clip(np.sqrt(ws['n'].values) * 14, 30, 500)
        for i, r in ws.iterrows():
            cv = r[cc] if pd.notna(r[cc]) else vmin
            ax.scatter(r['x'], r['y'], s=sizes[i], c=[cmap(norm(cv))],
                       edgecolors='#ffffff', linewidths=1.2, zorder=4, alpha=0.9)
            if show_labels:
                sid = re.sub(r'(?i)^well[_\-\s]*', '', str(r['well_id'])).strip() or str(r['well_id'])
                ax.text(r['x'], r['y'], sid, ha='center', va='bottom', fontsize=7.5, color='#e0e8ff',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='#1a1f2e', alpha=0.55, edgecolor='none'), zorder=5)
        sm = ScalarMappable(cmap=cm, norm=norm); sm.set_array([])
        cb = self.canvas_map.fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label(cl, color='#c0c8e0', fontsize=10)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='#c0c8e0')
        if self._map_selected_point:
            px, py = self._map_selected_point
            sc = ax.scatter(px, py, s=400, c='#f1c40f', marker='*', edgecolors='#ffffff', linewidths=2, zorder=12)
            sc._map_selected = True
        ax.set_title(f'Карта месторождения · {len(ws)} скважин\nПКМ — выбрать точку для ML-прогноза',
                     color='#ffffff', fontsize=12)
        ax.set_xlabel('X, м', color='#b0b8d0'); ax.set_ylabel('Y, м', color='#b0b8d0')
        ax.tick_params(colors='#9aa0b8'); ax.grid(True, color='#2e3550', linestyle='--', alpha=0.4)
        ax.set_aspect('equal', adjustable='datalim')
        self.canvas_map.fig.tight_layout(); self.canvas_map.draw()

    # ══ ВК 3: ПРОФИЛИ ═════════════════════════════════════════════════════
    def _build_tab_profiles(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(16,16,16,16); v.setSpacing(8)
        lbl = QLabel('Сравнение профилей скважин'); lbl.setObjectName('header'); v.addWidget(lbl)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('Скв. 1:')); self.combo_w1 = QComboBox(); self.combo_w1.setMinimumWidth(180); ctrl.addWidget(self.combo_w1)
        ctrl.addWidget(QLabel('Скв. 2:')); self.combo_w2 = QComboBox(); self.combo_w2.setMinimumWidth(180); ctrl.addWidget(self.combo_w2)
        ctrl.addWidget(QLabel('Свойство:')); self.combo_prop = QComboBox()
        self.combo_prop.addItems(['Пористость (porosity)','Проницаемость (permeability_mD)','Плотность (density_gcc)','Вода (water_saturation)'])
        ctrl.addWidget(self.combo_prop)
        btn = QPushButton('Сравнить'); btn.clicked.connect(self.update_profiles); ctrl.addWidget(btn)
        hint = QLabel('клик по легенде — скрыть/показать'); hint.setStyleSheet('color:#4a8a6a; font-size:11px; font-style:italic;')
        ctrl.addWidget(hint); ctrl.addStretch(); v.addLayout(ctrl)
        self.canvas_profiles = InteractiveCanvas(figsize=(14,8)); v.addWidget(self.canvas_profiles)
        return w

    def update_profiles(self):
        if self.df_all is None: return
        w1 = self.combo_w1.currentText().split('  ')[0].strip()
        w2 = self.combo_w2.currentText().split('  ')[0].strip()
        if not w1 or not w2: return
        prop_map = ['porosity','permeability_mD','density_gcc','water_saturation']
        prop = prop_map[self.combo_prop.currentIndex()]
        prop_label = self.combo_prop.currentText()
        df1 = self.df_all[self.df_all['well_id']==w1].dropna(subset=['depth_m']).sort_values('depth_m')
        df2 = self.df_all[self.df_all['well_id']==w2].dropna(subset=['depth_m']).sort_values('depth_m')
        self.canvas_profiles.fig.clear(); ax = self.canvas_profiles.fig.add_subplot(111)
        ax.set_facecolor('#232940')
        for df, wid, color, marker in [(df1, w1, '#3498db', 'o'), (df2, w2, '#e67e22', 's')]:
            if prop not in df.columns or df[prop].isna().all():
                ax.text(0.5, 0.5, f'Нет данных {prop} для {wid}', transform=ax.transAxes, ha='center', va='center', color='#9aa0b8', fontsize=13)
                continue
            df_p = df[df[prop].notna()]; n = len(df_p)
            if n > 200: ax.plot(df_p[prop], df_p['depth_m'], '-', color=color, linewidth=1.5, label=f'{wid}  (n={n})', alpha=0.85)
            else: ax.plot(df_p[prop], df_p['depth_m'], f'{marker}-', color=color, linewidth=1.8, markersize=4, label=f'{wid}  (n={n})', alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel(prop_label, color='#b0b8d0', fontsize=12); ax.set_ylabel('Глубина, м', color='#b0b8d0', fontsize=12)
        ax.set_title(f'{w1} vs {w2}', color='#ffffff', fontsize=13); ax.tick_params(colors='#9aa0b8')
        ax.grid(True, color='#2e3550', linestyle='--', alpha=0.5)
        leg = ax.legend(fontsize=11, facecolor='#2e3550', edgecolor='#3e4560', labelcolor='#d0d4e8')
        if leg: self.canvas_profiles.register_legend(leg)
        self.canvas_profiles.fig.tight_layout(); self.canvas_profiles.draw()

    # ══ ВК 4: ВАРИОГРАММА — баг с призрачными кружками исправлен ══════════
    def _build_tab_variogram(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(16,16,16,16); v.setSpacing(8)
        lbl = QLabel('Пространственная вариограмма пористости'); lbl.setObjectName('header'); v.addWidget(lbl)
        note = QLabel('Показывает пространственную однородность: насколько похожи скважины на разных расстояниях')
        note.setStyleSheet('color:#9aa0b8; font-size:11px;'); note.setWordWrap(True); v.addWidget(note)
        self.lbl_variogram = QLabel('')
        self.lbl_variogram.setStyleSheet('color:#9aa0b8; font-size:12px; padding:4px;')
        self.lbl_variogram.setWordWrap(True)
        v.addWidget(self.lbl_variogram)
        self.canvas_variogram = InteractiveCanvas(figsize=(14,6)); v.addWidget(self.canvas_variogram)
        return w

    def update_variogram(self):
        if self.df_all is None: return
        df = self.df_all
        if 'porosity' not in df.columns:
            self.lbl_variogram.setText('Нет данных по пористости'); return
        df_clean = df[df['porosity'].notna() & df['x_m'].notna() & df['y_m'].notna()]
        ws = df_clean.groupby('well_id').agg(por=('porosity','median'), x=('x_m','first'), y=('y_m','first')).reset_index().dropna()
        if len(ws) < 3:
            self.lbl_variogram.setText('Нужно ≥3 скважин с координатами и пористостью'); return
        coords = ws[['x','y']].values; por = ws['por'].values
        D = cdist(coords, coords); n = len(ws)
        distances, gammas = [], []
        for i in range(n):
            for j in range(i+1, n):
                distances.append(D[i,j]); gammas.append(0.5*(por[i]-por[j])**2)
        distances = np.array(distances); gammas = np.array(gammas)
        if len(distances) == 0: return
        max_d = distances.max()
        n_bins = min(20, max(5, len(distances)//3))
        bins = np.linspace(0, max_d * 0.95, n_bins + 1)
        bc, bm, bn = [], [], []
        for i in range(n_bins):
            mask = (distances >= bins[i]) & (distances < bins[i+1])
            if mask.sum() >= 2:
                bc.append((bins[i]+bins[i+1])/2); bm.append(gammas[mask].mean()); bn.append(mask.sum())

        self.canvas_variogram.fig.clear()
        ax = self.canvas_variogram.fig.add_subplot(111); ax.set_facecolor('#232940')

        # Scatter пар — БЕЗ label (прокси для легенды создаём отдельно)
        ax.scatter(distances, gammas, s=15, alpha=0.18, color='#3498db', edgecolors='none', zorder=2)

        import matplotlib.patches as mpatches
        pairs_proxy = mpatches.Patch(color='#3498db', alpha=0.5, label=f'Пары скважин ({len(distances)})')

        var_line = None
        bin_scatter = None
        if len(bc) >= 2:
            # Линия — с label → войдёт в легенду
            var_line = ax.plot(bc, bm, '-', color='#e74c3c', linewidth=2.5,
                               label='Экспериментальная вариограмма', zorder=5)[0]
            # Кружки бинов — БЕЗ label, скроются через extra_groups
            sizes_arr = np.array([max(40, min(cn*8, 300)) for cn in bn])
            bin_scatter = ax.scatter(bc, bm, s=sizes_arr, c='#e74c3c',
                                     edgecolors='#ffffff', linewidths=1.5, zorder=6, alpha=0.9)

        disp_line = ax.axhline(np.var(por), color='#f1c40f', linestyle=':', linewidth=1.5,
                               alpha=0.7, label=f'Дисперсия: {np.var(por):.4f}')

        ax.set_xlabel('Расстояние, м', color='#b0b8d0', fontsize=12)
        ax.set_ylabel('γ(h) = ½·(φ₁−φ₂)²', color='#b0b8d0', fontsize=12)
        ax.set_title(f'Вариограмма пористости · {len(ws)} скважин · {len(distances)} пар',
                     color='#ffffff', fontsize=13)
        ax.tick_params(colors='#9aa0b8'); ax.grid(True, color='#2e3550', linestyle='--', alpha=0.5)

        leg_handles = [pairs_proxy]
        if var_line: leg_handles.append(var_line)
        leg_handles.append(disp_line)
        leg = ax.legend(handles=leg_handles, fontsize=10,
                        facecolor='#2e3550', edgecolor='#3e4560', labelcolor='#d0d4e8')

        # Регистрируем: кружки бинов скрываются ВМЕСТЕ с линией
        extra = {}
        if var_line and bin_scatter:
            extra['Экспериментальная вариограмма'] = [bin_scatter]
        if leg: self.canvas_variogram.register_legend(leg, extra_groups=extra)

        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        self.canvas_variogram.fig.tight_layout(); self.canvas_variogram.draw()

        if len(bc) >= 2:
            slope = (bm[-1]-bm[0]) / (bc[-1]-bc[0]) if bc[-1] != bc[0] else 0
            if slope > 1e-6:
                txt = ('📈 Вариограмма нарастает — типичная пространственная неоднородность. '
                       'Пористость коррелирует на коротких расстояниях; радиус влияния скважин '
                       f'оценивается ~{bc[len(bc)//2]:.0f} м')
            elif slope < -1e-6:
                txt = '📉 Убывающая вариограмма — нетипично; возможны выбросы или структурный тренд в данных'
            else:
                txt = '📊 Стабильная вариограмма — пористость пространственно однородна, хорошая связность пласта'
            self.lbl_variogram.setText(f'{len(distances)} пар | {distances.min():.0f}–{max_d:.0f} м | {txt}')

    # ══ ВК 5: ОДИНОЧНАЯ СКВАЖИНА ══════════════════════════════════════════
    def _build_tab_single(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(16,16,16,16); v.setSpacing(8)
        lbl = QLabel('Анализ загруженной скважины'); lbl.setObjectName('header'); v.addWidget(lbl)
        self.lbl_single_status = QLabel('Нажмите «🛢 Загрузить скважину» для анализа')
        self.lbl_single_status.setStyleSheet('color:#9aa0b8; font-size:12px;')
        self.lbl_single_status.setWordWrap(True); v.addWidget(self.lbl_single_status)
        self.lbl_single_loading = QLabel('')
        self.lbl_single_loading.setStyleSheet('color:#e67e22; font-size:13px; font-weight:bold;')
        v.addWidget(self.lbl_single_loading)
        grid = QGridLayout(); self.single_metrics = {}
        for i, (k, t) in enumerate([('samples','Образцов'),('por_mean','Ср. φ'),('perm_mean','Ср. k, мД'),('r2','R² ML')]):
            card, val = self._metric_card(t); self.single_metrics[k] = val; grid.addWidget(card, 0, i)
        v.addLayout(grid)
        hint = QLabel('🖱 Двойной клик по графику — открыть на весь экран')
        hint.setStyleSheet('color:#5a6080; font-size:11px; font-style:italic;'); v.addWidget(hint)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvas_single_props = InteractiveCanvas(figsize=(7,7), allow_nav=False)
        self.canvas_single_ml    = InteractiveCanvas(figsize=(7,7), allow_nav=False)
        splitter.addWidget(self.canvas_single_props); splitter.addWidget(self.canvas_single_ml)
        splitter.setSizes([500,500]); v.addWidget(splitter, stretch=1)
        return w

        # ══ ВК 5: ОДИНОЧНАЯ СКВАЖИНА — метод update_single_view ══════════════
        # ══ ВК 5: ОДИНОЧНАЯ СКВАЖИНА — метод update_single_view ══════════════
        # ══ ВК 5: ОДИНОЧНАЯ СКВАЖИНА — метод update_single_view ══════════════
    def update_single_view(self):
        if self.df_single is None: return
        df = self.df_single; ml = self.df_ml_result
        self.single_metrics['samples'].setText(str(len(df)))
        for k, col in [('por_mean','porosity'),('perm_mean','permeability_mD')]:
            if col in df.columns and df[col].notna().any():
                vv = df[col].dropna().mean()
                self.single_metrics[k].setText(f'{vv:.4f}' if col=='porosity' else f'{vv:.2f}')
            else: self.single_metrics[k].setText('—')

        # Левый — профили свойств
        self.canvas_single_props.fig.clear()
        props = [(c,l) for c,l in [('porosity','Пористость φ'),('permeability_mD','Проницаемость, мД'),
                                    ('density_gcc','Плотность, г/см³'),('water_saturation','Sw'),
                                    ('GK','ГК'),('PZ','ПЗ'),('PS','ПС')]
                 if c in df.columns and df[c].notna().any()]
        if not props or 'depth_m' not in df.columns or df['depth_m'].isna().all():
            ax = self.canvas_single_props.fig.add_subplot(111); ax.set_facecolor('#232940')
            ax.text(0.5, 0.5, 'Нет данных', transform=ax.transAxes, ha='center', va='center', color='#9aa0b8', fontsize=13)
            self.canvas_single_props.draw()
        else:
            n_p = len(props)
            # ИСПРАВЛЕНО: squeeze=False чтобы всегда был двумерный массив
            axes = self.canvas_single_props.fig.subplots(1, n_p, sharey=True, squeeze=False)
            axes = axes[0]  # берём первую строку
            if n_p == 1: axes = [axes[0]]
            self.canvas_single_props.fig.patch.set_facecolor('#1a1f2e')
            colors_list = ['#3498db','#e67e22','#2ecc71','#9b59b6','#e74c3c','#1abc9c','#f1c40f']
            df_s = df.dropna(subset=['depth_m']).sort_values('depth_m')
            for ax, (col,lbl_txt), color in zip(axes, props, colors_list):
                ax.set_facecolor('#232940'); d = df_s[df_s[col].notna()]
                if len(d) > 200: ax.plot(d[col], d['depth_m'], '-', color=color, linewidth=1.2, alpha=0.85)
                else: ax.plot(d[col], d['depth_m'], 'o-', color=color, linewidth=1.5, markersize=3, alpha=0.85)
                
                # Все названия снизу, кроме проницаемости — её сверху
                if 'Проницаемость' in lbl_txt:
                    ax.set_title(lbl_txt, color='#b0b8d0', fontsize=9, pad=2)
                    ax.set_xlabel('')
                else:
                    ax.set_title('')
                    ax.set_xlabel(lbl_txt, color='#b0b8d0', fontsize=9)
                
                ax.tick_params(colors='#9aa0b8', labelsize=8)
                ax.grid(True, color='#2e3550', linestyle='--', alpha=0.5); ax.invert_yaxis()
            axes[0].set_ylabel('Глубина, м', color='#b0b8d0', fontsize=10)
            wn = df['well_id'].iloc[0] if 'well_id' in df.columns else ''
            self.canvas_single_props.fig.suptitle(f'Профили скважины: {wn}', color='#ffffff', fontsize=11, y=0.99)
            self.canvas_single_props.fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
            self.canvas_single_props.draw()

        # Правый — Факт vs ML
        self.canvas_single_ml.fig.clear()
        ax2 = self.canvas_single_ml.fig.add_subplot(111); ax2.set_facecolor('#232940')
        legend_items, all_vals = [], []

        if 'porosity' in df.columns and 'depth_m' in df.columns:
            df_p = df[df['porosity'].notna() & df['depth_m'].notna()].sort_values('depth_m')
            if len(df_p) > 0:
                lf = ax2.plot(df_p['porosity'], df_p['depth_m'], 'o-', color='#3498db',
                              linewidth=2, markersize=4, label=f'Факт (n={len(df_p)})', alpha=0.9, zorder=3)[0]
                legend_items.append(lf); all_vals.extend(df_p['porosity'].values)

        if ml is not None and 'depth_m' in ml.columns and 'porosity_pred' in ml.columns:
            ml_s = ml.sort_values('depth_m')
            lp = ax2.plot(ml_s['porosity_pred'], ml_s['depth_m'], '--', color='#e74c3c',
                          linewidth=2.5, label='ML-прогноз φ', alpha=0.9, zorder=2)[0]
            legend_items.append(lp); all_vals.extend(ml_s['porosity_pred'].values)

        if not legend_items:
            ax2.text(0.5,0.5,'Загрузите архив для ML-прогноза', transform=ax2.transAxes,
                     ha='center', va='center', color='#9aa0b8', fontsize=12)
        ax2.invert_yaxis()
        ax2.set_xlabel('Пористость φ', color='#b0b8d0', fontsize=13)
        ax2.set_ylabel('Глубина, м', color='#b0b8d0', fontsize=13)
        ax2.set_title('Факт vs ML-прогноз', color='#ffffff', fontsize=14)
        ax2.tick_params(colors='#9aa0b8', labelsize=11)
        ax2.grid(True, color='#2e3550', linestyle='--', alpha=0.5)
        if all_vals:
            mn, mx = min(all_vals), max(all_vals)
            pad = max((mx - mn) * 0.3, abs(mx) * 0.15, abs(mn) * 0.15, 0.02)
            ax2.set_xlim(max(0, mn - pad), min(1, mx + pad))
        if legend_items:
            leg = ax2.legend(handles=legend_items, fontsize=11, facecolor='#2e3550', 
                           edgecolor='#3e4560', labelcolor='#d0d4e8')
            if leg: self.canvas_single_ml.register_legend(leg)
        self.canvas_single_ml.fig.tight_layout(); self.canvas_single_ml.draw()
        self.lbl_single_loading.setText('')

    def on_ml_single_done(self, result_df, info):
        self.df_ml_result = result_df; self.update_single_view()
        if self.df_single is not None and 'porosity' in self.df_single.columns and 'porosity_pred' in result_df.columns:
            df_s = self.df_single[self.df_single['porosity'].notna() & self.df_single['depth_m'].notna()].sort_values('depth_m')
            df_r = result_df.sort_values('depth_m')
            if len(df_s) > 1 and len(df_r) > 1:
                merged = pd.merge_asof(df_s[['depth_m','porosity']], df_r[['depth_m','porosity_pred']],
                                       on='depth_m', direction='nearest', tolerance=50)
                valid = merged.dropna()
                if len(valid) > 1:
                    r2  = r2_score(valid['porosity'], valid['porosity_pred'])
                    mae = mean_absolute_error(valid['porosity'], valid['porosity_pred'])
                    self.single_metrics['r2'].setText(f'{r2:.3f}')
                    color = '#27ae60' if r2 > 0.6 else '#e67e22' if r2 > 0.3 else '#e74c3c'
                    verdict = '✅ Хорошее согласие' if r2 > 0.6 else '⚠️ Умеренное' if r2 > 0.3 else '❌ Слабое'
                    geo = geo_interpretation('porosity', info.get('mean_por',0), info.get('std_por',0))
                    self.lbl_single_status.setText(
                        f'{verdict}  |  R²={r2:.3f}  MAE={mae:.4f}\n'
                        f'Ближайший сосед: {info["nearest_distance"]:.0f} м  Соседей в 1 км: {info["n_neighbors"]}\n'
                        f'🪨 {geo}')
                    self.lbl_single_status.setStyleSheet(f'color:{color}; font-size:12px; font-weight:bold;')
                    return
        geo = geo_interpretation('porosity', info.get('mean_por',0), info.get('std_por',0))
        self.lbl_single_status.setText(
            f'ML-прогноз готов  |  Ср. φ = {info["mean_por"]:.4f} ± {info["std_por"]:.4f}\n'
            f'Ближайший сосед: {info["nearest_distance"]:.0f} м  Соседей в 1 км: {info["n_neighbors"]}\n'
            f'🪨 {geo}')
        self.lbl_single_status.setStyleSheet('color:#3498db; font-size:12px;')

    # ══ ВК 6: ВОССТАНОВЛЕНИЕ (φ + k + ρ) ═════════════════════════════════
    def _build_tab_restore(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(16,16,16,16); v.setSpacing(8)
        lbl = QLabel('Восстановление свойств из окружения'); lbl.setObjectName('header'); v.addWidget(lbl)
        note = QLabel('Random Forest прогнозирует пористость φ, проницаемость k и плотность ρ по данным соседних скважин')
        note.setStyleSheet('color:#9aa0b8; font-size:11px;'); note.setWordWrap(True); v.addWidget(note)
        top = QHBoxLayout()
        g1 = QGroupBox('Скважина из реестра'); fl1 = QFormLayout(g1)
        self.combo_restore = QComboBox(); self.combo_restore.setMinimumWidth(280); fl1.addRow('Скважина:', self.combo_restore)
        btn1 = QPushButton('🔄  Прогноз по скважине'); btn1.clicked.connect(self.restore_by_well); fl1.addRow(btn1)
        top.addWidget(g1)
        g2 = QGroupBox('Произвольные координаты (или выберите на карте ПКМ)'); fl2 = QFormLayout(g2)
        self.spin_x = QDoubleSpinBox(); self.spin_x.setRange(-1e7,1e7); self.spin_x.setDecimals(1); self.spin_x.setSingleStep(100)
        self.spin_y = QDoubleSpinBox(); self.spin_y.setRange(-1e7,1e7); self.spin_y.setDecimals(1); self.spin_y.setSingleStep(100)
        fl2.addRow('X, м:', self.spin_x); fl2.addRow('Y, м:', self.spin_y)
        btn2 = QPushButton('📍  Прогноз в точке'); btn2.clicked.connect(self.restore_by_coords); fl2.addRow(btn2)
        top.addWidget(g2); v.addLayout(top)
        self.lbl_restore_loading = QLabel('')
        self.lbl_restore_loading.setStyleSheet('color:#e67e22; font-size:13px; font-weight:bold;'); v.addWidget(self.lbl_restore_loading)
        self.txt_restore_result = QTextEdit(); self.txt_restore_result.setReadOnly(True)
        self.txt_restore_result.setMaximumHeight(180); v.addWidget(self.txt_restore_result)
        self.canvas_restore = InteractiveCanvas(figsize=(14,6)); v.addWidget(self.canvas_restore, stretch=1)
        return w

    def restore_by_well(self):
        if self.df_all is None: QMessageBox.warning(self,'Нет данных','Сначала загрузите архив'); return
        text = self.combo_restore.currentText()
        if not text or text.startswith('(') or not text.strip(): return
        well_id = text.split('  ')[0].strip()
        if well_id not in self.df_all['well_id'].values:
            QMessageBox.warning(self,'Ошибка',f'Скважина "{well_id}" не найдена'); return
        self.lbl_restore_loading.setText('⏳ ML обучается (φ + k + ρ)...')
        self.txt_restore_result.clear(); self._run_ml(well_id=well_id)

    def restore_by_coords(self):
        if self.df_all is None: QMessageBox.warning(self,'Нет данных','Сначала загрузите архив'); return
        self.lbl_restore_loading.setText('⏳ ML обучается (φ + k + ρ)...')
        self.txt_restore_result.clear(); self._run_ml(x=self.spin_x.value(), y=self.spin_y.value())

    def _run_ml(self, well_id=None, x=None, y=None, props=None):
        if self.df_all is None: return
        if props is None:
            props = [p for p in ['porosity','permeability_mD','density_gcc'] if p in self.df_all.columns]
        if not props: self.lbl_restore_loading.setText('❌ Нет нужных столбцов в данных'); return
        n_train = self.df_all[props[0]].notna().sum()
        if n_train < 10: self.lbl_restore_loading.setText('❌ Нужно ≥10 образцов'); return
        self.progress.setVisible(True); self.progress.setValue(0)
        self.ml_worker = MLWorker(self.df_all, well_id=well_id, x=x, y=y, target_props=props)
        if well_id: self.ml_worker.finished.connect(self.on_ml_restore_well_done)
        else: self.ml_worker.finished.connect(self.on_ml_restore_coords_done)
        self.ml_worker.error.connect(self.on_ml_error)
        self.ml_worker.progress.connect(self.progress.setValue)
        self.ml_worker.start()

    def on_ml_restore_well_done(self, result_df, info):
        self.progress.setVisible(False)
        self.lbl_restore_loading.setText('✅ Прогноз φ + k + ρ готов')
        self.lbl_restore_loading.setStyleSheet('color:#27ae60; font-size:13px; font-weight:bold;')
        self._show_restore_result(self.combo_restore.currentText().split('  ')[0].strip(), result_df, info, 'well')

    def on_ml_restore_coords_done(self, result_df, info):
        self.progress.setVisible(False)
        self.lbl_restore_loading.setText('✅ Прогноз φ + k + ρ готов')
        self.lbl_restore_loading.setStyleSheet('color:#27ae60; font-size:13px; font-weight:bold;')
        self._show_restore_result(f"X={info['target_x']:.0f} Y={info['target_y']:.0f}", result_df, info, 'coords')

        # ══ ВК 6: ВОССТАНОВЛЕНИЕ — метод _show_restore_result ═══════════════
    def _show_restore_result(self, label, result_df, info, mode):
        summary = info.get('summary', {})
        first = list(summary.values())[0] if summary else {}
        n = first.get('n_neighbors', info.get('n_neighbors',0))
        nd = first.get('nearest_distance', info.get('nearest_distance',9999))
        if n >= 3 and nd < 2000: rel, rc = '✅ Надёжный прогноз', '#27ae60'
        elif n >= 1 and nd < 5000: rel, rc = '⚠️ Приблизительный прогноз', '#e67e22'
        else: rel, rc = '❌ Ненадёжный — нет близких скважин', '#e74c3c'

        prop_names = {'porosity':'Пористость φ','permeability_mD':'Проницаемость k','density_gcc':'Плотность ρ'}
        units = {'porosity':'','permeability_mD':' мД','density_gcc':' г/см³'}
        html = f'<b>Цель:</b> {label}<br>'
        for prop, s in summary.items():
            u = units.get(prop,'')
            html += (f'<b>{prop_names.get(prop,prop)}:</b> {s["mean"]:.4f} ± {s["std"]:.4f}{u} '
                     f'[{s["lower"]:.4f} — {s["upper"]:.4f}]<br>')
            g = geo_interpretation(prop, s['mean'], s['std'])
            if g: html += f'<span style="color:#9aa0b8;font-size:11px;">🪨 {g}</span><br>'
        html += (f'<b>Ближайшая скважина:</b> {nd:.0f} м | <b>Соседей в 1 км:</b> {n}<br>'
                 f'<b style="color:{rc}">{rel}</b>')
        self.txt_restore_result.setHtml(html)

        # УВЕЛИЧИЛ шрифт текста результатов
        self.txt_restore_result.setStyleSheet('background:#232940; color:#d0d4e8; border:1px solid #2e3550; '
                                              'border-radius:6px; font-size:14px; padding:8px;')

        # ── Графики ──
        self.canvas_restore.fig.clear()
        self.canvas_restore.fig.patch.set_facecolor('#1a1f2e')
        prop_cols = [p for p in ['porosity','permeability_mD','density_gcc'] if f'{p}_pred' in result_df.columns]
        n_plots = len(prop_cols) + 1
        axes = self.canvas_restore.fig.subplots(1, n_plots)
        if n_plots == 1: axes = [axes]

        pinfo = {
            'porosity':         {'label':'Пористость φ',    'color':'#e74c3c'},
            'permeability_mD':  {'label':'Проницаемость, мД','color':'#9b59b6'},
            'density_gcc':      {'label':'Плотность, г/см³', 'color':'#2ecc71'},
        }

        for i, prop in enumerate(prop_cols):
            ax = axes[i]; ax.set_facecolor('#232940')
            pi = pinfo.get(prop, {'label':prop,'color':'#e74c3c'})
            rs = result_df.sort_values('depth_m')
            items, vals = [], []
            pc, lc, uc = f'{prop}_pred', f'{prop}_lower', f'{prop}_upper'

            # УБРАЛ fill_between — только линия прогноза
            lp = ax.plot(rs[pc], rs['depth_m'], '-', color=pi['color'],
                         linewidth=2.5, label='ML-прогноз', zorder=2)[0]
            items.append(lp); vals.extend(rs[pc].values)

            if mode == 'well':
                wid = self.combo_restore.currentText().split('  ')[0].strip()
                dfw = self.df_all[self.df_all['well_id'] == wid]
                if prop in dfw.columns:
                    dff = dfw[dfw[prop].notna() & dfw['depth_m'].notna()].sort_values('depth_m')
                    if len(dff) > 0:
                        lf = ax.plot(dff[prop], dff['depth_m'], 'o-', color='#3498db',
                                     linewidth=1.5, markersize=3, label='Факт', alpha=0.8, zorder=3)[0]
                        items.append(lf); vals.extend(dff[prop].values)

            ax.invert_yaxis()
            # УВЕЛИЧИЛ шрифты
            ax.set_xlabel(pi['label'], color='#b0b8d0', fontsize=13)
            if i == 0: ax.set_ylabel('Глубина, м', color='#b0b8d0', fontsize=13)
            ax.set_title(pi['label'], color='#ffffff', fontsize=14)
            ax.tick_params(colors='#9aa0b8', labelsize=11)
            ax.grid(True, color='#2e3550', linestyle='--', alpha=0.5)
            # ИСПРАВИЛ масштабирование — больше отступов
            if vals:
                vmn, vmx = min(vals), max(vals)
                pad = max((vmx - vmn) * 0.3, abs(vmx) * 0.15, abs(vmn) * 0.15, 0.05)
                ax.set_xlim(vmn - pad, vmx + pad)
            if items:
                leg_r = ax.legend(handles=items, fontsize=10, facecolor='#2e3550', 
                                edgecolor='#3e4560', labelcolor='#d0d4e8')
                if leg_r: self.canvas_restore.register_legend(leg_r)

        # Карта
        ax_m = axes[-1]; ax_m.set_facecolor('#232940')
        ws = self.df_all.groupby('well_id').agg(
            por=('porosity','median'), n=('depth_m','count'), x=('x_m','first'), y=('y_m','first')
        ).reset_index(); ws = ws[ws['x'].notna() & ws['y'].notna()]
        if len(ws) > 0:
            vp = ws[ws['por'].notna()]
            vmn = vp['por'].min() if len(vp) else 0; vmx = vp['por'].max() if len(vp) else 1
            if vmn == vmx: vmn -= 0.001; vmx += 0.001
            norm = Normalize(vmin=vmn, vmax=vmx); cmap = plt.get_cmap('plasma')
            sizes = np.clip(np.sqrt(ws['n'].values)*12, 20, 300)
            for i, r in ws.iterrows():
                cv = r['por'] if pd.notna(r['por']) else vmn
                ax_m.scatter(r['x'], r['y'], s=sizes[i], c=[cmap(norm(cv))],
                             edgecolors='#aaaaaa', linewidths=0.8, zorder=3, alpha=0.8)
                ax_m.text(r['x'], r['y'],
                          re.sub(r'(?i)^well[_\-\s]*','',str(r['well_id'])).strip() or str(r['well_id']),
                          ha='center', va='bottom', fontsize=6.5, color='#c0c8e0', zorder=4)
            sm = ScalarMappable(cmap='plasma', norm=norm); sm.set_array([])
            cb = self.canvas_restore.fig.colorbar(sm, ax=ax_m, fraction=0.04, pad=0.02)
            cb.set_label('Медианная φ', color='#c0c8e0', fontsize=9)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='#c0c8e0')
        tx, ty = info['target_x'], info['target_y']
        ax_m.scatter(tx, ty, s=450, c='#e74c3c', marker='*', edgecolors='#ffffff', linewidths=2, zorder=10)
        por_s = summary.get('porosity',{}); perm_s = summary.get('permeability_mD',{}); dens_s = summary.get('density_gcc',{})
        ann = f"φ={por_s.get('mean',0):.3f}±{por_s.get('std',0):.3f}"
        if perm_s: ann += f"\nk={perm_s.get('mean',0):.1f} мД"
        if dens_s: ann += f"\nρ={dens_s.get('mean',0):.3f} г/см³"
        # УВЕЛИЧИЛ шрифт аннотации на карте
        ax_m.annotate(ann, (tx,ty), textcoords='offset points', xytext=(12,12), fontsize=12, color='#fff',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#c0392b', alpha=0.85, edgecolor='none'), zorder=11)
        ax_m.set_title('Карта месторождения', color='#ffffff', fontsize=11)
        ax_m.set_xlabel('X, м', color='#b0b8d0'); ax_m.set_ylabel('Y, м', color='#b0b8d0')
        ax_m.tick_params(colors='#9aa0b8'); ax_m.grid(True, color='#2e3550', linestyle='--', alpha=0.4)
        ax_m.set_aspect('equal', adjustable='datalim')
        self.canvas_restore.fig.suptitle(f'Прогноз: {label}', color='#ffffff', fontsize=13, y=0.98)
        self.canvas_restore.fig.tight_layout(rect=[0, 0, 1, 0.94])
        self.canvas_restore.draw()

    def on_ml_error(self, msg):
        self.progress.setVisible(False)
        short = msg.split('\n')[0][:120]
        self.lbl_restore_loading.setText(f'❌ {short}')
        self.lbl_restore_loading.setStyleSheet('color:#e74c3c; font-size:12px;')
        self.lbl_single_loading.setText(f'❌ {short}')
        self.lbl_single_loading.setStyleSheet('color:#e74c3c; font-size:12px;')

    # ── Загрузка ─────────────────────────────────────────────────────────
    def load_archive(self):
        msg = QMessageBox(self); msg.setWindowTitle('Тип загрузки'); msg.setText('Что загружаем?')
        btn_folder = msg.addButton('📁  Папку', QMessageBox.ButtonRole.AcceptRole)
        btn_zip    = msg.addButton('🗜  ZIP/RAR архив', QMessageBox.ButtonRole.AcceptRole)
        msg.addButton('Отмена', QMessageBox.ButtonRole.RejectRole); msg.exec()
        if msg.clickedButton() == btn_folder:
            path = QFileDialog.getExistingDirectory(self, 'Выберите папку со скважинами')
            if path: self._load_from_folder(path)
        elif msg.clickedButton() == btn_zip:
            path, _ = QFileDialog.getOpenFileName(self,'Выберите архив','','Архивы (*.zip *.rar);;Все файлы (*)')
            if path:
                if path.lower().endswith('.rar'): self._load_from_rar(path)
                else: self._load_from_zip(path)

    def _extract_registry(self, folder, csv_files):
        keywords = ['registry','реестр','well_registry','coords','coordinates','survey','wells']
        for fname in csv_files:
            if any(kw in os.path.basename(fname).lower() for kw in keywords):
                try:
                    df_reg = read_csv_auto(fname); df_reg = normalize_columns(df_reg); df_reg = ensure_numeric(df_reg)
                    if 'well_id' in df_reg.columns:
                        self.df_registry = df_reg; self.well_coords = {}
                        for _, row in df_reg.iterrows():
                            wid = str(row['well_id']); x = row.get('x_m', np.nan); y = row.get('y_m', np.nan)
                            if pd.notna(x) and pd.notna(y): self.well_coords[wid] = (float(x), float(y))
                        return fname
                except Exception as e: print(f'Реестр {fname}: {e}')
        return None

    def _read_well_file(self, path, well_name=None):
        df = read_csv_auto(path); df = normalize_columns(df); df = ensure_numeric(df)
        if 'well_id' not in df.columns:
            df['well_id'] = well_name or os.path.splitext(os.path.basename(path))[0]
        return df

    def _load_from_folder(self, folder):
        self.lbl_status.setText('⏳ Читаю папку...'); QApplication.processEvents()
        all_paths = []
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith('.csv'): all_paths.append(os.path.join(root, f))
        all_paths = sorted(all_paths)
        if not all_paths: QMessageBox.warning(self,'Нет файлов','В папке нет CSV-файлов'); return
        reg = self._extract_registry(folder, all_paths)
        if reg: all_paths = [p for p in all_paths if p != reg]
        all_dfs, errors = [], []
        for i, p in enumerate(all_paths):
            try: all_dfs.append(self._read_well_file(p))
            except Exception as e: errors.append(f'{os.path.basename(p)}: {e}')
            if (i+1) % 5 == 0: self.lbl_status.setText(f'⏳ {i+1}/{len(all_paths)}...'); QApplication.processEvents()
        self._finish_load(all_dfs, errors)

    def _load_from_zip(self, zip_path):
        self.lbl_status.setText('⏳ Читаю ZIP...'); QApplication.processEvents()
        all_dfs, errors = [], []
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                csv_names = sorted([n for n in zf.namelist() if n.lower().endswith('.csv')])
                if not csv_names: QMessageBox.warning(self,'Пусто','В архиве нет CSV-файлов'); return
                with tempfile.TemporaryDirectory() as tmpdir:
                    zf.extractall(tmpdir)
                    full_paths = [os.path.join(tmpdir, n) for n in csv_names]
                    reg = self._extract_registry(tmpdir, full_paths)
                    full_paths = [p for p in full_paths if p != reg]
                    for i, p in enumerate(full_paths):
                        try: all_dfs.append(self._read_well_file(p))
                        except Exception as e: errors.append(f'{os.path.basename(p)}: {e}')
                        if (i+1) % 5 == 0: self.lbl_status.setText(f'⏳ {i+1}/{len(full_paths)}...'); QApplication.processEvents()
        except Exception as e: QMessageBox.critical(self,'Ошибка',f'Не удалось открыть ZIP: {e}'); return
        self._finish_load(all_dfs, errors)

    def _load_from_rar(self, rar_path):
        self.lbl_status.setText('⏳ Читаю RAR...'); QApplication.processEvents()
        result = try_read_rar(rar_path)
        if result is None: QMessageBox.critical(self,'Ошибка','Для RAR установите: pip install rarfile\nТакже нужен unrar или 7zip'); return
        if not result: QMessageBox.warning(self,'Пусто','В RAR нет CSV-файлов'); return
        self._finish_load(result, [])

    def _finish_load(self, all_dfs, errors):
        if errors and len(errors) <= 10:
            QMessageBox.warning(self,'Предупреждения','Не прочитаны:\n'+'\n'.join(errors[:10]))
        if not all_dfs:
            self.lbl_status.setText('❌ Ничего не загружено')
            self.lbl_status.setStyleSheet('color:#e74c3c; font-size:12px;'); return
        self.df_all = pd.concat(all_dfs, ignore_index=True)
        if self.well_coords:
            for wid, (x, y) in self.well_coords.items():
                mask = self.df_all['well_id'].astype(str) == str(wid)
                if mask.any() and ('x_m' not in self.df_all.columns or self.df_all.loc[mask,'x_m'].isna().all()):
                    self.df_all.loc[mask,'x_m'] = x; self.df_all.loc[mask,'y_m'] = y
        if 'x_m' not in self.df_all.columns or self.df_all['x_m'].isna().mean() > 0.9:
            wells = sorted(self.df_all['well_id'].unique(), key=natural_sort_key)
            nn = len(wells); angles = np.linspace(0, 2*np.pi, nn, endpoint=False)
            radii = 2000 + np.random.default_rng(42).uniform(-500, 500, nn)
            coords = {w: (r*np.cos(a), r*np.sin(a)) for w, a, r in zip(wells, angles, radii)}
            self.df_all['x_m'] = self.df_all['well_id'].map(lambda w: coords.get(w,(0,0))[0])
            self.df_all['y_m'] = self.df_all['well_id'].map(lambda w: coords.get(w,(0,0))[1])
        nw = self.df_all['well_id'].nunique(); nr = len(self.df_all)
        self.lbl_status.setText(f'✅ {nw} скважин · {nr} строк')
        self.lbl_status.setStyleSheet('color:#27ae60; font-size:12px; font-weight:bold;')
        self.btn_well.setEnabled(True); self.btn_export.setEnabled(True)
        self._refresh_combos(); self.update_summary(); self.update_maps(); self.update_variogram()

    def _refresh_combos(self):
        if self.df_all is None: return
        all_wells = sorted(self.df_all['well_id'].unique(), key=natural_sort_key)
        por_counts = (self.df_all[self.df_all['porosity'].notna()].groupby('well_id').size()
                      if 'porosity' in self.df_all.columns else pd.Series(dtype=int))
        items = [f'{w}  (φ: {por_counts.get(w,0)}/{(self.df_all["well_id"]==w).sum()})' for w in all_wells]
        self.combo_w1.blockSignals(True); self.combo_w2.blockSignals(True)
        self.combo_w1.clear(); self.combo_w1.addItems(items)
        self.combo_w2.clear(); self.combo_w2.addItems(items)
        if len(items) >= 2: self.combo_w2.setCurrentIndex(1)
        self.combo_w1.blockSignals(False); self.combo_w2.blockSignals(False)
        self.combo_restore.clear()
        if items: self.combo_restore.addItems(items)
        else: self.combo_restore.addItem('(нет данных)'); self.combo_restore.setEnabled(False)

    def load_single_well(self):
        path, _ = QFileDialog.getOpenFileName(self,'CSV скважины','','CSV (*.csv)')
        if not path: return
        try:
            self.df_single = self._read_well_file(path)
            self.df_ml_result = None
            well_name = str(self.df_single['well_id'].iloc[0])
            n = len(self.df_single)
            n_por = self.df_single['porosity'].notna().sum() if 'porosity' in self.df_single.columns else 0
            self.lbl_single_status.setText(f'Загружена: {well_name}  ·  {n} строк  ·  {n_por} с φ')
            self.lbl_single_status.setStyleSheet('color:#3498db; font-size:12px;')
            self.lbl_single_loading.setText('')
            self.tabs.setCurrentIndex(4); self.update_single_view()
            if self.df_all is not None:
                self.lbl_single_loading.setText('⏳ ML-прогноз...')
                matched = next((str(aw) for aw in self.df_all['well_id'].unique()
                                if str(aw).strip().lower() == well_name.strip().lower()), None)
                if matched:
                    self.ml_worker = MLWorker(self.df_all, well_id=matched, target_props=['porosity'])
                else:
                    x = self.df_single['x_m'].dropna().iloc[0] if 'x_m' in self.df_single.columns and self.df_single['x_m'].notna().any() else None
                    y = self.df_single['y_m'].dropna().iloc[0] if 'y_m' in self.df_single.columns and self.df_single['y_m'].notna().any() else None
                    if x is not None and y is not None:
                        self.ml_worker = MLWorker(self.df_all, x=float(x), y=float(y), target_props=['porosity'])
                    else:
                        self.lbl_single_loading.setText('ℹ️ Скважина не найдена в архиве (нет координат)'); return
                self.ml_worker.finished.connect(self.on_ml_single_done)
                self.ml_worker.error.connect(self.on_ml_error)
                self.ml_worker.start()
        except Exception as e:
            QMessageBox.critical(self,'Ошибка',str(e))

    def export_report(self):
        if self.df_all is None: return
        path, _ = QFileDialog.getSaveFileName(self,'Экспорт','field_report.xlsx','Excel (*.xlsx)')
        if not path: return
        try:
            with pd.ExcelWriter(path, engine='openpyxl') as wr:
                ws = well_level_stats(self.df_all)
                ws.sort_values('well_id', key=lambda x: x.apply(natural_sort_key)).to_excel(wr, sheet_name='Реестр', index=False)
                self.df_all.to_excel(wr, sheet_name='Все_данные', index=False)
                if self.df_single is not None: self.df_single.to_excel(wr, sheet_name='Загруженная_скважина', index=False)
                if self.df_ml_result is not None: self.df_ml_result.to_excel(wr, sheet_name='ML_прогноз', index=False)
            self.lbl_status.setText('✅ Отчёт сохранён')
            self.lbl_status.setStyleSheet('color:#27ae60; font-size:12px; font-weight:bold;')
        except Exception as e: QMessageBox.critical(self,'Ошибка экспорта',str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())