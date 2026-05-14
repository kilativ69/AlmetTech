# -*- coding: utf-8 -*-
"""
ИИ-детектив для нефтяных данных · AWW 2026
Запуск: python app_desktop.py
pip install PyQt6 matplotlib pandas numpy scipy
"""

import sys
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
from scipy.stats import mannwhitneyu

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QFileDialog, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QSizePolicy,
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QScrollArea,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

# ── Стили ──────────────────────────────────────────────────────────────────
STYLE = """
QMainWindow, QWidget {
    background-color: #1a1f2e;
    color: #d0d4e8;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QTabWidget::pane {
    border: 1px solid #2e3550;
    background: #1a1f2e;
}
QTabBar::tab {
    background: #232940;
    color: #9aa0b8;
    padding: 10px 22px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 13px;
}
QTabBar::tab:selected {
    background: #2e75b6;
    color: #ffffff;
    font-weight: bold;
}
QPushButton {
    background: #2e75b6;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 9px 22px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton:hover { background: #3a8fd1; }
QPushButton:pressed { background: #235f96; }
QPushButton#btn_save { background: #27ae60; }
QPushButton#btn_save:hover { background: #2ecc71; }
QPushButton#btn_expand_tbl {
    background: #3d2e75;
    color: #c0c8ff;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: bold;
    border-radius: 5px;
}
QPushButton#btn_expand_tbl:hover { background: #4e3d96; }
QPushButton#btn_secondary {
    background: #2e3550;
    color: #c0c8e0;
    padding: 7px 16px;
    font-size: 12px;
    font-weight: normal;
}
QTableWidget {
    background: #232940;
    color: #d0d4e8;
    gridline-color: #2e3550;
    border: none;
    font-size: 12px;
}
QHeaderView::section {
    background: #2e3550;
    color: #c0c8e0;
    padding: 6px;
    border: none;
    font-weight: bold;
}
QLabel#metric_val {
    font-size: 28px;
    font-weight: bold;
    color: #4d9de0;
}
QLabel#metric_lbl {
    font-size: 11px;
    color: #9aa0b8;
}
QFrame#metric_card {
    background: #232940;
    border-radius: 10px;
    padding: 10px;
}
QLabel#header {
    font-size: 15px;
    font-weight: bold;
    color: #ffffff;
    padding: 4px 0px;
}
QGroupBox {
    color: #c0c8e0;
    border: 1px solid #2e3550;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 8px;
    font-size: 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    color: #a0c0e0;
}
QDoubleSpinBox {
    background: #2e3550;
    color: #d0d4e8;
    border: 1px solid #3e4560;
    border-radius: 4px;
    padding: 3px 6px;
}
QCheckBox { color: #c0c8e0; font-size: 12px; }
QCheckBox::indicator { width: 14px; height: 14px; }
QScrollArea { border: none; background: transparent; }
"""

MPLSTYLE = {
    'figure.facecolor': '#1a1f2e',
    'axes.facecolor':   '#232940',
    'axes.edgecolor':   '#3e4560',
    'axes.labelcolor':  '#b0b8d0',
    'xtick.color':      '#9aa0b8',
    'ytick.color':      '#9aa0b8',
    'text.color':       '#d0d4e8',
    'grid.color':       '#2e3550',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
}
for k, v in MPLSTYLE.items():
    try: plt.rcParams[k] = v
    except: pass


# ── Холст matplotlib ────────────────────────────────────────────────────────
class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(8, 4)):
        self.fig = Figure(figsize=figsize, facecolor='#1a1f2e')
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


# ── Интерактивная легенда для профилей ─────────────────────────────────────
class ProfileCanvas(MplCanvas):
    """
    Canvas с интерактивной легендой: клик по элементу легенды
    скрывает / показывает соответствующую линию.
    """
    def __init__(self, figsize=(12, 7)):
        super().__init__(figsize)
        self._line_map = {}   # artist легенды → линия графика
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)

    def set_line_map(self, line_map: dict):
        """line_map: {legend_artist: line_artist}"""
        self._line_map = line_map

    def _on_pick(self, event):
        leg_artist = event.artist
        if leg_artist not in self._line_map:
            return
        line = self._line_map[leg_artist]
        visible = not line.get_visible()
        line.set_visible(visible)
        leg_artist.set_alpha(1.0 if visible else 0.3)
        self.fig.canvas.draw_idle()


# ── Inline expand panel (вместо отдельных окон) ─────────────────────────────
class ExpandPanel(QWidget):
    """
    Панель полного экрана внутри приложения.
    Показывается поверх вкладок через QStackedWidget в MainWindow.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 8, 12, 12)
        self._layout.setSpacing(8)

        hdr = QHBoxLayout()
        self._title_lbl = QLabel('')
        self._title_lbl.setObjectName('header')
        self._btn_back = QPushButton('← Назад')
        self._btn_back.setObjectName('btn_secondary')
        hdr.addWidget(self._title_lbl)
        hdr.addStretch()
        hdr.addWidget(self._btn_back)
        self._layout.addLayout(hdr)

        self._content_widget = None

    def _clear_content(self):
        if self._content_widget is not None:
            self._layout.removeWidget(self._content_widget)
            self._content_widget.setParent(None)
            self._content_widget = None

    def show_canvas(self, title: str, canvas: 'MplCanvas'):
        self._clear_content()
        self._title_lbl.setText(title)
        self._content_widget = canvas
        self._layout.addWidget(canvas)

    def show_table(self, title: str, source_table: 'QTableWidget'):
        self._clear_content()
        self._title_lbl.setText(title)
        tbl = QTableWidget()
        tbl.setAlternatingRowColors(True)
        tbl.setStyleSheet('alternate-background-color: #1e2438; font-size:12px;')
        tbl.setRowCount(source_table.rowCount())
        tbl.setColumnCount(source_table.columnCount())
        headers = [source_table.horizontalHeaderItem(c).text()
                   if source_table.horizontalHeaderItem(c) else ''
                   for c in range(source_table.columnCount())]
        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for r in range(source_table.rowCount()):
            for c in range(source_table.columnCount()):
                src = source_table.item(r, c)
                if src:
                    item = QTableWidgetItem(src.text())
                    item.setForeground(src.foreground())
                    item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                    tbl.setItem(r, c, item)
        self._content_widget = tbl
        self._layout.addWidget(tbl)


# ── Дефолтные правила ───────────────────────────────────────────────────────
DEFAULT_RULES = {
    'porosity':        {'enabled': True, 'min': 0.0,   'max': 1.0,   'label': 'Пористость'},
    'permeability_mD': {'enabled': True, 'min': 0.0,                  'label': 'Проницаемость'},
    'density_gcc':     {'enabled': True, 'min': 2.0,   'max': 2.9,   'label': 'Плотность'},
    'sat_sum':         {'enabled': True,               'max': 1, 'label': 'Sw+So'},
    'duplicates':      {'enabled': True,                              'label': 'Дубликаты'},
    'column_shift':    {'enabled': True, 'min': 10.0,                 'label': 'Смещение столбца'},
}


def run_verification(df, rules=None):
    if rules is None:
        rules = DEFAULT_RULES
    flags = pd.DataFrame(index=df.index)

    r = rules.get('porosity', {})
    flags['ошибка_пористость'] = (
        ~df['porosity'].between(r['min'], r['max']) if r.get('enabled') else pd.Series(False, index=df.index)
    )

    r = rules.get('permeability_mD', {})
    flags['ошибка_проницаемость'] = (
        df['permeability_mD'] < r['min'] if r.get('enabled') else pd.Series(False, index=df.index)
    )

    r = rules.get('density_gcc', {})
    flags['ошибка_плотность'] = (
        ~df['density_gcc'].between(r['min'], r['max']) if r.get('enabled') else pd.Series(False, index=df.index)
    )

    r = rules.get('sat_sum', {})
    sat_sum = df['water_saturation'] + df['oil_saturation']
    flags['ошибка_насыщенность'] = (
        sat_sum > r['max'] if r.get('enabled') else pd.Series(False, index=df.index)
    )

    r = rules.get('duplicates', {})
    if r.get('enabled'):
        key_cols = ['well_id', 'depth_m', 'porosity', 'permeability_mD', 'density_gcc']
        dup_mask = df.duplicated(subset=key_cols, keep=False)
        dup_partner = {}
        for _, grp in df[dup_mask].groupby(key_cols):
            idxs = list(grp.index)
            for idx in idxs:
                partners = [str(x) for x in idxs if x != idx]
                dup_partner[idx] = ', '.join(partners)
        flags['дубль'] = dup_mask
        flags['дубль_строки'] = pd.Series(index=df.index, dtype=str).fillna('')
        for idx, val in dup_partner.items():
            flags.at[idx, 'дубль_строки'] = val
    else:
        flags['дубль'] = pd.Series(False, index=df.index)
        flags['дубль_строки'] = pd.Series('', index=df.index)

    r = rules.get('column_shift', {})
    flags['смещение_столбца'] = (
        df['density_gcc'] > r['min'] if r.get('enabled') else pd.Series(False, index=df.index)
    )

    err_cols = ['ошибка_пористость', 'ошибка_проницаемость', 'ошибка_плотность',
                'ошибка_насыщенность', 'дубль', 'смещение_столбца']
    flags['any_error'] = flags[err_cols].any(axis=1)
    return flags


# ── Главное окно ────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df    = None
        self.flags = None
        self.rules = {k: dict(v) for k, v in DEFAULT_RULES.items()}
        self.setWindowTitle('ИИ-детектив для нефтяных данных · AWW 2026')
        self.resize(1380, 880)
        self._build_ui()

    # ── Шапка ──────────────────────────────────────────────────────────────
    def _build_header(self):
        bar = QWidget()
        bar.setStyleSheet('background:#161b29; padding:6px;')
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 8, 16, 8)

        title = QLabel('🛢 ИИ-детектив для нефтяных данных')
        title.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        title.setStyleSheet('color:#ffffff;')

        self.lbl_file = QLabel('Файл не загружен')
        self.lbl_file.setStyleSheet('color:#9aa0b8; font-size:12px;')

        self.btn_load = QPushButton('📂  Загрузить CSV')
        self.btn_load.clicked.connect(self.load_file)

        self.btn_save = QPushButton('💾  Сохранить отчёт')
        self.btn_save.setObjectName('btn_save')
        self.btn_save.clicked.connect(self.save_report)
        self.btn_save.setEnabled(False)

        h.addWidget(title)
        h.addWidget(self.lbl_file)
        h.addStretch()
        h.addWidget(self.btn_load)
        h.addWidget(self.btn_save)
        return bar

    # ── Вкладка 1: Верификация ──────────────────────────────────────────────
    def _build_tab_verify(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        lbl = QLabel('Верификация данных')
        lbl.setObjectName('header')
        v.addWidget(lbl)

        self.metric_total  = self._metric_card('Всего записей', '—')
        self.metric_errors = self._metric_card('С ошибками', '—')
        self.metric_clean  = self._metric_card('Чистых', '—')
        self.metric_pct    = self._metric_card('% ошибок', '—')

        grid = QGridLayout()
        grid.addWidget(self.metric_total[0],  0, 0)
        grid.addWidget(self.metric_errors[0], 0, 1)
        grid.addWidget(self.metric_clean[0],  0, 2)
        grid.addWidget(self.metric_pct[0],    0, 3)
        v.addLayout(grid)

        # График ошибок + подсказка про двойной клик
        chart_header = QHBoxLayout()
        lbl_chart = QLabel('График ошибок по типам')
        lbl_chart.setObjectName('header')
        hint_chart = QLabel('двойной клик — на весь экран')
        hint_chart.setStyleSheet('color:#5a6080; font-size:11px; font-style:italic;')
        chart_header.addWidget(lbl_chart)
        chart_header.addStretch()
        chart_header.addWidget(hint_chart)
        v.addLayout(chart_header)

        self.canvas_errors = MplCanvas(figsize=(10, 3))
        self.canvas_errors.mouseDoubleClickEvent = lambda e: self._expand_chart()
        self.canvas_errors.setCursor(Qt.CursorShape.PointingHandCursor)
        v.addWidget(self.canvas_errors)

        # Заголовок таблицы + кнопка развернуть + подсказка двойной клик
        tbl_header = QHBoxLayout()
        lbl2 = QLabel('Аномальные записи')
        lbl2.setObjectName('header')
        tbl_header.addWidget(lbl2)
        tbl_header.addStretch()

        hint_tbl = QLabel('двойной клик — на весь экран')
        hint_tbl.setStyleSheet('color:#5a6080; font-size:11px; font-style:italic;')
        tbl_header.addWidget(hint_tbl)

        self.btn_expand_tbl = QPushButton('⛶  Развернуть таблицу')
        self.btn_expand_tbl.setObjectName('btn_expand_tbl')
        self.btn_expand_tbl.clicked.connect(self._open_expand_table)
        tbl_header.addWidget(self.btn_expand_tbl)
        v.addLayout(tbl_header)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet('alternate-background-color: #1e2438;')
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellDoubleClicked.connect(lambda r, c: self._open_expand_table())
        self.table.viewport().mouseDoubleClickEvent = lambda e: self._open_expand_table()
        self.table.setCursor(Qt.CursorShape.PointingHandCursor)
        v.addWidget(self.table, stretch=1)
        return w

    def _expand_chart(self):
        """Разворачивает график ошибок во весь экран внутри приложения."""
        if self.flags is None:
            return
        flags = self.flags
        err_cols = ['ошибка_пористость','ошибка_проницаемость','ошибка_плотность',
                    'ошибка_насыщенность','дубль','смещение_столбца']
        counts = flags[err_cols].sum().sort_values(ascending=False)

        def draw(fig):
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor('#232940')
            bar_colors = ['#e74c3c','#e67e22','#f1c40f','#3498db','#9b59b6','#1abc9c']
            bars = ax.bar(range(len(counts)), counts.values,
                          color=bar_colors[:len(counts)], edgecolor='none', width=0.6)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=15, ha='right', fontsize=13, color='#c0c8e0')
            ax.set_ylabel('Количество записей', color='#b0b8d0', fontsize=13)
            ax.set_title('Ошибки по типам', color='#ffffff', fontsize=15)
            ax.tick_params(colors='#9aa0b8')
            ax.grid(axis='y', color='#2e3550', linestyle='--', alpha=0.5)
            for bar, val in zip(bars, counts.values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                            str(val), ha='center', va='bottom', color='#ffffff', fontsize=13)
            # Увеличенные отступы чтобы подписи не обрезались
            fig.tight_layout(pad=2.5)

        self._open_expand_canvas('График ошибок по типам', draw)

    def _expand_profiles(self):
        """Разворачивает профили пористости во весь экран."""
        if self.df is None:
            return
        df = self.df
        filter_on = self.chk_filter_profiles.isChecked()
        por_min = self.rules.get('porosity', {}).get('min', 0.0)
        por_max = self.rules.get('porosity', {}).get('max', 1.0)

        def draw(fig):
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor('#232940')
            cmap_tab = plt.get_cmap('tab10')

            for i, well in enumerate(sorted(df['well_id'].unique())):
                grp = df[df['well_id'] == well].sort_values('depth_m')
                if filter_on:
                    grp = grp[grp['porosity'].between(por_min, por_max)]
                if grp.empty:
                    continue
                ax.plot(grp['porosity'], grp['depth_m'],
                        color=cmap_tab(i % 10), alpha=0.85, linewidth=2.0, label=well)

            ax.invert_yaxis()
            ax.set_xlabel('Пористость', color='#b0b8d0', fontsize=12)
            ax.set_ylabel('Глубина, м', color='#b0b8d0', fontsize=12)
            ax.tick_params(colors='#9aa0b8')
            suffix = ' (только физически допустимые)' if filter_on else ''
            ax.set_title(f'Профили пористости по глубине{suffix}', color='#ffffff', fontsize=14)
            ax.legend(fontsize=9, facecolor='#2e3550', edgecolor='#3e4560', labelcolor='#d0d4e8', ncol=2)
            ax.grid(True, color='#2e3550', linestyle='--', alpha=0.5)

            if not filter_on:
                ax.axvline(por_min, color="#563ce7", linestyle=':', alpha=0.7)
                ax.axvline(por_max, color="#3c6de7", linestyle=':', alpha=0.7)

            fig.tight_layout(pad=1.8, h_pad=2.0, w_pad=2.0)

    def _expand_map_subplot(self, event):
        if self.df is None:
            return
        axes = self.canvas_map.fig.get_axes()
        if not axes:
            return
        pos = event.position()
        x_fig = pos.x() / self.canvas_map.width()
        y_fig = 1.0 - pos.y() / self.canvas_map.height()
        clicked_ax_idx = 0
        for i, ax in enumerate(axes):
            bbox = ax.get_position()
            if bbox.contains(x_fig, y_fig):
                clicked_ax_idx = i
                break

        df, flags = self.df, self.flags
        well_stats = df.groupby('well_id').agg(
            por_med=('porosity', 'median'),
            n=('porosity', 'count')
        ).reset_index()
        well_stats['x'] = df.groupby('well_id')['x_m'].first().values
        well_stats['y'] = df.groupby('well_id')['y_m'].first().values
        df2 = df.copy()
        df2['_err'] = flags['any_error'].values
        err_by_well = df2.groupby('well_id')['_err'].sum().reset_index()
        well_stats = well_stats.merge(err_by_well, on='well_id')

        if clicked_ax_idx == 0:
            title = 'Карта скважин — пористость и аномалии'
            def draw(fig):
                fig.clear()
                ax = fig.add_subplot(111)
                ax.set_facecolor('#232940')
                por_norm = Normalize(vmin=well_stats['por_med'].min(), vmax=well_stats['por_med'].max())
                por_cmap = plt.get_cmap('winter')
                err_norm = Normalize(vmin=0, vmax=max(well_stats['_err'].max(), 1))
                err_cmap = plt.get_cmap('Reds')
                for _, r in well_stats.iterrows():
                    fc = por_cmap(por_norm(r['por_med']))
                    ec = err_cmap(err_norm(r['_err']))
                    ax.scatter(r['x'], r['y'], s=r['n'] * 22,
                               c=[fc], edgecolors=[ec], linewidths=4.5, zorder=3)
                    ax.annotate(r['well_id'], (r['x'], r['y']),
                                textcoords='offset points', xytext=(0, 14),
                                fontsize=10, color='#d0d4e8')
                sm_por = ScalarMappable(cmap='winter', norm=por_norm)
                sm_por.set_array([])
                cb1 = fig.colorbar(sm_por, ax=ax, fraction=0.035, pad=0.01)
                cb1.set_label('Медианная пористость', color='#c0c8e0', fontsize=11)
                plt.setp(cb1.ax.yaxis.get_ticklabels(), color='#c0c8e0')
                sm_err = ScalarMappable(cmap='Reds', norm=err_norm)
                sm_err.set_array([])
                cb2 = fig.colorbar(sm_err, ax=ax, fraction=0.035, pad=0.12, location='left')
                cb2.set_label('Кол-во аномалий', color='#c0c8e0', fontsize=11)
                plt.setp(cb2.ax.yaxis.get_ticklabels(), color='#c0c8e0')
                ax.set_title('Медианная пористость + аномалии', color='#ffffff', fontsize=14)
                ax.set_xlabel('X, м', color='#b0b8d0', fontsize=12)
                ax.set_ylabel('Y, м', color='#b0b8d0', fontsize=12)
                ax.tick_params(colors='#9aa0b8')
                ax.grid(True, color='#2e3550', linestyle='--', alpha=0.4)
                fig.tight_layout(pad=2.0)
        else:
            title = 'Карта скважин — геологические зоны'
            def draw(fig):
                fig.clear()
                ax2 = fig.add_subplot(111)
                ax2.set_facecolor('#232940')
                if 'геол_зона' in df.columns:
                    well_zone = df.groupby('well_id')['геол_зона'].first().reset_index()
                    ws2 = well_stats.merge(well_zone, on='well_id')
                    zones = sorted(ws2['геол_зона'].unique())
                    palette = ['#27ae60','#e67e22','#e74c3c','#3498db','#9b59b6']
                    zcolors = {z: palette[i % len(palette)] for i, z in enumerate(zones)}
                    for zone, grp in ws2.groupby('геол_зона'):
                        ax2.scatter(grp['x'], grp['y'], s=150,
                                    color=zcolors[zone], label=zone,
                                    edgecolors='#ffffff', linewidths=0.8, zorder=3)
                        for _, r in grp.iterrows():
                            ax2.annotate(r['well_id'], (r['x'], r['y']),
                                         textcoords='offset points', xytext=(6, 4),
                                         fontsize=10, color='#d0d4e8')
                    ax2.legend(fontsize=10, facecolor='#2e3550', edgecolor='#3e4560', labelcolor='#d0d4e8')
                ax2.set_title('Геологические зоны', color='#ffffff', fontsize=14)
                ax2.set_xlabel('X, м', color='#b0b8d0', fontsize=12)
                ax2.set_ylabel('Y, м', color='#b0b8d0', fontsize=12)
                ax2.tick_params(colors='#9aa0b8')
                ax2.grid(True, color='#2e3550', linestyle='--', alpha=0.4)
                fig.tight_layout(pad=2.0)

        self._open_expand_canvas(title, draw)

    def _expand_spatial_subplot(self, event):
        if self.df is None:
            return
        axes = self.canvas_spatial.fig.get_axes()
        if not axes:
            return
        pos = event.position()
        x_fig = pos.x() / self.canvas_spatial.width()
        y_fig = 1.0 - pos.y() / self.canvas_spatial.height()
        clicked_ax_idx = 0
        for i, ax in enumerate(axes):
            bbox = ax.get_position()
            if bbox.contains(x_fig, y_fig):
                clicked_ax_idx = i
                break

        df = self.df
        flags = self.flags
        hide = self.chk_hide_anomalies.isChecked()
        df_plot = df[~flags['any_error']].copy() if hide else df.copy()
        suffix = ' (без аномалий)' if hide else ''

        if clicked_ax_idx == 0:
            title = f'Пористость по лабораториям{suffix}'
            labs   = sorted(df_plot['lab_id'].unique())
            groups = [df_plot[df_plot['lab_id'] == l]['porosity'].dropna().values for l in labs]
            labs   = [l for l, g in zip(labs, groups) if len(g) > 0]
            groups = [g for g in groups if len(g) > 0]

            def draw(fig):
                fig.clear()
                ax = fig.add_subplot(111)
                ax.set_facecolor('#232940')
                if groups:
                    bp = ax.boxplot(groups, labels=labs, patch_artist=True,
                                    medianprops=dict(color='#f1c40f', linewidth=2),
                                    whiskerprops=dict(color='#9aa0b8'),
                                    capprops=dict(color='#9aa0b8'),
                                    flierprops=dict(markerfacecolor='#e74c3c', markersize=5))
                    box_colors = ['#3498db','#e67e22','#27ae60']
                    for patch, color in zip(bp['boxes'], box_colors):
                        patch.set_facecolor(color); patch.set_alpha(0.7)
                    if len(groups) >= 2:
                        stat_kw, p_kw = stats.kruskal(*groups)
                        pairwise = []
                        for i in range(len(labs)):
                            for j in range(i+1, len(labs)):
                                _, p_mw = mannwhitneyu(groups[i], groups[j])
                                mark = '⚠' if p_mw < 0.05 else '✓'
                                pairwise.append(f'{labs[i]} vs {labs[j]}: p={p_mw:.4f} {mark}')
                        ax.set_xlabel('\n'.join(pairwise), color='#9aa0b8', fontsize=11)
                ax.set_title(title, color='#ffffff', fontsize=14)
                ax.set_ylabel('Пористость', color='#b0b8d0', fontsize=12)
                ax.tick_params(colors='#9aa0b8')
                ax.grid(axis='y', color='#2e3550', linestyle='--', alpha=0.5)
                fig.tight_layout(pad=2.5)
        else:
            title = f'Медианная пористость по скважинам{suffix}'
            well_por = df_plot.groupby('well_id')['porosity'].median().sort_values()

            def draw(fig):
                fig.clear()
                ax2 = fig.add_subplot(111)
                ax2.set_facecolor('#232940')
                bar_colors = ['#e74c3c' if v > 0.3 else '#3498db' for v in well_por.values]
                ax2.barh(well_por.index, well_por.values, color=bar_colors, edgecolor='none')
                ax2.set_title(title, color='#ffffff', fontsize=14)
                ax2.set_xlabel('Медианная пористость', color='#b0b8d0', fontsize=12)
                ax2.tick_params(colors='#9aa0b8')
                ax2.grid(axis='x', color='#2e3550', linestyle='--', alpha=0.5)
                med = well_por.median()
                ax2.axvline(med, color='#f1c40f', linestyle='--', linewidth=1.5,
                            label=f'Медиана: {med:.3f}')
                ax2.legend(fontsize=10, facecolor='#2e3550', edgecolor='#3e4560', labelcolor='#d0d4e8')
                fig.tight_layout(pad=2.0)

        self._open_expand_canvas(title, draw)

    def _metric_card(self, label, value):
        card = QFrame()
        card.setObjectName('metric_card')
        vl = QVBoxLayout(card)
        vl.setContentsMargins(16, 12, 16, 12)
        lv = QLabel(value)
        lv.setObjectName('metric_val')
        ll = QLabel(label)
        ll.setObjectName('metric_lbl')
        vl.addWidget(lv)
        vl.addWidget(ll)
        return card, lv

    # ── Вкладка 2: Карта скважин ────────────────────────────────────────────
    def _build_tab_map(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(16, 16, 16, 16)

        hdr = QHBoxLayout()
        lbl = QLabel('Карта скважин')
        lbl.setObjectName('header')
        hint = QLabel('двойной клик — на весь экран')
        hint.setStyleSheet('color:#5a6080; font-size:11px; font-style:italic;')
        hdr.addWidget(lbl)
        hdr.addStretch()
        hdr.addWidget(hint)
        v.addLayout(hdr)

        note = QLabel('Размер шара = кол-во образцов · Заливка = медианная пористость · Каёмка = кол-во аномалий  |  ◀ шкала аномалий слева')
        note.setStyleSheet('color:#9aa0b8; font-size:11px;')
        v.addWidget(note)
        self.canvas_map = MplCanvas(figsize=(13, 7))
        self.canvas_map.mouseDoubleClickEvent = lambda e: self._expand_map_subplot(e)
        self.canvas_map.setCursor(Qt.CursorShape.PointingHandCursor)
        v.addWidget(self.canvas_map)
        return w

    # ── Вкладка 3: Профили глубины ──────────────────────────────────────────
    def _build_tab_profiles(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(8)

        lbl = QLabel('Профили пористости по глубине')
        lbl.setObjectName('header')
        v.addWidget(lbl)

        ctrl = QHBoxLayout()
        self.chk_filter_profiles = QCheckBox(
            'Скрыть точки вне физических границ пористости  →  увеличить масштаб'
        )
        self.chk_filter_profiles.setChecked(False)
        self.chk_filter_profiles.stateChanged.connect(self._on_profile_filter)
        ctrl.addWidget(self.chk_filter_profiles)
        ctrl.addStretch()

        hint_legend = QLabel('🖱 клик по легенде — скрыть/показать скважину')
        hint_legend.setStyleSheet('color:#4a8a6a; font-size:11px; font-style:italic;')
        ctrl.addWidget(hint_legend)

        hint_pr = QLabel('  |  двойной клик по графику — на весь экран')
        hint_pr.setStyleSheet('color:#5a6080; font-size:11px; font-style:italic;')
        ctrl.addWidget(hint_pr)
        v.addLayout(ctrl)

        # Используем ProfileCanvas с поддержкой интерактивной легенды
        self.canvas_profiles = ProfileCanvas(figsize=(12, 7))
        self.canvas_profiles.mouseDoubleClickEvent = lambda e: self._expand_profiles()
        self.canvas_profiles.setCursor(Qt.CursorShape.PointingHandCursor)
        v.addWidget(self.canvas_profiles)
        return w

    def _on_profile_filter(self):
        if self.df is not None:
            self.update_profiles()

    # ── Вкладка 4: Анализ лабораторий ──────────────────────────────────────
    def _build_tab_spatial(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(8)

        lbl = QLabel('Пространственный анализ и лаборатории')
        lbl.setObjectName('header')
        v.addWidget(lbl)

        ctrl = QHBoxLayout()
        self.chk_hide_anomalies = QCheckBox('Убрать аномальные данные с графиков  →  крупнее масштаб')
        self.chk_hide_anomalies.setChecked(False)
        self.chk_hide_anomalies.stateChanged.connect(self._on_spatial_filter)
        ctrl.addWidget(self.chk_hide_anomalies)
        ctrl.addStretch()
        hint_sp = QLabel('двойной клик — на весь экран')
        hint_sp.setStyleSheet('color:#5a6080; font-size:11px; font-style:italic;')
        ctrl.addWidget(hint_sp)
        v.addLayout(ctrl)

        self.lbl_kruskal = QLabel('Загрузите данные для анализа')
        self.lbl_kruskal.setStyleSheet('color:#9aa0b8; font-size:12px; padding:4px;')
        v.addWidget(self.lbl_kruskal)

        self.canvas_spatial = MplCanvas(figsize=(12, 6))
        self.canvas_spatial.mouseDoubleClickEvent = lambda e: self._expand_spatial_subplot(e)
        self.canvas_spatial.setCursor(Qt.CursorShape.PointingHandCursor)
        v.addWidget(self.canvas_spatial)
        return w

    def _on_spatial_filter(self):
        if self.df is not None:
            self.update_spatial()

    # ── Вкладка 5: Правила ─────────────────────────────────────────────────
    def _build_tab_rules(self):
        outer = QWidget()
        outer_v = QVBoxLayout(outer)
        outer_v.setContentsMargins(16, 16, 16, 16)
        outer_v.setSpacing(10)

        lbl = QLabel('Настройка правил верификации')
        lbl.setObjectName('header')
        outer_v.addWidget(lbl)

        hint = QLabel(
            'Измените пороговые значения и нажмите «Применить» — все вкладки пересчитаются.'
        )
        hint.setStyleSheet('color:#9aa0b8; font-size:12px;')
        hint.setWordWrap(True)
        outer_v.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_w = QWidget()
        scroll_v = QVBoxLayout(scroll_w)
        scroll_v.setSpacing(10)

        self._rw = {}

        self._rw['porosity'] = self._rule_range(
            'Пористость', 'porosity',
            self.rules['porosity']['min'], self.rules['porosity']['max'], 0.01,
            self.rules['porosity']['enabled']
        )
        scroll_v.addWidget(self._rw['porosity']['group'])

        self._rw['permeability_mD'] = self._rule_min(
            'Проницаемость — минимум', 'permeability_mD',
            self.rules['permeability_mD']['min'], 0.1,
            self.rules['permeability_mD']['enabled']
        )
        scroll_v.addWidget(self._rw['permeability_mD']['group'])

        self._rw['density_gcc'] = self._rule_range(
            'Плотность (г/см³)', 'density_gcc',
            self.rules['density_gcc']['min'], self.rules['density_gcc']['max'], 0.1,
            self.rules['density_gcc']['enabled']
        )
        scroll_v.addWidget(self._rw['density_gcc']['group'])

        self._rw['sat_sum'] = self._rule_max(
            'Сумма насыщенностей Sw+So — максимум', 'sat_sum',
            self.rules['sat_sum']['max'], 0.001,
            self.rules['sat_sum']['enabled']
        )
        scroll_v.addWidget(self._rw['sat_sum']['group'])

        self._rw['duplicates'] = self._rule_toggle(
            'Дубликаты строк', 'duplicates',
            self.rules['duplicates']['enabled']
        )
        scroll_v.addWidget(self._rw['duplicates']['group'])

        self._rw['column_shift'] = self._rule_min(
            'Смещение столбца — density_gcc > X', 'column_shift',
            self.rules['column_shift']['min'], 1.0,
            self.rules['column_shift']['enabled']
        )
        scroll_v.addWidget(self._rw['column_shift']['group'])

        scroll_v.addStretch()
        scroll.setWidget(scroll_w)
        outer_v.addWidget(scroll)

        btn_row = QHBoxLayout()
        btn_apply = QPushButton('✅  Применить правила и пересчитать')
        btn_apply.clicked.connect(self.apply_rules)
        btn_reset = QPushButton('↺  Сбросить к дефолту')
        btn_reset.setObjectName('btn_secondary')
        btn_reset.clicked.connect(self.reset_rules)
        btn_row.addWidget(btn_apply)
        btn_row.addWidget(btn_reset)
        outer_v.addLayout(btn_row)
        return outer

    def _rule_range(self, title, key, mn, mx, step, enabled):
        group = QGroupBox(title)
        fl = QFormLayout(group)
        chk = QCheckBox('Включить')
        chk.setChecked(enabled)
        s_min = QDoubleSpinBox(); s_min.setRange(-1e9, 1e9); s_min.setSingleStep(step); s_min.setValue(mn)
        s_max = QDoubleSpinBox(); s_max.setRange(-1e9, 1e9); s_max.setSingleStep(step); s_max.setValue(mx)
        fl.addRow(chk)
        fl.addRow('Минимум:', s_min)
        fl.addRow('Максимум:', s_max)
        return {'group': group, 'chk': chk, 'spin_min': s_min, 'spin_max': s_max}

    def _rule_min(self, title, key, mn, step, enabled):
        group = QGroupBox(title)
        fl = QFormLayout(group)
        chk = QCheckBox('Включить')
        chk.setChecked(enabled)
        s_min = QDoubleSpinBox(); s_min.setRange(-1e9, 1e9); s_min.setSingleStep(step); s_min.setValue(mn)
        fl.addRow(chk)
        fl.addRow('Минимум:', s_min)
        return {'group': group, 'chk': chk, 'spin_min': s_min}

    def _rule_max(self, title, key, mx, step, enabled):
        group = QGroupBox(title)
        fl = QFormLayout(group)
        chk = QCheckBox('Включить')
        chk.setChecked(enabled)
        s_max = QDoubleSpinBox(); s_max.setRange(-1e9, 1e9); s_max.setSingleStep(step)
        s_max.setDecimals(4); s_max.setValue(mx)
        fl.addRow(chk)
        fl.addRow('Максимум:', s_max)
        return {'group': group, 'chk': chk, 'spin_max': s_max}

    def _rule_toggle(self, title, key, enabled):
        group = QGroupBox(title)
        fl = QFormLayout(group)
        chk = QCheckBox('Включить')
        chk.setChecked(enabled)
        fl.addRow(chk)
        return {'group': group, 'chk': chk}

    def apply_rules(self):
        rw = self._rw
        self.rules['porosity']       = {'enabled': rw['porosity']['chk'].isChecked(),
                                        'min': rw['porosity']['spin_min'].value(),
                                        'max': rw['porosity']['spin_max'].value()}
        self.rules['permeability_mD']= {'enabled': rw['permeability_mD']['chk'].isChecked(),
                                        'min': rw['permeability_mD']['spin_min'].value()}
        self.rules['density_gcc']    = {'enabled': rw['density_gcc']['chk'].isChecked(),
                                        'min': rw['density_gcc']['spin_min'].value(),
                                        'max': rw['density_gcc']['spin_max'].value()}
        self.rules['sat_sum']        = {'enabled': rw['sat_sum']['chk'].isChecked(),
                                        'max': rw['sat_sum']['spin_max'].value()}
        self.rules['duplicates']     = {'enabled': rw['duplicates']['chk'].isChecked()}
        self.rules['column_shift']   = {'enabled': rw['column_shift']['chk'].isChecked(),
                                        'min': rw['column_shift']['spin_min'].value()}
        if self.df is not None:
            self.flags = run_verification(self.df, self.rules)
            self.update_all()

    def reset_rules(self):
        self.rules = {k: dict(v) for k, v in DEFAULT_RULES.items()}
        rw = self._rw
        rw['porosity']['chk'].setChecked(True)
        rw['porosity']['spin_min'].setValue(0.0)
        rw['porosity']['spin_max'].setValue(1.0)
        rw['permeability_mD']['chk'].setChecked(True)
        rw['permeability_mD']['spin_min'].setValue(0.0)
        rw['density_gcc']['chk'].setChecked(True)
        rw['density_gcc']['spin_min'].setValue(1.5)
        rw['density_gcc']['spin_max'].setValue(3.5)
        rw['sat_sum']['chk'].setChecked(True)
        rw['sat_sum']['spin_max'].setValue(1.001)
        rw['duplicates']['chk'].setChecked(True)
        rw['column_shift']['chk'].setChecked(True)
        rw['column_shift']['spin_min'].setValue(10.0)
        if self.df is not None:
            self.flags = run_verification(self.df, self.rules)
            self.update_all()

    # ── Сборка UI ───────────────────────────────────────────────────────────
    def _build_ui(self):
        from PyQt6.QtWidgets import QStackedWidget
        central = QWidget()
        self.setCentralWidget(central)
        main_v = QVBoxLayout(central)
        main_v.setContentsMargins(0, 0, 0, 0)
        main_v.setSpacing(0)
        main_v.addWidget(self._build_header())

        self._stack = QStackedWidget()

        tabs_widget = QWidget()
        tabs_v = QVBoxLayout(tabs_widget)
        tabs_v.setContentsMargins(0, 0, 0, 0)
        tabs_v.setSpacing(0)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_verify(),   '🔍  Верификация')
        self.tabs.addTab(self._build_tab_map(),      '🗺  Карта скважин')
        self.tabs.addTab(self._build_tab_profiles(), '📈  Профили глубины')
        self.tabs.addTab(self._build_tab_spatial(),  '📊  Анализ лабораторий')
        self.tabs.addTab(self._build_tab_rules(),    '⚙  Правила')
        tabs_v.addWidget(self.tabs)

        self._expand_panel = ExpandPanel()
        self._expand_panel._btn_back.clicked.connect(self._close_expand)

        self._stack.addWidget(tabs_widget)
        self._stack.addWidget(self._expand_panel)

        main_v.addWidget(self._stack)

    def _open_expand_canvas(self, title: str, draw_fn):
        """Рисует график в новый canvas и показывает его внутри приложения."""
        # figsize чуть меньше прежнего (18×9 вместо 20×10) — больше отступов для подписей
        canvas = MplCanvas(figsize=(18, 9))
        draw_fn(canvas.fig)
        canvas.draw()
        self._expand_panel.show_canvas(title, canvas)
        self._stack.setCurrentIndex(1)

    def _open_expand_table(self):
        self._expand_panel.show_table('Аномальные записи', self.table)
        self._stack.setCurrentIndex(1)

    def _close_expand(self):
        self._expand_panel._clear_content()
        self._stack.setCurrentIndex(0)

    # ── Загрузка файла ──────────────────────────────────────────────────────
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Открыть CSV', '', 'CSV файлы (*.csv);;Все файлы (*)'
        )
        if not path:
            return
        try:
            self.df    = pd.read_csv(path)
            self.flags = run_verification(self.df, self.rules)
            fname = path.replace('\\', '/').split('/')[-1]
            self.lbl_file.setText(f'📄 {fname}')
            self.btn_save.setEnabled(True)
            self.update_all()
        except Exception as e:
            self.lbl_file.setText(f'Ошибка: {e}')

    def update_all(self):
        self.update_verify()
        self.update_map()
        self.update_profiles()
        self.update_spatial()

    # ── Вкладка 1 ──────────────────────────────────────────────────────────
    def update_verify(self):
        df, flags = self.df, self.flags
        n_total = len(df)
        n_err   = flags['any_error'].sum()
        n_clean = n_total - n_err
        pct     = n_err / n_total * 100

        self.metric_total[1].setText(str(n_total))
        self.metric_errors[1].setText(str(n_err))
        self.metric_errors[1].setStyleSheet('font-size:28px;font-weight:bold;color:#e74c3c;')
        self.metric_clean[1].setText(str(n_clean))
        self.metric_clean[1].setStyleSheet('font-size:28px;font-weight:bold;color:#27ae60;')
        self.metric_pct[1].setText(f'{pct:.1f}%')

        err_cols = ['ошибка_пористость','ошибка_проницаемость','ошибка_плотность',
                    'ошибка_насыщенность','дубль','смещение_столбца']
        counts = flags[err_cols].sum().sort_values(ascending=False)

        self.canvas_errors.fig.clear()
        ax = self.canvas_errors.fig.add_subplot(111)
        ax.set_facecolor('#232940')
        bar_colors = ['#e74c3c','#e67e22','#f1c40f','#3498db','#9b59b6','#1abc9c']
        bars = ax.bar(range(len(counts)), counts.values,
                      color=bar_colors[:len(counts)], edgecolor='none', width=0.6)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=20, ha='right', fontsize=10, color='#c0c8e0')
        ax.set_ylabel('Количество записей', color='#b0b8d0')
        ax.set_title('Ошибки по типам', color='#ffffff', fontsize=12)
        ax.tick_params(colors='#9aa0b8')
        ax.grid(axis='y', color='#2e3550', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, counts.values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        str(val), ha='center', va='bottom', color='#ffffff', fontsize=10)
        self.canvas_errors.fig.tight_layout()
        self.canvas_errors.draw()

        # Таблица
        suspicious = df[flags['any_error']].copy()
        all_cols = list(df.columns) + ['типы ошибок']
        self.table.setColumnCount(len(all_cols))
        self.table.setHorizontalHeaderLabels(all_cols)
        self.table.setRowCount(len(suspicious))

        for row_i, (idx, row) in enumerate(suspicious.iterrows()):
            for col_i, col in enumerate(df.columns):
                item = QTableWidgetItem(str(row[col]))
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row_i, col_i, item)

            errs = []
            for c in err_cols:
                if flags.at[idx, c]:
                    if c == 'дубль':
                        partner = str(flags.at[idx, 'дубль_строки'])
                        errs.append(f'дубль (стр. {partner})' if partner else 'дубль')
                    else:
                        errs.append(c)
            item_e = QTableWidgetItem(', '.join(errs))
            item_e.setForeground(QColor('#ff7070'))
            item_e.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row_i, len(df.columns), item_e)

    # ── Вкладка 2 ──────────────────────────────────────────────────────────
    def update_map(self):
        df, flags = self.df, self.flags

        well_stats = df.groupby('well_id').agg(
            por_med=('porosity', 'median'),
            n=('porosity', 'count')
        ).reset_index()
        well_stats['x'] = df.groupby('well_id')['x_m'].first().values
        well_stats['y'] = df.groupby('well_id')['y_m'].first().values

        df2 = df.copy()
        df2['_err'] = flags['any_error'].values
        err_by_well = df2.groupby('well_id')['_err'].sum().reset_index()
        well_stats  = well_stats.merge(err_by_well, on='well_id')

        self.canvas_map.fig.clear()
        axes = self.canvas_map.fig.subplots(1, 2)
        self.canvas_map.fig.patch.set_facecolor('#1a1f2e')

        por_norm = Normalize(vmin=well_stats['por_med'].min(), vmax=well_stats['por_med'].max())
        por_cmap = plt.get_cmap('winter')
        err_norm = Normalize(vmin=0, vmax=max(well_stats['_err'].max(), 1))
        err_cmap = plt.get_cmap('Reds')

        ax = axes[0]
        ax.set_facecolor('#232940')
        y_min, y_max = well_stats['y'].min(), well_stats['y'].max()
        y_offset = (y_max - y_min) * 0.03

        for _, r in well_stats.iterrows():
            fc = por_cmap(por_norm(r['por_med']))
            ec = err_cmap(err_norm(r['_err']))
            ax.scatter(r['x'], r['y'],
                       s=r['n'] * 18,
                       c=[fc], edgecolors=[ec], linewidths=4.0, zorder=3)
            ax.text(r['x'], r['y'] + y_offset + 10, r['well_id'],
                    ha='center', va='bottom', fontsize=8, color='#d0d4e8')

        sm_por = ScalarMappable(cmap='winter', norm=por_norm)
        sm_por.set_array([])
        cb1 = self.canvas_map.fig.colorbar(sm_por, ax=ax, fraction=0.035, pad=0.01)
        cb1.set_label('Медианная пористость', color='#c0c8e0')
        plt.setp(cb1.ax.yaxis.get_ticklabels(), color='#c0c8e0')

        sm_err = ScalarMappable(cmap='Reds', norm=err_norm)
        sm_err.set_array([])
        cb2 = self.canvas_map.fig.colorbar(sm_err, ax=ax, fraction=0.035, pad=0.12, location='left')
        cb2.set_label('Кол-во аномалий', color='#c0c8e0')
        plt.setp(cb2.ax.yaxis.get_ticklabels(), color='#c0c8e0')

        ax.set_title('Медианная пористость + аномалии', color='#ffffff', fontsize=11)
        ax.set_xlabel('X, м', color='#b0b8d0'); ax.set_ylabel('Y, м', color='#b0b8d0')
        ax.tick_params(colors='#9aa0b8')
        ax.grid(True, color='#2e3550', linestyle='--', alpha=0.4)

        ax2 = axes[1]
        ax2.set_facecolor('#232940')
        if 'геол_зона' in df.columns:
            well_zone = df.groupby('well_id')['геол_зона'].first().reset_index()
            ws2 = well_stats.merge(well_zone, on='well_id')
            zones = sorted(ws2['геол_зона'].unique())
            palette = ['#27ae60','#e67e22','#e74c3c','#3498db','#9b59b6']
            zcolors = {z: palette[i % len(palette)] for i, z in enumerate(zones)}
            for zone, grp in ws2.groupby('геол_зона'):
                ax2.scatter(grp['x'], grp['y'], s=130,
                            color=zcolors[zone], label=zone,
                            edgecolors='#ffffff', linewidths=0.8, zorder=3)
                for _, r in grp.iterrows():
                    ax2.text(r['x'], r['y'] + y_offset, r['well_id'],
                            ha='center', va='bottom', fontsize=8, color='#d0d4e8')
            ax2.legend(fontsize=9, facecolor='#2e3550', edgecolor='#3e4560', labelcolor='#d0d4e8')
        ax2.set_title('Геологические зоны', color='#ffffff', fontsize=11)
        ax2.set_xlabel('X, м', color='#b0b8d0'); ax2.set_ylabel('Y, м', color='#b0b8d0')
        ax2.tick_params(colors='#9aa0b8')
        ax2.grid(True, color='#2e3550', linestyle='--', alpha=0.4)

        self.canvas_map.fig.tight_layout()
        self.canvas_map.draw()

    # ── Вкладка 3 — профили с интерактивной легендой ───────────────────────
    def update_profiles(self):
        df         = self.df
        filter_on  = self.chk_filter_profiles.isChecked()
        por_min    = self.rules.get('porosity', {}).get('min', 0.0)
        por_max    = self.rules.get('porosity', {}).get('max', 1.0)

        self.canvas_profiles.fig.clear()
        ax = self.canvas_profiles.fig.add_subplot(111)
        ax.set_facecolor('#232940')

        cmap_tab = plt.get_cmap('tab10')
        lines = {}  # well_id → line artist
        for i, well in enumerate(sorted(df['well_id'].unique())):
            grp = df[df['well_id'] == well].sort_values('depth_m')
            if filter_on:
                grp = grp[grp['porosity'].between(por_min, por_max)]
            if grp.empty:
                continue
            line, = ax.plot(grp['porosity'], grp['depth_m'],
                            color=cmap_tab(i % 10), alpha=0.85, linewidth=1.5, label=well)
            lines[well] = line

        ax.invert_yaxis()
        ax.set_xlabel('Пористость', color='#b0b8d0')
        ax.set_ylabel('Глубина, м', color='#b0b8d0')
        ax.tick_params(colors='#9aa0b8')
        suffix = ' (только физически допустимые)' if filter_on else ''
        ax.set_title(f'Профили пористости по глубине{suffix}', color='#ffffff', fontsize=12)
        ax.grid(True, color='#2e3550', linestyle='--', alpha=0.5)

        if not filter_on:
            ax.axvline(por_min, color="#563ce7", linestyle=':', alpha=0.7)
            ax.axvline(por_max, color="#3c6de7", linestyle=':', alpha=0.7)

        # ── Интерактивная легенда ───────────────────────────────────────────
        leg = ax.legend(fontsize=8, facecolor='#2e3550', edgecolor='#3e4560',
                        labelcolor='#d0d4e8', ncol=2)
        # Строим map: элемент легенды → линия графика
        line_map = {}
        for leg_line, orig_well in zip(leg.get_lines(), lines.keys()):
            leg_line.set_pickradius(8)   # зона захвата клика, пикселей
            leg_line.set_picker(True)
            line_map[leg_line] = lines[orig_well]
        self.canvas_profiles.set_line_map(line_map)

        self.canvas_profiles.fig.tight_layout()
        self.canvas_profiles.draw()

    # ── Вкладка 4 ──────────────────────────────────────────────────────────
    def update_spatial(self):
        df    = self.df
        flags = self.flags
        hide  = self.chk_hide_anomalies.isChecked()
        df_plot = df[~flags['any_error']].copy() if hide else df.copy()

        self.canvas_spatial.fig.clear()
        axes = self.canvas_spatial.fig.subplots(1, 2)
        self.canvas_spatial.fig.patch.set_facecolor('#1a1f2e')

        labs   = sorted(df_plot['lab_id'].unique())
        groups = [df_plot[df_plot['lab_id'] == l]['porosity'].dropna().values for l in labs]
        labs   = [l for l, g in zip(labs, groups) if len(g) > 0]
        groups = [g for g in groups if len(g) > 0]

        ax = axes[0]
        ax.set_facecolor('#232940')
        if groups:
            bp = ax.boxplot(groups, labels=labs, patch_artist=True,
                            medianprops=dict(color='#f1c40f', linewidth=2),
                            whiskerprops=dict(color='#9aa0b8'),
                            capprops=dict(color='#9aa0b8'),
                            flierprops=dict(markerfacecolor='#e74c3c', markersize=4))
            box_colors = ['#3498db','#e67e22','#27ae60']
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color); patch.set_alpha(0.7)

            if len(groups) >= 2:
                stat_kw, p_kw = stats.kruskal(*groups)
                txt = f'Крускал–Уоллис: H={stat_kw:.2f}, p={p_kw:.5f}'
                txt += (' — ЗНАЧИМО ⚠' if p_kw < 0.05 else ' — норма ✓')
                color_txt = '#e74c3c' if p_kw < 0.05 else '#27ae60'
                self.lbl_kruskal.setText(txt)
                self.lbl_kruskal.setStyleSheet(
                    f'color:{color_txt}; font-size:12px; padding:4px; font-weight:bold;')

                pairwise = []
                for i in range(len(labs)):
                    for j in range(i+1, len(labs)):
                        _, p_mw = mannwhitneyu(groups[i], groups[j])
                        mark = '⚠' if p_mw < 0.05 else '✓'
                        pairwise.append(f'{labs[i]} vs {labs[j]}: p={p_mw:.4f} {mark}')
                ax.set_xlabel('\n'.join(pairwise), color='#9aa0b8', fontsize=9)

        suffix = ' (без аномалий)' if hide else ''
        ax.set_title(f'Пористость по лабораториям{suffix}', color='#ffffff', fontsize=11)
        ax.set_ylabel('Пористость', color='#b0b8d0')
        ax.tick_params(colors='#9aa0b8')
        ax.grid(axis='y', color='#2e3550', linestyle='--', alpha=0.5)

        ax2 = axes[1]
        ax2.set_facecolor('#232940')
        well_por = df_plot.groupby('well_id')['porosity'].median().sort_values()
        bar_colors = ['#e74c3c' if v > 0.3 else '#3498db' for v in well_por.values]
        ax2.barh(well_por.index, well_por.values, color=bar_colors, edgecolor='none')
        ax2.set_title(f'Медианная пористость по скважинам{suffix}', color='#ffffff', fontsize=11)
        ax2.set_xlabel('Медианная пористость', color='#b0b8d0')
        ax2.tick_params(colors='#9aa0b8')
        ax2.grid(axis='x', color='#2e3550', linestyle='--', alpha=0.5)
        med = well_por.median()
        ax2.axvline(med, color='#f1c40f', linestyle='--', linewidth=1.5,
                    label=f'Медиана: {med:.3f}')
        ax2.legend(fontsize=8, facecolor='#2e3550', edgecolor='#3e4560', labelcolor='#d0d4e8')

        self.canvas_spatial.fig.tight_layout()
        self.canvas_spatial.draw()

    # ── Сохранение ─────────────────────────────────────────────────────────
    def save_report(self):
        if self.df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Сохранить отчёт', 'anomaly_report.csv', 'CSV файлы (*.csv)'
        )
        if not path:
            return
        err_cols = ['ошибка_пористость','ошибка_проницаемость','ошибка_плотность',
                    'ошибка_насыщенность','дубль','смещение_столбца','дубль_строки']
        suspicious = self.df[self.flags['any_error']].copy()
        for col in err_cols:
            if col in self.flags.columns:
                suspicious[col] = self.flags.loc[self.flags['any_error'], col].values
        suspicious.to_csv(path, index=True, encoding='utf-8-sig')
        fname = path.replace('\\', '/').split('/')[-1]
        self.lbl_file.setText(f'✅ Сохранено: {fname}')


# ── Точка входа ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
