"""Visual language for the console - one place to retune the whole interface.

Colours are grouped by role rather than by hue, so a re-skin (an asset with a
different corporate palette, a light-mode build) means editing this file only.
The stylesheet is a plain Qt style sheet string; widgets that need painting
(charts, badges) read the same constants.
"""

from __future__ import annotations

__all__ = ["C", "STYLESHEET", "BAND_COLORS", "SEVERITY_COLORS", "CATEGORICAL"]


class C:
    """Named colours. ``C.x`` reads better than a dictionary at call sites."""

    # Surfaces, darkest to lightest.
    APP = "#0a1524"
    SIDEBAR = "#0d1b2e"
    PANEL = "#122238"
    PANEL_ALT = "#16293f"
    CARD = "#182f49"
    HEADER = "#102035"
    BORDER = "#1f3a57"
    BORDER_SOFT = "#193049"

    # Text.
    TEXT = "#e8f0f8"
    TEXT_DIM = "#8fa8c0"
    TEXT_FAINT = "#5f7894"

    # Brand and accents.
    BRAND = "#e63329"
    ACCENT = "#22d3c5"
    ACCENT_DIM = "#0e7a72"
    BLUE = "#3b82f6"
    PURPLE = "#8b5cf6"

    # Status.
    DANGER = "#ef4444"
    WARN = "#f59e0b"
    OK = "#22c55e"
    INFO = "#38bdf8"


#: Risk band -> colour, used by every table and chart.
BAND_COLORS = {"Critical": C.DANGER, "High": "#fb923c", "Medium": C.WARN, "Low": C.OK}

#: Severity hint -> colour.
SEVERITY_COLORS = {"High": C.DANGER, "Medium": C.WARN, "Low": C.OK}

#: Fixed categorical order for the energy donut - assigned by position, never
#: cycled, so a category keeps its colour as the mix changes.
CATEGORICAL = ("#3b82f6", "#ef4444", "#f59e0b", "#22c55e", "#8b5cf6",
               "#38bdf8", "#fb923c", "#e879f9", "#64748b")

STYLESHEET = f"""
QWidget {{
    background-color: {C.APP};
    color: {C.TEXT};
    font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif;
    font-size: 13px;
}}
QLabel {{ background: transparent; border: none; }}
QFrame#Sidebar {{ background-color: {C.SIDEBAR}; border-right: 1px solid {C.BORDER}; }}
QFrame#Header {{ background-color: {C.HEADER}; border-bottom: 1px solid {C.BORDER}; }}
QFrame#Footer {{ background-color: {C.HEADER}; border-top: 1px solid {C.BORDER}; }}
QFrame#Panel {{
    background-color: {C.PANEL};
    border: 1px solid {C.BORDER};
    border-radius: 12px;
}}
QFrame#Card {{
    background-color: {C.PANEL};
    border: 1px solid {C.BORDER};
    border-radius: 12px;
}}
QFrame#Tile {{
    background-color: {C.PANEL_ALT};
    border: 1px solid {C.BORDER};
    border-radius: 10px;
}}
QLabel#AppTitle {{ font-size: 27px; font-weight: 700; letter-spacing: 0.3px; }}
QLabel#AppSubtitle {{ font-size: 12.5px; color: {C.TEXT_DIM}; }}
QLabel#BrandName {{ font-size: 15px; font-weight: 700; color: {C.TEXT}; }}
QLabel#BrandSub {{ font-size: 8px; color: {C.TEXT_DIM}; letter-spacing: 0.4px; }}
QLabel#SectionTitle {{ font-size: 14.5px; font-weight: 600; color: {C.TEXT}; }}
QLabel#Caption {{ font-size: 10.5px; color: {C.TEXT_DIM}; letter-spacing: 0.9px; }}
QLabel#Muted {{ color: {C.TEXT_DIM}; }}
QLabel#Faint {{ color: {C.TEXT_FAINT}; font-size: 11.5px; }}
QLabel#KpiValue {{ font-size: 30px; font-weight: 700; }}
QLabel#KpiUnit {{ font-size: 12px; color: {C.TEXT_DIM}; }}

QPushButton {{
    background-color: {C.CARD};
    border: 1px solid {C.BORDER};
    border-radius: 9px;
    padding: 9px 14px;
    font-weight: 600;
}}
QPushButton:hover {{ background-color: #1e3a58; }}
QPushButton:pressed {{ background-color: #142a42; }}
QPushButton:disabled {{ color: {C.TEXT_FAINT}; border-color: {C.BORDER_SOFT}; }}
QPushButton#Primary {{
    background-color: {C.ACCENT};
    border: 1px solid {C.ACCENT};
    color: #04211f;
}}
QPushButton#Primary:hover {{ background-color: #2ee9d9; }}
QPushButton#Primary:disabled {{ background-color: {C.ACCENT_DIM}; color: #9fc9c5; }}
QPushButton#Warning {{
    background-color: {C.WARN};
    border: 1px solid {C.WARN};
    color: #2a1a00;
}}
QPushButton#Nav {{
    background-color: transparent;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0px;
    padding: 11px 16px;
    text-align: left;
    font-weight: 600;
    color: {C.TEXT_DIM};
}}
QPushButton#Nav:hover {{ background-color: {C.PANEL}; color: {C.TEXT}; }}
QPushButton#Nav:checked {{
    background-color: {C.PANEL};
    color: {C.TEXT};
    border-left: 3px solid {C.ACCENT};
}}

QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {C.PANEL_ALT};
    border: 1px solid {C.BORDER};
    border-radius: 9px;
    padding: 8px 10px;
    selection-background-color: {C.ACCENT_DIM};
}}
QComboBox {{
    background-color: {C.PANEL_ALT};
    border: 1px solid {C.BORDER};
    border-radius: 9px;
    padding: 7px 10px;
}}
QComboBox QAbstractItemView {{
    background-color: {C.PANEL_ALT};
    border: 1px solid {C.BORDER};
    selection-background-color: {C.ACCENT_DIM};
}}
QCheckBox {{ spacing: 8px; }}

QTableWidget {{
    background-color: {C.PANEL};
    alternate-background-color: {C.PANEL_ALT};
    gridline-color: {C.BORDER_SOFT};
    border: 1px solid {C.BORDER};
    border-radius: 10px;
    selection-background-color: #1c4a5e;
}}
QHeaderView::section {{
    background-color: {C.PANEL_ALT};
    color: {C.TEXT_DIM};
    padding: 8px;
    border: none;
    border-right: 1px solid {C.BORDER_SOFT};
    border-bottom: 1px solid {C.BORDER};
    font-weight: 600;
}}
QTableWidget::item {{ padding: 4px; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent;
    color: {C.TEXT_DIM};
    padding: 9px 18px;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:selected {{ color: {C.TEXT}; border-bottom: 2px solid {C.ACCENT}; }}

QProgressBar {{
    background-color: {C.PANEL_ALT};
    border: 1px solid {C.BORDER};
    border-radius: 6px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{ background-color: {C.ACCENT}; border-radius: 5px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C.BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {C.BORDER}; border-radius: 5px; min-width: 30px; }}
QStatusBar {{ background-color: {C.HEADER}; color: {C.TEXT_DIM}; }}
QToolTip {{
    background-color: {C.PANEL_ALT};
    color: {C.TEXT};
    border: 1px solid {C.BORDER};
    padding: 6px;
}}
"""
