"""The second skin: near-black navy with a teal accent.

Same console, different clothes. :mod:`ui.theme` is the console's own look -
mid navy, blue accent, 12px cards. This is the alternative design: a much
darker ground (#050d1a), teal as the interactive colour, hairline borders drawn
in white alpha rather than a solid navy, tighter corners, and a navigation item
that tints rather than fills when it is selected.

It exists as a separate module rather than as options inside ``ui.theme``
because the two are whole designs, not variants of one: almost every rule
differs, and interleaving them behind conditionals would leave neither
readable.

Two halves, and both are needed for a complete re-skin:

``PALETTE``
    Handed to :func:`ui.theme.apply_palette` *before* a window is built, so the
    widgets that paint themselves - badges, KPI values, chart series, the brand
    mark - pick these colours up too. A style sheet alone cannot reach them.
``STYLESHEET``
    Set on the window after it is built. Structure as well as colour: corner
    radii, the tinted selection, the flat scroll bars.

Fonts are named in the design as Instrument Sans, Inter and DM Mono. Those are
web fonts and will not be installed on a plant workstation, so each is given
the platform's own stack behind it and the layout is built to survive whichever
one actually resolves.
"""

from __future__ import annotations

__all__ = ["PALETTE", "STYLESHEET", "NAME"]

#: What to call this look where an operator can see it.
NAME = "Deep navy / teal"

#: Design tokens, straight from the reference. Names match :class:`ui.theme.C`
#: so the mapping can be applied without translation.
PALETTE = {
    # Surfaces. The reference ground is near black, and the cards sit only just
    # above it - the separation is carried by the border, not by brightness.
    "APP": "#050d1a",
    "SIDEBAR": "#080e1f",
    "HEADER": "#080e1f",
    "PANEL": "#0d1829",
    "PANEL_ALT": "#0f1d30",
    "CARD": "#132035",
    "BORDER": "#1d2c42",
    "BORDER_SOFT": "#152134",

    "TEXT": "#e6ebf2",
    "TEXT_DIM": "#8a97a8",
    "TEXT_FAINT": "#5d6a7c",

    "BRAND": "#e63329",
    "ACCENT": "#14b8a6",
    "ACCENT_SOFT": "#2dd4bf",
    "ACCENT_DIM": "#0d9488",
    "BLUE": "#60a5fa",
    "PURPLE": "#a78bfa",

    "SCROLL_TRACK": "#050d1a",
    "SCROLL_THUMB": "#243d5e",
    "SCROLL_THUMB_HOVER": "#31527d",

    "DANGER": "#ef4444",
    "WARN": "#fbbf24",
    "OK": "#2dd4bf",
    "INFO": "#60a5fa",
}

_C = PALETTE
#: Hairlines. The reference draws every border in white alpha, which keeps a
#: card readable against the near-black ground without a visible frame round it.
_EDGE = "rgba(255, 255, 255, 0.07)"
_EDGE_SOFT = "rgba(255, 255, 255, 0.04)"
#: The selected navigation item is a teal wash, not a solid block.
_TINT = "rgba(20, 184, 166, 0.13)"
_FONT = '"Inter", "Segoe UI", "DejaVu Sans", Arial, sans-serif'
_DISPLAY = '"Instrument Sans", "Inter", "Segoe UI", Arial, sans-serif'

STYLESHEET = f"""
QWidget {{
    background-color: {_C["APP"]};
    color: {_C["TEXT"]};
    font-family: {_FONT};
    font-size: 13px;
}}
QLabel {{ background: transparent; border: none; }}
QFrame#Sidebar {{ background-color: {_C["SIDEBAR"]}; }}
QSplitter#Shell::handle {{ background-color: {_EDGE}; }}
QSplitter#Shell::handle:hover {{ background-color: {_C["ACCENT"]}; }}
QFrame#Header {{
    background-color: {_C["HEADER"]};
    border-bottom: 1px solid {_EDGE};
}}
QWidget#HeaderBrand {{ background: transparent; }}
QFrame#Footer {{
    background-color: {_C["HEADER"]};
    border-top: 1px solid {_EDGE};
}}
QFrame#Panel, QFrame#Card {{
    background-color: {_C["PANEL"]};
    border: 1px solid {_EDGE};
    border-radius: 8px;
}}
QFrame#Tile {{
    background-color: {_C["PANEL"]};
    border: 1px solid {_EDGE};
    border-radius: 8px;
}}

QLabel#AppTitle {{ font-family: {_DISPLAY}; font-size: 26px; font-weight: 700; }}
QLabel#AppSubtitle {{ font-size: 12.5px; color: {_C["TEXT_DIM"]}; }}
QLabel#BrandName {{
    font-family: {_DISPLAY};
    font-size: 15px;
    font-weight: 700;
    color: {_C["TEXT"]};
    letter-spacing: 0.4px;
}}
QLabel#BrandSub {{ font-size: 8px; color: {_C["TEXT_DIM"]}; letter-spacing: 0.4px; }}
QLabel#PageTitle {{
    font-family: {_DISPLAY};
    font-size: 21px;
    font-weight: 700;
    letter-spacing: -0.2px;
}}
QLabel#SectionTitle {{
    font-family: {_DISPLAY};
    font-size: 14.5px;
    font-weight: 600;
    color: {_C["TEXT"]};
}}
QLabel#Caption {{
    font-size: 10px;
    color: {_C["TEXT_DIM"]};
    letter-spacing: 1.1px;
}}
QLabel#Muted {{ color: {_C["TEXT_DIM"]}; }}
QLabel#Faint {{ color: {_C["TEXT_FAINT"]}; font-size: 11.5px; }}
/* 32px and tucked in tight, the way the reference sets a headline metric. */
QLabel#KpiValue {{
    font-family: {_DISPLAY};
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.6px;
}}
QLabel#KpiUnit {{ font-size: 12px; color: {_C["TEXT_DIM"]}; }}

/* Bright enough to read as pressable: dim text on a transparent ground is
   indistinguishable from a disabled control. */
QPushButton {{
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 5px;
    padding: 9px 15px;
    font-family: {_DISPLAY};
    font-weight: 500;
    color: {_C["TEXT"]};
}}
QPushButton:hover {{
    background-color: rgba(255, 255, 255, 0.09);
    border-color: {_C["ACCENT"]};
    color: {_C["TEXT"]};
}}
QPushButton:pressed {{ background-color: rgba(255, 255, 255, 0.03); }}
QPushButton:disabled {{ color: {_C["TEXT_FAINT"]}; border-color: {_EDGE_SOFT}; }}
QPushButton#Primary {{
    background-color: {_C["ACCENT"]};
    border: 1px solid {_C["ACCENT"]};
    color: {_C["APP"]};
    font-weight: 600;
}}
QPushButton#Primary:hover {{
    background-color: {_C["ACCENT_SOFT"]};
    border-color: {_C["ACCENT_SOFT"]};
    color: {_C["APP"]};
}}
QPushButton#Primary:disabled {{ background-color: {_C["ACCENT_DIM"]}; color: #9ec9c3; }}
QPushButton#Warning {{
    background-color: {_C["WARN"]};
    border: 1px solid {_C["WARN"]};
    color: #2a1a00;
    font-weight: 600;
}}
/* Selected pages tint rather than fill: against a near-black ground a solid
   block of accent is louder than the content it is pointing at. */
QPushButton#Nav {{
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 9px 14px;
    margin: 1px 8px;
    text-align: left;
    font-family: {_DISPLAY};
    font-weight: 500;
    color: {_C["TEXT_DIM"]};
}}
QPushButton#Nav:hover {{
    background-color: rgba(255, 255, 255, 0.05);
    color: {_C["TEXT"]};
}}
QPushButton#Nav:checked {{
    background-color: {_TINT};
    color: {_C["ACCENT_SOFT"]};
    font-weight: 600;
}}

QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {_C["PANEL"]};
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 5px;
    padding: 8px 10px;
    selection-background-color: {_C["ACCENT_DIM"]};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid rgba(20, 184, 166, 0.5);
}}
QComboBox {{
    background-color: {_C["PANEL"]};
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 5px;
    padding: 7px 10px;
}}
QComboBox:focus {{ border: 1px solid rgba(20, 184, 166, 0.5); }}
QComboBox QAbstractItemView {{
    background-color: {_C["PANEL"]};
    border: 1px solid {_EDGE};
    selection-background-color: {_C["ACCENT_DIM"]};
}}
QCheckBox {{ spacing: 8px; background: transparent; }}

QTableWidget {{
    background-color: {_C["PANEL"]};
    alternate-background-color: {_C["PANEL_ALT"]};
    gridline-color: {_EDGE_SOFT};
    border: 1px solid {_EDGE};
    border-radius: 8px;
    selection-background-color: rgba(20, 184, 166, 0.16);
}}
QHeaderView::section {{
    background-color: {_C["PANEL"]};
    color: {_C["TEXT_DIM"]};
    padding: 9px 8px;
    border: none;
    border-bottom: 1px solid {_EDGE};
    font-family: {_DISPLAY};
    font-weight: 600;
    font-size: 11.5px;
}}
QTableWidget::item {{ padding: 4px; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent;
    color: {_C["TEXT_DIM"]};
    padding: 9px 18px;
    border-bottom: 2px solid transparent;
    font-family: {_DISPLAY};
    font-weight: 600;
}}
QTabBar::tab:selected {{
    color: {_C["ACCENT_SOFT"]};
    border-bottom: 2px solid {_C["ACCENT"]};
}}

QProgressBar {{
    background-color: {_C["PANEL"]};
    border: 1px solid {_EDGE};
    border-radius: 4px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{ background-color: {_C["ACCENT"]}; border-radius: 3px; }}

/* Flat and slim, with no stepper arrows - the reference has no furniture on
   its scroll bars. Kept at 10px rather than the reference's 4px, because 4px
   is a mouse target nobody hits on a workstation. */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {_C["SCROLL_THUMB"]};
    border-radius: 5px;
    min-height: 36px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{ background: {_C["SCROLL_THUMB_HOVER"]}; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {_C["SCROLL_THUMB"]};
    border-radius: 5px;
    min-width: 36px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{ background: {_C["SCROLL_THUMB_HOVER"]}; }}
QScrollBar::add-line, QScrollBar::sub-line {{
    background: transparent;
    border: none;
    width: 0px;
    height: 0px;
}}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background-color: {_C["HEADER"]}; color: {_C["TEXT_DIM"]}; }}
QMenuBar {{ background-color: {_C["HEADER"]}; color: {_C["TEXT_DIM"]}; }}
QMenuBar::item:selected {{ background: {_TINT}; color: {_C["ACCENT_SOFT"]}; }}
QMenu {{
    background-color: {_C["PANEL"]};
    border: 1px solid {_EDGE};
    padding: 4px;
}}
QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {_TINT}; color: {_C["ACCENT_SOFT"]}; }}
QToolTip {{
    background-color: {_C["PANEL"]};
    color: {_C["TEXT"]};
    border: 1px solid {_EDGE};
    padding: 6px;
}}
"""
