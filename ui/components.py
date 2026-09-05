"""Reusable interface pieces: KPI tiles, panels, navigation, header, tables.

Everything here is presentation only - no pipeline, no threads, no I/O - so the
views stay short and the widgets can be exercised without starting an analysis.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFrame,
    QScrollArea,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .theme import BAND_COLORS, C

__all__ = ["KpiTile", "Panel", "Sidebar", "HeaderBar", "DataTable", "Pill", "FieldRow"]


class Pill(QLabel):
    """A small coloured status chip (SIF-POTENTIAL, risk band, trigger)."""

    def __init__(self, text: str = "", colour: str = C.ACCENT, parent=None) -> None:
        super().__init__(text, parent)
        self.set_colour(colour)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        """Resize with the text: a fixed-size chip must re-measure when relabelled."""
        super().setText(text)
        self.adjustSize()
        self.updateGeometry()

    def set_colour(self, colour: str) -> None:
        """Recolour the chip, keeping the translucent-fill / solid-text pairing."""
        self.setStyleSheet(
            f"color: {colour}; border: 1px solid {colour}; border-radius: 9px;"
            f"padding: 3px 10px; font-size: 11px; font-weight: 700;")


class KpiTile(QFrame):
    """Headline metric: caption, big value, optional unit and delta note."""

    def __init__(self, caption: str, value: str = "0", accent: str = C.TEXT,
                 unit: str = "", note: str = "", glyph: str = "") -> None:
        super().__init__()
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(96)

        self._glyph = QLabel(glyph)
        self._glyph.setFixedSize(42, 42)
        self._glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._glyph.setStyleSheet(
            f"background-color: {C.PANEL_ALT}; border: 1px solid {C.BORDER};"
            f"border-radius: 11px; font-size: 18px; color: {accent};")

        caption_label = QLabel(caption)
        caption_label.setObjectName("Caption")

        self._value = QLabel(value)
        self._value.setObjectName("KpiValue")
        self._value.setStyleSheet(f"color: {accent};")

        self._unit = QLabel(unit)
        self._unit.setObjectName("KpiUnit")

        self._note = QLabel(note)
        self._note.setObjectName("Faint")

        value_row = QHBoxLayout()
        value_row.setSpacing(6)
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.addWidget(self._value)
        value_row.addWidget(self._unit, alignment=Qt.AlignmentFlag.AlignBottom)
        value_row.addStretch(1)

        text_column = QVBoxLayout()
        text_column.setSpacing(1)
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.addWidget(caption_label)
        text_column.addLayout(value_row)
        text_column.addWidget(self._note)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)
        if glyph:
            layout.addWidget(self._glyph, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(text_column, stretch=1)

    def set_value(self, value: str, note: Optional[str] = None) -> None:
        """Update the metric, and optionally its footnote."""
        self._value.setText(value)
        if note is not None:
            self._note.setText(note)


class Panel(QFrame):
    """Titled container with an optional right-hand action widget."""

    def __init__(self, title: str = "", action: Optional[QWidget] = None,
                 margins: Tuple[int, int, int, int] = (14, 12, 14, 14)) -> None:
        super().__init__()
        self.setObjectName("Panel")

        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        self.body.setContentsMargins(0, 0, 0, 0)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(*margins)
        outer.setSpacing(10)

        if title:
            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            label = QLabel(title)
            label.setObjectName("SectionTitle")
            header.addWidget(label)
            header.addStretch(1)
            if action is not None:
                header.addWidget(action)
            outer.addLayout(header)
        outer.addLayout(self.body, stretch=1)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        """Add a widget to the panel body and return it."""
        self.body.addWidget(widget, stretch)
        return widget


class Sidebar(QFrame):
    """Branded navigation rail. Emits :attr:`navigated` with the view key."""

    navigated = pyqtSignal(str)

    def __init__(self, items: Sequence[Tuple[str, str, str]]) -> None:
        super().__init__()
        self.setObjectName("Sidebar")
        self.setFixedWidth(228)
        self._buttons: Dict[str, QPushButton] = {}
        self._badges: Dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._brand())

        # The nav list scrolls on short screens so the footer card and the last
        # entries stay reachable instead of being clipped.
        nav = QWidget()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroller.setWidget(nav)

        group = QButtonGroup(self)
        group.setExclusive(True)
        for key, glyph, label in items:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 12, 0)
            row_layout.setSpacing(0)

            button = QPushButton(f"  {glyph}   {label}")
            button.setObjectName("Nav")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked, name=key: self.navigated.emit(name))
            group.addButton(button)

            badge = QLabel("")
            badge.setVisible(False)
            badge.setFixedHeight(19)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(
                f"background-color: {C.DANGER}; color: white; border-radius: 9px;"
                "padding: 0 7px; font-size: 10px; font-weight: 700;")

            row_layout.addWidget(button, stretch=1)
            row_layout.addWidget(badge)
            nav_layout.addWidget(row)

            self._buttons[key] = button
            self._badges[key] = badge

        nav_layout.addStretch(1)
        layout.addWidget(scroller, stretch=1)
        layout.addWidget(self._footer_card())

    def select(self, key: str) -> None:
        """Check the button for ``key`` without emitting a navigation signal."""
        button = self._buttons.get(key)
        if button is not None:
            button.setChecked(True)

    def set_badge(self, key: str, count: int) -> None:
        """Show a count badge beside a nav item (hidden when zero)."""
        badge = self._badges.get(key)
        if badge is None:
            return
        badge.setText(str(count))
        badge.setVisible(count > 0)

    @staticmethod
    def _brand() -> QWidget:
        """Corporate lock-up at the top of the rail."""
        widget = QWidget()
        widget.setStyleSheet(f"border-bottom: 1px solid {C.BORDER};")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(16, 16, 12, 16)
        layout.setSpacing(10)

        mark = QLabel("◧")
        mark.setFixedSize(34, 34)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setStyleSheet(
            f"background-color: {C.BRAND}; color: white; border-radius: 8px;"
            "font-size: 17px; font-weight: 700;")

        name = QLabel("Oil India Limited")
        name.setObjectName("BrandName")
        tagline = QLabel("CONQUERING NEWER HORIZONS")
        tagline.setObjectName("BrandSub")

        text = QVBoxLayout()
        text.setSpacing(0)
        text.setContentsMargins(0, 0, 0, 0)
        text.addWidget(name)
        text.addWidget(tagline)

        layout.addWidget(mark)
        layout.addLayout(text, stretch=1)
        return widget

    @staticmethod
    def _footer_card() -> QWidget:
        """The safety-values card that closes the rail."""
        card = QFrame()
        card.setStyleSheet(
            f"background-color: rgba(34, 197, 94, 0.10); border: 1px solid {C.OK};"
            "border-radius: 10px;")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(2)

        leaf = QLabel("🌿")
        leaf.setStyleSheet("font-size: 17px; border: none;")
        layout.addWidget(leaf)
        for line in ("Safety", "People", "Environment", "Sustainable Growth"):
            label = QLabel(line)
            label.setStyleSheet(f"color: {C.TEXT_DIM}; font-size: 11px; border: none;")
            layout.addWidget(label)

        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(14, 8, 14, 16)
        holder_layout.addWidget(card)
        return holder


class HeaderBar(QFrame):
    """Application header: title block, search, notifications, user chip."""

    search_changed = pyqtSignal(str)

    def __init__(self, title: str, subtitle: str, user_name: str = "",
                 user_role: str = "") -> None:
        super().__init__()
        self.setObjectName("Header")
        self.setFixedHeight(92)

        title_label = QLabel(title)
        title_label.setObjectName("AppTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("AppSubtitle")

        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.setContentsMargins(0, 0, 0, 0)
        titles.addWidget(title_label)
        titles.addWidget(subtitle_label)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search reports, sites, activities...")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(310)
        self.search.textChanged.connect(self.search_changed.emit)

        self.notifications = QLabel("🔔")
        self.notifications.setStyleSheet("font-size: 16px;")

        initials = "".join(part[0] for part in user_name.split()[:2]).upper() or "HSE"
        avatar = QLabel(initials)
        avatar.setFixedSize(34, 34)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setStyleSheet(
            f"background-color: {C.BLUE}; color: white; border-radius: 17px;"
            "font-weight: 700; font-size: 12px;")

        name = QLabel(user_name)
        name.setStyleSheet("font-weight: 600;")
        role = QLabel(user_role)
        role.setObjectName("Faint")
        user_text = QVBoxLayout()
        user_text.setSpacing(0)
        user_text.setContentsMargins(0, 0, 0, 0)
        user_text.addWidget(name)
        user_text.addWidget(role)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setSpacing(14)
        layout.addLayout(titles)
        layout.addStretch(1)
        layout.addWidget(self.search)
        layout.addWidget(self.notifications)
        layout.addWidget(avatar)
        layout.addLayout(user_text)


class DataTable(QTableWidget):
    """Read-only table driven by a ``(header, key, width)`` column schema.

    Cell rendering is centralised here - band colouring, boolean rendering,
    numeric alignment - so every table in the app formats identically.
    """

    CENTRED_KEYS = {"_index", "sif_potential", "risk_score", "p_sif", "reference",
                    "reports", "sif_reports", "sif_rate", "mean_risk", "max_risk",
                    "confidence", "ml_probability", "rule_confidence"}
    BAND_KEYS = {"risk_score", "mean_risk", "max_risk", "risk_band"}

    def __init__(self, columns: Sequence[Tuple[str, str, int]],
                 on_select: Optional[Callable[[int], None]] = None) -> None:
        super().__init__(0, len(columns))
        self._columns = list(columns)
        self.setHorizontalHeaderLabels([label for label, _, _ in columns])
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.setShowGrid(False)
        self.verticalHeader().setDefaultSectionSize(30)
        # Per-pixel scrolling so the styled bars track the content smoothly
        # instead of jumping a whole row (or column) per step.
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        for index, (_, _, width) in enumerate(columns):
            self.setColumnWidth(index, width)
        if on_select is not None:
            self.itemSelectionChanged.connect(
                lambda: on_select(self.currentRow()) if self.currentRow() >= 0 else None)

    def set_rows(self, payloads: Sequence[Dict[str, object]]) -> None:
        """Replace all rows."""
        self.setRowCount(0)
        for payload in payloads:
            self.append_row(payload)

    def append_row(self, payload: Dict[str, object]) -> None:
        """Append one row, formatted per the shared conventions."""
        row = self.rowCount()
        self.insertRow(row)
        for column, (_, key, _) in enumerate(self._columns):
            item = QTableWidgetItem(self._text(payload, key, row + 1))
            tooltip = payload.get("explanation") or payload.get("reason") or ""
            if tooltip:
                item.setToolTip(str(tooltip))
            if key in self.CENTRED_KEYS or key in self.BAND_KEYS:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._colour(item, payload, key)
            self.setItem(row, column, item)

    # -- formatting --------------------------------------------------------

    @staticmethod
    def _text(payload: Dict[str, object], key: str, index: int) -> str:
        if key == "_index":
            return str(index)
        value = payload.get(key, "")
        if key == "sif_potential":
            return "YES" if value else "no"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if key in {"p_sif", "confidence", "rule_confidence", "ml_probability"}:
            return "-" if value in (None, "") else f"{float(value):.2f}"
        if key in {"risk_score", "mean_risk", "max_risk", "sif_rate"}:
            return "" if value in (None, "") else f"{float(value):.1f}"
        if key in {"reference", "review_trigger", "trigger"}:
            return str(value or "-")
        if key == "raw_text" or key == "summary":
            text = str(value)
            return text if len(text) <= 260 else text[:257] + "..."
        return str(value)

    @staticmethod
    def _colour(item: QTableWidgetItem, payload: Dict[str, object], key: str) -> None:
        if key == "sif_potential":
            item.setForeground(QColor(C.DANGER if payload.get("sif_potential") else C.OK))
            font = QFont()
            font.setBold(True)
            item.setFont(font)
        elif key in DataTable.BAND_KEYS:
            band = payload.get("risk_band")
            if not band:
                value = float(payload.get(key) or 0.0)
                band = ("Critical" if value >= 70 else "High" if value >= 50
                        else "Medium" if value >= 30 else "Low")
            item.setForeground(QColor(BAND_COLORS.get(str(band), C.OK)))
        elif key in {"review_trigger", "trigger"}:
            item.setForeground(QColor(C.WARN if payload.get(key) else C.TEXT_FAINT))
        elif key == "level":
            level = str(payload.get("level", ""))
            item.setForeground(QColor({"ERROR": C.DANGER, "CRITICAL": C.DANGER,
                                       "WARNING": C.WARN, "DEBUG": C.TEXT_FAINT}
                                      .get(level, C.INFO)))


class FieldRow(QWidget):
    """A ``label: value`` line used by the report detail panel."""

    def __init__(self, glyph: str, label: str, value: str = "-",
                 accent: str = C.TEXT_DIM) -> None:
        super().__init__()
        self._value = QLabel(value)
        self._value.setWordWrap(False)
        self._value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._value.setStyleSheet(f"color: {C.TEXT}; font-weight: 600;")
        self._full_value = value

        icon = QLabel(glyph)
        icon.setFixedWidth(18)
        icon.setStyleSheet(f"color: {accent};")

        name = QLabel(label)
        name.setObjectName("Muted")
        name.setFixedWidth(112)

        self.setMinimumHeight(26)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)
        layout.addWidget(icon)
        layout.addWidget(name)
        layout.addWidget(self._value, stretch=1)

    def set_value(self, value: str) -> None:
        """Update the right-hand value, eliding to one line with a full tooltip."""
        self._full_value = value
        self._value.setToolTip(value)
        self._elide()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._elide()
        super().resizeEvent(event)

    def _elide(self) -> None:
        metrics = self._value.fontMetrics()
        width = max(self._value.width(), 60)
        self._value.setText(
            metrics.elidedText(self._full_value, Qt.TextElideMode.ElideRight, width))
