"""Emoji-free navigation and header for the second build.

Everything else - panels, KPI tiles, tables, charts - is reused from :mod:`ui`
unchanged; only the two widgets that carried pictographs are replaced here.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.theme import C

__all__ = ["Sidebar", "HeaderBar", "scrollable"]


def scrollable(widget: QWidget, minimum_height: int = 0) -> QScrollArea:
    """Wrap a page so it scrolls rather than compressing on a short screen.

    Lives here rather than in :mod:`ui2.views` so every page in the build can
    reach it - the workflow map and the review bench need it as much as the
    dashboard does. The scrollbars themselves are styled once, globally, in
    :mod:`ui.theme`, so a wrapped page gets the same stepper arrows as the
    tables without asking for them.

    ``minimum_height`` is the height below which the content stops shrinking and
    the bar appears instead; leave it at zero for content that has no natural
    floor.
    """
    if minimum_height:
        widget.setMinimumHeight(minimum_height)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setWidget(widget)
    return area


class Sidebar(QFrame):
    """Navigation rail. Items are ``(key, label)`` - no icons, by design."""

    navigated = pyqtSignal(str)

    def __init__(self, items: Sequence[Tuple[str, str]]) -> None:
        super().__init__()
        self.setObjectName("Sidebar")
        self.setFixedWidth(238)
        self._buttons: Dict[str, QPushButton] = {}
        self._badges: Dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._brand())

        nav = QWidget()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 6, 0, 6)
        nav_layout.setSpacing(0)

        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroller.setWidget(nav)

        group = QButtonGroup(self)
        group.setExclusive(True)
        for key, label in items:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 12, 0)
            row_layout.setSpacing(0)

            button = QPushButton(f"   {label}")
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
        layout.addWidget(self._footer())

    def select(self, key: str) -> None:
        """Check a nav item without emitting a navigation signal."""
        button = self._buttons.get(key)
        if button is not None:
            button.setChecked(True)

    def set_badge(self, key: str, count: int) -> None:
        """Show a count beside a nav item; hidden at zero."""
        badge = self._badges.get(key)
        if badge is None:
            return
        badge.setText(str(count))
        badge.setVisible(count > 0)

    @staticmethod
    def _brand() -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(f"border-bottom: 1px solid {C.BORDER};")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(16, 16, 12, 16)
        layout.setSpacing(10)

        mark = QLabel("OIL")
        mark.setFixedSize(38, 34)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setStyleSheet(
            f"background-color: {C.BRAND}; color: white; border-radius: 8px;"
            "font-size: 12px; font-weight: 700; letter-spacing: 1px;")

        name = QLabel("Oil India Limited")
        name.setObjectName("BrandName")
        tagline = QLabel("SIF INSIGHT CONSOLE  ·  PS 26165")
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
    def _footer() -> QWidget:
        card = QFrame()
        card.setStyleSheet(
            f"background-color: rgba(34, 197, 94, 0.10); border: 1px solid {C.OK};"
            "border-radius: 10px;")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        title = QLabel("SAFETY FIRST")
        title.setStyleSheet(f"color: {C.OK}; font-size: 10px; font-weight: 700;"
                            "letter-spacing: 1px; border: none;")
        layout.addWidget(title)
        for line in ("No report is closed by the engine.",
                     "Every flag carries its evidence."):
            label = QLabel(line)
            label.setWordWrap(True)
            label.setStyleSheet(f"color: {C.TEXT_DIM}; font-size: 10.5px; border: none;")
            layout.addWidget(label)

        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(14, 8, 14, 16)
        holder_layout.addWidget(card)
        return holder


class HeaderBar(QFrame):
    """Application header: titles, search, and the live engine state."""

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
        self.search.setPlaceholderText("Search reports, sites, activities")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(300)
        self.search.textChanged.connect(self.search_changed.emit)

        self.engine_label = QLabel("Engines: starting")
        self.engine_label.setObjectName("Faint")
        self.engine_label.setAlignment(Qt.AlignmentFlag.AlignRight)

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
        layout.addWidget(self.engine_label)
        layout.addWidget(self.search)
        layout.addWidget(avatar)
        layout.addLayout(user_text)

    def set_engines(self, text: str) -> None:
        """Show which engines are live, in the header."""
        self.engine_label.setText(text)
