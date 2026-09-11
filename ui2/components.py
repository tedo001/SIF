"""Emoji-free navigation and header for the second build.

Everything else - panels, KPI tiles, tables, charts - is reused from :mod:`ui`
unchanged; only the two widgets that carried pictographs are replaced here.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from PyQt6.QtCore import QSize, Qt, pyqtSignal
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

from ui.theme import PAGE_MARGIN, C
from ui2.icons import nav_icon

__all__ = ["Sidebar", "HeaderBar", "scrollable", "titled"]


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
    """Navigation rail. Items are ``(key, label)``; icons come from the key."""

    navigated = pyqtSignal(str)

    def __init__(self, items: Sequence[Tuple[str, str]]) -> None:
        super().__init__()
        self.setObjectName("Sidebar")
        # A range rather than a fixed width, so the rail can be dragged wider
        # for long page names or narrower to buy the tables room. The floor
        # keeps every label readable; the ceiling stops it eating the content.
        self.setMinimumWidth(212)
        self.setMaximumWidth(330)
        self.resize(238, self.height())
        self._buttons: Dict[str, QPushButton] = {}
        self._badges: Dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

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

            button = QPushButton(f"  {label}")
            button.setObjectName("Nav")
            button.setCheckable(True)
            button.setIcon(nav_icon(key))
            button.setIconSize(QSize(18, 18))
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


def titled(page: QWidget, title: str, subtitle: str) -> QWidget:
    """Put a page behind its own heading.

    The header band names the operator's organisation, not the current page, so
    each page says what it is here instead. Pages that already open with their
    own heading - the workflow map, the review bench - are passed through
    untouched rather than given a second one.
    """
    if not title:
        return page

    heading = QLabel(title)
    heading.setObjectName("PageTitle")
    caption = QLabel(subtitle)
    caption.setObjectName("Muted")
    caption.setWordWrap(True)

    # The heading carries the page margin and the page keeps its own, rather
    # than the wrapper indenting both: nesting one inside the other put the
    # heading a full margin to the left of the panels it belongs to.
    head = QWidget()
    head_layout = QVBoxLayout(head)
    head_layout.setContentsMargins(PAGE_MARGIN, 14, PAGE_MARGIN, 0)
    head_layout.setSpacing(1)
    head_layout.addWidget(heading)
    head_layout.addWidget(caption)

    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(head)
    layout.addWidget(page, stretch=1)
    return wrapper


def _divider() -> QLabel:
    """A thin vertical rule between header items."""
    rule = QLabel()
    rule.setFixedSize(1, 20)
    rule.setStyleSheet(f"background-color: {C.BORDER};")
    return rule


class HeaderBar(QFrame):
    """Full-width header: the product mark, the operator's identity, search.

    It spans the whole window rather than only the content column, so the
    product is named once at the top and the rail below it carries nothing but
    navigation. :meth:`set_rail_width` keeps the mark's block exactly as wide as
    that rail, so the two share one vertical edge however the rail is dragged.
    """

    search_changed = pyqtSignal(str)

    def __init__(self, product: str, title: str, subtitle: str, user_name: str = "",
                 user_role: str = "") -> None:
        super().__init__()
        self.setObjectName("Header")
        self.setFixedHeight(58)

        self.brand = QWidget()
        self.brand.setObjectName("HeaderBrand")
        brand_layout = QHBoxLayout(self.brand)
        # 22 = the nav pill's 8px margin plus its 14px padding, so the mark sits
        # on the same vertical line as the icons beneath it.
        brand_layout.setContentsMargins(22, 0, 12, 0)
        brand_layout.setSpacing(10)

        # "S" rather than a glyph: an ASCII letter is on every machine, which a
        # pictograph is not, and this mark has to survive a plant workstation.
        self.mark = QLabel("S")
        mark = self.mark
        mark.setFixedSize(30, 30)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setStyleSheet(
            f"background-color: {C.ACCENT}; color: white; border-radius: 8px;"
            "font-size: 16px; font-weight: 700;")
        product_label = QLabel(product)
        product_label.setObjectName("BrandName")
        brand_layout.addWidget(mark)
        brand_layout.addWidget(product_label)
        brand_layout.addStretch(1)

        organisation = QLabel(title)
        organisation.setObjectName("BrandName")
        context_label = QLabel(subtitle)
        context_label.setObjectName("Muted")

        titles = QHBoxLayout()
        titles.setSpacing(10)
        titles.setContentsMargins(PAGE_MARGIN, 0, 0, 0)
        titles.addWidget(organisation)
        titles.addWidget(_divider())
        titles.addWidget(context_label)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search reports, sites, activities")
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(180)
        self.search.setMaximumWidth(420)
        self.search.textChanged.connect(self.search_changed.emit)

        self.engine_label = QLabel("Engines: starting")
        self.engine_label.setObjectName("Faint")
        self.engine_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        initials = "".join(part[0] for part in user_name.split()[:2]).upper() or "HSE"
        self.avatar = QLabel(initials)
        avatar = self.avatar
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
        layout.setContentsMargins(0, 10, PAGE_MARGIN, 10)
        layout.setSpacing(14)
        layout.addWidget(self.brand)
        layout.addLayout(titles)
        layout.addStretch(1)
        layout.addWidget(self.engine_label)
        layout.addWidget(self.search)
        layout.addWidget(avatar)
        layout.addLayout(user_text)

    def set_rail_width(self, width: int) -> None:
        """Match the brand block to the navigation rail beneath it."""
        self.brand.setFixedWidth(max(width, 0))

    def set_engines(self, text: str) -> None:
        """Show which engines are live, in the header."""
        self.engine_label.setText(text)
