"""Painted chart widgets - no charting dependency, no rasterised images.

Two forms cover everything the dashboard needs:

``HBarChart``
    Ranked magnitude by category (exposure per IOGP rule, failed barriers).
    Horizontal, because the category labels are long safety phrases that would
    be unreadable rotated under a vertical axis. Bars carry a second, optional
    segment so a total can show its SIF-potential share in place, and each bar
    is value-labelled, so identity never rests on colour alone.

``DonutChart``
    Composition of a small, fixed set of categories (the energy mix) with the
    total in the middle and a value-labelled legend beside it - the legend, not
    the ring, is what people actually read the numbers from.

Both paint with ``QPainter``: they stay crisp at any DPI, follow the theme, and
add no wheels to the install.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from .theme import CATEGORICAL, C

__all__ = ["HBarChart", "DonutChart"]


class HBarChart(QWidget):
    """Ranked horizontal bars, optionally split into a highlighted share.

    Data is a list of ``(label, total, highlighted)`` triples where
    ``highlighted <= total``; pass ``highlighted = 0`` for a plain bar.
    """

    LABEL_WIDTH = 168
    ROW_HEIGHT = 22
    BAR_HEIGHT = 13
    VALUE_WIDTH = 46

    def __init__(self, highlight_color: str = C.DANGER, base_color: str = C.BLUE,
                 max_rows: int = 10, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rows: List[Tuple[str, float, float]] = []
        self._highlight = QColor(highlight_color)
        self._base = QColor(base_color)
        self._max_rows = max_rows
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(120)

    def set_data(self, rows: Sequence[Tuple[str, float, float]]) -> None:
        """Replace the chart's data and repaint."""
        self._rows = list(rows)[: self._max_rows]
        self.updateGeometry()
        self.update()

    def sizeHint(self):  # noqa: D102 - Qt API
        from PyQt6.QtCore import QSize

        return QSize(320, max(len(self._rows), 1) * self.ROW_HEIGHT + 10)

    def paintEvent(self, event) -> None:  # noqa: N802, D102 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(C.PANEL))

        if not self._rows:
            painter.setPen(QColor(C.TEXT_FAINT))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data yet - analyse a report")
            painter.end()
            return

        font = QFont(self.font())
        font.setPointSizeF(8.2)
        painter.setFont(font)

        label_width = min(self.LABEL_WIDTH, int(self.width() * 0.42))
        plot_left = label_width + 10
        plot_width = max(self.width() - plot_left - self.VALUE_WIDTH, 30)
        peak = max((total for _, total, _ in self._rows), default=1.0) or 1.0
        row_height = min(self.ROW_HEIGHT,
                         max(self.height() / max(len(self._rows), 1), 16.0))

        for index, (label, total, highlighted) in enumerate(self._rows):
            top = index * row_height
            middle = top + row_height / 2

            painter.setPen(QColor(C.TEXT_DIM))
            painter.drawText(QRectF(0, top, label_width, row_height),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                             self._elide(painter, label, label_width))

            full = max(total / peak * plot_width, 2.0)
            bar_top = middle - self.BAR_HEIGHT / 2
            painter.setPen(Qt.PenStyle.NoPen)
            if highlighted > 0:
                share = max(highlighted / peak * plot_width, 2.0)
                painter.setBrush(self._highlight)
                painter.drawRoundedRect(QRectF(plot_left, bar_top, share, self.BAR_HEIGHT), 3, 3)
                if total > highlighted:
                    painter.setBrush(self._base)
                    painter.drawRoundedRect(
                        QRectF(plot_left + share + 2, bar_top, full - share - 2,
                               self.BAR_HEIGHT), 3, 3)
            else:
                painter.setBrush(self._base)
                painter.drawRoundedRect(QRectF(plot_left, bar_top, full, self.BAR_HEIGHT), 3, 3)

            painter.setPen(QColor(C.TEXT))
            painter.drawText(
                QRectF(plot_left + full + 6, top, self.VALUE_WIDTH, row_height),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"{total:g}")
        painter.end()

    @staticmethod
    def _elide(painter: QPainter, text: str, width: float) -> str:
        metrics = painter.fontMetrics()
        return metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(width))


class DonutChart(QWidget):
    """Composition ring with a centred total and a value-labelled legend."""

    RING_THICKNESS = 26
    LEGEND_ROW = 19

    def __init__(self, centre_caption: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rows: List[Tuple[str, float]] = []
        self._caption = centre_caption
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(190)

    def set_data(self, rows: Sequence[Tuple[str, float]]) -> None:
        """Replace the chart's data (``(label, value)`` pairs) and repaint."""
        self._rows = [(label, float(value)) for label, value in rows if value > 0]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802, D102 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(C.PANEL))

        if not self._rows:
            painter.setPen(QColor(C.TEXT_FAINT))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data yet - analyse a report")
            painter.end()
            return

        total = sum(value for _, value in self._rows) or 1.0
        size = min(self.height() - 12, self.width() * 0.44, 190.0)
        ring = QRectF(8, (self.height() - size) / 2, size, size)

        start = 90 * 16
        for index, (_, value) in enumerate(self._rows):
            span = int(-value / total * 360 * 16)
            colour = QColor(CATEGORICAL[index % len(CATEGORICAL)])
            pen = QPen(colour, self.RING_THICKNESS)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            inset = self.RING_THICKNESS / 2
            painter.drawArc(ring.adjusted(inset, inset, -inset, -inset), start, span)
            start += span

        font = QFont(self.font())
        font.setPointSizeF(15)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(C.TEXT))
        painter.drawText(ring, Qt.AlignmentFlag.AlignCenter, f"{total:g}")
        if self._caption:
            font.setPointSizeF(7.6)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QColor(C.TEXT_DIM))
            painter.drawText(ring.adjusted(0, 26, 0, 26), Qt.AlignmentFlag.AlignCenter,
                             self._caption)

        legend_left = ring.right() + 18
        legend_width = max(self.width() - legend_left - 8, 60)
        rows = self._rows[:9]
        legend_top = max((self.height() - len(rows) * self.LEGEND_ROW) / 2, 4)
        font.setPointSizeF(8.2)
        painter.setFont(font)
        for index, (label, value) in enumerate(rows):
            top = legend_top + index * self.LEGEND_ROW
            colour = QColor(CATEGORICAL[index % len(CATEGORICAL)])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colour)
            painter.drawEllipse(QRectF(legend_left, top + 5, 8, 8))
            painter.setPen(QColor(C.TEXT_DIM))
            text_rect = QRectF(legend_left + 15, top, legend_width - 46, self.LEGEND_ROW)
            painter.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                painter.fontMetrics().elidedText(label, Qt.TextElideMode.ElideRight,
                                                 int(text_rect.width())))
            painter.setPen(QColor(C.TEXT))
            painter.drawText(
                QRectF(legend_left + legend_width - 30, top, 26, self.LEGEND_ROW),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                f"{value:g}")
        painter.end()
