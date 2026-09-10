"""Navigation icons, drawn rather than typed.

An earlier build carried pictographs in the nav labels and they arrived on a
plant workstation as empty boxes: the machine had no emoji font, and no amount
of styling fixes a glyph that is not installed. These are QPainter paths, so
they render identically on any machine with Qt and cannot fall back to
anything.

Each icon is drawn inside an 18x18 logical box with a rounded 1.6px stroke, and
is produced twice - dim for a resting item, white for the selected one - so the
filled pill in :mod:`ui.theme` gets an icon that stays legible on top of it.
"""

from __future__ import annotations

from typing import Callable, Dict

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

from ui.theme import C

__all__ = ["nav_icon", "ICON_NAMES"]

BOX = 18.0
STROKE = 1.6
#: Drawn at this multiple and tagged with the matching device-pixel ratio, so
#: the stroke stays crisp on a high-DPI display instead of blurring.
SCALE = 2


def _workflow(paint: QPainter) -> None:
    paint.drawRoundedRect(QRectF(2.0, 2.5, 6.5, 4.5), 1.5, 1.5)
    paint.drawRoundedRect(QRectF(9.5, 11.0, 6.5, 4.5), 1.5, 1.5)
    paint.drawLine(QPointF(5.2, 7.0), QPointF(5.2, 13.2))
    paint.drawLine(QPointF(5.2, 13.2), QPointF(9.5, 13.2))


def _ingest(paint: QPainter) -> None:
    paint.drawLine(QPointF(9.0, 2.5), QPointF(9.0, 11.0))
    paint.drawLine(QPointF(5.8, 5.7), QPointF(9.0, 2.5))
    paint.drawLine(QPointF(9.0, 2.5), QPointF(12.2, 5.7))
    paint.drawLine(QPointF(3.0, 15.0), QPointF(15.0, 15.0))
    paint.drawLine(QPointF(3.0, 11.5), QPointF(3.0, 15.0))
    paint.drawLine(QPointF(15.0, 11.5), QPointF(15.0, 15.0))


def _dashboard(paint: QPainter) -> None:
    for x, y in ((2.5, 2.5), (10.0, 2.5), (2.5, 10.0), (10.0, 10.0)):
        paint.drawRoundedRect(QRectF(x, y, 5.5, 5.5), 1.4, 1.4)


def _reports(paint: QPainter) -> None:
    paint.drawRoundedRect(QRectF(3.5, 2.0, 11.0, 14.0), 1.6, 1.6)
    for y in (6.0, 9.0, 12.0):
        paint.drawLine(QPointF(6.0, y), QPointF(12.0, y))


def _hotspots(paint: QPainter) -> None:
    paint.drawEllipse(QRectF(3.0, 3.0, 12.0, 12.0))
    paint.drawEllipse(QRectF(7.0, 7.0, 4.0, 4.0))
    paint.drawLine(QPointF(9.0, 1.2), QPointF(9.0, 3.4))
    paint.drawLine(QPointF(9.0, 14.6), QPointF(9.0, 16.8))


def _review(paint: QPainter) -> None:
    paint.drawEllipse(QRectF(4.0, 2.5, 5.5, 5.5))
    paint.drawArc(QRectF(2.0, 9.0, 9.5, 9.0), 20 * 16, 140 * 16)
    paint.drawLine(QPointF(10.8, 13.0), QPointF(12.6, 14.8))
    paint.drawLine(QPointF(12.6, 14.8), QPointF(16.2, 10.4))


def _analytics(paint: QPainter) -> None:
    paint.drawLine(QPointF(2.5, 15.2), QPointF(15.5, 15.2))
    for x, top in ((4.2, 10.0), (8.2, 5.5), (12.2, 8.0)):
        paint.drawLine(QPointF(x, 15.2), QPointF(x, top))


def _engines(paint: QPainter) -> None:
    paint.drawRoundedRect(QRectF(5.0, 5.0, 8.0, 8.0), 1.5, 1.5)
    for offset in (7.0, 9.0, 11.0):
        paint.drawLine(QPointF(offset, 2.4), QPointF(offset, 5.0))
        paint.drawLine(QPointF(offset, 13.0), QPointF(offset, 15.6))
        paint.drawLine(QPointF(2.4, offset), QPointF(5.0, offset))
        paint.drawLine(QPointF(13.0, offset), QPointF(15.6, offset))


def _settings(paint: QPainter) -> None:
    paint.drawEllipse(QRectF(5.5, 5.5, 7.0, 7.0))
    for x1, y1, x2, y2 in ((9.0, 2.6, 9.0, 4.4), (9.0, 13.6, 9.0, 15.4),
                           (2.6, 9.0, 4.4, 9.0), (13.6, 9.0, 15.4, 9.0),
                           (4.6, 4.6, 5.8, 5.8), (12.2, 12.2, 13.4, 13.4),
                           (13.4, 4.6, 12.2, 5.8), (5.8, 12.2, 4.6, 13.4)):
        paint.drawLine(QPointF(x1, y1), QPointF(x2, y2))


#: Nav key -> the routine that draws it.
PAINTERS: Dict[str, Callable[[QPainter], None]] = {
    "workflow": _workflow,
    "ingest": _ingest,
    "dashboard": _dashboard,
    "reports": _reports,
    "hotspots": _hotspots,
    "review": _review,
    "analytics": _analytics,
    "engines": _engines,
    "settings": _settings,
}

ICON_NAMES = tuple(PAINTERS)


def _pixmap(key: str, colour: str) -> QPixmap:
    """One icon, stroked in ``colour`` on a transparent ground."""
    pixmap = QPixmap(int(BOX * SCALE), int(BOX * SCALE))
    pixmap.fill(Qt.GlobalColor.transparent)
    pixmap.setDevicePixelRatio(SCALE)

    # No explicit scale(): a QPainter on a pixmap already works in logical
    # coordinates once the device-pixel ratio is set, so scaling here again
    # draws at double size and only the top-left quarter of the icon survives.
    paint = QPainter(pixmap)
    paint.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(colour))
    pen.setWidthF(STROKE)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    paint.setPen(pen)
    paint.setBrush(Qt.BrushStyle.NoBrush)
    PAINTERS[key](paint)
    paint.end()
    return pixmap


def nav_icon(key: str) -> QIcon:
    """An icon for one nav key, carrying both its resting and selected forms.

    Qt picks ``State.On`` for a checked button, which is exactly the item
    wearing the filled pill, so the white stroke arrives without the sidebar
    having to swap icons when the page changes.
    """
    icon = QIcon()
    if key not in PAINTERS:
        return icon
    icon.addPixmap(_pixmap(key, C.TEXT_DIM), QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(_pixmap(key, C.TEXT), QIcon.Mode.Active, QIcon.State.Off)
    icon.addPixmap(_pixmap(key, "#ffffff"), QIcon.Mode.Normal, QIcon.State.On)
    icon.addPixmap(_pixmap(key, "#ffffff"), QIcon.Mode.Active, QIcon.State.On)
    return icon
