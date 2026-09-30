"""App-owned vector glyphs for the permanent native tool rail."""

from termin.gui_native import Point, Rect, SrgbColor, Widget


class ToolGlyph(Widget):
    def __init__(self, key, model, command_id):
        self.key = key
        self.model = model
        self.command_id = command_id

    def paint(self, context):
        enabled = self.model.command(self.command_id).data.enabled
        color = SrgbColor(.92, .94, .97, 1 if enabled else .4)
        paint_tool_icon(self.key, context, self.bounds, color)


def paint_tool_icon(key, context, bounds, color):
    # Logical geometry, scaled by the UI renderer at the display density.
    extent = min(24, bounds.width, bounds.height)
    scale = extent / 24
    x0 = bounds.x + (bounds.width - extent) / 2
    y0 = bounds.y + (bounds.height - extent) / 2

    def point(x, y):
        return Point(x0 + x * scale, y0 + y * scale)

    def line(points, width=1.6):
        context.draw_polyline([point(x, y) for x, y in points], color, width * scale)

    def rect(box, width=1.5):
        x, y, right, bottom = box
        context.stroke_rect(Rect(x0 + x * scale, y0 + y * scale,
                                 (right - x) * scale, (bottom - y) * scale),
                            color, width * scale)

    def ellipse(box):
        x, y, right, bottom = box
        context.stroke_circle(point((x + right) / 2, (y + bottom) / 2),
                              (right - x) * scale / 2, color, 1.5 * scale)

    def brush():
        line([(10, 14), (18, 3), (21, 6), (13, 17), (10, 14)])
        line([(10, 14), (7, 14), (5, 17), (3, 20), (8, 21), (12, 18), (10, 14)])

    def erase():
        line([(3, 15), (13, 4), (21, 11), (12, 21), (9, 21), (3, 15)])
        line([(8, 10), (16, 17)])
        line([(9, 21), (22, 21)])

    def selection():
        for a, b in ((3, 6), (10, 14), (18, 21)):
            line([(a, 3), (b, 3)])
            line([(a, 21), (b, 21)])
            line([(3, a), (3, b)])
            line([(21, a), (21, b)])

    if key == "paint":
        brush()
    elif key == "eraser":
        erase()
    elif key == "smudge":
        line([(5, 16), (5, 10), (8, 10), (8, 5), (11, 4), (12, 9),
              (15, 8), (17, 10), (20, 10), (21, 13), (18, 20), (10, 21), (5, 16)])
        line([(2, 18), (5, 21)])
    elif key == "select_rect":
        selection()
    elif key == "select_brush":
        selection()
        line([(7, 16), (15, 7), (18, 10), (10, 19), (6, 20), (7, 16)])
    elif key == "move":
        line([(12, 2), (12, 22)])
        line([(2, 12), (22, 12)])
        for points in ([(8, 6), (12, 2), (16, 6)], [(8, 18), (12, 22), (16, 18)],
                       [(6, 8), (2, 12), (6, 16)], [(18, 8), (22, 12), (18, 16)]):
            line(points)
    elif key == "transform":
        rect((5, 5, 19, 19))
        for x, y in ((3, 3), (17, 3), (3, 17), (17, 17)):
            rect((x, y, x + 4, y + 4))
        line([(8, 16), (16, 8), (12, 8)])
        line([(16, 8), (16, 12)])
    elif key in ("mask", "mask_eraser"):
        ellipse((3, 3, 21, 21))
        for x in (6, 9, 12):
            line([(x, 7), (x, 17)])
        line([(14, 12), (20, 12)])
        if key == "mask":
            line([(17, 9), (17, 15)])
    elif key == "patch":
        rect((3, 3, 21, 21))
        line([(8, 3), (8, 21)], 1)
        line([(16, 3), (16, 21)], 1)
        line([(3, 8), (21, 8)], 1)
        line([(3, 16), (21, 16)], 1)
    else:
        raise ValueError(f"unknown tool icon: {key}")
