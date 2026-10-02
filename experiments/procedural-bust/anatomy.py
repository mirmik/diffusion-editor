"""An original, deliberately stylised study; no imported models or textures.

X is left/right, -Y is forward, Z is up. The head has its own local frame.
"""
import numpy as np
from sdf import Sculpt, ellipsoid, loft, smooth_min


def head_frame(p, params):
    yaw = np.deg2rad(params['head_yaw_degrees'])
    c, s = np.cos(yaw), np.sin(yaw)
    return ((c * p[0] + s * p[1]) / 1.08, -s * p[0] + c * p[1], p[2] - params['neck_extension'] + .035)


def field(p, params):
    sw = params['shoulder_width']
    neck = params['neck_extension']
    q = (p[0] / sw, p[1], p[2])
    body = Sculpt(q)
    body.add(loft(q, [
        (.055, .095, .064, .055, .012),
        (.10, .112, .066, .063, .012),
        (.18, .155, .076, .072, .012),
        (.25, .193, .082, .077, .016),
        (.30, .191, .071, .073, .021),
        (.335, .150, .060, .064, .027),
        (.36, .106, .043, .058, .025),
        (.395, .050, .039, .047, .025),
    ]), .014)
    for side in (-1, 1):
        body.ell((side * .183, .017, .287), (.057, .058, .066), .033)
        body.ell((side * .090, -.043, .270), (.089, .031, .044), .030)
        body.muscle((side * .033, .026, .414 + neck * .6), (side * .172, .032, .312), .040, .050, .030)
        body.muscle((side * .015, -.042, .322), (side * .155, -.025, .326), .007, .008, .014)
        body.ell((side * .092, .065, .263), (.081, .029, .070), .025)
    # Neck is not widened with the shoulder parameter.
    n = Sculpt(p)
    n.muscle((0, .016, .318), (0, .018, .505 + neck), .049, .052, .018)
    for side in (-1, 1):
        n.muscle((side * .043, .004, .465 + neck), (side * .017, -.038, .337), .009, .010, .014)
    n.ell((0, -.029, .395 + neck * .4), (.014, .010, .020), .012)
    body.add(n.field, .022)
    # Shallow suprasternal notch and pectoral separation.
    body.cut(ellipsoid(p, (0, -.064, .329), (.015, .017, .013)), .005)
    body.cut(ellipsoid(p, (0, -.091, .246), (.006, .012, .052)), .003)

    h = head_frame(p, params)
    head = Sculpt(h)
    jaw = params['jaw_width']
    head.add(loft(h, [
        (.485, .019 * jaw, .027, .018, -.016),
        (.500, .041 * jaw, .041, .031, -.007),
        (.520, .055 * jaw, .049, .047, -.003),
        (.546, .062 * jaw, .057, .062, .000),
        (.578, .067, .062, .076, .001),
        (.613, .073, .062, .083, .002),
        (.647, .075, .064, .082, .002),
        (.676, .069, .058, .073, .003),
        (.700, .050, .042, .055, .005),
        (.714, .022, .019, .027, .008),
        (.718, .002, .002, .002, .009),
    ]), .008)
    for side in (-1, 1):
        head.ell((side * .048, -.032, .586), (.026, .022, .018), .018)
        head.muscle((side * .012, -.056, .633), (side * .062, -.036, .631), .009, .010, .011)
        head.ell((side * .047 * jaw, -.006, .526), (.016, .028, .026), .019)
        # External ears and their concha; the ear joins the skull before cutting.
        head.ell((side * .077, .005, .594), (.015, .020, .036), .006)
        head.cut(ellipsoid(h, (side * .086, -.009, .598), (.008, .014, .022)), .002)
        head.ell((side * .082, -.008, .577), (.008, .010, .011), .003)
        # Orbital socket; eyeballs are separate explicit objects.
        head.ell((side * .032, -.050, .613), (.024, .024, .018), .011)
        head.cut(ellipsoid(h, (side * .032, -.080, .613), (.020, .025, .006)), .0015)
    # Nose: bridge, tip, wings, then nostrils.
    head.ell((0, -.057, .602), (.014, .018, .033), .010)
    head.muscle((0, -.060, .634), (0, -.080, .580), .010, .010, .009)
    head.ell((0, -.081, .581), (.011, .016, .010), .006)
    for side in (-1, 1):
        head.ell((side * .011, -.071, .575), (.009, .013, .007), .005)
        head.cut(ellipsoid(h, (side * .010, -.081, .572), (.0035, .005, .003)), .001)
    # Muzzle, lips and chin; the mouth is a shallow subtractive crease.
    head.ell((0, -.049, .546), (.030, .017, .021), .010)
    head.ell((0, -.052, .516), (.025 * jaw, .014, .017), .014)
    for side in (-1, 1):
        head.ell((side * .009, -.065, .552), (.014, .007, .0045), .003)
    head.ell((0, -.065, .542), (.021, .008, .005), .003)
    head.cut(ellipsoid(h, (0, -.070, .548), (.023, .008, .0017)), .0008)
    head.cut(ellipsoid(h, (0, -.070, .532), (.016, .005, .002)), .001)
    return smooth_min(body.field, head.field, .014).astype(np.float32)


def head_to_world(point, params):
    yaw = np.deg2rad(params['head_yaw_degrees'])
    c, s = np.cos(yaw), np.sin(yaw)
    x, y, z = point
    x *= 1.08
    return (c * x - s * y, s * x + c * y, z + params['neck_extension'] - .035)
