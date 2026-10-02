"""Vaan proportions estimated from the front and both generated profiles.

Front is -Y; height is about 1.83 m. A-pose is baked into the construction.
"""
import numpy as np
from geometry import Sculpt, ellipsoid, loft, tapered_segment


def fingers(side):
    across = np.array([.925*side, 0, .380])
    down = np.array([.380*side, -.02, -.925])
    for i, (offset, length) in enumerate(zip((-.024,-.008,.009,.024),(.071,.083,.077,.060))):
        root = np.array([.493*side, -.011, .975]) + offset * across
        tip = root + length*down + np.array([side*offset*.18, -.006, 0])
        yield i, root, tip


def body_field(p):
    s = Sculpt(p)
    abdominal=np.zeros(np.broadcast_shapes(p[0].shape,p[2].shape),dtype=np.float32)
    for z in (1.313,1.259,1.207):
        abdominal+=.0045*np.exp(-((np.abs(p[0])-.029)/.028)**2-((p[2]-z)/.026)**2)
    abdominal-=.0012*np.exp(-(p[0]/.0035)**2-((p[2]-1.26)/.082)**4)
    abdomen=(p[0],p[1]+abdominal*np.clip(-p[1]/.035,0,1),p[2])
    s.add(loft(abdomen, [
        (1.010,.114,.072,.071,.008),
        (1.070,.121,.074,.074,.008),
        (1.145,.108,.068,.074,.008),
        (1.230,.116,.075,.080,.010),
        (1.320,.138,.081,.082,.010),
        (1.420,.158,.080,.078,.013),
        (1.460,.151,.062,.064,.016),
        (1.500,.082,.043,.047,.019),
        (1.515,.048,.036,.040,.019),
    ]), .012)
    s.add(tapered_segment(p,(0,.016,1.477),(0,.027,1.626),.043,.036), .014)
    for side in (-1,1):
        s.ell((side*.105,.014,1.479),(.055,.044,.035),.020)
        s.ell((side*.067,-.053,1.398),(.068,.024,.045),.019)
        s.muscle((side*.020,-.015,1.57),(side*.018,-.025,1.485),.007,.007,.016)
        s.muscle((side*.016,-.028,1.483),(side*.143,.002,1.477),.008,.010,.016)
        s.muscle((side*.085,-.037,1.272),(side*.109,-.012,1.081),.018,.022,.017)
        shoulder=(side*.170,.012,1.465)
        elbow=(side*.292,.003,1.258)
        wrist=(side*.448,-.008,1.048)
        s.add(tapered_segment(p,shoulder,elbow,.043,.030),.021)
        s.ell((side*.173,.009,1.456),(.049,.048,.061),.025)
        s.muscle((side*.183,-.012,1.423),(side*.275,-.008,1.286),.035,.043,.013)
        s.muscle((side*.197,.027,1.418),(side*.283,.028,1.284),.032,.028,.014)
        s.add(tapered_segment(p,elbow,wrist,.031,.020),.016)
        s.muscle((side*.309,.001,1.240),(side*.409,-.006,1.105),.031,.034,.012)
        s.muscle((side*.439,-.009,1.067),(side*.496,-.011,.970),.030,.016,.009)
        s.muscle((side*.453,-.010,1.040),(side*.501,-.011,.960),.026,.014,.008)
        knuckles=list(fingers(side))
        s.add(tapered_segment(p,knuckles[0][1],knuckles[-1][1],.011,.010),.005)
        web=np.array([.380*side,0,-.925])*.010
        s.add(tapered_segment(p,knuckles[0][1]+web,knuckles[-1][1]+web,.0135,.0125),.005)
        for _, root, tip in fingers(side):
            mid=root*.45+tip*.55
            mid[1]-=.004
            s.add(tapered_segment(p,root,mid,.008,.0065),.003)
            s.add(tapered_segment(p,mid,tip,.0065,.0045),.002)
        s.add(tapered_segment(p,(side*.454,-.013,1.014),(side*.445,-.017,.980),.011,.008),.007)
        s.add(tapered_segment(p,(side*.445,-.017,.980),(side*.455,-.022,.953),.008,.0055),.004)
    s.cut(ellipsoid(p,(0,-.064,1.150),(.0035,.007,.006)),.001)
    s.cut(ellipsoid(p,(0,-.087,1.379),(.0025,.008,.025)),.003)
    return s.field


def pants_field(p):
    s=Sculpt(p)
    s.ell((0,.011,.968),(.143,.098,.123),.020)
    for side in (-1,1):
        z=p[2]
        cx=side*np.interp(z,[.19,.4,.7,.9,1.08],[.153,.150,.138,.100,.093])
        q=(p[0]-cx,p[1],z)
        d=loft(q,[
            (.191,.042,.052,.052,.010),
            (.220,.057,.063,.065,.010),
            (.263,.080,.080,.080,.010),
            (.335,.088,.082,.084,.011),
            (.460,.089,.083,.086,.014),
            (.570,.087,.081,.086,.008),
            (.700,.089,.083,.089,.008),
            (.825,.091,.088,.094,.010),
            (.980,.080,.084,.091,.008),
            (1.074,.064,.075,.077,.008),
        ])
        angle=np.arctan2(p[1]-.01,q[0])
        folds=.0025*np.sin(angle*5+z*6)*np.sin(z*8+.7)
        # Unequal oblique compression folds; no single periodic ribbing axis.
        for index,(height,amp,width) in enumerate([(.235,.007,.010),(.270,.009,.013),(.313,.006,.016),(.360,.003,.020),(.548,.003,.022),(.598,.002,.019)]):
            line=height+.013*np.sin(angle*(1 if index%2 else 2)+index*1.7)+.007*np.cos(angle*3-index)
            azimuth=.45+.55*np.sin(angle*1.5+index*.8)**2
            folds+=amp*azimuth*(np.exp(-((z-line)/width)**2)-.5*np.exp(-((z-line-width*1.5)/(width*.7))**2))
        folds+=.0008*np.sin(z*57+angle*9)*np.sin(z*31-angle*4)
        folds*=np.clip((z-.19)/.04,0,1)*np.clip((1.06-z)/.08,0,1)
        s.add(d-folds,.017)
    s.field=np.maximum(s.field,p[2]-1.074)
    return s.field
