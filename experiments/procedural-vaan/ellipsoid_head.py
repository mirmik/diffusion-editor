"""Head from named ellipsoid volumes only: no loft or facial displacement.

All coordinates are metres, front is -Y. Ellipsoid fields are approximate
distances. Eyelids and lip colour can ray-project onto the resulting volume.
"""
import math
from pathlib import Path
import sys
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]/'procedural-bust'))
from sdf import Sculpt, ellipsoid


EYE_X=.0315
EYE_Z=1.664
EYE_Y=-.0315
EYE_R=.025
EAR_Y=.027
EAR_Z=1.654


def part(name,center,radii,blend,rx=0,rz=0):
    return dict(name=name,center=list(center),radii=list(radii),blend=blend,rx=rx,rz=rz)


PARTS=[
    part('Cranial vault',(0,.024,1.708),(.071,.079,.075),0),
    part('Occiput',(0,.045,1.680),(.061,.055,.052),.018),
    part('Nuchal transition',(0,.045,1.651),(.038,.035,.043),.018),
    part('Forehead',(0,-.020,1.692),(.062,.036,.058),.024),
    part('Facial core',(0,-.005,1.645),(.060,.053,.066),.020),
    part('Maxilla',(0,-.025,1.631),(.045,.032,.044),.016),
    part('Mandible body',(0,.002,1.608),(.037,.044,.033),.016),
    part('Chin',(0,-.032,1.587),(.023,.023,.020),.014),
    part('Chin transition',(0,-.035,1.596),(.027,.019,.022),.014),
    part('Mouth support',(0,-.040,1.612),(.029,.018,.020),.012),
    part('Glabella',(0,-.042,1.676),(.014,.019,.017),.010),
    part('Nasal bridge',(0,-.058,1.652),(.0075,.010,.027),.007,rx=-20),
    part('Nasal dorsum',(0,-.066,1.638),(.0075,.009,.017),.005,rx=-15),
    part('Nasal tip',(0,-.073,1.625),(.008,.0055,.0065),.004),
    part('Columella',(0,-.069,1.621),(.003,.004,.005),.003),
    part('Lower lip',(0,-.0555,1.598),(.019,.0045,.003),.003),
]
for side in (-1,1):
    PARTS.extend([
        part(f'Temple {side}',(side*.054,.014,1.680),(.014,.044,.040),.016),
        part(f'Zygomatic cheek {side}',(side*.043,-.018,1.644),(.019,.027,.024),.018,rz=side*12),
        part(f'Mandibular ramus {side}',(side*.035,.017,1.628),(.017,.029,.030),.016),
        part(f'Mandibular edge {side}',(side*.024,.004,1.609),(.020,.048,.018),.016,rx=35),
        part(f'Masseter {side}',(side*.037,.012,1.622),(.013,.025,.023),.014),
        part(f'Brow ridge {side}',(side*.030,-.037,1.682),(.028,.020,.011),.009),
        part(f'Nasal ala {side}',(side*.008,-.064,1.623),(.005,.007,.005),.004),
        part(f'Upper lip {side}',(side*.009,-.0568,1.602),(.011,.004,.0025),.003),
        part(f'Ear root {side}',(side*.065,.025,EAR_Z),(.014,.016,.026),.005),
        part(f'Ear pinna {side}',(side*.073,EAR_Y,EAR_Z),(.011,.015,.028),.004),
        part(f'Ear lobe {side}',(side*.074,EAR_Y-.004,EAR_Z-.021),(.008,.009,.009),.003),
    ])

CUTS=[]
for side in (-1,1):
    CUTS.extend([
        part(f'Ear concha {side}',(side*.081,EAR_Y-.004,EAR_Z+.002),(.008,.010,.019),.0012),
        part(f'Orbit {side}',(side*EYE_X,-.050,EYE_Z),(.0195,.024,.0068),.0007),
        part(f'Nostril {side}',(side*.008,-.071,1.621),(.0022,.004,.0019),.0006),
    ])


def rotation(spec):
    a,b=math.radians(spec['rx']),math.radians(spec['rz'])
    cx,sx,cz,sz=math.cos(a),math.sin(a),math.cos(b),math.sin(b)
    return np.array([[cz,-sz*cx,sz*sx],[sz,cz*cx,-cz*sx],[0,sx,cx]])


def distance(p,spec):
    q=[p[i]-spec['center'][i] for i in range(3)]
    if spec['rx'] or spec['rz']:
        r=rotation(spec)
        q=[sum(q[j]*r[j,i] for j in range(3)) for i in range(3)]
    return ellipsoid(q,(0,0,0),spec['radii'])


def volume(p):
    s=Sculpt(p)
    for spec in PARTS:
        s.add(distance(p,spec),spec['blend'])
    return s.field


def head_field(p):
    s=Sculpt(p)
    s.field=volume(p)
    for spec in CUTS:
        s.cut(distance(p,spec),spec['blend'])
    return s.field


_front_cache=None


def prepare_front():
    """Batch ray intersections; interpolation avoids thousands of scalar SDF calls."""
    global _front_cache
    xs=np.linspace(-.063,.063,253)
    zs=np.linspace(1.587,1.706,239)
    x,z=np.meshgrid(xs,zs,indexing='ij')
    lo=np.full_like(x,-.145)
    hi=np.full_like(x,.020)
    valid=volume((x,hi,z))<0
    for _ in range(23):
        mid=(lo+hi)*.5
        outside=volume((x,mid,z))>0
        lo=np.where(outside,mid,lo)
        hi=np.where(outside,hi,mid)
    y=(lo+hi)*.5
    y[~valid]=np.nan
    _front_cache=(xs,zs,y)


def skin_front(x,z):
    if _front_cache is None:
        prepare_front()
    xs,zs,grid=_front_cache
    u=(float(x)-xs[0])/(xs[1]-xs[0])
    v=(float(z)-zs[0])/(zs[1]-zs[0])
    i,j=int(np.floor(u)),int(np.floor(v))
    if not (0<=i<len(xs)-1 and 0<=j<len(zs)-1):
        raise ValueError(f'Facial ray outside cache: {x}, {z}')
    u,v=u-i,v-j
    result=(grid[i,j]*(1-u)*(1-v)+grid[i+1,j]*u*(1-v)
            +grid[i,j+1]*(1-u)*v+grid[i+1,j+1]*u*v)
    if not np.isfinite(result):
        raise ValueError(f'Facial ray missed ellipsoid volume: {x}, {z}')
    return float(result)


def configure_face(portrait):
    """Reuse eye/lid/colour surfaces, fitted to this field, never the old loft."""
    for name in ('EYE_X','EYE_Z','EYE_Y','EYE_R','EAR_Y','EAR_Z'):
        setattr(portrait,name,globals()[name])
    portrait.skin_front=skin_front
    portrait.head_field=head_field
