"""Reference-led face and swept silver hair, in metres, front towards -Y.

The face is a warped implicit loft. Eyes occupy recessed sockets; their lids
interpolate from an eyeball to the same analytic skin surface. Hair locks are
closed, grooved lenticular ribbons with a shared scalp underneath.
"""
import math
import bpy
import numpy as np
from geometry import Sculpt, ellipsoid, mesh, curve, bezier, surface, material
from sdf import interpolate


HEAD = np.array([
    (1.565,.003,.007,.006,-.042),
    (1.573,.019,.025,.033,-.029),
    (1.585,.034,.038,.052,-.015),
    (1.602,.049,.050,.067,-.009),
    (1.622,.063,.067,.074,.006),
    (1.645,.072,.080,.078,.015),
    (1.672,.074,.089,.081,.024),
    (1.699,.0753,.097,.082,.029),
    (1.725,.0706,.0925,.0784,.032),
    (1.750,.060,.082,.0672,.030),
    (1.771,.042,.054,.0479,.028),
    (1.782,.001,.001,.001,.026),
])
EAR_Y=.035
EAR_Z=1.653


def head_sections(z):
    z=np.asarray(z)
    rx,rf,rb,cy=[interpolate(z, HEAD[:,0], HEAD[:,i]) for i in range(1,5)]
    # A genuine rounded vault. Finite end slopes in a loft produce a cone at
    # the crown; analytic ellipse radii retain the horizontal tangent there.
    blend=np.clip((z-1.693)/.028,0,1)
    blend=blend*blend*(3-2*blend)
    dome=[r*np.sqrt(np.maximum(1e-8,1-((z-center)/height)**2))
          for r,center,height in [(.0756,1.690,.092),(.097,1.698,.084),(.082,1.699,.083)]]
    return [old*(1-blend)+new*blend for old,new in zip((rx,rf,rb),dome)]+[cy]


def mouth_height(x):
    # Quiet, slightly stern mouth, with a shallow cupid's bow.
    return 1.600 + .00045*np.exp(-((np.abs(x)-.005)/.004)**2) - .0014*(x/.023)**2


def relief(x,z):
    """Continuous facial planes, rather than intersecting feature ellipsoids."""
    x,z=np.asarray(x),np.asarray(z)
    nz=np.array([1.606,1.612,1.618,1.624,1.633,1.651,1.666,1.679])
    nose=interpolate(z,nz,np.array([0,.001,.009,.0175,.015,.007,.001,0]))
    width=interpolate(z,nz,np.array([.009,.009,.011,.009,.008,.007,.010,.014]))
    d=nose*np.exp(-(np.abs(x)/width)**2.3)
    for side in (-1,1):
        # Malar plane, orbital hollow, eyebrow ridge and nasal alae.
        d+=.0030*np.exp(-((x-side*.046)/.020)**2-((z-1.637)/.016)**2)
        d-=.0020*np.exp(-((x-side*.032)/.020)**2-((z-1.662)/.011)**2)
        d+=.0020*np.exp(-((x-side*.030)/.023)**2-((z-1.679)/.006)**2)
        d+=.0040*np.exp(-((x-side*.011)/.0045)**2-((z-1.619)/.004)**2)
    mask=np.exp(-(np.abs(x)/.021)**4)
    d+=.0040*np.exp(-(x/.021)**2-((z-1.611)/.010)**4)
    line=mouth_height(x)
    d+=mask*(.0030*np.exp(-((z-line-.0020)/.0020)**2)
             +.0039*np.exp(-((z-line+.0026)/.0028)**2)
             -.0011*np.exp(-((z-line)/.0007)**2))
    d+=.0025*np.exp(-(x/.021)**2-((z-1.581)/.012)**2)
    # Philtrum and slight lower-lip / chin hollow.
    d-=.0010*np.exp(-(x/.0023)**2-((z-1.609)/.006)**2)
    d-=.0006*np.exp(-(x/.017)**2-((z-1.590)/.0035)**2)
    return d


def skin_front(x,z):
    rx,rf,rb,cy=head_sections(z)
    return cy-rf*np.sqrt(np.maximum(0,1-(np.abs(x)/rx)**2.5))-relief(x,z)


EYE_X=.0325
EYE_Z=1.6605
EYE_HALF=.0175
EYE_R=.0310
EYE_Y=-.0285


def aperture(u,upper):
    return (1.0-np.minimum(1,np.abs(u))**2)**.72*(.0053 if upper else -.0041)


def eye_point(side,u,v):
    x=side*(EYE_X+EYE_HALF*u)
    z=EYE_Z+.0022*u+aperture(u,v>=0)*abs(v)
    y=EYE_Y+.30*u*EYE_HALF-math.sqrt(max(.00001,EYE_R**2-(x-side*EYE_X)**2-(z-EYE_Z)**2))
    return np.array([x,y,z])


def head_field(p):
    x,y,z=p
    rx,rf,rb,cy=head_sections(z)
    # Warp only the facial half, retaining an ordinary rounded occiput.
    displacement=relief(x,z)*np.clip((cy-y)/.025,0,1)
    yy=y+displacement-cy
    ry=np.where(yy<0,rf,rb)
    power=np.where(yy<0,2.5,2.0)
    radial=(np.sqrt((np.abs(x)/rx)**power+(yy/ry)**2)-1)*np.minimum(rx,ry)
    cap=np.maximum(HEAD[0,0]-z,z-HEAD[-1,0])
    s=Sculpt(p)
    s.add(np.minimum(np.maximum(radial,cap),0)+np.sqrt(np.maximum(radial,0)**2+np.maximum(cap,0)**2),0)
    for side in (-1,1):
        s.ell((side*.073,EAR_Y,EAR_Z),(.012,.016,.028),.005)
        s.cut(ellipsoid(p,(side*.082,EAR_Y-.005,EAR_Z+.002),(.008,.010,.019)),.0012)
        s.ell((side*.075,EAR_Y-.005,EAR_Z-.022),(.008,.009,.010),.003)
        s.cut(ellipsoid(p,(side*.008,-.0705,1.6175),(.0022,.0040,.0017)),.0007)
    # Blind orbital pockets, closed at the rear. No through-holes in the head.
    u=(np.abs(x)-EYE_X)/EYE_HALF
    mid=EYE_Z+.0022*u
    h=np.where(z>=mid,.0056,.0044)
    opening=np.maximum((np.abs(u)-1)*EYE_HALF,
                       np.abs(z-mid)-h*np.maximum(0,1-np.minimum(1,np.abs(u))**2)**.72)
    pocket=np.maximum(opening,y+.028)
    s.cut(pocket,.0006)
    return s.field


def eye_material():
    mat=material('Eyes | continuous iris and sclera',(.7,.73,.69),.26)
    n,l=mat.node_tree.nodes,mat.node_tree.links
    shader=n.get('Principled BSDF')
    uv=n.new('ShaderNodeTexCoord')
    distance=n.new('ShaderNodeVectorMath'); distance.operation='DISTANCE'
    distance.inputs[1].default_value=(.5,.55,0)
    l.new(uv.outputs['UV'],distance.inputs[0])
    ramp=n.new('ShaderNodeValToRGB'); cr=ramp.color_ramp
    colors=[(0,(.004,.008,.009,1)),(.048,(.004,.008,.009,1)),
            (.054,(.025,.033,.027,1)),(.092,(.055,.060,.043,1)),
            (.107,(.009,.014,.012,1)),(.118,(.009,.014,.012,1)),
            (.124,(.56,.58,.55,1)),(.5,(.62,.64,.60,1))]
    cr.elements.remove(cr.elements[1])
    for i,(pos,color) in enumerate(colors):
        e=cr.elements[0] if i==0 else cr.elements.new(pos)
        e.position=pos; e.color=color
    l.new(distance.outputs['Value'],ramp.inputs[0]); l.new(ramp.outputs[0],shader.inputs['Base Color'])
    shader.inputs['Coat Weight'].default_value=.22
    shader.inputs['Coat Roughness'].default_value=.15
    return mat


def face(m):
    eyes=eye_material()
    for side in (-1,1):
        obj=surface(f'Recessed eyeball aperture {side}',lambda u,v:eye_point(side,2*u-1,2*v-1),100,40,eyes)
        uv=obj.data.uv_layers.new(name='Iris coordinates')
        for poly in obj.data.polygons:
            for li in poly.loop_indices:
                p=obj.data.vertices[obj.data.loops[li].vertex_index].co
                uv.data[li].uv=((p.x-side*EYE_X)/.05+.5,(p.z-EYE_Z)/.05+.5)
        for upper in (False,True):
            sign=1 if upper else -1
            def lid(u,v):
                t=2*u-1
                inner=eye_point(side,t,sign)
                # A skin band merging into the facial plane, with a rounded rim.
                x=side*(EYE_X+EYE_HALF*t*(1+.19*v))
                z=inner[2]+sign*v*(.0038 if upper else .0030)*max(0,1-t*t)**.4
                outer_y=float(skin_front(x,z))-.00018
                y=(inner[1]-.0006)*(1-v)+outer_y*v-.00065*math.sin(math.pi*v)
                return x,y,z
            surface(f'Anatomical eyelid {side} {upper}',lid,64,10,m['skin'])
            rim=[eye_point(side,t,sign)+np.array([0,-.00075,0]) for t in np.linspace(-.985,.985,28)]
            curve(f'Lid wet margin {side} {upper}',rim,.00052 if upper else .00024,m['eyeline'] if upper else m['liplight'])
            if upper:
                crease=[]
                for t in np.linspace(-.82,.85,22):
                    p=eye_point(side,t,1)
                    p[2]+=.0045*max(0,1-t*t)**.5
                    p[1]=float(skin_front(p[0],p[2]))-.00025
                    crease.append(p)
                curve(f'Upper lid fold {side}',crease,.00023,m['lip'])
        # Tapered, slightly angled eyebrows. They follow the actual forehead.
        def brow(u,v):
            x=side*(.013+.043*u)
            z=1.679+.004*math.sin(u*2.1)+(.5-v)*.0038*(1-.65*u)
            return x,float(skin_front(x,z))-.00045,z
        surface(f'Silver eyebrow plane {side}',brow,35,4,m['brow'])
        for j in range(27):
            u=j/28
            points=[brow(min(.99,u+.035*t),.92-.85*t) for t in np.linspace(0,1,4)]
            curve(f'Brow filament {side} {j}',points,.00010,m['hairshade'])
        # Cartilage on the exposed outer ear, avoiding the old spherical lobes.
        helix=[]
        for t in np.linspace(-2.0,2.8,34):
            helix.append((side*(.080+.002*math.cos(t)),EAR_Y-.001+.010*math.sin(t),EAR_Z+.002+.023*math.cos(t)))
        curve(f'Ear helix {side}',helix,.0018,m['skin'])
    # Coloured lip surfaces inherit the relief; no separate floating ellipsoid.
    for upper in (True,False):
        def lip(u,v):
            x=(u*2-1)*.022
            w=max(0,1-(x/.022)**2)**.65
            z=float(mouth_height(x))+(1 if upper else -1)*v*(.0032 if upper else .0040)*w
            return x,float(skin_front(x,z))-.00015,z
        surface('Upper lip vermilion' if upper else 'Lower lip vermilion',lip,64,8,m['lip'] if upper else m['liplight'])
    points=[]
    for x in np.linspace(-.022,.022,40):
        z=float(mouth_height(x)); points.append((x,float(skin_front(x,z))-.00045,z))
    curve('Closed mouth line',points,.00038,m['mouth'])


def hair(m):
    centre=np.array([0,.021,1.712])
    radii=np.array([.087,.108,.100])
    def fitted_guide(points):
        """Re-seat authored guides on the deeper skull, preserving front Y."""
        p=np.array(points,dtype=float)
        p[:,1]=centre[1]+(p[:,1]-.010)*(radii[1]/.096)
        p[:,2]=centre[2]+(p[:,2]-1.715)*(radii[2]/.098)
        return p
    def scalp(theta,angle,inflate=0):
        return centre+(radii+inflate)*np.array([math.sin(theta)*math.sin(angle),-math.sin(theta)*math.cos(angle),math.cos(theta)])
    def hairline(a):
        frontal=min(a,math.tau-a)
        return 2.60-1.80*math.exp(-(frontal/1.20)**4)
    def cap(u,v):
        a=u*math.tau
        end=hairline(a)
        theta=.01+v*(end-.01)
        return scalp(theta,a,.0005*math.sin(105*a+theta*17))
    surface('Continuous silver hair foundation',cap,112,48,m['hairshade'],.003)
    strands=[]
    rng=np.random.default_rng(174)

    def on_foundation(p,clearance):
        # Prevent the scalp from slicing off the middle of a curved lock.
        # This projects only buried surface samples, preserving free tips.
        q=(p-centre)/radii
        length=np.linalg.norm(q)
        theta=math.acos(np.clip(q[2]/length,-1,1))
        angle=math.atan2(q[0],-q[1])%math.tau
        edge=hairline(angle)
        if theta>edge+.18:
            return p
        floor=scalp(theta,angle,.0005*math.sin(105*angle+theta*17)+clearance)
        if np.linalg.norm((floor-centre)/radii)>length:
            weight=np.clip((edge+.18-theta)/.18,0,1)
            weight=weight*weight*(3-2*weight)
            return p+(floor-p)*weight
        return p

    def lock(name,control,width=.014,depth=.0028):
        path=bezier(control,45)
        sides,normals,widths,depths=[],[],[],[]
        vertices,faces=[],[]
        # An upper ribbed lens and a flatter underside make a thin, sharp lock.
        cross=np.r_[np.linspace(-1,1,25),np.linspace(1,-1,13)[1:-1]]
        top_count=25; nc=len(cross)
        for i,p in enumerate(path):
            t=i/(len(path)-1)
            tangent=path[min(i+1,len(path)-1)]-path[max(0,i-1)]
            tangent/=np.linalg.norm(tangent)
            n=(p-centre)/(radii*radii)
            n-=tangent*np.dot(n,tangent); n/=max(1e-8,np.linalg.norm(n))
            axis=np.cross(tangent,n); axis/=np.linalg.norm(axis)
            factor=max(.001,max(0,math.sin(math.pi*t))**.50*(1-t)**.30)
            w=width*factor*(1+.05*math.sin(t*9))
            d=depth*factor
            sides.append(axis); normals.append(n); widths.append(w); depths.append(d)
            for j,u in enumerate(cross):
                lens=max(0,1-u*u)**.62
                if j<top_count:
                    h=d*lens+.00022*factor*lens*(math.cos(u*7*math.pi+t*1.4)-.4)
                else:
                    h=-d*.55*lens
                point=p+w*u*axis+h*n
                if j<top_count:
                    point=on_foundation(point,.0017+d*lens*.70)
                vertices.append(tuple(point))
        for i in range(len(path)-1):
            for j in range(nc):
                a=i*nc+j; b=i*nc+(j+1)%nc
                faces.append((a,b,b+nc,a+nc))
        faces.extend([tuple(reversed(range(nc))),tuple(range((len(path)-1)*nc,len(path)*nc))])
        mat=m['hair'] if rng.random()>.22 else m['hairshade']
        mesh(name,vertices,faces,mat)
        for u0 in [-.77,-.50,-.22,.12,.42,.70]:
            points=[]
            for i in range(3,43,2):
                t=i/44; u=u0+.035*math.sin(t*math.pi)
                h=depths[i]*max(0,1-u*u)**.62+.00012
                p=path[i]+sides[i]*widths[i]*u+normals[i]*h
                p=on_foundation(p,.0018+depths[i]*max(0,1-u*u)**.62*.70)
                points.append((p,max(.08,math.sin(t*math.pi)**.5)))
            strands.append(points)

    # Back and side paths wrap around the skull and converge towards the nape.
    # Staggered root heights and unequal lengths avoid rows of identical leaves.
    for side in (-1,1):
        for row,(theta,length) in enumerate([(.30,1.01),(.78,.95),(1.28,1.02)]):
            for j,a in enumerate(np.linspace(.68,2.48,7)):
                angle=side*(a+.095*math.sin(j*3.7+row*2.0))
                th=theta+.075*math.sin(j*2.1+row)
                end=th+length+.08*math.cos(j*1.8+row)
                turn=side*(.68+.18*math.sin(j+row))*(1 if a<2.1 else .5)
                p0=scalp(th,angle,-.004)
                p1=scalp(th+.26,angle+turn*.2,.012)
                p2=scalp(end-.21,angle+turn*.7,.014)
                p3=scalp(end,angle+turn,.013 if row<2 else .006)
                # Upper tips flick outward, lower nape tips fall close to neck.
                p3[0]+=side*(.013 if a<1.9 and row<3 else .002)
                p3[2]+=.012 if row<2 else -.005
                lock(f'Swept silver layer {side} {row} {j}',[p0,p1,p2,p3],.0225-row*.0020,.0040)
        for j in range(3):
            y=-.038+j*.029
            lock(f'Temple and nape wisp {side} {j}',fitted_guide([(side*.052,y+.010,1.769),(side*.082,y-.004,1.713),(side*.083,y+.001,1.665),(side*(.068+j*.007),y+.017,1.624+j*.007)]),.0105,.0020)
        for j,a in enumerate(np.linspace(1.55,3.08,6)):
            angle=side*a
            end=2.55+.16*j/5
            points=[scalp(1.64,angle,-.004),scalp(1.98,angle+.12*side,.008),scalp(2.38,angle+.18*side,.009),scalp(end,angle+.21*side,.011)]
            points[-1][2]-=.016+.006*j/5
            lock(f'Fine tapered nape {side} {j}',points,.013,.0024)
    # Long, uneven back flow replaces the mirrored scallops of the first pass.
    for j in range(-4,5):
        a=j/4
        points=[(.016*a,.032,1.805-.006*abs(a)),
                (.043*a-.008,.084,1.776),
                (.082*a,.110,1.692+.013*abs(a)),
                (.062*a+.004*math.sin(j*2),.052,1.604+.029*abs(a)+.007*math.sin(j))]
        lock(f'Long back hair flow {j}',fitted_guide(points),.0185,.0040)
    for j in range(-2,3):
        a=j/2
        points=[(-.008,.039,1.808),(.027*a,.088,1.795),(.068*a,.111,1.750),(.084*a+.012,.108,1.700+.016*abs(a))]
        lock(f'Back crown overlap {j}',fitted_guide(points),.020,.0040)
    # Hand-directed fringe flow traced from the front, checked in both profiles.
    bangs=[
        ([(-.019,-.045,1.807),(-.048,-.080,1.794),(-.061,-.097,1.727),(-.086,-.061,1.663)],.018),
        ([(-.024,-.044,1.809),(-.063,-.069,1.788),(-.079,-.073,1.755),(-.117,-.038,1.743)],.017),
        ([(-.029,-.032,1.806),(-.071,-.052,1.802),(-.074,-.061,1.777),(-.111,-.022,1.775)],.014),
        ([(-.016,-.052,1.806),(-.034,-.087,1.778),(-.019,-.100,1.724),(.008,-.087,1.671)],.0175),
        ([(-.010,-.051,1.805),(.006,-.089,1.780),(.011,-.093,1.741),(.027,-.079,1.703)],.012),
        ([(-.013,-.042,1.810),(.025,-.078,1.793),(.043,-.093,1.750),(.069,-.060,1.704)],.0185),
        ([(-.019,-.031,1.813),(.038,-.050,1.817),(.061,-.076,1.776),(.112,-.029,1.760)],.021),
        ([(-.016,-.013,1.817),(.023,-.021,1.841),(.075,-.035,1.799),(.108,.011,1.796)],.017),
    ]
    for i,(p,w) in enumerate(bangs):
        p=fitted_guide(p)
        # Keep the fringe close to the forehead in profile.
        p[:,1]=np.maximum(p[:,1],-.079)
        lock(f'Asymmetric fringe {i}',p,w,.0038)
        # Fine secondary tip split, displaced along the primary surface.
        if i in (0,3,5,6):
            q=np.array(p,dtype=float)
            q[:,0]+=[.003,.006,.007,.003]; q[:,1]-=[.0005,.002,.002,.001]
            q[-1,2]+=.012
            lock(f'Fringe split {i}',q,w*.40,.0015)
    lock('Fine crown cowlick',fitted_guide([(-.024,.002,1.804),(-.028,-.005,1.827),(-.016,.003,1.833),(-.001,.005,1.839)]),.005,.0014)
    lock('Swept crown wisp',fitted_guide([(-.018,.016,1.812),(.004,.018,1.831),(.029,.020,1.832),(.044,.029,1.835)]),.006,.0015)
    # Batch fine filaments in one curves object, with tapered ends.
    data=bpy.data.curves.new('Directional silver filaments','CURVE')
    data.dimensions='3D'; data.bevel_depth=.000095; data.bevel_resolution=1
    for points in strands:
        s=data.splines.new('POLY'); s.points.add(len(points)-1)
        for vertex,(p,radius) in zip(s.points,points):
            vertex.co=(*p,1); vertex.radius=radius
    obj=bpy.data.objects.new('Directional silver filaments',data)
    bpy.context.collection.objects.link(obj); data.materials.append(m['hairgroove'])
