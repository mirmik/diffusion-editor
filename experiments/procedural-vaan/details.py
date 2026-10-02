"""Eyes, hair, garments, leatherwork and footwear from procedural surfaces."""
import math
import numpy as np
import bpy
from geometry import mesh, box, uv_ellipsoid, curve, bezier, sweep, surface


def vest_point(angle,z):
    rx=np.interp(z,[1.19,1.27,1.4,1.5],[.139,.145,.165,.146])
    depths=[.093,.105,.126,.082] if math.cos(angle)>=0 else [.082,.088,.087,.076]
    ry=np.interp(z,[1.19,1.3,1.43,1.5],depths)
    folds=.0018*math.sin(z*65+angle*5)*math.sin(angle*3)**2
    return ((rx+folds)*math.sin(angle),.014-(ry+folds)*math.cos(angle),z)


def clothes(m):
    def panel(u,v):
        angle=.46+u*(math.tau-.92)
        z=1.194+v*(.310-.012*math.cos(angle))
        return vest_point(angle,z)
    def armhole(u,v):
        angle=.46+u*(math.tau-.92)
        z=panel(u,v)[2]
        distance=min(abs(angle-math.pi/2),abs(angle-3*math.pi/2))
        return (distance/.47)**2+((z-1.425)/.068)**2<1
    surface('Open blue sleeveless vest',panel,100,64,m['blue'],.004,armhole)
    for side in (-1,1):
        a=.46 if side==1 else math.tau-.46
        curve(f'Front vest piping {side}',[vest_point(a,z) for z in np.linspace(1.197,1.494,16)],.0025,m['blueedge'])
        curve(f'Vest vertical seam {side}',[vest_point(side*.93%math.tau,z) for z in np.linspace(1.203,1.350,10)],.0009,m['bluedark'])
        # Thin leather binding follows the actual armhole ellipse.
        path=[]
        for a0 in np.linspace(0,math.tau,40,endpoint=False):
            angle=(math.pi/2 if side==1 else 3*math.pi/2)+.47*math.cos(a0)
            path.append(vest_point(angle,1.425+.068*math.sin(a0)))
        curve(f'Leather armhole binding {side}',path,.003,m['leatherdark'],True)
        def front_y(x,z,offset=.008):
            rx=np.interp(z,[1.19,1.27,1.4,1.5],[.139,.145,.165,.146])
            angle=math.asin(min(.97,abs(x)/rx))
            return vest_point(angle,z)[1]-offset
        def patch(name,outline,mat):
            offset=.014 if 'flap' in name else .008
            # Subdivide in the XZ plane before projecting: a single nonplanar
            # ngon's chord would disappear beneath the curved blue fabric.
            center=np.mean(outline,axis=0)
            verts,faces=[],[]
            for index in range(len(outline)):
                a=np.array(outline[index]); b=np.array(outline[(index+1)%len(outline)])
                for i in range(6):
                    for j in range(6-i):
                        def pt(i,j):
                            x,z=center+(a-center)*i/6+(b-center)*j/6
                            return (side*x,front_y(x,z,offset),z)
                        base=len(verts); verts.extend([pt(i,j),pt(i+1,j),pt(i,j+1)])
                        faces.append((base,base+1,base+2))
                        if i+j<5:
                            base=len(verts); verts.extend([pt(i+1,j),pt(i+1,j+1),pt(i,j+1)])
                            faces.append((base,base+1,base+2))
            obj=mesh(name,verts,faces,mat)
            mod=obj.modifiers.new('Leather thickness','SOLIDIFY'); mod.thickness=.003
            points=[]
            for i,(x,z) in enumerate(outline):
                bx,bz=outline[(i+1)%len(outline)]
                for t in np.linspace(0,1,8,endpoint=False):
                    px,pz=x*(1-t)+bx*t,z*(1-t)+bz*t
                    points.append((side*px,front_y(px,pz,offset+.002),pz))
            curve(name+' edging',points,.0011,m['leatherdark'],True)
        patch(f'Brown shoulder yoke {side}',[(.074,1.494),(.137,1.500),(.150,1.434),(.142,1.357),(.111,1.340),(.079,1.365)],m['leather'])
        patch(f'Chest pocket flap {side}',[(.081,1.418),(.143,1.428),(.142,1.391),(.113,1.370),(.081,1.388)],m['leatherlight'])
        x,z=.113,1.384
        uv_ellipsoid(f'Pocket press stud {side}',(side*x,front_y(x,z,.019),z),(.003,.0015,.003),m['metal'],24)
        for z in [1.478,1.454]:
            curve(f'Yoke stitching {side} {z}',[(side*x,front_y(x,z,.011),z) for x in np.linspace(.080,.137,9)],.0006,m['stitch'])
    for z in [1.198,1.218]:
        curve(f'Vest hem {z}',[vest_point(a,z) for a in np.linspace(.46,math.tau-.46,80)],.002,m['blueedge'])
    # The relaxed hood is a bag: a low fold between neckline and turned-up lip.
    def hood(u,v):
        angle=-math.pi/2+u*math.pi
        c=max(0,math.cos(angle))
        inner=np.array([.049*math.sin(angle),.015+.050*c,1.552-.006*c])
        outer=np.array([.126*math.sin(angle),.014+.108*c,1.535-.046*c])
        p=inner*(1-v)+outer*v
        p[1]+=.034*math.sin(math.pi*v)*c
        p[2]-=.049*math.sin(math.pi*v)*c**.7
        p[2]+=.003*math.sin(angle*7+v*4)*math.sin(math.pi*v)
        return tuple(p)
    surface('Folded hood draped on back',hood,80,48,m['blue'],.004)
    curve('Hood rear folded edge',[hood(u,1) for u in np.linspace(0,1,50)],.004,m['blueedge'])
    curve('Hood edge stitching',[np.array(hood(u,.96))+np.array([0,.001,.001]) for u in np.linspace(.02,.98,55)],.00055,m['bluedark'])
    curve('Hood centre seam',[np.array(hood(.5,v))+np.array([0,.002,0]) for v in np.linspace(.03,.98,36)],.00065,m['blueedge'])
    for side in (-1,1):
        path=[(side*.065,-.080,1.402),(side*.055,-.081,1.479),(side*.072,-.046,1.552),(side*.119,.013,1.535)]
        sweep(f'Rolled front hood lapel {side}',bezier(path,32),.017,.004,m['blueedge'])
    # Ribbed waist and cuffs.
    surface('Waistband',lambda u,v:(.139*math.sin(u*math.tau),.008-.089*math.cos(u*math.tau),1.046+.029*v),80,3,m['pantsedge'],.004)
    for side in (-1,1):
        uv_ellipsoid(f'Exposed ankle {side}',(side*.153,.012,.165),(.033,.040,.060),m['skin'])
        surface(f'Elastic cuff {side}',lambda u,v:(side*.153+(.043+.0007*math.sin(40*u*math.tau))*math.sin(u*math.tau),.010+.054*math.cos(u*math.tau),.195+.029*v),100,4,m['pantsedge'],.003)
        # Outer cargo pockets occupy the front/side quadrant, as in both profiles.
        pocket=box(f'Cargo pocket {side}',(side*.186,-.026,.676),(.071,.103,.153),m['pants'],.008)
        pocket.rotation_euler[2]=side*.32
        flap=box(f'Cargo flap {side}',(side*.190,-.029,.745),(.077,.112,.032),m['pantsedge'],.004)
        flap.rotation_euler[2]=side*.32
        for x in [.165,.207]:
            uv_ellipsoid(f'Cargo rivet {side} {x}',(side*x,-.081,.739),(.0025,.0013,.0025),m['metal'],20)
        curve(f'Cargo pleat {side}',[(side*.190,-.083,z) for z in [.610,.670,.724]],.0011,m['pantsedge'])
        # Leather pouch and the long hanging strap seen in the side views.
        pouch=box(f'Hip leather pouch {side}',(side*.176,.012,.963),(.044,.089,.135),m['leather'],.010)
        box(f'Hip pouch flap {side}',(side*.200,.006,1.005),(.008,.084,.045),m['leatherlight'],.004)
        points=bezier([(side*.109,-.083,1.06),(side*.149,-.104,.940),(side*.192,-.080,.862),(side*.204,.010,.846)],24)
        points=np.concatenate([points,bezier([(side*.204,.010,.846),(side*.193,.106,.868),(side*.147,.115,.946),(side*.109,.094,1.06)],24)[1:]])
        sweep(f'Hanging leather strap {side}',points,.013,.003,m['leatherlight'],normal=(side,0,.1))
        for y in [-.084,.094]:
            curve(f'Strap buckle {side} {y}',[(side*.099,y,1.058),(side*.119,y,1.058),(side*.119,y,1.078),(side*.099,y,1.078)],.0015,m['metal'],True)
        for x in [.087,.130]:
            box(f'Belt loop {side} {x}',(side*x,-.072,1.061),(.009,.007,.039),m['pantsedge'],.001)
    uv_ellipsoid('Waist button',(0,-.084,1.062),(.006,.002,.006),m['metal'],24)
    curve('Trouser fly',[(.007,-.091,1.04),(.008,-.092,.99),(.011,-.085,.92)],.0013,m['pantsedge'])
    # Alternating turquoise / silver necklace and diamond pendant.
    for side in (-1,1):
        pts=bezier([(side*.060,-.036,1.509),(side*.066,-.073,1.474),(side*.031,-.091,1.451),(0,-.100,1.425)],32)
        curve(f'Necklace thread {side}',pts,.0018,m['metal'])
        for i,t in enumerate(np.linspace(.23,.9,7)):
            idx=int(t*31)
            obj=box(f'Necklace bead {side} {i}',pts[idx],(.011,.005,.016),m['cyan'] if i%2==0 else m['metal'],.001)
            delta=pts[min(idx+1,31)]-pts[max(0,idx-1)]
            obj.rotation_euler[1]=math.atan2(delta[0],delta[2])
    for size,mat,y in [(.024,m['metal'],-.101),(.017,m['cyan'],-.105),(.008,m['gem'],-.108)]:
        obj=box('Diamond pendant',(0,y,1.424),(size,.004,size),mat,.001)
        obj.rotation_euler[1]=math.pi/4


def shoes(m):
    for side in (-1,1):
        cx=side*.153
        # Closed loft along Z, elongated toe toward negative Y.
        for name,profile,mat in [
            ('Rubber outsole',[(.012,.062,.124,-.059),(.022,.070,.131,-.060),(.037,.071,.132,-.059),(.043,.066,.126,-.056)],m['sole']),
            ('Blue sculpted midsole',[(.033,.070,.129,-.058),(.049,.068,.129,-.057),(.064,.062,.118,-.052)],m['blueedge']),
            ('Leather sneaker upper',[(.060,.062,.116,-.052),(.081,.062,.119,-.049),(.104,.057,.099,-.031),(.135,.045,.058,.003),(.151,.041,.050,.010)],m['shoe']),
        ]:
            def point(u,v):
                z=np.interp(v,np.linspace(0,1,len(profile)),[p[0] for p in profile])
                rx,ry,cy=[np.interp(z,[p[0] for p in profile],[p[i] for p in profile]) for i in [1,2,3]]
                a=u*math.tau
                # Rounded superellipse gives a broad toe instead of a pointed oval.
                ss=math.copysign(abs(math.sin(a))**.8,math.sin(a))
                cc=math.copysign(abs(math.cos(a))**.8,math.cos(a))
                return (cx+rx*ss,cy-ry*cc,z)
            obj=surface(f'{name} {side}',point,64,16,mat)
            # Cap via original boundary loops.
            bm=bmesh_for_caps(obj)
            bm.to_mesh(obj.data); bm.free()
        upper=[(.060,.062,.116,-.052),(.081,.062,.119,-.049),(.104,.057,.099,-.031),(.135,.045,.058,.003),(.151,.041,.050,.010)]
        def lacepoint(x,z):
            rx,ry,cy=[np.interp(z,[p[0] for p in upper],[p[i] for p in upper]) for i in [1,2,3]]
            y=cy-ry*max(.001,1-(abs(x)/rx)**2.5)**.4-.0015
            return (cx+x,y,z)
        for i,(z,w) in enumerate([(.093,.041),(.106,.038),(.119,.034),(.132,.030),(.144,.026)]):
            # Project the entire curve; a three-point spline cut into the vamp.
            for direction in (-1,1):
                points=[]
                for t in np.linspace(0,1,20):
                    x=(2*t-1)*w
                    p=np.array(lacepoint(x,z+direction*(t-.5)*.007))
                    p[1]-=.0012
                    points.append(p)
                curve(f'Cross lace {side} {i} {direction}',points,.00135,m['laces'])
            for sign in (-1,1):
                p=np.array(lacepoint(sign*w,z)); p[1]-=.001
                uv_ellipsoid(f'Lace eyelet {side} {i} {sign}',p,(.0035,.001,.0025),m['sole'],20)
        curve(f'Toe cap seam {side}',[lacepoint(x,.080+.006*(x/.052)**2) for x in np.linspace(-.052,.052,30)],.0009,m['laces'])
        for sign in (-1,1):
            for i,y in enumerate(np.linspace(-.135,.037,10)):
                x=cx+sign*(.057+.009*math.sin((y+.17)/.22*math.pi))
                lug=box(f'Outsole tread {side} {sign} {i}',(x,y,.025),(.011,.010,.017),m['sole'],.002)
        tongue=box(f'Shoe tongue {side}',(cx,-.044,.144),(.057,.016,.048),m['shoe'],.009)
        box(f'Blue tongue label {side}',(cx,-.054,.159),(.024,.003,.010),m['blueedge'],.001)
        for sign in (-1,1):
            curve(f'Sneaker panel {side} {sign}',[(cx+sign*.045,.022,.135),(cx+sign*.062,-.040,.096),(cx+sign*.057,-.129,.084)],.0018,m['laces'])
            curve(f'Sole side groove {side} {sign}',[(cx+sign*.069,y,.035+.005*math.sin(y*30)) for y in np.linspace(-.15,.052,12)],.0015,m['shoe'])


def bmesh_for_caps(obj):
    import bmesh
    bm=bmesh.new(); bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6)
    bmesh.ops.holes_fill(bm,edges=[e for e in bm.edges if e.is_boundary],sides=0)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    return bm
