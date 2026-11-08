import numpy as np, trimesh, math
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from shapely.ops import unary_union
from shapely import affinity
from trimesh.visual.material import PBRMaterial
from PIL import Image
import arm_px as A

W=3.0; S=0.955/1020.0
def M(x,y): return ((1052-x)*S, (340-y)*S)        # px -> (z, v) metres
def conv(g):
    if g.is_empty: return g
    if isinstance(g,MultiPolygon): return unary_union([conv(p) for p in g.geoms])
    ext=[M(*p) for p in g.exterior.coords]
    ints=[[M(*p) for p in i.coords] for i in g.interiors]
    return Polygon(ext,ints).buffer(0)

def extr(poly,x0,x1,axes='zy'):
    """extrude 2D poly (z,y) along x"""
    parts=poly.geoms if isinstance(poly,MultiPolygon) else [poly]
    ms=[]
    for p in parts:
        if p.area<1e-8: continue
        m=trimesh.creation.extrude_polygon(p,height=x1-x0)
        v=m.vertices
        mm=trimesh.Trimesh(np.column_stack([v[:,2]+x0,v[:,1],v[:,0]]),m.faces.copy(),process=True)
        mm.invert(); ms.append(mm)
    return trimesh.util.concatenate(ms)

def pbr(name,rgb,metal,rough,alpha=255,tex=None):
    m=PBRMaterial(name=name,baseColorFactor=[*rgb,alpha],metallicFactor=metal,roughnessFactor=rough,
                  alphaMode='BLEND' if alpha<255 or tex is not None else 'OPAQUE',doubleSided=True)
    if tex is not None: m.baseColorTexture=tex
    return m

abs_mat=pbr('ABS_szurke',[0.50,0.51,0.53],0.0,0.5)
alu_mat=pbr('aluminium',[0.80,0.81,0.83],0.85,0.32)
dark_mat=pbr('fekete_el',[0.12,0.12,0.13],0.0,0.6)

scene=trimesh.Scene()
thick=conv(A.thick); thin=conv(A.thin)
PLATE_T=0.034; PLATE_W=0.057; PLATE_H=0.28
# rounded top end-cap + mounting plate in (x,y) plane, extruded along z
plate2d=box(-PLATE_W/2,0,PLATE_W/2,PLATE_H).buffer(-0.012).buffer(0.012)  # rounded corners
plate2d=plate2d.intersection(box(-1,0,1,PLATE_H+0.0))
holes=[Point(0,PLATE_H-0.047).buffer(0.0065),Point(0,PLATE_H-0.047-0.185).buffer(0.0065)]
plate2d=plate2d.difference(unary_union(holes))
pm=trimesh.creation.extrude_polygon(plate2d,height=PLATE_T)   # (x,y,z)
plate_base=pm.copy()
RAIL_W=0.040; RIB_W=0.014

def arm(xc,name):
    parts=[]
    a=extr(thick,xc-RAIL_W/2,xc+RAIL_W/2); parts.append(a)
    b=extr(thin,xc-RIB_W/2,xc+RIB_W/2); parts.append(b)
    p=plate_base.copy(); p.apply_translation([xc,0,0]); parts.append(p)
    # slim ribs along the web underside for a hollow look: skip
    m=trimesh.util.concatenate(parts)
    m.visual=trimesh.visual.TextureVisuals(material=abs_mat)
    scene.add_geometry(m,node_name=name,geom_name=name)
for xc in (-W/2+PLATE_W/2,0.0,W/2-PLATE_W/2):
    arm(xc,f'kar_{xc:+.2f}')

# ---------- roof panels with UV + stripe texture ----------
# stripe texture: multiwall channels (alpha varies)
tw,th=64,8
arr=np.zeros((th,tw,4),np.uint8)
arr[:,:,:3]=(222,236,240)
arr[:,:,3]=95
arr[:,0:4,3]=150    # channel wall
arr[:,tw//2-2:tw//2+2,3]=130
tex_img=Image.fromarray(arr,'RGBA'); tex_img.save('stripes.png')
import PIL
PC=pbr('polikarbonat',[1,1,1],0.0,0.2,tex=tex_img)
PC.baseColorFactor=[255,255,255,255]

Tl=A.Tline
Tpts=np.array([M(*p) for p in Tl.coords])            # (z,v), wall -> front as x decreases
# ordered wall->front already (x decreasing? Tline from x=32 to 1052 => tip->wall); reverse
Tpts=Tpts[np.argsort(Tpts[:,0])]
def offset_pts(P,d):
    t=np.gradient(P,axis=0); t/=np.linalg.norm(t,axis=1)[:,None]
    n=np.column_stack([t[:,1],-t[:,0]])   # normal pointing 'up' for z increasing, v decreasing
    return P+n*d
def up_off(P,d):
    a=offset_pts(P,0.01)
    sg=1 if a[:,1].mean()>P[:,1].mean() else -1
    return offset_pts(P,sg*d)
topP=Tpts.copy(); botP=up_off(Tpts,-0.0075)
# arc length param
seg=np.linalg.norm(np.diff(Tpts,axis=0),axis=1); sl=np.concatenate([[0],np.cumsum(seg)])
GAP=0.004
def panel(x0,x1,name):
    nx=2; n=len(Tpts)
    verts=[];uvs=[];faces=[]
    def add_grid(P,xs,flip):
        base=len(verts)
        for i in range(n):
            for xx in xs:
                verts.append([xx,P[i,1],P[i,0]]); uvs.append([(xx-x0)/0.0105, sl[i]/0.0105])
        for i in range(n-1):
            for j in range(len(xs)-1):
                a=base+i*len(xs)+j; b=a+1; c=a+len(xs); d=c+1
                faces.extend([[a,c,b],[b,c,d]] if not flip else [[a,b,c],[b,d,c]])
    add_grid(topP,[x0,x1],False); add_grid(botP,[x0,x1],True)
    # side walls: left/right (x=x0,x1) strips between top & bottom
    for xx,flip in ((x0,False),(x1,True)):
        base=len(verts)
        for i in range(n):
            verts.append([xx,topP[i,1],topP[i,0]]); uvs.append([0,sl[i]/0.0105])
            verts.append([xx,botP[i,1],botP[i,0]]); uvs.append([0.5,sl[i]/0.0105])
        for i in range(n-1):
            a=base+2*i;b=a+1;c=a+2;d=a+3
            faces.extend([[a,b,c],[b,d,c]] if flip else [[a,c,b],[b,c,d]])
    # front & back edges
    for i,flip in ((0,False),(n-1,True)):
        base=len(verts)
        for xx in (x0,x1):
            verts.append([xx,topP[i,1],topP[i,0]]); uvs.append([(xx-x0)/0.0105,0])
            verts.append([xx,botP[i,1],botP[i,0]]); uvs.append([(xx-x0)/0.0105,0.5])
        a=base;b=a+1;c=a+2;d=a+3
        faces.extend([[a,b,c],[b,d,c]] if flip else [[a,c,b],[b,c,d]])
    m=trimesh.Trimesh(np.array(verts),np.array(faces),process=False)
    m.visual=trimesh.visual.TextureVisuals(uv=np.array(uvs),material=PC)
    scene.add_geometry(m,node_name=name,geom_name=name)
panel(-W/2,-GAP/2,'tetolap_1'); panel(GAP/2,W/2,'tetolap_2')

# ---------- aluminium profiles ----------
def prof(poly,x0,x1,mat_,name):
    m=extr(poly,x0,x1); m.visual=trimesh.visual.TextureVisuals(material=mat_)
    scene.add_geometry(m,node_name=name,geom_name=name)
# front edge: U-channel cap (rounded)
zf,vf=Tpts[-1]
front=Point(zf-0.004,vf-0.004).buffer(0.012).union(box(zf-0.030,vf-0.013,zf-0.004,vf+0.007))
prof(front,-W/2,W/2,alu_mat,'elso_profil')
# rear: dark/grey wall-side edge profile on top of panel at wall
z0,v0=Tpts[0]
rear=box(z0-0.002,v0-0.004,z0+0.045,v0+0.016).buffer(0.003)
prof(rear,-W/2,W/2,alu_mat,'hatso_profil')
# seam bar following the roof arc, centred on the middle arm
idx=np.arange(10,len(Tpts)-6)
o=up_off(Tpts,0.011)[idx]; t=Tpts[idx]
seam=Polygon([tuple(p) for p in t]+[tuple(p) for p in o[::-1]]).buffer(0)
prof(seam,-0.018,0.018,alu_mat,'osszekoto_sin')
# bottom H-profile (under the seam, between arm rail and panel is occupied by arm) skip
bb=scene.bounds; print('bounds',bb.round(3).tolist(),'size',(bb[1]-bb[0]).round(3).tolist())
scene.export('eloteto_300x100.glb')
