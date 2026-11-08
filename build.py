import numpy as np, trimesh, math
from shapely.geometry import Polygon, LineString, MultiPolygon, box
from shapely.ops import unary_union
from trimesh.visual.material import PBRMaterial

W=3.0; D=0.955; H=0.28; DROP=0.30
R=(D*D+DROP*DROP)/(2*DROP)          # roof arc radius
CY=H-R                              # circle centre height (centre at z=0)
def arc_pts(r,n=40,z0=0.0,z1=D):
    a0=math.asin(z0/r) if z0>0 else 0; a1=math.asin(min(z1/r,1))
    return [(r*math.sin(a),CY+r*math.cos(a)) for a in np.linspace(a0,a1,n)]

def to_mesh(poly,x0,x1):
    """poly in (z,y) metres -> extruded along x from x0 to x1"""
    m=trimesh.creation.extrude_polygon(poly,height=x1-x0)   # (u=z,v=y,w)
    v=m.vertices.copy()
    nv=np.column_stack([v[:,2]+x0,v[:,1],v[:,0]])
    mm=trimesh.Trimesh(nv,m.faces.copy(),process=True)
    mm.invert()
    return mm

def mat(name,rgb,metal=0.0,rough=0.5,alpha=255):
    return PBRMaterial(name=name,baseColorFactor=[*rgb,alpha],metallicFactor=metal,roughnessFactor=rough,
        alphaMode='BLEND' if alpha<255 else 'OPAQUE',doubleSided=alpha<255)

# ---------- arm (ABS bracket) ----------
upper=arc_pts(R,50,0.0,D-0.012)
tip=upper[-1]
# lower edge: quadratic bezier from wall bottom to tip
p0=np.array([0.0,0.0]);p2=np.array(tip)-np.array([0.0,0.045]);p1=np.array([0.50,0.02])
lower=[tuple((1-t)**2*p0+2*(1-t)*t*p1+t**2*p2) for t in np.linspace(1,0,40)]
fan=Polygon(upper+lower).buffer(0)
plate=box(0.0,0.0,0.035,H)
outer=unary_union([fan,plate]).buffer(0)
inner=fan.buffer(-0.022)
ribs=[]
for zt in (0.18,0.40,0.62,0.82):
    a=math.asin(zt/R); top=(R*math.sin(a),CY+R*math.cos(a))
    ribs.append(LineString([(0.05,0.12),(zt*0.55+0.05,0.12+ (top[1]-0.12)*0.35),top]).buffer(0.010))
ribs.append(LineString([(0.05,0.03),(0.5,0.03),(0.9,tip[1]-0.03)]).buffer(0.008))
ribs=unary_union(ribs)
holes=inner.difference(ribs).difference(box(-1,-1,0.07,1))
geoms=list(holes.geoms) if hasattr(holes,'geoms') else [holes]
geoms=[g for g in geoms if g.area>0.0006]
arm_poly=outer
for g in geoms: arm_poly=arm_poly.difference(g.buffer(-0.004))
arm_poly=arm_poly.buffer(0)
if isinstance(arm_poly,MultiPolygon): arm_poly=max(arm_poly.geoms,key=lambda g:g.area)
ARM_W=0.045
grey=mat('ABS_szurke',[0.42,0.44,0.46],0.0,0.6)

scene=trimesh.Scene()
for xc in (0.035,W/2,W-0.035):
    m=to_mesh(arm_poly,xc-ARM_W/2-W/2,xc+ARM_W/2-W/2)
    m.visual=trimesh.visual.TextureVisuals(material=grey)
    scene.add_geometry(m,node_name=f'kar_{xc:.2f}',geom_name=f'kar_{xc:.2f}')

# ---------- polycarbonate panels (2 x 150x100) ----------
T=0.0052
for i,(a,b) in enumerate(((0,W/2),(W/2,W))):
    gap=0.004
    o=arc_pts(R+T+0.001,50,0.0,D); inn=arc_pts(R+0.001,50,0.0,D)
    poly=Polygon(o+inn[::-1]).buffer(0)
    m=to_mesh(poly,a-W/2+gap/2,b-W/2-gap/2)
    m.visual=trimesh.visual.TextureVisuals(material=mat('polikarbonat',[0.70,0.80,0.84],0.0,0.12,85))
    scene.add_geometry(m,node_name=f'tetolap_{i+1}',geom_name=f'tetolap_{i+1}')

# ---------- aluminium profiles ----------
alu=mat('aluminium',[0.74,0.76,0.78],0.9,0.35)
dark=mat('hatso_profil',[0.30,0.31,0.33],0.2,0.5)
def add(poly,x0,x1,material,name):
    m=to_mesh(poly,x0,x1); m.visual=trimesh.visual.TextureVisuals(material=material)
    scene.add_geometry(m,node_name=name,geom_name=name)
# back profile along the wall edge
add(box(-0.004,H-0.004,0.040,H+0.026),-W/2,W/2,dark,'hatso_profil')
# front edge cap
a=math.asin((D-0.001)/R)
fz,fy=R*math.sin(a),CY+R*math.cos(a)
add(Polygon([(fz-0.030,fy+0.020),(fz+0.006,fy+0.010),(fz+0.006,fy-0.020),(fz-0.030,fy-0.008)]).buffer(0),-W/2,W/2,alu,'elso_profil')
# seam bar between the two panels (follows the arc, centred on the middle arm)
sp=Polygon(arc_pts(R+T+0.012,50,0.04,D-0.03)+arc_pts(R+T,50,0.04,D-0.03)[::-1]).buffer(0)
add(sp,-0.016,0.016,alu,'osszekoto_sin')

scene.export('/home/ninja/ar/eloteto_300x100.glb')
b=scene.bounds; print('bounds m',b.round(3).tolist(),'size',(b[1]-b[0]).round(3).tolist())
