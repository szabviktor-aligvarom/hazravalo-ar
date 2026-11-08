import numpy as np
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union

# measured top edge of the roof panel in the product photo (px, 4x crop of altpic_5)
TX=[40,80,120,160,200,240,280,320,360,400,440,480,520,560,600,640,680,720,760,800,840,880,920,960,1000,1040,1052]
TY=[436,412,384,356,321,305,281,258,236,214,196,177,161,145,131,116,104,93,84,74,67,60,55,51,48,45,44]
# smooth
c=np.polyfit(TX,TY,5)
def T(x): return np.polyval(c,x)
xs=np.linspace(32,1052,200)
Tpts=[(x,T(x)) for x in xs]
Tline=LineString(Tpts)

def offs(line,d):
    return line.offset_curve(d)   # + = left of direction of travel

def band(line,d0,d1):
    a=offs(line,d0); b=offs(line,d1)
    return Polygon(list(a.coords)+list(b.coords)[::-1]).buffer(0)

# direction of travel is x increasing (towards wall) => left side is "up" in image coords? test
t=offs(Tline,8)
sign = 1 if np.mean([p[1] for p in t.coords])>np.mean([p[1] for p in Tline.coords]) else -1  # image y down: want +y (downwards)
def below(line,d0,d1): return band(line,sign*d0,sign*d1)

PANEL=7
A=Tline
rail=below(Tline,PANEL,PANEL+15)          # upper rail band
# front web (solid band toward the tip)
fx=np.linspace(32,620,90)
top=[(x,T(x)+PANEL) for x in fx]
dd=lambda x: np.interp(x,[32,60,150,300,440,520,620],[14,17,30,38,38,22,2])
bot=[(x,T(x)+PANEL+15+dd(x)) for x in fx]
web=Polygon(top+bot[::-1]).buffer(2)
# lower rail
lr_top=[(1020,258),(900,238),(800,225),(700,221),(560,222),(440,222)]
lr_bot=[(1020,284),(900,260),(800,247),(700,246),(560,247),(440,246)]
lower=Polygon(lr_top+lr_bot[::-1]).buffer(1)
spokesL=[
 [(596,140),(604,170),(622,200),(645,222),(655,232)],                      # A
 [(733,100),(737,130),(750,165),(770,198),(790,224),(795,236)],            # B
 [(900,64),(880,82),(864,115),(864,160),(878,200),(900,232),(905,248)],   # C thick
 [(832,76),(858,102)], [(630,138),(620,150)]
]
widths=[8,15,22,6,5]
spokes=unary_union([LineString(p).buffer(w/2,cap_style=1) for p,w in zip(spokesL,widths)])
bosses=[(655,232),(795,236),(905,249)]
boss_out=unary_union([Point(b).buffer(15) for b in bosses])
boss_hole=unary_union([Point(b).buffer(6) for b in bosses])
thin=unary_union([lower,spokes,boss_out]).difference(boss_hole)
thick=unary_union([rail,web])
if __name__=='__main__':
    from PIL import Image, ImageDraw
    im=Image.open('/tmp/img/arm5.png').convert('RGB')
    d=ImageDraw.Draw(im,'RGBA')
    def draw(g,col):
        gs=g.geoms if hasattr(g,'geoms') else [g]
        for p in gs:
            d.polygon(list(p.exterior.coords),outline=col,fill=col[:3]+(60,))
            for h in p.interiors: d.polygon(list(h.coords),outline=(0,0,255,255))
    draw(thick,(255,0,0,255)); draw(thin,(0,160,0,255))
    d.line(Tpts,fill=(255,0,255,255),width=1)
    im.save('overlay.png')
