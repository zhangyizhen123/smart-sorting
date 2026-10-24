#!/usr/bin/env python3
"""Generate nine closed, outward-facing binary STL solids in millimetres.
Python 3 standard library only. Run: python3 generate_models.py
"""
import json
import math
from pathlib import Path
import struct
import zipfile
from collections import Counter, defaultdict
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
SIZE = 40.0
SEGMENTS = 256  # divisible by four, preserving exact X/Y extrema
SPHERE_LATITUDES = 128


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def prism(n, radius, height):
    vertices = [(radius*math.cos(2*math.pi*i/n), radius*math.sin(2*math.pi*i/n), z)
                for z in (0.0, height) for i in range(n)]
    faces = []
    for i in range(1, n-1):
        faces += [(0, i, i+1), (n, n+i, n+i+1)]
    for i in range(n):
        j = (i+1) % n
        faces += [(i, j, n+j), (i, n+j, n+i)]
    return vertices, faces


def box(x, y, z):
    vertices = [(a, b, c) for c in (0.0, z) for b in (-y/2, y/2) for a in (-x/2, x/2)]
    quads = [(0,1,3,2), (4,5,7,6), (0,1,5,4), (2,3,7,6), (0,2,6,4), (1,3,7,5)]
    faces = [(q[0], q[1], q[2]) for q in quads] + [(q[0], q[2], q[3]) for q in quads]
    return vertices, faces


def cone(n, radius, height):
    vertices = [(radius*math.cos(2*math.pi*i/n), radius*math.sin(2*math.pi*i/n), 0.0)
                for i in range(n)] + [(0.0, 0.0, height)]
    faces = [(0, i, i+1) for i in range(1, n-1)]
    faces += [(i, (i+1) % n, n) for i in range(n)]
    return vertices, faces


def tetrahedron(edge):
    base_height = edge*math.sqrt(3)/2
    vertices = [(-edge/2, -base_height/3, 0.0), (edge/2, -base_height/3, 0.0),
                (0.0, 2*base_height/3, 0.0), (0.0, 0.0, edge*math.sqrt(2/3))]
    return vertices, [(0,1,2), (0,1,3), (1,2,3), (2,0,3)]


def sphere(n, latitudes, radius):
    vertices = [(0.0, 0.0, 0.0)]
    for j in range(1, latitudes):
        theta = math.pi*j/latitudes
        for i in range(n):
            phi = 2*math.pi*i/n
            vertices.append((radius*math.sin(theta)*math.cos(phi),
                             radius*math.sin(theta)*math.sin(phi), radius*(1-math.cos(theta))))
    top = len(vertices)
    vertices.append((0.0, 0.0, 2*radius))
    faces = [(0, 1+i, 1+(i+1)%n) for i in range(n)]
    for j in range(latitudes-2):
        lower, upper = 1+j*n, 1+(j+1)*n
        for i in range(n):
            k = (i+1)%n
            faces += [(lower+i, lower+k, upper+k), (lower+i, upper+k, upper+i)]
    last = 1+(latitudes-2)*n
    faces += [(last+i, last+(i+1)%n, top) for i in range(n)]
    return vertices, faces


def models(segments=SEGMENTS, latitudes=SPHERE_LATITUDES):
    s, r = SIZE, SIZE/2
    return {
        'cylinder': prism(segments, r, s),
        'cube': box(s, s, s),
        'cuboid': box(s, s*0.75, s*0.5),
        'tri_prism': prism(3, r, s),
        'pent_prism': prism(5, r, s),
        'hex_prism': prism(6, r, s),
        'cone': cone(segments, r, s),
        'tetrahedron': tetrahedron(s),
        'sphere': sphere(segments, latitudes, r),
    }


def outward(vertices, faces):
    # All solids are convex, so the vertex average lies inside them.
    center = tuple(sum(v[k] for v in vertices)/len(vertices) for k in range(3))
    result = []
    for a,b,c in faces:
        normal = cross(sub(vertices[b], vertices[a]), sub(vertices[c], vertices[a]))
        if dot(normal, sub(vertices[a], center)) < 0:
            b,c = c,b
        result.append((a,b,c))
    return result


def write_stl(path, vertices, faces):
    with path.open('wb') as f:
        f.write(b'Closed solid; coordinates in mm; nominal maximum dimension 40 mm'.ljust(80, b' '))
        f.write(struct.pack('<I', len(faces)))
        for a,b,c in faces:
            normal = cross(sub(vertices[b], vertices[a]), sub(vertices[c], vertices[a]))
            norm = math.sqrt(dot(normal, normal))
            normal = tuple(x/norm for x in normal)
            f.write(struct.pack('<12fH', *normal, *vertices[a], *vertices[b], *vertices[c], 0))


def validate_stl(path, expected_volume):
    # Read the actual float32 output, not the original modelling data.
    raw = path.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    assert len(raw) == 84+count*50
    vertex_ids, vertices, faces = {}, [], []
    for t in range(count):
        values = struct.unpack_from('<12fH', raw, 84+t*50)
        ids = []
        for k in (3,6,9):
            vertex = tuple(values[k:k+3])
            assert all(math.isfinite(x) for x in vertex)
            if vertex not in vertex_ids:
                vertex_ids[vertex] = len(vertices)
                vertices.append(vertex)
            ids.append(vertex_ids[vertex])
        a,b,c = (vertices[i] for i in ids)
        normal = cross(sub(b,a), sub(c,a))
        assert dot(normal, normal) > 1e-16, 'degenerate triangle'
        assert dot(normal, values[:3]) > 0, 'incorrect normal'
        faces.append(ids)
    edges, balance, neighbors = Counter(), Counter(), defaultdict(set)
    volume = 0.0
    face_keys = set()
    for a,b,c in faces:
        key = tuple(sorted((a,b,c)))
        assert key not in face_keys, 'duplicate face'
        face_keys.add(key)
        for u,v in ((a,b),(b,c),(c,a)):
            key = (min(u,v), max(u,v))
            edges[key] += 1
            balance[key] += 1 if u < v else -1
            neighbors[u].add(v)
            neighbors[v].add(u)
        volume += dot(vertices[a], cross(vertices[b],vertices[c]))/6
    assert all(n == 2 for n in edges.values()), 'non-manifold/open edge'
    assert all(n == 0 for n in balance.values()), 'inconsistent winding'
    seen, stack = {0}, [0]
    while stack:
        for v in neighbors[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    assert len(seen) == len(vertices), 'disconnected mesh'
    assert len(vertices)-len(edges)+len(faces) == 2, 'unexpected topology'
    low = [min(v[k] for v in vertices) for k in range(3)]
    high = [max(v[k] for v in vertices) for k in range(3)]
    size = [b-a for a,b in zip(low,high)]
    assert abs(low[2]) < 1e-6, 'not seated at Z=0'
    assert max(size) <= SIZE+1e-5
    assert abs(max(size)-SIZE) < 1e-5
    assert volume > 0
    error = abs(volume-expected_volume)/expected_volume
    assert error < 0.001, 'volume differs from analytical solid by >=0.1%'
    return {'file': path.name, 'units': 'mm', 'dimensions_mm': [round(x,6) for x in size],
            'bounds_mm': [low,high], 'triangles': count, 'vertices': len(vertices),
            'watertight': True, 'consistent_winding': True, 'connected_components': 1,
            'euler_characteristic': 2, 'volume_mm3': round(volume,6),
            'analytical_volume_error_percent': round(error*100,6), 'bytes': len(raw)}


def preview():
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1080" viewBox="0 0 1080 1080">',
           '<rect width="1080" height="1080" fill="#f1f5f9"/>',
           '<text x="36" y="40" font-family="sans-serif" font-size="24" fill="#0f172a">Sorting blocks | nominal maximum size 40 mm</text>']
    colors = [(59,130,246),(245,158,11),(16,185,129),(139,92,246),(239,68,68),
              (6,182,212),(249,115,22),(99,102,241),(236,72,153)]
    camera = (1.0,-1.0,0.82)
    for index,(name,(vertices,faces)) in enumerate(models(32,16).items()):
        faces = outward(vertices,faces)
        ox,oy = 20+(index%3)*355, 62+(index//3)*335
        svg.append(f'<rect x="{ox}" y="{oy}" width="335" height="315" rx="16" fill="white"/>')
        projected = [((x+y)*0.7071, (x-y)*0.40-z*0.82) for x,y,z in vertices]
        cx = (min(x for x,y in projected)+max(x for x,y in projected))/2
        cy = (min(y for x,y in projected)+max(y for x,y in projected))/2
        visible = []
        for a,b,c in faces:
            normal = cross(sub(vertices[b],vertices[a]),sub(vertices[c],vertices[a]))
            if dot(normal,camera) <= 0:
                continue
            light = 0.60+0.40*max(0,dot(normal,(0.25,-0.45,0.86))/math.sqrt(dot(normal,normal)))
            color = '#'+''.join(f'{min(255,int(v*light)):02x}' for v in colors[index])
            points = ' '.join(f'{ox+167+(projected[v][0]-cx)*3.7:.2f},{oy+133+(projected[v][1]-cy)*3.7:.2f}' for v in (a,b,c))
            depth = sum(dot(vertices[v],camera) for v in (a,b,c))
            visible.append((depth, f'<polygon points="{points}" fill="{color}" stroke="{color}" stroke-width="0.4"/>'))
        svg.extend(p for _,p in sorted(visible))
        svg.append(f'<text x="{ox+20}" y="{oy+278}" font-family="monospace" font-size="20" fill="#0f172a">{escape(name)}.stl</text>')
    svg.append('</svg>')
    (OUT/'preview.svg').write_text('\n'.join(svg), encoding='utf-8')


def main():
    s, r = SIZE, SIZE/2
    expected = {'cylinder': math.pi*r*r*s, 'cube': s**3, 'cuboid': s**3*0.75*0.5,
                'cone': math.pi*r*r*s/3, 'tetrahedron': s**3/(6*math.sqrt(2)),
                'sphere': 4*math.pi*r**3/3}
    for name,n in [('tri_prism',3),('pent_prism',5),('hex_prism',6)]:
        expected[name] = n*r*r*math.sin(2*math.pi/n)*s/2
    results = []
    for name,(vertices,faces) in models().items():
        path = OUT/f'{name}.stl'
        write_stl(path, vertices, outward(vertices,faces))
        result = validate_stl(path, expected[name])
        results.append(result)
        print(f"PASS {path.name:20s} {result['dimensions_mm']}  triangles={result['triangles']}")
    (OUT/'validation_report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    preview()
    with zipfile.ZipFile(OUT/'blocks_40mm.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for pattern in ('*.stl','*.md','*.py','*.json','*.svg'):
            for path in sorted(OUT.glob(pattern)):
                archive.write(path,path.name)
    print('All 9 STLs validated. Archive: blocks_40mm.zip')


if __name__ == '__main__':
    main()
