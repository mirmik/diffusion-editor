"""Meet an exact triangle count by conforming, surface-preserving subdivision."""
from __future__ import annotations

import numpy as np


def refine_to_face_count(vertices, faces, target: int):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if target < len(faces) or not len(faces):
        raise ValueError('Face budget requires a nonempty mesh already below the target')
    deficit = target - len(faces)
    if deficit % 2:
        # Splitting every incident face at one edge is conforming. Prefer a
        # boundary edge (+1). A closed manifold cannot have an odd face count.
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        candidates = np.flatnonzero((counts % 2 == 1) & (counts <= deficit))
        if not len(candidates):
            raise ValueError('Exact face count cannot preserve this topology; use an even target for a closed mesh')
        edge = unique[candidates[np.argmin(counts[candidates])]]
        midpoint = len(vertices)
        vertices = np.vstack([vertices, vertices[edge].mean(axis=0)])
        affected = np.flatnonzero(np.sum(np.isin(faces, edge), axis=1) == 2)
        replacement = []
        for face in faces[affected]:
            for i in range(3):
                a, b, c = face[i], face[(i + 1) % 3], face[(i + 2) % 3]
                if a in edge and b in edge:
                    replacement.extend([(a, midpoint, c), (midpoint, b, c)])
                    break
        keep = np.ones(len(faces), dtype=bool); keep[affected] = False
        faces = np.vstack([faces[keep], replacement])
    while len(faces) < target:
        count = min((target - len(faces)) // 2, len(faces))
        if not count:
            raise RuntimeError('Could not satisfy exact face budget')
        triangles = vertices[faces]
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        selected = np.argsort(area)[-count:]
        if np.any(area[selected] == 0):
            raise ValueError('Cannot subdivide degenerate faces')
        centers = triangles[selected].mean(axis=1)
        ids = np.arange(len(vertices), len(vertices) + count)
        a, b, c = faces[selected].T
        replacement = np.vstack([np.column_stack([a, b, ids]), np.column_stack([b, c, ids]), np.column_stack([c, a, ids])])
        keep = np.ones(len(faces), dtype=bool); keep[selected] = False
        vertices = np.vstack([vertices, centers])
        faces = np.vstack([faces[keep], replacement])
    return vertices, faces
