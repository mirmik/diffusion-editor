import numpy as np
import pytest
from diffusion_editor.multiview_studio.mesh_face_budget import refine_to_face_count


def metrics(v, f):
    triangles = v[f]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    edges, counts = np.unique(np.sort(np.concatenate([f[:, [0,1]],f[:, [1,2]],f[:, [2,0]]]),axis=1),axis=0,return_counts=True)
    return np.linalg.norm(cross,axis=1).sum()/2, counts, cross


@pytest.mark.parametrize('target', [4, 6, 100])
def test_closed_surface_count_area_and_topology_preserved(target):
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    out, faces = refine_to_face_count(v,f,target)
    assert len(faces)==target
    area, counts, cross = metrics(out,faces)
    assert area == pytest.approx(metrics(v,f)[0])
    assert np.all(counts==2)
    assert np.all(np.linalg.norm(cross,axis=1)>0)
    # Signed volume checks winding and geometric surface conservation.
    volume = np.einsum('ij,ij->i',out[faces[:,0]],np.cross(out[faces[:,1]],out[faces[:,2]])).sum()/6
    assert volume == pytest.approx(1/6)
    np.testing.assert_array_equal(out.min(axis=0), v.min(axis=0))
    np.testing.assert_array_equal(out.max(axis=0), v.max(axis=0))


def test_boundary_split_can_change_parity_without_holes():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    out,faces=refine_to_face_count(v,[[0,1,2]],10)
    assert len(faces)==10
    area, counts,cross=metrics(out,faces)
    assert area==pytest.approx(.5)
    assert (counts==1).sum()==4
    assert np.all(cross[:,2]>0)


def test_closed_mesh_odd_budget_rejected_and_overbudget_rejected():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    with pytest.raises(ValueError,match='even target'):
        refine_to_face_count(v,f,5)
    with pytest.raises(ValueError,match='below'):
        refine_to_face_count(v,f,2)
