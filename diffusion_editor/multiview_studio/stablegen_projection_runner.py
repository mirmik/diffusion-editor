"""Isolated CUDA render/projection worker; uses the exact Studio camera matrix."""
import json
import sys
from pathlib import Path


def main(root, operation):
    import copy
    import numpy as np
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from PIL import Image
    torch.set_grad_enabled(False)
    request=json.loads((root/'request.json').read_text())
    width,height=request['camera']['size']
    device='cuda'
    def tensor(a,dtype=torch.float32): return torch.as_tensor(np.array(a,copy=True,order="C"),dtype=dtype,device=device)
    def save(name,array): Image.fromarray(array).save(root/name)
    def png(t): return np.rint(np.clip(t.flip(0).cpu().numpy(),0,1)*255).astype(np.uint8)
    def load_image(name): return tensor(np.array(Image.open(root/name).convert('RGBA'),copy=True)/255).flip(0)
    source=root/('input.glb' if operation=='prepare' else 'base.glb')
    scene=trimesh.load(source,force='scene',process=False)
    if len(scene.geometry)!=1:
        raise ValueError('Texture passes currently require a single mesh/material GLB')
    mesh=scene.to_mesh()
    if not len(mesh.faces): raise ValueError('Empty mesh')
    if mesh.visual.kind == 'vertex':
        raise ValueError('Bake vertex colors to a base-color UV atlas before starting texture passes')
    material=copy.deepcopy(getattr(mesh.visual,'material',None))
    uv=getattr(mesh.visual,'uv',None)
    if uv is None:
        from cumesh import Atlas
        atlas=Atlas();atlas.add_mesh(torch.tensor(np.asarray(mesh.vertices),dtype=torch.float32),torch.tensor(np.asarray(mesh.faces),dtype=torch.int32))
        atlas.compute_charts(fix_winding=False,verbose=False)
        atlas.pack_charts(resolution=2048,padding=2,bilinear=True,verbose=False)
        mapping,faces,uv=atlas.get_mesh(0)
        mesh=trimesh.Trimesh(np.asarray(mesh.vertices)[np.asarray(mapping)],np.asarray(faces),process=False)
        uv=np.asarray(uv)
    uv=np.asarray(uv,dtype=np.float32)
    if not np.isfinite(uv).all() or uv.min()<0 or uv.max()>1:
        raise ValueError('Texture passes require finite UVs inside the 0..1 atlas')
    if material is None:
        material=trimesh.visual.material.PBRMaterial(baseColorFactor=[55,55,55,255],roughnessFactor=1.)
    elif not isinstance(material,trimesh.visual.material.PBRMaterial):
        material=material.to_pbr()
    if operation=='prepare':
        image=material.baseColorTexture
        base=np.array(image.convert('RGBA'),copy=True) if image is not None else np.full((2048,2048,4),255,np.uint8)
        factor=np.asarray(material.baseColorFactor if material.baseColorFactor is not None else [255]*4)/255
        colors=base[:,:,:3]/255
        linear=np.where(colors<=.04045,colors/12.92,((colors+.055)/1.055)**2.4)*factor[:3]
        encoded=np.where(linear<=.0031308,linear*12.92,1.055*np.maximum(linear,0)**(1/2.4)-.055)
        base[:,:,:3]=np.rint(np.clip(encoded,0,1)*255).astype(np.uint8)
        base[:,:,3]=np.rint(base[:,:,3]*factor[3]).astype(np.uint8)
        save('base-color.png',base)
        material.baseColorTexture=Image.fromarray(base)
        material.baseColorFactor=np.array([255]*4,np.uint8)
        mesh.visual=trimesh.visual.TextureVisuals(uv=uv,material=material)
        mesh.export(root/'base.glb')
    base=load_image('base-color.png')
    th,tw=base.shape[:2]
    vertices=tensor(mesh.vertices);faces=tensor(mesh.faces,torch.int32);uvs=tensor(uv)
    homo=torch.cat([vertices,torch.ones_like(vertices[:,:1])],dim=1)
    # Termin camera: Vulkan NDC, y down, z 0..1. nvdiffrast: y up, z -1..1.
    clip=homo @ tensor(request['camera']['mvp']).T
    glclip=clip.clone();glclip[:,1]*=-1;glclip[:,2]=2*clip[:,2]-clip[:,3]
    ctx=dr.RasterizeCudaContext()
    # Give each UV triangle a distinct depth so depth peeling detects even
    # perfectly coincident islands (equal-depth fragments would be skipped).
    expanded=uvs[faces.long()].reshape(-1,2)
    z=torch.linspace(-.9,.9,len(faces),device=device).repeat_interleave(3)[:,None]
    uvclip=torch.cat([expanded*2-1,z,torch.ones_like(z)],dim=1)
    uvfaces=torch.arange(len(expanded),device=device,dtype=torch.int32).reshape(-1,3)
    with dr.DepthPeeler(ctx,uvclip[None],uvfaces,[th,tw]) as peeler:
        atlas_rast,_=peeler.rasterize_next_layer()
        overlap,_=peeler.rasterize_next_layer()
    # Freeze ambiguous texels and their bilinear-filter footprint. Small UV
    # overlaps from generated atlases must not block otherwise safe edits.
    ambiguous=(overlap[...,3]>0).float()
    protected=torch.nn.functional.max_pool2d(ambiguous[:,None],3,stride=1,padding=1)[0,0]>0
    exclusive=(atlas_rast[0,:,:,3]>0)&~protected
    if not bool(exclusive.any()):
        raise ValueError('Overlapping UV islands leave no safely editable texels; unwrap the mesh first')
    save('protected-uv-texels.png',png(protected.float()))
    rast,_=dr.rasterize(ctx,glclip[None],faces,resolution=[height,width])
    screen_uv,_=dr.interpolate(uvs[None],rast,faces)
    rgb=dr.texture(base[None],screen_uv,filter_mode='linear',boundary_mode='clamp')[0]
    silhouette=rast[0,:,:,3]>0
    clipw,_=dr.interpolate(clip[:,3:4][None].contiguous(),rast,faces)
    distance=clipw[0,:,:,0]
    eye=tensor(request['camera']['eye'])
    normals=tensor(mesh.face_normals)
    centers=vertices[faces.long()].mean(dim=1)
    direction=torch.nn.functional.normalize(eye-centers,dim=1)
    angles=(normals*direction).sum(dim=1).abs()
    angle_image=angles[(rast[0,:,:,3].long()-1).clamp(min=0)]
    visible=silhouette & (angle_image>=.2)
    if operation=='prepare':
        rgb[~silhouette]=torch.tensor([.5,.5,.5,1],device=device)
        save('input-rgb.png',png(rgb))
        near=distance[silhouette].min();far=distance[silhouette].max()
        depth=torch.where(silhouette,1-(distance-near)/(far-near).clamp(min=1e-6),0)
        save('control-depth.png',png(depth).astype(np.uint8))
        save('visible-mask.png',png(visible.float()))
        save('mask.png',np.zeros((height,width),np.uint8))
        (root/'prepared.json').write_text(json.dumps(dict(status='success',faces=len(mesh.faces),atlas_size=[tw,th],protected_uv_texels=int(protected.sum()))))
        return
    if operation!='project': raise ValueError(operation)
    # UV-space raster maps each editable texel to the original surface.
    projected,_=dr.interpolate(glclip[None],atlas_rast,faces)
    p=projected[0];ndc=p[:,:,:3]/p[:,:,3:4].clamp(min=1e-8)
    coords=(ndc[:,:,:2]+1)/2
    ids=atlas_rast[0,:,:,3]
    allowed=exclusive&(p[:,:,3]>0)&(coords.min(dim=-1).values>=0)&(coords.max(dim=-1).values<=1)
    # A raster pixel can contain several tiny triangles. Compare the exact
    # first ray hit at each atlas texel instead of requiring equal pixel IDs.
    from cumesh import cuBVH
    bvh=cuBVH(vertices,faces)
    points,_=dr.interpolate(vertices[None],atlas_rast,faces)
    selected=torch.nonzero(allowed.reshape(-1),as_tuple=False)[:,0]
    flat_points=points[0].reshape(-1,3)
    visibility=torch.zeros(th*tw,dtype=torch.bool,device=device)
    tolerance=max(float(np.max(mesh.extents))*1e-5,1e-7)
    for chunk in selected.split(262144):
        targets=flat_points[chunk]
        origins=eye.expand_as(targets)
        rays=torch.nn.functional.normalize(targets-origins,dim=1)
        hits,face_ids,_=bvh.ray_trace(origins,rays)
        visibility[chunk]=(face_ids>=0)&(torch.linalg.vector_norm(hits-targets,dim=1)<=tolerance)
    allowed &= visibility.reshape(th,tw)
    angle=angles[(ids.long()-1).clamp(min=0)]
    allowed &= angle>=.2
    mask=load_image('mask.png')[:,:,:1].contiguous()
    alpha=dr.texture(mask[None],coords[None],filter_mode='linear',boundary_mode='zero')[0,:,:,0]
    alpha*=allowed.float()*((angle-.2)/.2).clamp(0,1)
    candidate=load_image('candidate.png')
    color=dr.texture(candidate[None],coords[None],filter_mode='linear',boundary_mode='zero')[0,:,:,:3]
    def linear(c): return torch.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
    def srgb(c): return torch.where(c<=.0031308,c*12.92,1.055*c.clamp(min=0)**(1/2.4)-.055)
    result=base.clone()
    result[:,:,:3]=srgb(linear(base[:,:,:3])*(1-alpha[:,:,None])+linear(color)*alpha[:,:,None])
    result[alpha==0]=base[alpha==0]
    result_image=png(result);old=png(base);changed=np.any(result_image!=old,axis=2)
    permitted=alpha.flip(0).cpu().numpy()>0
    assert not changed[~permitted].any()
    if not permitted.any(): raise ValueError('Mask does not cover any visible atlas texels')
    save('application-mask.png',png(alpha))
    save('allowed-texels.png',permitted.astype(np.uint8)*255)
    save('result-color.png',result_image)
    material.baseColorTexture=Image.fromarray(result_image)
    material.baseColorFactor=np.array([255]*4,np.uint8)
    mesh.visual=trimesh.visual.TextureVisuals(uv=uv,material=material)
    mesh.export(root/'candidate.glb')
    (root/'projection.json').write_text(json.dumps(dict(status='success',shape='candidate.glb',faces=len(mesh.faces),changed_texels=int(changed.sum()),permitted_texels=int(permitted.sum()),outside_mask_changes=0),indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve(),sys.argv[2])
