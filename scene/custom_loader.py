import numpy as np
from scene.colmap_loader import rotmat2qvec, create_extrinsics

def read_custom_intrinsics(path):
    intrinsics = {}
    intrinsics = np.loadtxt(path)
    intrinsics_dict = {}
    for intr in intrinsics:
        cam_id = intr[0]
        cam_intrinsics = intr[1:].reshape(3,3)
        intrinsics_dict[cam_id] = cam_intrinsics
    return intrinsics_dict

def read_custom_extrinsics(path):
    extrinsics = np.loadtxt(path).reshape(-1, 4, 4)
    extrinsics_dict = {}
    for idx, extr in enumerate(extrinsics):
        extrinsics_dict[idx] = extr
    return extrinsics_dict

def create_colmap_extrinsic_format(cam_extrinsics):
    """
    @desc:
        - set imagename to 'image_{camid}/{06:d}.png'
        - set qvec and tvec
        - create Image namedTuple by using the colmap_loader
        -> "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
    @param:
        - cam_extrinsics: {cam_id(int): extr(Nx4x4)}

    @return:
        - extrinsics: {idx: Image(qvec, tvec, name, camera_id, ...)}
    """

    extrinsics = {}
    extr_id = 0
    for _, cam_id in enumerate(cam_extrinsics):
        extrinsics_per_cam = cam_extrinsics[cam_id]
        for idx, (key, extr) in enumerate(extrinsics_per_cam.items()):
            image_name = f'{"image_%02d" % cam_id}/{idx:06d}.png'
            qvec = rotmat2qvec(extr[:3, :3])
            tvec = extr[:3, 3]
            extrinsics[extr_id] = create_extrinsics(id=extr_id, qvec=qvec, tvec=tvec,
                    camera_id=cam_id, name=image_name,
                    xys=None, point3D_ids=None)
            extr_id += 1

    return extrinsics