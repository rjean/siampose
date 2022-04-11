import numpy as np
import open3d as o3d
from siampose.data.objectron.dataset.box import EDGES, FACES
import PIL
from PIL import ImageDraw

def get_dist_from_plane(plane_normal : np.ndarray, plane_center: np.ndarray, point: np.ndarray) -> float:
    """Get the smallest distance between a point and a plane.

    Args:
        plane_normal (np.ndarray): Normal vector of the plane. (Perpendicular to the plane.)
        plane_center (np.ndarray): A point lying on the plane. 
        point (np.ndarray): The point on which we want to evaluate the distance.

    Returns:
        float: Smallest distance between the plane and the point.
    """
    plane = get_plane_equation_center_normal(plane_center, plane_normal)
    a, b, c, d = plane
    X,Y,Z = point
    return abs(a*X + b*Y + c*Z + d)/np.sqrt(a**2+b**2+c**2)

def get_plane_equation_center_normal(plane_center : np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Get a plane equation from it's normal vector and a point on the plane.

    Args:
        plane_center (np.ndarray): Normal vector of the plane. (Perpendicular to the plane.)
        plane_normal (np.ndarray): A point lying on the plane. 

    Returns:
        np.ndarray: The plane equation is [ax + by +cz -d]=0
    """
    #ax +by + cz = d 
    #where d= ax0 + by0 +czo
    #https://tutorial.math.lamar.edu/classes/calciii/eqnsofplanes.aspx

    #The plane equation is [ax + by +cz -d]=0
    return np.hstack((plane_normal, -np.dot(plane_center,plane_normal)))

def get_planes_intersections(plane1 : np.ndarray, plane2: np.ndarray, plane3: np.ndarray) -> np.ndarray:
    """Get the intersection point between 3 planes.

    Args:
        plane1 (np.ndarray): Equation of the first plane. [ax + by +cz -d]=0
        plane2 (np.ndarray): Equation of the second plane. [ax + by +cz -d]=0
        plane3 (np.ndarray): Equation of the third plane. [ax + by +cz -d]=0
    
    Returns:
        np.ndarray: Intersection point.
    """
    #See Hartley, p.589
    #The plane equation is [ax + by +cz -d]=0
    A = np.vstack((plane1, plane2, plane3))
    u,d,v= np.linalg.svd(A)
    v = v #Numpy returns V transposed. We need v.
    u,d,v
    point = v[-1]
    point = point / point[-1] #Normalization of homogenous coordinates.
    return point[0:3]

#https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
def get_intersect_relative_to_camera(plane_normal: np.ndarray, plane_center: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Gets the intersection between a line going for (0,0,0) to point and the plane.

    Args:
        plane_center (np.ndarray): Normal vector of the plane. (Perpendicular to the plane.)
        plane_normal (np.ndarray): A point lying on the plane. 
        point (np.ndarray): Point defining the line to the origin (0,0,0)

    Returns:
        np.ndarray: Intersection point.
    """
    p0 = plane_center
    n = plane_normal
    l0 = np.array([0,0,0]) #camera center
    l = point #Relative to camera center
    d = np.dot((p0-l0), n)/np.dot(l,n)
    intersect = l0 + d*l
    return intersect



def get_3d_bbox_center(points3d: np.ndarray) -> np.ndarray:
    """Gets the center of a 3d bounding box.

    Args:
        points3d (np.ndarray): 3D Bounding box (8x3)

    Returns:
        np.ndarray: Center of the bounding box.
    """
    return 0.5 * (points3d[1:].min(axis=0) + points3d[1:].max(axis=0))


def project_3d_to_2d(points3d, intrinsics):
    """Projects 3d points, (in the Google format) to pixel
    location using the intrinsic matrix. The intrinsic matrix
    must be scaled if the image is scaled.

    Args:
        points3d ([list]): List of 3D points in camera frame.
        intrinsics ([np.array]): Intrinsic parameters of the camera.

    Returns:
        [np.array]: Location of the project points in pixels.
    """
    p3d_fixed = np.array(points3d) #3d points from the dataset. 
    p3d_fixed[:,2]=-p3d_fixed[:,2] #Reverse z axis
    res = np.dot(intrinsics, p3d_fixed.T)
    res=res/res[2]
    x = res[1]
    y = res[0]
    return np.swapaxes(np.vstack([x, y]),0,1)

def scale_intrinsics(intrinsics: np.array, scale_x: float, scale_y:float):
    """Scale an intrinsic matrix to simulate a lower resolution sensor. This is 
    useful for scene reconstruction when the original image was taken at a resolution
    and the analysed image is in another resolution.

    Args:
        intrinsics (np.array): Camera intrinsic matrix.
        scale_x (float): Scaling factor for x axis.
        scale_y (float): Scaling factor for y axis
    """
    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics[0,:]*=scale_x
    scaled_intrinsics[1,:]*=scale_y
    return scaled_intrinsics



def intersect_plane_with_line(plane_normal, plane_center, point1, point0=np.array([0,0,0])):
    p0 = plane_center
    n = plane_normal
    l0 = point0 #Line start.
    l = point1 #Line end
    d = np.dot((p0-l0), n)/np.dot(l,n)
    intersect = l0 + d*l
    return intersect

def get_bbox_area(bbox):
    dx = bbox[2]-bbox[0]
    dy = bbox[3]-bbox[1]
    return np.sqrt(dx**2+dy**2)


def scale_3d_bbox(bbox, factor):
    assert factor>0
    center = bbox[0]
    bbox = (bbox-center)*factor + center 
    return bbox

def snap_to_plane(points3d, plane_normal_query, plane_center_query, point_on_center_line, obj_radius=None):
    #if obj_radius is None:
    #    obj_radius = get_dist_from_plane(plane_normal_result, plane_center_result, points3d[0])
    offset = obj_radius * plane_normal_query
    new_center = intersect_plane_with_line(plane_normal_query, plane_center_query+offset, point_on_center_line) 
    snapped = points3d - points3d[0] + new_center
    return snapped

# %%
# From Stack Overflow: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -float(v[2]), float(v[1])], [float(v[2]), 0, -float(v[0])], [-float(v[1]), float(v[0]), 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


# %%
#https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

def get_rotation_matrix_vector_angle(u, theta):
    R = np.zeros((3,3))
    ux, uy, uz = u/np.linalg.norm(u)
    #Diagonal
    R[0,0] = np.cos(theta) + (ux**2)*(1-np.cos(theta))
    R[1,1] = np.cos(theta) + (uy**2)*(1-np.cos(theta))
    R[2,2] = np.cos(theta) + (uz**2)*(1-np.cos(theta))
    # Row 1
    R[0,1] = ux*uy*(1-np.cos(theta)) - uz*np.sin(theta)
    R[0,2] = ux*uz*(1-np.cos(theta)) + uy*np.sin(theta)
    #Row 2
    R[1,0] = uy*ux*(1-np.cos(theta)) + uz*np.sin(theta)
    R[1,2] = uy*uz*(1-np.cos(theta)) - ux*np.sin(theta)
    #Row 3
    R[2,0] = uz*ux*(1-np.cos(theta)) - uy*np.sin(theta)
    R[2,1] = uz*uy*(1-np.cos(theta)) + ux*np.sin(theta)
    return R

def rotate_around_center(points, R, center=(0,0,0)):
    center = points[0]
    original_centered = points-center
    original_centered_rotated = np.dot(R, original_centered.T).T
    rotated = original_centered_rotated + center
    return rotated

def rotate_bbox_around_its_center(points, theta):
    center = points[0]
    axis = points[3]-points[1]
    R = get_rotation_matrix_vector_angle(axis, theta)
    return rotate_around_center(points, R, center)

def get_middle_bottom_point(points3d, plane_normal):
    face = find_face_facing_plane_v2(points3d,plane_normal)
    return points3d[face].mean(axis=0)
    #return points3d[BOTTOM_POINTS].mean(axis=0)

def snap_box_to_plane(points3d, plane_normal, plane_center, align_axis=True):
    bottom_middle=get_middle_bottom_point(points3d, plane_normal)
    intersect = get_intersect_relative_to_camera(plane_normal, plane_center, bottom_middle)
    snapped= points3d-(bottom_middle-intersect)
    if align_axis:
        box_normal = snapped[0] - intersect
        #box_normal[0]=-box_normal
        box_rotation = rotation_matrix_from_vectors(box_normal, plane_normal)
        pcd_snapped = o3d.geometry.PointCloud()
        pcd_snapped.points = o3d.utility.Vector3dVector(np.array(snapped))
        pcd_snapped.rotate(box_rotation, intersect)
        snapped_rotated = np.array(pcd_snapped.points)
        snapped = snapped_rotated
    return snapped, intersect

def get_cube(scaling_factor=1, center=(0,0,0)):
    cube= (
        (0, 0, 0 ),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, -1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, -1, 1),
        (-1, 1, 1)
        )
    cube = np.array(cube)*scaling_factor
    cube += np.array(center)
    return cube

def get_plane_points_z(scene_plane: np.ndarray, plane_center: np.ndarray, xmin=-3, ymin=-3, xmax=3, ymax=3):
    """Create a bounding box representing the ground plane around the origin for visualisation.

    Args:
        scene_plane (np.ndarray): Scene plane equation [ax by cz d]
        plane_center (np.ndarray): Plane center of the scene.
        xmin (int, optional): [description]. Defaults to -3.
        ymin (int, optional): [description]. Defaults to -3.
        xmax (int, optional): [description]. Defaults to 3.
        ymax (int, optional): [description]. Defaults to 3.

    Returns:
        np.array: Ground plane bounding box.
    """
    bottom_plane = [0,1,0,-ymin] #y=-1
    top_plane = [0,1,0,-ymax] #y=1
    left_plane = [1,0,0,-xmin] #x=-1
    right_plane = [1,0,0,-xmax] #x=1
    top_left = get_planes_intersections(top_plane, left_plane, scene_plane)
    bottom_left = get_planes_intersections(bottom_plane, left_plane, scene_plane)
    top_right = get_planes_intersections(top_plane, right_plane, scene_plane)
    bottom_right = get_planes_intersections(bottom_plane, right_plane, scene_plane)

    eps=0.01
    #EDGES = (
    #[1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    #[1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    #[1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
    #)
    plane_points = np.vstack([top_left, bottom_left, top_right, bottom_right])
    plane_points_offset = np.vstack([top_left-eps, bottom_left-eps, top_right-eps, bottom_right-eps])
    plane_rect = np.vstack((plane_points, plane_points_offset, plane_center))
    return plane_rect


def find_face_facing_plane_v2(points, plane_normal):
    min_dist=1e6
    most_probable_face = [1, 2, 6, 5]
    for face in FACES:
        points_from_current_face = points[face]
        closest_point_idx = int(np.argmin(np.dot(plane_normal, (points_from_current_face+plane_normal*20).T)))
        closest_corner = points_from_current_face[closest_point_idx]
        points_relative_to_closest_corner = points_from_current_face-closest_corner
        normalized = points_relative_to_closest_corner/np.linalg.norm(points_relative_to_closest_corner, axis=1).reshape(4,1)
        normalized[closest_point_idx] = np.array([0,0,0])
        dist = np.dot(plane_normal, (closest_corner+normalized+plane_normal*20).T).mean()
        if dist<min_dist:
            most_probable_face=face
            min_dist = dist
    return most_probable_face


def get_bbox(points2d_px: list, width: int, height: int, clip=True):
    """Get 2d bounding box in pixel for a normalized 2d point list.

    Args:
        points2d_px (list): List of normalized 2d points.
        width (int): Image width in pixels.
        height (int): Image heigh in pixels.
        clip (bool, optional): Clip values outside of picture. Defaults to True.

    Returns:
        tuple: x_min, y_min, x_max, y_max in pixels.
    """
    x_min = 10000
    x_max = 0
    y_min = 10000
    y_max = 0
    for point2d_px in points2d_px:
        x,y = point2d_px
        #x*=width
        #y*=height
        if x < x_min:
            x_min=x
        if x > x_max:
            x_max=x
        if y < y_min:
            y_min=y
        if y > y_max:
            y_max=y
    if clip:
        x_min=max(x_min,0)
        y_min=max(y_min,0)
        x_max=min(x_max,width)
        y_max=min(y_max,height)
    return x_min, y_min, x_max, y_max


def get_center(x_min: int,y_min: int,x_max:int,y_max:int):
    """Get center of bounding box.

    Args:
        x_min (int): Minimum x value in pixels.
        y_min (int): Minimum y value in pixels.
        x_max (int): Maximum x value in pixels.
        y_max (int): Maximum y value in pixels.

    Returns:
        tuple: Center (x,y) in pixels.
    """
    x = (x_min+x_max)/2
    y = (y_min+y_max)/2
    return x,y

def points_2d_to_points2d_px(points2d:list, width:int, height:int):
    """Convert normalzied 2d point list into pixel point list.

    Args:
        points2d (list): Normalized 2d points list (x, y, depth)
        width (int): Image width in pixels.
        height (int): Image heigh in pixels.

    Returns:
        list: (x,y) in pixels relative to image. 
    """
    points2d_px=[]
    for i, point2d in enumerate(points2d):
        x = point2d[0]*width
        y = point2d[1]*height
        points2d_px.append((x,y))
    return np.array(points2d_px)

def get_scale_factors(bbox1:tuple, bbox2:tuple):
    """Get x and y scaling factors between 2 bounding boxes.

    Args:
        bbox1 (tuple): xmin1, ymin1, xmax1, ymax1
        bbox2 (tuple): xmin2, ymin2, xmax2, ymax2

    Returns:
        tuple: (x,y) scale factor.
    """
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    width1 = xmax1-xmin1
    width2 = xmax2-xmin2
    height1 = ymax1-ymin1
    height2 = ymax2-ymin2
    try:
        return float(width2/width1), float(height2/height1)
    except ZeroDivisionError:
        return (1,1)

def align_with_bbox(train_point2d_px, train_bbox, valid_bbox):
    """Aligns a projected 3d bounding box in pixel space.

    Args:
        train_point2d_px (list): Projected points in pixels of a 3d bounding box relative to image.
        train_bbox (tuple): Result 2d bounding box, in pixels.
        valid_bbox ([type]): Query 2d bounding box, in pixels.
    """
    x_center_valid, y_center_valid = get_center(*valid_bbox)
    x_center_train, y_center_train = get_center(*train_bbox)
    
    #x_center_train, y_center_train = train_point2d_px[0][0], train_point2d_px[0][1]
    dx = x_center_valid - x_center_train
    dy = y_center_valid - y_center_train
    aligned_points2d_px = []
    scale_x, scale_y = get_scale_factors(train_bbox, valid_bbox)
#    scale_x = 1.2
#    scale_y = 1.2
    for train_point2d_px in train_point2d_px:
        x,y = train_point2d_px
        x+=dx
        y+=dy
        x = (x-x_center_train)*scale_x + x_center_train
        y = (y-y_center_train)*scale_y + y_center_train
        aligned_points2d_px.append((x,y))
    
    return aligned_points2d_px


def draw_bbox(im: PIL.Image, points2d_px: list, line_color=(0,255,0), pixel_center_color=(0,255,0), object_center_color=(255,0,0), line_width=5):
    """Draw a projected 2d bounding box (in pixels) over a Pillow Image

    Args:
        im (PIL.Image): Query image.
        points2d_px (list): Aligned projected 2d bounding box in pixels.

    Returns:
        Pil.Image: Image with drawn bounding box.
    """
    #with Image.open(image_file) as im:
    #points2d, points3d = 
    draw = ImageDraw.Draw(im)
    #points2d_px=[]
    #points2d_px = points_2d_to_points2d_px(points2d, im.width, im.height)
    for i, point2d_px in enumerate(points2d_px):
        x, y = point2d_px
        fill=(0,0,255)
        if i==0: #Center point
            fill=object_center_color
        draw.ellipse((x-5 , y-5,x+5 , y+5), fill=fill)
    #points2d_px.append((x,y))
    
    x_min, y_min, x_max, y_max = get_bbox(points2d_px, im.width, im.height)
    x_center = (x_max+x_min)/2
    y_center = (y_max+y_min)/2
    draw.ellipse((x_center-10 , y_center-10,x_center+10 , y_center+10), fill=pixel_center_color)
    for edge in EDGES:
        p1, p2 = edge
        x1, y1 = points2d_px[p1]
        x2, y2 = points2d_px[p2]
        draw.line((x1,y1,x2,y2), fill=line_color, width=line_width)
    return im
import math 

def align_with_bbox_3d(points3d_train: list, train_bbox: tuple, valid_bbox: tuple, 
                        alpha_x=368.0, alpha_y:float=None, verbose=False):
    """Aligns a 3d bounding box (result) according to the query bounding box location and the
    result bounding box in 2d (pixels).

    Args:
        points3d_train (list): 3d bounding box relative to camera. First point is center.
        train_bbox (tuple): Result 2d bounding box location in pixels.
        valid_bbox (tuple): Query 2d bounding box location in pixels.
        alpha_x (float, optional): Alpha x intrinsic camera parameter. Defaults to 368.0.
        alpha_y ([type], optional): Alpha y intrinsic camera parameter. Defaults to None:float.
        verbose (bool, optional): [description]. Defaults to False.
    """
    
    if alpha_y is None:
        alpha_y = alpha_x #Square CCD
    x_center_valid, y_center_valid = get_center(*valid_bbox)
    x_center_train, y_center_train = get_center(*train_bbox)
    
    #x_center_train, y_center_train = train_point2d_px[0][0], train_point2d_px[0][1]
    dx_px = x_center_valid - x_center_train
    dy_px = y_center_valid - y_center_train
    
    translated_points3d = []
    scale_x, scale_y = get_scale_factors(train_bbox, valid_bbox)

    scale = 1-(scale_x+scale_y)/2
    distance_scale = 1/((scale_x+scale_y)/2)
    c_x, c_y, c_z = points3d_train[0]
    c_depth = np.sqrt(c_x**2+c_y**2+c_z**2) 
    #print(dx_px, dy_px, c_z)
    z_offset = c_z*scale
    #x_offset = -dx_px/368*c_z
    #y_offset = -dy_px/368*c_z


    x_offset = math.atan(dx_px/alpha_x*c_depth)
    y_offset = math.atan(dy_px/alpha_y*c_depth)
    #x_offset = dx_px/alpha_x*c_depth
    #y_offset = dy_px/alpha_y*c_depth
#     
    scale_x = 1.2
#    scale_y = 1.2
    for point3d in points3d_train:
        x,y,z = point3d
        y+=x_offset #x/y are reversed most of the time !!!!!
        x+=y_offset
        z+=z_offset
        translated_points3d.append((x,y,z))

    alpha = x_offset/c_depth #Small angle approx. sin(alpha)=alpha
    beta = -y_offset/c_depth
    if verbose:
        print(f"alpha: {alpha} beta: {beta} dx_px: {dx_px}, dy_px: {dy_px}")
    #print(alpha,beta)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #rotation = mesh.get_rotation_matrix_from_xyz([beta,alpha,0])
    center_rotation = np.array(mesh.get_rotation_matrix_from_xyz([alpha,beta,0]))
    axis_rotation = np.array(mesh.get_rotation_matrix_from_xyz([0,0,0]))
    
    #center = np.array(translated_points3d[0])


    pcd = o3d.geometry.PointCloud()
    

    #translated_bbox = np.array(translated_points3d)
    #pcd.points = o3d.utility.Vector3dVector(translated_bbox)
    
    offset = -(1-distance_scale) * np.array(points3d_train[0])
    points_3d_offset = np.array(points3d_train) + offset
    old_center = points_3d_offset[0]
    new_center = np.dot(center_rotation, points_3d_offset[0])

    #pcd.points= o3d.utility.Vector3dVector(np.array(points3d_train))
    pcd.points =  o3d.utility.Vector3dVector(points_3d_offset)
    center = np.array([0,0,0])

    pcd.rotate(center_rotation, center)
    #pcd.rotate([beta,alpha,0])
    rotated_points3d = np.asarray(pcd.points)

    rotated_points3d_v2 = points_3d_offset - old_center + new_center
    pcd_v2 = o3d.geometry.PointCloud()
    pcd_v2.points =  o3d.utility.Vector3dVector(rotated_points3d_v2)
    pcd_v2.rotate(axis_rotation, new_center)

    rotated_points3d_v2 = np.asarray(pcd_v2.points)
    #offset =scale * np.array(points3d_train[0])
    #offset =scale * rotated_points3d[0]
    #translated_rotated_points3d=rotated_points3d+offset
    #return translated_points3d
    #return rotated_points3d, points3d_train #translated_points3d
    rotated_points3d = list(map(tuple, rotated_points3d))
    return rotated_points3d_v2, points3d_train
    #return rotated_points3d, points3d_train
