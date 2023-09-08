import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree
from evaluation.implicit_waterproofing import implicit_waterproofing

# mostly apdopted from occupancy_networks/im2mesh/common.py and occupancy_networks/im2mesh/eval.py

def f1(actual, predicted, label=1):

    """ A helper function to calculate f1-score for the given `label` """

    # F1 = 2 * (precision * recall) / (precision + recall)
    tp = np.sum((actual==label) & (predicted==label))
    fp = np.sum((actual!=label) & (predicted==label))
    fn = np.sum((predicted!=label) & (actual==label))
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {'f1':f1, 'tp': tp, 'fp':fp, 'fn':fn, 
            'precision': precision, 'recall': recall}

def eval_mesh( mesh_pred, mesh_gt, bb_min, bb_max, n_points=100000):
    if len(mesh_pred.vertices) < 10:
        print('Pred mesh have to data. Exiting evaluation.')
        return {}

    pointcloud_pred, idx = mesh_pred.sample(n_points, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    # normals_pred = mesh_pred.face_normals[idx]
    # print(pointcloud_pred.shape)
    pointcloud_gt, idx = mesh_gt.sample(n_points, return_index=True)
    pointcloud_gt = pointcloud_gt.astype(np.float32)
    # normals_gt = mesh_gt.face_normals[idx]
    # print(pointcloud_gt.shape)

    # out_dict = eval_pointcloud(pointcloud_pred, pointcloud_gt, normals_pred, normals_gt)
    out_dict = eval_pointcloud(pointcloud_pred, pointcloud_gt)
    # print('Pointcloud eval complete')

    bb_len = bb_max - bb_min
    bb_samples = np.random.rand(n_points*10, 3) * bb_len + bb_min

    occ_pred = implicit_waterproofing(mesh_pred, bb_samples)[0]
    occ_gt = implicit_waterproofing(mesh_gt, bb_samples)[0]

    area_union = (occ_pred | occ_gt).astype(np.float32).sum()
    area_intersect = (occ_pred & occ_gt).astype(np.float32).sum()

    out_dict['iou'] =  (area_intersect / area_union)
    return out_dict


def eval_pointcloud(pointcloud_pred, pointcloud_gt,
                    normals_pred=None, normals_gt=None):

    pointcloud_pred = np.asarray(pointcloud_pred)
    pointcloud_gt = np.asarray(pointcloud_gt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    # print('Calculating p2p distance - completeness')
    completeness, completeness_normals = distance_p2p(
        pointcloud_gt, pointcloud_pred,
        normals_gt, normals_pred
    )
    # print('completeness done') 
    completeness2 = completeness ** 2

    precision = {}
    for p in [0.005, 0.01, 0.05]:
        precision['precision_'+str(p*100)] = len(completeness[completeness < p])/len(pointcloud_pred)
    
    # print('precision done') 
    completeness = completeness.mean()
    completeness2 = completeness2.mean()

   # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    # print('Calculating p2p distance - accuracy')
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, pointcloud_gt,
        normals_pred, normals_gt
    )
    accuracy2 = accuracy**2
    
    # print('accuracy done') 
    recall = {}
    for p in [0.005, 0.01, 0.05]:
        recall['recall_'+str(p*100)] = len(accuracy[accuracy < p])/len(pointcloud_pred)

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()


    # Chamfer distance
    chamfer_l2 = 0.5 * completeness2 + 0.5 * accuracy2
    chamfer_l2 = chamfer_l2*10000

    fscore = {}
    for p in [0.005, 0.01, 0.05]:
        fscore['fscore_'+str(p*100)] = 2*(precision['precision_'+str(p*100)]*\
                                        recall['recall_'+str(p*100)])/(precision['precision_'+str(p*100)]+\
                                        recall['recall_'+str(p*100)]+1e-5)
    # print('fscore done') 
    if not normals_pred is None:
        accuracy_normals = accuracy_normals.mean()
        completeness_normals = completeness_normals.mean()
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
    else:
        accuracy_normals = np.nan
        completeness_normals = np.nan
        normals_correctness = np.nan

    '''
    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals': normals_correctness,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer_l2': chamfer_l2,
        'iou': np.nan
    }
    '''
    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer_l2': chamfer_l2,
    }
    out_dict.update(precision)
    out_dict.update(recall)
    out_dict.update(fscore)
    # out_dict = out_dict | precision | recall | fscore
    return out_dict


def distance_p2p(pointcloud_pred, pointcloud_gt,
                    normals_pred, normals_gt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    # print('Building kdtree')
    kdtree = KDTree(pointcloud_gt)
    
    # print('Querying kdtree')
    dist, idx = kdtree.query(pointcloud_pred)

    if normals_pred is None:
        return dist, None

    normals_pred = normals_pred / np.linalg.norm(normals_pred, axis=-1, keepdims=True)
    normals_gt = normals_gt / np.linalg.norm(normals_gt, axis=-1, keepdims=True)

    normals_dot_product = (normals_gt[idx] * normals_pred).sum(axis=-1)
    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    normals_dot_product = np.abs(normals_dot_product)

    return dist, normals_dot_product
