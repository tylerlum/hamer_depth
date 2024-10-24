import numpy as np


def transform_pts(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.hstack([pts, np.ones((len(pts), 1))])
    pts = np.dot(T, pts.T).T
    return pts[:, :3]



def transform_pt(pt: np.ndarray, T: np.ndarray) -> np.ndarray:
    pt = np.array(pt)
    pt = np.append(pt, 1)
    pt = np.dot(T, pt)
    return pt[:3]
