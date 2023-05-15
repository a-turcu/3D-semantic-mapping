import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from itearative_global_registration import pcd_pipeline


pcd_list = []


for i in range(56):
    pcd = pcd_pipeline(i)
    pcd_list.append(pcd)


def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_coarse,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    icp_fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine, icp_fine.transformation
    )
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, uncertain=False)
                )
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, uncertain=True)
                )
    return pose_graph


voxel_size = 0.00008

pcds_down = []
# pcd_list = []
for pcd in pcd_list:
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down.estimate_normals()
    pcds_down.append(pcd_down)
    # pcd_list.append(pcd)
    # print("PCD")
    # print(pcd)
    # print(pcd_down)
    # print(np.asarray(pcd_down.points))


print("Full registration ...")
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine)


print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine, edge_prune_threshold=0.25, reference_node=0
)
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

print("Transform points and display")
pcd_combined = o3d.geometry.PointCloud()
pcd_combined_down = o3d.geometry.PointCloud()
for point_id in range(len(pcds_down)):
    # print(pose_graph.nodes[point_id].pose)
    pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_list[point_id].transform(pose_graph.nodes[point_id].pose)

    pcd_combined += pcd_list[point_id]
    pcd_combined_down += pcds_down[point_id]

o3d.visualization.draw_geometries([pcd_combined_down])
o3d.visualization.draw_geometries([pcd_combined])

print(pcd_combined.points)


# o3d.visualization.draw_geometries([pcds[point_id]])

# o3d.visualization.draw_geometries([pcd_combined])


# pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
# o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
# # o3d.visualization.draw_geometries([pcd_combined_down],
# #                                   zoom=0.3412,
# #                                   front=[0.4257, -0.2125, -0.8795],
# #                                   lookat=[2.6172, 2.0475, 1.532],
# #                                   up=[-0.0694, -0.9768, 0.2024])


# combined = o3d.io.read_point_cloud("multiway_registration.pcd")
# print(combined)

# for pcd in pcds_down:
#     print("PCD")
#     print(pcd_down)

# o3d.visualization.draw_geometries([combined])
