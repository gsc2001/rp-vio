#pragma once

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>


struct PlaneFactor {
    Eigen::Vector3d pts;

    PlaneFactor(const Eigen::Vector3d &_pts) : pts(_pts) {}

    template<typename T>
    bool operator()(const T *const pose_i, const T *const para_n, const T *const para_depth, const T *const ex_pose,
                    const T *const para_feature, T *residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pi(pose_i);
        Eigen::Quaternion<T> qi;
        qi.coeffs() << pose_i[3], pose_i[4], pose_i[5], pose_i[6];

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> tic(ex_pose);
        Eigen::Quaternion<T> qic;
        qic.coeffs() << ex_pose[3], ex_pose[4], ex_pose[5], ex_pose[6];

        Eigen::Matrix<T, 3, 1> n;
        n << para_n[0], para_n[1], para_n[2];

        Eigen::Map<const Eigen::Matrix<T, 1, 1>> d(para_depth);
        Eigen::Map<const Eigen::Matrix<T, 1, 1>> inv_dep_i(para_feature);


        Eigen::Matrix<T, 3, 1> pts_camera_i = pts / inv_dep_i(0, 0);
        Eigen::Matrix<T, 3, 1> pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Matrix<T, 3, 1> pts_imu_0 = qi * pts_imu_i + pi;
        Eigen::Matrix<T, 3, 1> pts_w = qic.inverse() * (pts_imu_0 - tic);

        residuals[0] = n.dot(pts_w) - d(0, 0);
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d &_pts) {
        return (new ceres::AutoDiffCostFunction<PlaneFactor, 1, 7, 3, 1, 7, 1>(new PlaneFactor(_pts)));
    }


};
