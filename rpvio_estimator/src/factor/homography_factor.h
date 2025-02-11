#pragma once

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

struct HomographyFactor {
    HomographyFactor(const Vector3d &_pts_i, const Vector3d &_pts_j) : pts_i(_pts_i), pts_j(_pts_j) {}

    template<typename T>
    bool
    operator()(const T *const pose_i, const T *const pose_j, const T *const para_n, const T *const para_depth,
               const T *const ex_pose, T *residuals) const {
        Map<const Matrix<T, 3, 1>> ti(pose_i);
        Quaternion<T> qi;
        qi.coeffs() << pose_i[3], pose_i[4], pose_i[5], pose_i[6];

        Map<const Matrix<T, 3, 1>> tj(pose_j);
        Quaternion<T> qj;
        qj.coeffs() << pose_j[3], pose_j[4], pose_j[5], pose_j[6];

        Map<const Matrix<T, 3, 1>> tic(ex_pose);
        Quaternion<T> qic;
        qic.coeffs() << ex_pose[3], ex_pose[4], ex_pose[5], ex_pose[6];

        Map<const Matrix<T, 1, 1>> depth(para_depth);

        Map<const Matrix<T, 3, 1>> n(para_n);

        Quaternion<T> qji = qj.inverse() * qi;
        Matrix<T, 3, 1> tji = qj.inverse() * (ti - tj);

        Matrix<T, 3, 1> n_imu_0 = qic * n;
        Matrix<T, 3, 1> n_imu_i = qi.inverse() * n_imu_0;

        Matrix<T, 1, 1> di0, di;

        di0(0, 0) = depth(0, 0) + tic.dot(n_imu_0);
        di(0, 0) = di0(0, 0) - ti.dot(n_imu_0);

        Matrix<T, 3, 1> pts_imu_i = qic * pts_i.cast<T>() + tic;

        Matrix<T, 3, 1> pts_pred_imu_j = qji * pts_imu_i + (tji * 1.0 / di(0, 0)) * n_imu_i.transpose() * pts_imu_i;
        Matrix<T, 3, 1> pts_pred_cam_j = qic.inverse() * pts_pred_imu_j;

        pts_pred_cam_j[0] /= pts_pred_cam_j[2];
        pts_pred_cam_j[1] /= pts_pred_cam_j[2];
        residuals[0] = pts_pred_cam_j[0] - T(pts_j[0]);
        residuals[1] = pts_pred_cam_j[1] - T(pts_j[1]);
        return true;
    }


    static ceres::CostFunction *Create(const Vector3d &_pts_i, const Vector3d &_pts_j) {
        return (new ceres::AutoDiffCostFunction<HomographyFactor, 2, 7, 7, 3, 1, 7>(
                new HomographyFactor(_pts_i, _pts_j)));
    }

    Vector3d pts_i, pts_j;
};