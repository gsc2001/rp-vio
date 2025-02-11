#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace Eigen;
using namespace std;



struct SFMFeature {
    bool state;
    int id;
    int plane_id;
    vector<pair<int, Vector2d>> observation;
    double position[3];
    double depth;
};

struct ReprojectionError3D {
    ReprojectionError3D(double observed_u, double observed_v)
            : observed_u(observed_u), observed_v(observed_v) {}

    template<typename T>
    bool operator()(const T *const camera_R, const T *const camera_T, const T *point, T *residuals) const {
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0];
        p[1] += camera_T[1];
        p[2] += camera_T[2];
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        residuals[0] = xp - T(observed_u);
        residuals[1] = yp - T(observed_v);
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y) {
        return (new ceres::AutoDiffCostFunction<
                ReprojectionError3D, 2, 4, 3, 3>(
                new ReprojectionError3D(observed_x, observed_y)));
    }

    double observed_u;
    double observed_v;
};


struct ReprojectionErrorH {
    double observed_x, observed_y;

    ReprojectionErrorH(double observed_x, double observed_y) : observed_x(observed_x), observed_y(observed_y) {}

    template<typename T>
    bool operator()(const T *R, const T *t, const T *n, const T *point, T *residuals) const {
        T norm_point[3] = {point[0] / point[2], point[1] / point[2], point[2] / point[2]};
        T rotated_point[3];
        ceres::QuaternionRotatePoint(R, norm_point, rotated_point);

        T nu = n[0] * norm_point[0] + n[1] * norm_point[1] + n[2] * norm_point[2];
        T np[3] = {t[0] * nu, t[1] * nu, t[2] * nu};
        np[0] += rotated_point[0];
        np[1] += rotated_point[1];
        np[2] += rotated_point[2];
        T x = np[0] / np[2];
        T y = np[1] / np[2];
        residuals[0] = x - T(observed_x);
        residuals[1] = y - T(observed_y);
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorH, 2, 4, 3, 3, 3>(
                new ReprojectionErrorH(observed_x, observed_y)));
    }
};

class GlobalSFM {
public:
    GlobalSFM();

    bool construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
                   const Matrix3d relative_R, const Vector3d relative_T,
                   vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

    bool constructH(int frame_num, Quaterniond *q, Vector3d *T, int l,
                    const Matrix3d relative_R, const Vector3d relative_T, Vector3d &n,
                    vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
    bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                          Vector2d &point0, Vector2d &point1, Vector3d &point_3d);

    void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                              int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                              vector<SFMFeature> &sfm_f);

    int feature_num;
};