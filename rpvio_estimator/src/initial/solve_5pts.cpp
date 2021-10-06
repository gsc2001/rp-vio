#include "solve_5pts.h"
#include <algorithm>

namespace cv {
    void decomposeEssentialMat(InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t) {

        Mat E = _E.getMat().reshape(1, 3);
        CV_Assert(E.cols == 3 && E.rows == 3);

        Mat D, U, Vt;
        SVD::compute(E, D, U, Vt);

        if (determinant(U) < 0) U *= -1.;
        if (determinant(Vt) < 0) Vt *= -1.;

        Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
        W.convertTo(W, E.type());

        Mat R1, R2, t;
        R1 = U * W * Vt;
        R2 = U * W.t() * Vt;
        t = U.col(2) * 1.0;

        R1.copyTo(_R1);
        R2.copyTo(_R2);
        t.copyTo(_t);
    }

    int recoverPose(InputArray E, InputArray _points1, InputArray _points2, InputArray _cameraMatrix,
                    OutputArray _R, OutputArray _t, InputOutputArray _mask) {

        Mat points1, points2, cameraMatrix;
        _points1.getMat().convertTo(points1, CV_64F);
        _points2.getMat().convertTo(points2, CV_64F);
        _cameraMatrix.getMat().convertTo(cameraMatrix, CV_64F);

        int npoints = points1.checkVector(2);
        CV_Assert(npoints >= 0 && points2.checkVector(2) == npoints &&
                  points1.type() == points2.type());

        CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3 && cameraMatrix.channels() == 1);

        if (points1.channels() > 1) {
            points1 = points1.reshape(1, npoints);
            points2 = points2.reshape(1, npoints);
        }

        double fx = cameraMatrix.at<double>(0, 0);
        double fy = cameraMatrix.at<double>(1, 1);
        double cx = cameraMatrix.at<double>(0, 2);
        double cy = cameraMatrix.at<double>(1, 2);

        points1.col(0) = (points1.col(0) - cx) / fx;
        points2.col(0) = (points2.col(0) - cx) / fx;
        points1.col(1) = (points1.col(1) - cy) / fy;
        points2.col(1) = (points2.col(1) - cy) / fy;

        points1 = points1.t();
        points2 = points2.t();

        Mat R1, R2, t;
        decomposeEssentialMat(E, R1, R2, t);
        Mat P0 = Mat::eye(3, 4, R1.type());
        Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
        P1(Range::all(), Range(0, 3)) = R1 * 1.0;
        P1.col(3) = t * 1.0;
        P2(Range::all(), Range(0, 3)) = R2 * 1.0;
        P2.col(3) = t * 1.0;
        P3(Range::all(), Range(0, 3)) = R1 * 1.0;
        P3.col(3) = -t * 1.0;
        P4(Range::all(), Range(0, 3)) = R2 * 1.0;
        P4.col(3) = -t * 1.0;

        // Do the cheirality check.
        // Notice here a threshold dist is used to filter
        // out far away points (i.e. infinite points) since
        // there depth may vary between postive and negtive.
        double dist = 50.0;
        Mat Q;
        triangulatePoints(P0, P1, points1, points2, Q);
        Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask1 = (Q.row(2) < dist) & mask1;
        Q = P1 * Q;
        mask1 = (Q.row(2) > 0) & mask1;
        mask1 = (Q.row(2) < dist) & mask1;

        triangulatePoints(P0, P2, points1, points2, Q);
        Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask2 = (Q.row(2) < dist) & mask2;
        Q = P2 * Q;
        mask2 = (Q.row(2) > 0) & mask2;
        mask2 = (Q.row(2) < dist) & mask2;

        triangulatePoints(P0, P3, points1, points2, Q);
        Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask3 = (Q.row(2) < dist) & mask3;
        Q = P3 * Q;
        mask3 = (Q.row(2) > 0) & mask3;
        mask3 = (Q.row(2) < dist) & mask3;

        triangulatePoints(P0, P4, points1, points2, Q);
        Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);
        mask4 = (Q.row(2) < dist) & mask4;
        Q = P4 * Q;
        mask4 = (Q.row(2) > 0) & mask4;
        mask4 = (Q.row(2) < dist) & mask4;

        mask1 = mask1.t();
        mask2 = mask2.t();
        mask3 = mask3.t();
        mask4 = mask4.t();

        // If _mask is given, then use it to filter outliers.
        if (!_mask.empty()) {
            Mat mask = _mask.getMat();
            CV_Assert(mask.size() == mask1.size());
            bitwise_and(mask, mask1, mask1);
            bitwise_and(mask, mask2, mask2);
            bitwise_and(mask, mask3, mask3);
            bitwise_and(mask, mask4, mask4);
        }
        if (_mask.empty() && _mask.needed()) {
            _mask.create(mask1.size(), CV_8U);
        }

        CV_Assert(_R.needed() && _t.needed());
        _R.create(3, 3, R1.type());
        _t.create(3, 1, t.type());

        int good1 = countNonZero(mask1);
        int good2 = countNonZero(mask2);
        int good3 = countNonZero(mask3);
        int good4 = countNonZero(mask4);

        if (good1 >= good2 && good1 >= good3 && good1 >= good4) {
            R1.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask1.copyTo(_mask);
            return good1;
        } else if (good2 >= good1 && good2 >= good3 && good2 >= good4) {
            R2.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask2.copyTo(_mask);
            return good2;
        } else if (good3 >= good1 && good3 >= good2 && good3 >= good4) {
            t = -t;
            R1.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask3.copyTo(_mask);
            return good3;
        } else {
            t = -t;
            R2.copyTo(_R);
            t.copyTo(_t);
            if (_mask.needed()) mask4.copyTo(_mask);
            return good4;
        }
    }

    int recoverPose(InputArray E, InputArray _points1, InputArray _points2, OutputArray _R,
                    OutputArray _t, double focal, Point2d pp, InputOutputArray _mask) {
        Mat cameraMatrix = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
        return cv::recoverPose(E, _points1, _points2, cameraMatrix, _R, _t, _mask);
    }
}


bool MotionEstimator::solveRelativeHRT(const vector<pair<Vector3d, Vector3d>> &corres, const Matrix3d &R_imu,
                                       const Matrix4d &TrIC, Matrix3d &R, Vector3d &T, Vector3d &n) {
    if (corres.size() >= 15) {
        vector<cv::Point2f> left, right;

        for (int i = 0; i < corres.size(); i++) {
            left.emplace_back(corres[i].first(0), corres[i].first(1));
            right.emplace_back(corres[i].second(0), corres[i].second(1));
        }

        cv::Mat mask;
        cv::Mat H = cv::findHomography(left, right, cv::RANSAC, 0.3 / 460, mask, 2000, 0.99);
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

        Vector3d mean_left(0, 0, 1);
        int cnt = 0;

        for (int i = 0; i < left.size(); i++) {
            if (mask.at<uchar>(i, 0) != 0) {
                cnt++;
                mean_left(0) += left[i].x;
                mean_left(1) += left[i].y;
            }
        }
        mean_left(0) /= cnt;
        mean_left(1) /= cnt;

        Matrix4d est_Tr;
        Vector3d est_n;

        decomposeH(H, K, R_imu, TrIC, mean_left, est_Tr, est_n);
        R = est_Tr.block(0, 0, 3, 3);
        T = est_Tr.block(0, 3, 3, 1);
        n = est_n;
        if (n.isZero()) return false;

        return true;
    }

    return false;
}

void RTntoTrn(const cv::Mat &cv_R, const cv::Mat &cv_T, const cv::Mat &cv_n, Matrix4d &Tr, Vector3d &n) {
    Matrix3d R;
    Vector3d t;
    cv::cv2eigen(cv_R, R);
    cv::cv2eigen(cv_T, t);

    Tr = Matrix4d::Identity();
    Tr.block(0, 0, 3, 3) = R;
    Tr.block(0, 3, 3, 1) = t;

    cv::cv2eigen(cv_n, n);

    Tr = Tr.inverse().eval();
    n.normalize();
}


void MotionEstimator::decomposeH(const cv::Mat &H, const cv::Mat &K, const Matrix3d &R_imu, const Matrix4d &TrIC,
                                 const Vector3d &mean_l, Matrix4d &est_Tr, Vector3d &est_n) {


    vector<cv::Mat> cv_Rs, cv_Ts, cv_ns;
    int n_sols = cv::decomposeHomographyMat(H, K, cv_Rs, cv_Ts, cv_ns);
    Matrix4d Tr;
    Vector3d n;
    vector<Matrix4d> positive_transformations;
    vector<Vector3d> positive_normals;
    if (n_sols == 1) {
        RTntoTrn(cv_Rs[0], cv_Ts[0], cv_ns[0], Tr, n);
        est_Tr = Tr;
        est_n = n.normalized();
    } else {
        for (int i = 0; i < n_sols; i++) {
            RTntoTrn(cv_Rs[i], cv_Ts[i], cv_ns[i], Tr, n);
            if (n.dot(mean_l) > 0) {
                positive_normals.push_back(n);
                positive_transformations.push_back(Tr);
            }
        }
        vector<double> rotation_loss;
        for (int i = 0; i < positive_transformations.size(); i++) {
            Matrix4d Tr = TrIC * positive_transformations[i] * TrIC.inverse();
            Matrix3d R_imu_est = Tr.block(0,0,3,3);
            double loss = (R_imu.transpose() * R_imu_est - Matrix3d::Identity()).norm();
            rotation_loss.push_back(loss);
        }

        int min_loss_el = std::min_element(rotation_loss.begin(), rotation_loss.end()) - rotation_loss.begin();

        est_Tr = positive_transformations[min_loss_el];
        est_n = positive_normals[min_loss_el];
    }
}

