#include "ekf_node/ekf_node.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std::chrono_literals;

EKFNode::EKFNode()
    : Node("ekf_node"), pose_msg_(nullptr), imu_msg_(nullptr), L_(2.64)  // L_ initialized with an example wheelbase
{
    pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/localization/pose_estimator/pose_with_covariance", 10, std::bind(&EKFNode::poseCallback, this, std::placeholders::_1));

    imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/sensing/gnss/sbg/ros/imu/data", 10, std::bind(&EKFNode::imuCallback, this, std::placeholders::_1));

    publisher_pose_with_cov_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("output_pose_with_cov", 10);
    publisher_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("output_pose", 10);

    timer_ = this->create_wall_timer(
        10ms, std::bind(&EKFNode::publishPose, this));

    // Initial state vector [x, y, yaw, v, steering_angle, acceleration]
    state_ = VectorXd(6);
    state_.setZero();

    // State covariance matrix
    P_ = MatrixXd(6, 6);
    P_.setIdentity();

    // Process noise covariance matrix
    Q_ = MatrixXd(6, 6);
    Q_.setIdentity();
    Q_ *= 0.01; // Example process noise

    // Measurement noise covariance matrix for pose
    R_pose_ = MatrixXd(6, 6);
    R_pose_.setIdentity();
    R_pose_ *= 0.1; // Example measurement noise

    // Measurement noise covariance matrix for IMU
    R_imu_ = MatrixXd(3, 3);
    R_imu_.setIdentity();
    R_imu_ *= 0.1; // Example measurement noise

    // Identity matrix
    I_ = MatrixXd::Identity(6, 6);

    last_time_ = this->now();  // Değişiklik: Zaman değişkeni başlatıldı
}

void EKFNode::poseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    pose_msg_ = msg;

    // Measurement vector [x, y, yaw, v, steering_angle, acceleration]
    VectorXd z(6);
    z(0) = msg->pose.pose.position.x;
    z(1) = msg->pose.pose.position.y;

    tf2::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);

    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    z(2) = yaw;

    // Prediction step
    predict();

    // Update step for pose
    updatePose(z);
}

void EKFNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    imu_msg_ = msg;

    auto now = this->now();
    double dt = (now - last_time_).seconds();  // Değişiklik: Zaman farkı hesaplandı
    last_time_ = now;

    // Integrate IMU data to update roll, pitch, and yaw
    integrateIMU(msg, dt);  // Değişiklik: IMU verileri entegre edildi

    // Measurement vector [roll_rate, pitch_rate, yaw_rate]
    VectorXd z(3);
    z(0) = msg->angular_velocity.x;
    z(1) = msg->angular_velocity.y;
    z(2) = msg->angular_velocity.z;

    // Update step for IMU
    updateIMU(z);
}

void EKFNode::integrateIMU(const sensor_msgs::msg::Imu::SharedPtr msg, double dt)
{
    // Integrate angular velocities to get roll, pitch, and yaw
    state_(3) += msg->angular_velocity.x * dt;  // Değişiklik: Roll entegrasyonu
    state_(4) += msg->angular_velocity.y * dt;  // Değişiklik: Pitch entegrasyonu
    state_(5) += msg->angular_velocity.z * dt;  // Değişiklik: Yaw entegrasyonu
}

void EKFNode::predict()
{
    // Time delta (assuming fixed 20ms for simplicity)
    double dt = 0.01;

    // State transition matrix for bicycle model
    MatrixXd F(6, 6);
    F.setIdentity();
    F(0, 2) = -state_(3) * sin(state_(2)) * dt;  // dx/dtheta
    F(0, 3) = cos(state_(2)) * dt;               // dx/dv
    F(1, 2) = state_(3) * cos(state_(2)) * dt;   // dy/dtheta
    F(1, 3) = sin(state_(2)) * dt;               // dy/dv
    F(2, 4) = state_(3) / L_ * dt;               // dtheta/dsteering_angle

    // Predicted state estimate using bicycle model
    state_(0) += state_(3) * cos(state_(2)) * dt;
    state_(1) += state_(3) * sin(state_(2)) * dt;
    state_(2) += (state_(3) / L_) * tan(state_(4)) * dt;

    // Predicted covariance estimate
    P_ = F * P_ * F.transpose() + Q_;
}

void EKFNode::updatePose(const VectorXd &z)
{
    // Measurement matrix
    MatrixXd H(6, 6);
    H.setIdentity();

    // Innovation or measurement residual
    VectorXd y = z - H * state_;

    // Innovation covariance
    MatrixXd S = H * P_ * H.transpose() + R_pose_;

    // Kalman gain
    MatrixXd K = P_ * H.transpose() * S.inverse();

    // Update the state
    state_ = state_ + K * y;

    // Update the state covariance
    P_ = (I_ - K * H) * P_;
}

void EKFNode::updateIMU(const VectorXd &z)
{
    // Measurement matrix
    MatrixXd H(3, 6);
    H.setZero();
    H(0, 3) = 1;
    H(1, 4) = 1;
    H(2, 5) = 1;

    // Innovation or measurement residual
    VectorXd y = z - H * state_;

    // Innovation covariance
    MatrixXd S = H * P_ * H.transpose() + R_imu_;

    // Kalman gain
    MatrixXd K = P_ * H.transpose() * S.inverse();

    // Update the state
    state_ = state_ + K * y;

    // Update the state covariance
    P_ = (I_ - K * H) * P_;
}

void EKFNode::publishPose()
{
    if (pose_msg_ == nullptr)
        return;

    auto new_pose = geometry_msgs::msg::PoseWithCovarianceStamped();
    new_pose.header.stamp = this->now();
    new_pose.header.frame_id = pose_msg_->header.frame_id;

    new_pose.pose.pose.position.x = state_(0);
    new_pose.pose.pose.position.y = state_(1);

    tf2::Quaternion new_q;
    new_q.setRPY(0, 0, state_(2));
    new_pose.pose.pose.orientation = tf2::toMsg(new_q);

    publisher_pose_with_cov_->publish(new_pose);

    geometry_msgs::msg::PoseStamped new_pose_stamped;
    new_pose_stamped.header = new_pose.header;
    new_pose_stamped.pose = new_pose.pose.pose;
    publisher_pose_->publish(new_pose_stamped);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFNode>());
    rclcpp::shutdown();
    return 0;
}
