#ifndef EKF_NODE_HPP
#define EKF_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class EKFNode : public rclcpp::Node
{
public:
    EKFNode();

private:
    void poseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void predict();
    void updatePose(const VectorXd &z);
    void updateIMU(const VectorXd &z);
    void publishPose();
    void integrateIMU(const sensor_msgs::msg::Imu::SharedPtr msg, double dt);

    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr publisher_pose_with_cov_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr publisher_pose_;
    rclcpp::TimerBase::SharedPtr timer_;
    geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose_msg_;
    sensor_msgs::msg::Imu::SharedPtr imu_msg_;

    VectorXd state_;
    MatrixXd P_, Q_, R_pose_, R_imu_, I_;
    rclcpp::Time last_time_;  // Değişiklik: Zaman değişkeni eklendi

    double L_; // Wheelbase of the vehicle (aracın dingil mesafesi)
};

#endif // EKF_NODE_HPP
