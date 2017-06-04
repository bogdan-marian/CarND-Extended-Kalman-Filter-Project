#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
 FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  H_laser_ << 1, 0, 0, 0,
		0, 1, 0, 0;

   tools = Tools();
 }

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

	ekf_.x_ = VectorXd(4);
	ekf_.x_ << 1, 1, 1, 1;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    //Convert radar from polar to cartesian coordinates and initialize state.
    //ekf_.Init(VectorXd(4), MatrixXd(4, 4), MatrixXd(4, 4),
    //        MatrixXd(3, 4), MatrixXd(3, 3), MatrixXd(4, 4));

    //raw_measurements should be a Vector(3) if RADAR
    ekf_.x_ << tools.PolarToCart(measurement_pack.raw_measurements_);

    //init Hj based on the statespace:
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = MatrixXd(3, 4);
    ekf_.H_ << Hj_;

    //init measurement covariance R
    ekf_.R_ = MatrixXd(3, 3);
    ekf_.R_ << R_radar_;
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    //Initialize state.
	  //ekf_.Init(VectorXd(4), MatrixXd(4, 4), MatrixXd(4, 4), MatrixXd(2, 4),
    //      MatrixXd(2, 2), MatrixXd(4, 4));
	  //set the state with the initial location and zero velocity
	  ekf_.x_ << measurement_pack.raw_measurements_[0],
                measurement_pack.raw_measurements_[1], 0, 0;

	  //init measurement covariance R
	  ekf_.R_ = MatrixXd(2, 2);
	  ekf_.R_ << R_laser_;

	  //init measurement matrix H
	  ekf_.H_ = MatrixXd(2, 4);
	  ekf_.H_ << H_laser_;
  }

	//init state covariance matrix P
	ekf_.P_ = MatrixXd(4, 4);
	ekf_.P_ << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1000, 0,
		0, 0, 0, 1000;

	//the initial transition matrix F_
	ekf_.F_ = MatrixXd(4, 4);
	ekf_.F_ << 1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1;

  // done initializing, no need to predict or update
	previous_timestamp_ = measurement_pack.timestamp_;
  is_initialized_ = true;
  return;
}

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  //compute the time elapsed between the current and previous measurements
  //dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  //for convenience:
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //set the acceleration noise components as per instruction above
  float noise_ax = 9;
  float noise_ay = 9;

  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
 	  0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
 	  dt_3 / 2 * noise_ax, 0, dt_2*noise_ax, 0,
 	  0, dt_3 / 2 * noise_ay, 0, dt_2*noise_ay;

  // Call the Kalman Filter predict() function
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
	  // Need to resize H and R matrices here and update them in ekf
	  // Update Hj based on the current statespace:
	  Hj_ = tools.CalculateJacobian(ekf_.x_);
	  ekf_.H_ = MatrixXd(3, 4);
	  ekf_.H_ << Hj_;

	  // Update R matrix size
	  ekf_.R_ = MatrixXd(3, 3);
	  ekf_.R_ << R_radar_;

	  // Update state using Radar
	  ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
	  // Need to resize H and R matrices here and update them in ekf
	  // Refresh H based on the current statespace:
	  ekf_.H_ = MatrixXd(2, 4);
	  ekf_.H_ << H_laser_;

	  // Update R matrix size
	  ekf_.R_ = MatrixXd(2, 2);
	  ekf_.R_ << R_laser_;

	  // Update state using Laser
	  ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
