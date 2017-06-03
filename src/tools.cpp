#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
    rmse << 0,0,0,0;

    if(estimations.size() < 1) {
      throw std::invalid_argument( "Error: Estimation vector size should not be zero");
    }
    else if(estimations.size() != ground_truth.size()) {
      throw std::invalid_argument( "Error: Estimation vector size should equal ground truth size");
    } else {
      //accumulate squared residuals
      for(int i=0; i < estimations.size(); ++i){
        VectorXd current_estimate = estimations[i];
        VectorXd cur_ground_truth = ground_truth[i];
        VectorXd residual = current_estimate - cur_ground_truth;
        for(int j=0; j<residual.size(); ++j) {
          residual[j] = residual[j] * residual[j];
        }
        rmse += residual;
      }

      //calculate the mean
      rmse /= estimations.size();

      //calculate the squared root
      for(int i=0; i<rmse.size(); ++i) {
          rmse[i] = sqrt(rmse[i]);
      }

      //return the result
      return rmse;
  }
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	//check division by zero
	if(fabs(c1) < 0.0001){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	//compute the Jacobian matrix
	Hj << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}
