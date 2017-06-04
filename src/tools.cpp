#include <iostream>
#include "tools.h"
#include <math.h>

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

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size()
      || estimations.size() == 0){
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){

    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
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

VectorXd Tools::CalculateNonlinearH(const VectorXd& x_state) {
  //init the return vector h(x')
  VectorXd zpred(3);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //check division by zero

    //check division by zero
    if(px == 0) {
        throw std::invalid_argument("Division by zero! Please fix this");
    }

  //compute the h(x') matrix
	float pxpy2 = pow(px, 2) + pow(py, 2);
	zpred << pow(pxpy2, 0.5),
		atan2(py, px),
		(px*vx + py*vy) / pow(pxpy2, 0.5);

	//cout <<"zpred:\n " << zpred << endl;

	return zpred;
}

VectorXd Tools::PolarToCart(const VectorXd& x_state) {
	//init the return vector cartesian coords
	VectorXd cartesian(4);

	//recover state parameters
	float rho = x_state(0);
	float phi = x_state(1);
	float rhodot = x_state(2);

	//compute the cartesian coords
	float px;
	float py;

	// deriving px and py from polar coords:
	float denom = pow(1 + pow(tan(phi), 2), 0.5);
	px = rho / denom;
	py = tan(phi)*px;
	cartesian << px, py, 0, 0;

	return cartesian;
}
