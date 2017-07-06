
#include "DBSCAN.hpp"

#include <vector>
#include <iostream>

#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

int main()
{
	DBSCAN<VectorXd, MatrixXd> dbscan(0.5, 5);

	MatrixXd r = dbscan.gen_cluster_data(2, 100);
	cout << r << endl;

	dbscan.fit(r);

	vector<int> labels = dbscan.get_labels();

	for(auto& pid : labels){
		cout << "pid:" << pid << endl;
	}
}