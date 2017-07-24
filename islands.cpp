#include <iostream>
#include <queue>
#include <vector>

using namespace std;

struct vec2 
{
	vec2(int x, int y){
		this->x = x;
		this->y = y;
	}
	int x; 
	int y;
};

int main(){
	const int N = 7;
	const int M = N;

	int arr[N][M] =
	{
	{ 0, 0, 0, 0, 1, 1, 1, },
	{ 1, 0, 0, 1, 1, 1, 1, },
	{ 1, 1, 0, 1, 0, 1, 1, },
	{ 0, 0, 0, 1, 1, 1, 1, },
	{ 0, 1, 1, 0, 0, 0, 1, },
	{ 0, 1, 1, 0, 0, 0, 1, },
	{ 0, 1, 1, 0, 0, 0, 1, },
	};

	vector<vector<int> > visited(N, vector<int>(M, 0));
	vector<vector<int> > res(N, vector<int>(M, 0));

	vec2 start(N >> 1 , M >> 1);

	int area = 0;

	queue<vec2> queue_;
	queue_.push(start);
	visited[start.x][start.y] = 1;
	res[start.x][start.y] = 1;

	while(!queue_.empty()) {
		vec2 pt = queue_.front();
		queue_.pop();
		++area;

		vector<vec2> points;
		{
			int x = pt.x - 1;
			int y = pt.y;
			if(x >= 0) {
				points.push_back(vec2(x, y));
			}

			x = pt.x + 1;
			y = pt.y;
			if(x < N) {
				points.push_back(vec2(x, y));
			}

			x = pt.x;
			y = pt.y - 1;
			if(y > 0) {
				points.push_back(vec2(x, y));
			}

			x = pt.x;
			y = pt.y + 1;
			if(y < M) {
				points.push_back(vec2(x, y));
			}
		}

		for(auto& npt: points) {
			if( arr[npt.x][npt.y] == 1 && !visited[npt.x][npt.y] ){
				res[npt.x][npt.y] = 2;
				visited[npt.x][npt.y] = 1;
				queue_.push(npt);
			}
		}
	}

	for(auto& x: res){
		for(auto& v: x){
			cout << v << ", ";
		}
		cout << endl;
	}
	cout << "area:" << area << endl;
}
