#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <numeric>

using namespace std;

template<typename C>
size_t len(const C& c){
	return c.size();
}

// Get the median of an unordered set of numbers of arbitrary 
// type (this will modify the underlying dataset).
// fixme: const not working
template <typename It>
auto median(It begin, It end)   //-> // fixme: what put here
{
    const auto size = std::distance(begin, end);
    std::nth_element(begin, begin + size / 2, end);
    return *std::next(begin, size / 2);
}

float clamp_denom(float x, float eps=0.0001){
    if (abs(x) < eps)
        x = eps;  // bug!!!!!!!!!!!
    return x;
}

float clamp_max(float x, float maxx)
{
    if (x > maxx)
        x = maxx;
    return x;
}

float ro(float x, float xi, float arg=0)
{
	float delta = arg;
	float a = x - xi;
	float rel = (a / delta);
	rel *= rel;
	return delta * delta * (sqrt(1 + rel) - 1);
}


float K(float x, float h)
{
    return exp(-2 * x / h);
}

float Kq(float x, vector<float>& xs)
{
    float denom = (6 * median(xs.begin(), xs.end()));
    cout << "" << xs[0] << endl;
    cout << "denom:" << denom << endl;
    return x / clamp_denom(denom);
}




//=====================================================

class RobustNadarayaWatson
{
public:
	RobustNadarayaWatson(const  vector<float>& xs, 
		const vector<float>& ys, float max_gamma);

	float estimate(float x, int excluded = -1);

	void iterate();

	vector<float> xs;
	vector<float> ys;
	vector<float> hs;
	vector<float> gammas;
	float max_gamma;
};


RobustNadarayaWatson::RobustNadarayaWatson(const  vector<float>& xs, 
		const vector<float>& ys, float max_gamma)
{
	this->xs = xs;
	this->ys = ys;
	int N = len(xs);
	hs = vector<float>(N, 0);
	for (int i = 1; i < N; ++i){
		hs[i-1] = xs[i] - xs[i-1];
	}
	hs[N-1] = hs[N-2];

	gammas = vector<float>(N, 1);

	this->max_gamma = max_gamma;
}

void RobustNadarayaWatson::iterate()
{
	vector<float> eps(len(xs), 0);
	for(int i = 0; i < len(xs); ++i){
		float xi = xs[i];
		float yi = ys[i];
		float epi = abs(estimate(xi, i) - yi);
		eps[i] = epi;
	}

	for(int i = 0; i < len(xs); ++i){
		float g = Kq(eps[i], eps);
		gammas[i] = clamp_max(1 / g, max_gamma);
	}
}

float RobustNadarayaWatson::estimate(float x, int excluded)
{
	float num = 0;
	float denum = 0;
	int N = len(xs);

	for (int i = 0; i < N; ++i){
		if (excluded != -1){
			if (excluded == i){
				continue;
			}
		}
		float yi = ys[i];
		float xi = xs[i];
		float wi = K(ro(x, xi, hs[i]), hs[i]);
		float gi = gammas[i];
		// wi *= gi;
		num += yi*wi;
		denum += wi;
	}
	return num / clamp_denom(denum);
}

int main(){
	vector<float> xs = {
	    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
          30, 31, 32, 33, 34, 35, 36, 37, 38, 39, };
    vector<float> ys = {6.62978140876, 4.49678706376, 4.01898549999, 203.0, 9.71220875091, 6.81199054582, 10.3687737888,
          9.34003423625, 11.0943398334, 10.0852958726, 13.595412343, 12.4766620769, 14.2148696106, 15.7052702602,
          16.7081265738, 17.2904090912, 26.0336394742, 21.5965172674, 20.6046784136, 22.8558238898, 22.7599266781,
          22.9025721765, 27.6978857255, 26.0563418143, 27.3580205328, 26.9803286751, 28.9957041814, 28.7742286869,
          30.8504571937, 33.9582208526, 32.9446982772, 32.3383795971, 36.2575900683, 33.1961117655, 38.4606023971,
          36.6657461786, 38.2616113453, 43.806062633, 42.1742207406, 46.7033592699, };

  	RobustNadarayaWatson nw(xs, ys, 10);

  	for(int i = 0; i < 2; ++i){
  		nw.iterate();
  		cout << nw.gammas[0] * 100 << " 1: " << i << endl;
  		cout << nw.gammas[1] * 100 << " 2: " << i << endl;
  	}

  	for (int i = 0; i < len(xs); ++i){
  		float yi_est = nw.estimate(xs[i]);
  		cout << yi_est << endl;
  	}
}
