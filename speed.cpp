

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
template <typename It>
auto median(It begin, It end)   //-> // fixme: what put here
{
    const auto size = std::distance(begin, end);
    std::nth_element(begin, begin + size / 2, end);
    return *std::next(begin, size / 2);
}

auto speeds(std::vector<float>& xs, 
	std::vector<float>& ts)	-> vector<float>
{
	std::partial_sum(ts.begin(), ts.end(), 
		ts.begin(), std::plus<int>());


	int buffer_size = len(xs);
	assert(buffer_size == len(ts) + 1);

	vector<float> dxs(buffer_size - 1, 0);

	vector<float> vxs;
	for(int i = 1; i < buffer_size; ++i){
		for(int j = 1; j < buffer_size; ++j){
			if (j < i) {
				continue;
			}

			float dxv = xs[j] - xs[i-1];
			float dt = ts[j-1] - ts[i-2];

			float v = dxv / dt;

			vxs.push_back(v);
		}
	}
	return vxs;
}

int main()
{
	vector<float> xs{1, -2, 3, 4, -10};
	vector<float> ts{1, 1, 1, 1};
	int buffer_size = len(xs);

	auto vxs = speeds(xs, ts);

	for (auto v: vxs){
		cout << v << endl;
	}

	cout << "med:" << median(vxs.begin(), vxs.end()) << endl;

	int N = (buffer_size - 1) * (buffer_size - 1) - (buffer_size - 1);
	// assert(N == len(vxs));


	return 0;
}