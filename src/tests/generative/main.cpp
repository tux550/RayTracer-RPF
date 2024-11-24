#include <custom/mi.h>
#include <rapidcheck.h>

#include <algorithm>
#include <vector>

#include "rapidcheck/Assertions.h"

int main() {
    rc::check("mutual information of own vector is 1",
              [](std::vector<double> v1) {
                  RC_PRE(v1.size() >= 1);

                  double mi = MutualInformation(v1, v1, 5, 5);

                  RC_ASSERT(mi == 1.0);
              });
    return 0;
}
