using System;
using System.Collections.Generic;

namespace ZeroProximity.MachineLearning.Utility
{
    public interface INormalizedData
    {
        double[][] Data { get; }

        IDictionary<int, Tuple<double, double>> MinMaxValues { get; }
    }
}