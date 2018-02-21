using System;
using System.Collections.Generic;

namespace ZeroProximity.MachineLearning.Utility
{
    public class NormalizedData : INormalizedData
    {
        public NormalizedData(double[][] data, IDictionary<int, Tuple<double, double>> minMaxValues)
        {
            Data = data;
            MinMaxValues = minMaxValues;
        }

        public double[][] Data { get; }

        public IDictionary<int, Tuple<double, double>> MinMaxValues { get; }
    }
}