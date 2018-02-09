using System;
using System.Collections.Generic;
using System.Linq;

namespace ZeroProximity.MachineLearning.Utility
{
    public static class NormalizeExtensions
    {
        /// <summary>
        /// Normalizes a matrix of data
        /// </summary>
        /// <returns>Returns the normalized matrix</returns>
        public static double[][] Normalize(this double[][] data)
        {
            var cols = GetMinMaxDictionary(data);
            return Normalize(cols, data);
        }

        /// <summary>
        /// Gets the min and max value of each column
        /// </summary>
        private static Dictionary<int, Tuple<double, double>> GetMinMaxDictionary(double[][] data)
        {
            return data.SelectMany(row => row.Select((col, i) => Tuple.Create<int, double>(i, col)))
                .GroupBy(x => x.Item1)
                .ToDictionary(x => x.Key, x => Tuple.Create(x.Select(y => y.Item2).Min(), x.Select(y => y.Item2).Max()));
        }

        /// <summary>
        /// Performs normalization on the dataset using the min/max values
        /// </summary>
        private static double[][] Normalize(Dictionary<int, Tuple<double, double>> minMax, double[][] data)
        {
            return data
                .Select((row, i) => row
                    .Select((col, j) => (col - minMax[j].Item1) / (minMax[j].Item2 - minMax[j].Item1))
                    .ToArray())
                .ToArray();
        }
    }
}