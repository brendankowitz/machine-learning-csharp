using System;
using System.Collections.Generic;
using System.Linq;

namespace ZeroProximity.MachineLearning.Utility
{
    public static class Normalization
    {
        /// <summary>
        /// Normalizes a matrix of data
        /// </summary>
        /// <returns>Returns the normalized matrix</returns>
        public static INormalizedData Normalize(this double[][] data)
        {
            var cols = GetMinMaxDictionary(data);
            return new NormalizedData(Normalize(cols, data), cols);
        }

        /// <summary>
        /// Normalizes a row of data given an existing min/max value
        /// </summary>
        /// <returns>Returns the normalized matrix</returns>
        public static double[] Normalize(this INormalizedData data, double[] row)
        {
            return NormalizeTuple(data.MinMaxValues, row);
        }

        /// <summary>
        /// Gets the min and max value of each column
        /// </summary>
        public static Dictionary<int, Tuple<double, double>> GetMinMaxDictionary(double[][] data)
        {
            return data.SelectMany(row => row.Select((col, i) => Tuple.Create<int, double>(i, col)))
                .GroupBy(x => x.Item1)
                .ToDictionary(x => x.Key, x => Tuple.Create(x.Select(y => y.Item2).Min(), x.Select(y => y.Item2).Max()));
        }

        /// <summary>
        /// Performs normalization on the dataset using the min/max values
        /// </summary>
        public static double[][] Normalize(IDictionary<int, Tuple<double, double>> minMax, double[][] data)
        {
            return data
                .Select((row, i) => NormalizeTuple(minMax, row))
                .ToArray();
        }

        public static double[] NormalizeTuple(IDictionary<int, Tuple<double, double>> minMax, double[] row)
        {
            return row
                .Select((col, j) => NormalizeValue(minMax[j].Item1, minMax[j].Item2, col))
                .ToArray();
        }

        public static double NormalizeValue(double min, double max, double value)
        {
            return (value - min) / (max - min);
        }
    }
}