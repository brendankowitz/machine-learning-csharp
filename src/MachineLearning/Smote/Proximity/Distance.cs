using System;

namespace ZeroProximity.MachineLearning.Proximity
{
    public static class Distance
    {
        /// <summary>
        /// Calculates the Euclidean Distance Measure between two data points
        /// </summary>
        public static double EuclideanDistance(double[] x, double[] y)
        {
            int count;
            double distance;
            double sum = 0.0;

            if (x.GetUpperBound(0) != y.GetUpperBound(0))
            {
                throw new ArgumentException("The number of elements in X must match the number of elements in Y");
            }

            count = x.Length;

            for (int i = 0; i < count; i++)
            {
                sum = sum + Math.Pow(Math.Abs(x[i] - y[i]), 2);
            }

            distance = Math.Sqrt(sum);

            return distance;
        }
    }
}