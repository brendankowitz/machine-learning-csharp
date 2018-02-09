using System.Linq;
using ZeroProximity.MachineLearning.Utility;

namespace ZeroProximity.MachineLearning.Proximity
{
    public class NearestNeighbours
    {
        /// <summary>
        /// Finds the specified number of neighbours to the specified data point
        /// </summary>
        /// <returns>Indecies for nearest</returns>
        public static int[] Calculate(double[][] data, double[] point, int limit)
        {
            var normalized = data.Normalize();

            var nearestList = normalized
                .Select((col, i) => new
                {
                    i,
                    Distance = Distance.EuclideanDistance(col, point)
                })
                .OrderBy(x => x.Distance)
                .Take(limit)
                .Select(x => x.i)
                .ToArray();

            return nearestList;
        }
    }
}