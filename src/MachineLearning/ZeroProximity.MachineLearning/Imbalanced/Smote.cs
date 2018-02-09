using System;
using System.Collections.Generic;
using System.Linq;
using ZeroProximity.MachineLearning.Proximity;

namespace ZeroProximity.MachineLearning.Imbalanced
{
    /// <summary>
    /// SMOTE (Synthetic Minority Over-sampling Technique)
    /// </summary>
    public class Smote
    {
        private static readonly Random Random = new Random();

        public static Tuple<double[][], double[]> AutoBalance(double[][] dataset, double[] labels, int k = 5)
        {
            var labelGroups = labels.GroupBy(x => x).OrderBy(x => x.Count()).ToArray();
            var first = labelGroups.First().Count();
            var next = labelGroups.Skip(1).First().Count();

            var needed = (int)((next - first) / (double) first) * 100;

            return Balance(dataset, labels, Math.Max(100, needed), k);
        }

        public static Tuple<double[][], double[]> Balance(double[][] dataset, double[] labels, int n = 100, int k = 5)
        {
            if (dataset?.Any() != true) throw new ArgumentNullException(nameof(dataset));
            if (labels?.Any() != true) throw new ArgumentNullException(nameof(labels));

            if(k < 1) throw new ArgumentOutOfRangeException(nameof(k), "Must be greater than 1");
            if(n < 100) throw new ArgumentOutOfRangeException(nameof(n), "Must be greater than 100");
            if(n % 100.0 != 0) throw new ArgumentOutOfRangeException(nameof(n), "Must in multiples of 100");

            // Number of synthetic examples will be: (N/100) * T

            var labelGroups = labels.GroupBy(x => x);
            var minority = labelGroups.OrderBy(x => x.Count()).First();
            var minorityLabel = minority.Key;

            var t = minority.Count();

            if (n < 100)
            {
                t = (int)(n / 100.0 * t);
                n = 100;
            }
            n = (int)(n / 100.0);

            var sample = labels.Select((d, i) => new { d, i })
                .Where(x => x.d == minorityLabel)
                .Select(x => dataset[x.i]).ToArray();

            var synthetic = new List<double[]>();

            for (int i = 0; i < t; i++)
            {
                var nnarray = NearestNeighbours.Calculate(sample, sample[i], k);
                var synthRow = GenerateSynthetic(n, i, nnarray, sample);
                synthetic.AddRange(synthRow);
            }

            var result = Tuple.Create(
                dataset.Concat(synthetic).ToArray(),
                labels.Concat(Enumerable.Repeat(minorityLabel, synthetic.Count)).ToArray());

            return result;
        }

        private static IEnumerable<double[]> GenerateSynthetic(int n, int i, int[] nnarray, double[][] sample)
        {
            for (int j = n; j > 0; j--)
            {
                var nn = Random.Next(0, nnarray.Length);

                var length = sample.First().Length;
                var row = new double[length];

                for (int attr = 0; attr < length; attr++)
                {
                    var diff = sample[nnarray[nn]][attr] - sample[i][attr];
                    var gap = Random.NextDouble();
                    row[attr] = sample[i][attr] + gap * diff;
                }

                yield return row;
            }
        }
    }
}
