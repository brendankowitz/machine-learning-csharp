using System.Linq;
using Xunit;
using ZeroProximity.MachineLearning.Imbalanced;

namespace ZeroProximity.MachineLearning.Tests
{
    public class UnitTest1
    {
        private double[][] ds;
        private double[] labels;

        public UnitTest1()
        {
            ds = new[]
            {
                new[] { 1.0, 1.0, 0.0 },
                new[] { 1.0, 1.0, 0.0 },
                new[] { 1.0, 1.0, 0.0 },
                new[] { 1.0, 0.0, 1.0 },
                new[] { 1.0, 0.0, 1.0 },
                new[] { 1.0, 0.0, 1.0 },
                new[] { 0.0, 1.0, 1.0 },
                new[] { 0.0, 1.0, 1.0 },
                new[] { 0.0, 1.0, 1.0 },
                new[] { 0.0, 1.0, 1.0 },
                new[] { 0.0, 1.0, 1.0 },
                new[] { 0.0, 1.0, 1.0 },
                new[] { 2.0, 0.0, 2.0 },
                new[] { 2.0, 2.0, 0.0 },
                new[] { 0.0, 2.0, 0.0 },
            };

            labels = new[]
            {
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0
            };
        }

        [Fact]
        public void AutoBalance()
        {
            var balanced = Smote.AutoBalance(ds, labels, k: 2);
            var groups = balanced.Item2.GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count());
            Assert.Equal(12, groups[1.0]);
        }

        [Fact]
        public void Test1()
        {
            var balanced = Smote.Balance(ds, labels, k: 2);
            var groups = balanced.Item2.GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count());
            Assert.Equal(6, groups[1.0]);  
        }

        [Fact]
        public void Test2()
        {
            var balanced = Smote.Balance(ds, labels, n: 50, k: 2);
            var groups = balanced.Item2.GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count());
            Assert.Equal(4, groups[1.0]);
        }

        [Fact]
        public void Test3()
        {
            var balanced = Smote.Balance(ds, labels, n: 150, k: 2);
            var groups = balanced.Item2.GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count());
            Assert.Equal(6, groups[1.0]);
        }
    }
}
