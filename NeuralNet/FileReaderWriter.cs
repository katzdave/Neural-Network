using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class FileReaderWriter
    {
        public Net loadNeuralNet(string filePath)
        {
            try
            {
                using (StreamReader sr = new StreamReader(filePath))
                {
                    String line = sr.ReadToEnd();
                    var words = line.Split('\n');
                    var nums = words.First().Split(' ').Select(Int32.Parse).ToList();
                    var neuralNet = new Net(nums[0], nums[1], nums[2]);
                    for (int i = 1; i < 1 + nums[1]; i++)
                    {
                        var weights = words[i].Split(' ').Select(Double.Parse).ToList();
                        neuralNet.inpToHiddenWeight.Add(weights);
                    }
                    for (int i = nums[1] + 1; i < 1 + nums[1] + nums[2]; i++)
                    {
                        var weights = words[i].Split(' ').Select(Double.Parse).ToList();
                        neuralNet.hidToOutputWeight.Add(weights);
                    }
                    return neuralNet;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("The file could not be read:");
                Console.WriteLine(e.Message);
                return null;
            }
        }

        public void exportNeuralNet(Net n, string filePath)
        {
            var lines = new List<string>();
            lines.Add(n.numInput.ToString() + ' ' + n.numHidden.ToString() + ' ' + n.numOutput.ToString());
            foreach (var lw in n.inpToHiddenWeight)
            {
                string line = "";
                foreach (var weight in lw)
                {
                    line = line + String.Format("{0:#,0.000}", weight) + ' ';
                }
                lines.Add(line.TrimEnd(' '));
            }
            foreach (var lw in n.hidToOutputWeight)
            {
                string line = "";
                foreach (var weight in lw)
                {
                    line = line + String.Format("{0:#,0.000}", weight) + ' ';
                }
                lines.Add(line.TrimEnd(' '));
            }
            System.IO.File.WriteAllLines(filePath, lines);
        }

        public List<Tuple<List<double>, List<bool>>> getTrainingData(string filePath)
        {
            try
            {
                using (StreamReader sr = new StreamReader(filePath))
                {
                    String everything = sr.ReadToEnd();
                    var words = everything.Split('\n');
                    var nums = words.First().Split(' ').Select(Int32.Parse).ToList();
                    var output = new List<Tuple<List<double>, List<bool>>>();
                    for (int i = 1; i <= nums[0]; i++)
                    {
                        var vals = words[i].Split(' ').Select(Double.Parse).ToList();
                        var dubs = new List<double>();
                        var bools = new List<bool>();
                        for (int j = 0; j < nums[1]; j++)
                        {
                            dubs.Add(vals[j]);
                        }
                        for (int j = nums[1]; j < nums[1] + nums[2]; j++)
                        {
                            if (vals[j] == 0)
                                bools.Add(false);
                            else
                                bools.Add(true);
                        }
                        var tuple = new Tuple<List<double>, List<bool>>(dubs, bools);
                        output.Add(tuple);
                    }
                    return output;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("The file could not be read:");
                Console.WriteLine(e.Message);
                return null;
            }
        }

        public void exportTestResults(List<double> A, List<double> B, List<double> C, List<double> D, string filePath)
        {
            var allAcc = new List<double>();
            var allPrec = new List<double>();
            var allRec = new List<double>();
            var allF1 = new List<double>();

            var lines = new List<string>();

            for (int i = 0; i < A.Count(); i++)
            {
                var acc = accuracy(A[i], B[i], C[i], D[i]);
                var prec = precision(A[i], B[i], C[i], D[i]);
                var rec = recall(A[i], B[i], C[i], D[i]);
                var f1 = F1(prec, rec);

                allAcc.Add(acc);
                allPrec.Add(prec);
                allRec.Add(rec);
                allF1.Add(f1);

                lines.Add(A[i].ToString() + ' '
                    + B[i].ToString() + ' '
                    + C[i].ToString() + ' '
                    + D[i].ToString() + ' '
                    + String.Format("{0:#,0.000}", acc) + ' '
                    + String.Format("{0:#,0.000}", prec) + ' '
                    + String.Format("{0:#,0.000}", rec) + ' '
                    + String.Format("{0:#,0.000}", f1));
            }
            
            var Asum = A.Sum();
            var Bsum = B.Sum();
            var Csum = C.Sum();
            var Dsum = D.Sum();

            lines.Add(String.Format("{0:#,0.000}", accuracy(Asum, Bsum, Csum, Dsum)) + ' '
                + String.Format("{0:#,0.000}", precision(Asum, Bsum, Csum, Dsum)) + ' '
                + String.Format("{0:#,0.000}", recall(Asum, Bsum, Csum, Dsum)) + ' '
                + String.Format("{0:#,0.000}", F1(precision(Asum, Bsum, Csum, Dsum), recall(Asum, Bsum, Csum, Dsum))));

            lines.Add(String.Format("{0:#,0.000}", allAcc.Average()) + ' '
                + String.Format("{0:#,0.000}", allPrec.Average()) + ' '
                + String.Format("{0:#,0.000}", allRec.Average()) + ' '
                + String.Format("{0:#,0.000}", F1(allPrec.Average(), allRec.Average())));

            System.IO.File.WriteAllLines(filePath, lines);
        }

        public double accuracy(double A, double B, double C, double D)
        {
            return (A + D) / (A + B + C + D);
        }
        public double precision(double A, double B, double C, double D)
        {
            return A / (A + B);
        }
        public double recall(double A, double B, double C, double D)
        {
            return A / (A + C);
        }
        public double F1(double precision, double recall)
        {
            return (2 * precision * recall) / (precision + recall);
        }
    }
}
