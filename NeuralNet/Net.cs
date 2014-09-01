using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class Net
    {
        // First weight in each list bias weight
        public List<List<double>> inpToHiddenWeight = new List<List<double>>();
        public List<List<double>> hidToOutputWeight = new List<List<double>>();

        public List<double> inp_act = new List<double>();
        public List<double> hid_act = new List<double>();
        public List<double> outp_act = new List<double>();

        public List<double> outp_delta = new List<double>();
        public List<double> hid_delta = new List<double>();

        public int numInput;
        public int numHidden;
        public int numOutput;

        public List<double> A = new List<double>();
        public List<double> B = new List<double>();
        public List<double> C = new List<double>();
        public List<double> D = new List<double>();

        public double BiasDelta;

        public Net(int inpNodes, int hiddenNodes, int outputNodes)
        {
            numInput = inpNodes;
            numHidden = hiddenNodes;
            numOutput = outputNodes;
        }

        public void Train(List<Tuple<List<double>, List<bool>>> trainingData, int epochs, double lRate)
        {
            for (int k = 0; k < epochs; k++)
            {
                foreach(var examp in trainingData)
                {
                    // Reset all the lists
                    inp_act.Clear();
                    hid_act.Clear();
                    outp_act.Clear();
                    hid_delta.Clear();
                    outp_delta.Clear();

                    // Add Bias Weights
                    inp_act.Add(-1);
                    hid_act.Add(-1);

                    // Propogate forward activations
                    foreach (var amt in examp.Item1)
                    {
                        inp_act.Add(amt);
                    }
                    for (int j = 0; j < numHidden; j++)
                    {
                        double inj = 0;
                        for (int i = 0; i <= numInput; i++)
                        {
                            inj += inp_act[i] * inpToHiddenWeight[j][i];
                        }
                        hid_act.Add(sigmoid(inj));
                    }
                    for (int j = 0; j < numOutput; j++)
                    {
                        double inj = 0;
                        for (int i = 0; i <= numHidden; i++)
                        {
                            inj += hid_act[i] * hidToOutputWeight[j][i];
                        }
                        outp_act.Add(sigmoid(inj));
                    }

                    //Back propogate to get deltas
                    for (int j = 0; j < numOutput; j++)
                    {
                        double delta;
                        if (examp.Item2[j])
                            delta = outp_act[j] * (1 - outp_act[j]) * (1 - outp_act[j]);
                        else
                            delta = outp_act[j] * (1 - outp_act[j]) * (-outp_act[j]);
                        outp_delta.Add(delta);
                    }

                    for (int j = 1; j <= numHidden; j++)
                    {
                        double sumdelta = 0;
                        for (int i = 0; i < numOutput; i++)
                        {
                            sumdelta += hidToOutputWeight[i][j] * outp_delta[i];
                        }
                        hid_delta.Add(hid_act[j] * (1 - hid_act[j]) * sumdelta);
                    }

                    //Scale weights
                    for (int j = 0; j <= numHidden; j++)
                    {
                        for (int i = 0; i < numOutput; i++)
                        {
                            hidToOutputWeight[i][j] += lRate * hid_act[j] * outp_delta[i];
                        }
                    }

                    for (int j = 0; j <= numInput; j++)
                    {
                        for (int i = 0; i < numHidden; i++)
                        {
                            inpToHiddenWeight[i][j] += lRate * inp_act[j] * hid_delta[i];
                        }
                    }
                }
            }
        }

        public void Test(List<Tuple<List<double>, List<bool>>> testSet)
        {
            A.Clear();
            B.Clear();
            C.Clear();
            D.Clear();
            for (int i = 0; i < numOutput; i++)
            {
                A.Add(0);
                B.Add(0);
                C.Add(0);
                D.Add(0);
            }
            foreach (var examp in testSet)
            {
                // Reset the lists
                inp_act.Clear();
                hid_act.Clear();
                outp_act.Clear();

                // Add Bias Weights
                inp_act.Add(-1);
                hid_act.Add(-1);

                // Propogate forward to generate output activations
                foreach (var amt in examp.Item1)
                {
                    inp_act.Add(amt);
                }
                for (int j = 0; j < numHidden; j++)
                {
                    double inj = 0;
                    for (int i = 0; i <= numInput; i++)
                    {
                        inj += inp_act[i] * inpToHiddenWeight[j][i];
                    }
                    hid_act.Add(sigmoid(inj));
                }
                for (int j = 0; j < numOutput; j++)
                {
                    double inj = 0;
                    for (int i = 0; i <= numHidden; i++)
                    {
                        inj += hid_act[i] * hidToOutputWeight[j][i];
                    }
                    outp_act.Add(sigmoid(inj));
                }
                
                // Threshold
                for (int i = 0; i < numOutput; i++)
                {
                    if (outp_act[i] >= .5)
                        if (examp.Item2[i])
                            A[i]++;
                        else
                            B[i]++;
                    else
                        if (examp.Item2[i])
                            C[i]++;
                        else
                            D[i]++;
                }
            }
        }

        //Computes Sigmoid Function
        double sigmoid(double d)
        {
            return 1 / (1 + Math.Exp(-d));
        }
    }
}


