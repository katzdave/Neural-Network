using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace NeuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            var reader = new FileReaderWriter();
            List<Tuple<List<double>, List<bool>>> inputData = null;
            Net net = null;
            string inp, outputFile;

            start:
            Console.WriteLine("Do you wish to train?");
            inp = Console.ReadLine();
            if (inp[0] == 'y' || inp[0] == 'Y')
                goto train;
            Console.WriteLine("Do you wish to test?");
            inp = Console.ReadLine();
            if (inp[0] == 'y' || inp[0] == 'Y')
                goto test;
            goto start;

            train:
            while(net == null)
            {
                Console.WriteLine("Enter name of initial network file:");
                inp = Console.ReadLine();
                net = reader.loadNeuralNet(inp);
            }
            while(inputData == null)
            {
                Console.WriteLine("Enter name training data file:");
                inp = Console.ReadLine();
                inputData = reader.getTrainingData(inp);
            }

            Console.WriteLine("Enter name of output file:");
            outputFile = Console.ReadLine();

            Console.WriteLine("Enter number of epochs:");
            inp = Console.ReadLine();
            int epochs;
            if (!int.TryParse(inp, out epochs))
            {
                Console.WriteLine("Parse Failed defaulting to 100");
                epochs = 100;
            }

            Console.WriteLine("Enter learning rate:");
            inp = Console.ReadLine();
            double lrate;
            if (!double.TryParse(inp, out lrate))
            {
                Console.WriteLine("Parse Failed defaulting to .1");
                lrate = .1;
            }
            net.Train(inputData, epochs, lrate);
            reader.exportNeuralNet(net, outputFile);
            Console.WriteLine("Training Complete!");
            Environment.Exit(0);

            test:
            while (net == null)
            {
                Console.WriteLine("Enter name of trained network file:");
                inp = Console.ReadLine();
                net = reader.loadNeuralNet(inp);
            }
            while (inputData == null)
            {
                Console.WriteLine("Enter name test file:");
                inp = Console.ReadLine();
                inputData = reader.getTrainingData(inp);
            }
            Console.WriteLine("Enter name of output file:");
            outputFile = Console.ReadLine();

            net.Test(inputData);
            reader.exportTestResults(net.A, net.B, net.C, net.D, outputFile);
            Console.WriteLine("Testing Complete!");
            Environment.Exit(0);
        }
    }
}
