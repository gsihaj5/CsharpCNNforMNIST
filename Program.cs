namespace NonGUICNN
{
    class Program
    {
        public static void Main()
        {
            int[] networkShape = { 784, 100, 100, 100, 100, 10 };
            NeuralNetwork nn = new NeuralNetwork(networkShape, .01f, 1000, "Sigmoid");

            //raw processing directly from grayscale 
            Train(nn, @"./mnist_train.csv");
            EvaluateNN(nn, @"./mnist_test.csv");
        }

        private static void Train(NeuralNetwork nn, string filename)
        {
            using (var reader = new StreamReader(filename))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line?.Split(',');

                    float[] oneHotEncoding = new float[10];
                    int label = Convert.ToInt16(values?[0]);
                    oneHotEncoding[label] = 1;
                    float[] grayscales = Array.ConvertAll(values, float.Parse);
                    Console.WriteLine(grayscales.Length);

                    Console.WriteLine(label);
                    oneHotEncoding.ToList().ForEach(i => Console.Write(i.ToString()));
                    Console.WriteLine();

                    nn.Process(grayscales);
                    nn.BackPropagate(oneHotEncoding, grayscales);
                }
            }
        }

        private static void EvaluateNN(NeuralNetwork nn, string filename)
        {
            int numberOfTruth = 0;
            int numberOfFalse = 0;
            using (var reader = new StreamReader(filename))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line?.Split(',');

                    float[] oneHotEncoding = new float[10];
                    int label = Convert.ToInt16(values?[0]);
                    oneHotEncoding[label] = 1;
                    float[] grayscales = Array.ConvertAll(values, float.Parse);

                    float[] result = nn.Process(grayscales);
                    float maxValue = result.Max();
                    float maxIndex = result.ToList().IndexOf(maxValue);

                    if (maxIndex == label) numberOfTruth++;
                    else numberOfFalse++;
                }
            }

            float accuracy = numberOfTruth / ((float)numberOfTruth + (float)numberOfFalse);
            Console.WriteLine("Accuracy");
            Console.WriteLine(accuracy);
        }
    }
}