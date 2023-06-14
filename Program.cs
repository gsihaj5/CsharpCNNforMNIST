namespace NonGUICNN
{
    class Program
    {
        public static void Main()
        {
            int[] networkShape = { 16, 
                128,
                128,
                10 };
            NeuralNetwork nn = new NeuralNetwork(networkShape, .001f, 1000, "Sigmoid");
            CNN cnn = new CNN();

            //raw processing directly from grayscale 
            Train(nn, cnn, @"./mnist_train.csv");
            EvaluateNN(nn, cnn, @"./mnist_test.csv");
        }

        private static void Train(NeuralNetwork nn, CNN cnn, string filename)
        {
            float[,] sumWeightErrorBridge = new float[1000, 16];
            using (var reader = new StreamReader(filename))
            {
                int counter = 0;
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line?.Split(',');

                    float[] oneHotEncoding = new float[10];
                    int label = Convert.ToInt16(values?[0]);
                    oneHotEncoding[label] = 1;
                    float[] float_values = Array.ConvertAll(values, float.Parse);
                    float[] grayscales = new float[784];
                    Array.Copy(float_values, 1, grayscales, 0, 784);

                    float[] preparedImage = cnn.getPreparedImage(grayscales);

                    nn.Process(preparedImage);

                    // get the sumweight error from input layer on NN

                    nn.BackPropagate(oneHotEncoding, preparedImage, ref sumWeightErrorBridge);

                    cnn.maxPoolSumWeightError(sumWeightErrorBridge, 8, counter);
                    cnn.backPropKernel(4);
                    if (counter == 999)
                    {
                        cnn.updateKernel2();
                        counter = 0;
                    }

                    counter++;
                }
            }
        }

        private static void EvaluateNN(NeuralNetwork nn, CNN cnn, string filename)
        {
            int numberOfTruth = 0;
            int numberOfFalse = 0;
            using (var reader = new StreamReader(filename))
            {
                int counter = 0;
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line?.Split(',');

                    float[] oneHotEncoding = new float[10];
                    int label = Convert.ToInt16(values?[0]);
                    oneHotEncoding[label] = 1;
                    float[] float_values = Array.ConvertAll(values, float.Parse);
                    float[] grayscales = new float[784];
                    Array.Copy(float_values, 1, grayscales, 0, 784);

                    float[] preparedImage = cnn.getPreparedImage(grayscales);

                    float[] result = nn.Process(preparedImage);
                    float maxValue = result.Max();
                    float maxIndex = result.ToList().IndexOf(maxValue);

                    if (maxIndex == label) numberOfTruth++;
                    else numberOfFalse++;
                }
            }

            float accuracy = (float)numberOfTruth / ((float)numberOfTruth + (float)numberOfFalse);
            Console.WriteLine("Accuracy");
            Console.WriteLine(accuracy);
        }
    }
}