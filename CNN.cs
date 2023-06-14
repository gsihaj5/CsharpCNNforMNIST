namespace NonGUICNN;

public class CNN
{
    private float[,] kernel1, kernel2;
    private float[,] kernel1deltaWeight, kernelDeltaWeight;
    private float[,] maxPooledImage1cache;
    private float[,] feature2sumWeightError;

    public CNN()
    {
        kernel1 = createRandomKernel();
        kernel2 = createRandomKernel();
        kernel1deltaWeight = new float[5, 5];
        kernelDeltaWeight = new float[5, 5];
        feature2sumWeightError = new float[64, 64];
    }

    private float[,] createRandomKernel()
    {
        float[,] newKernel = new float[5, 5];
        RandomNormal rand = new RandomNormal();
        for (int y = 0; y < 5; y++)
        {
            for (int x = 0; x < 5; x++)
            {
                newKernel[x, y] = (float)rand.NextNormal(0, 5);
            }
        }

        return newKernel;
    }

    private float[,] reshapeImage(float[] originalImage, int dimension)
    {
        float[,] reshapedImage = new float[dimension, dimension];

        for (int y = 0; y < dimension; y++)
        {
            for (int x = 0; x < dimension; x++)
            {
                int originalIndex = x + dimension * y;
                reshapedImage[x, y] = originalImage[originalIndex];
            }
        }

        return reshapedImage;
    }


    private float[] flattenImage(float[,] images, int dimension)
    {
        float[] flattenedImage = new float[dimension * dimension];
        for (int y = 0; y < dimension; y++)
        {
            for (int x = 0; x < dimension; x++)
            {
                flattenedImage[x + y * dimension] = images[x, y];
            }
        }

        return flattenedImage;
    }

    public void backPropKernel(int dimension)
    {
        for (int centerY = 2; centerY < dimension - 2; centerY++)
        {
            for (int centerX = 2; centerX < dimension - 2; centerX++)
            {
                float sum = 0;
                // looptrough kernel
                for (int y = -2; y < 2; y++)
                {
                    for (int x = -2; x < 2; x++)
                        kernelDeltaWeight[x + 2, y + 2] += maxPooledImage1cache[centerX + x, centerY + y] *
                                                            feature2sumWeightError[centerX + x, centerY + y];
                }
            }
        }
    }

    public void updateKernel2()
    {
        for (int y = 0; y < 5; y++)
        {
            for (int x = 0; x < 5; x++)
            {
                kernel2[x, y] += kernelDeltaWeight[x, y];
                kernelDeltaWeight[x, y] = 0;
            }
        }
    }

    public void maxPoolSumWeightError(float[,] sumWeightError, int originalImageDimension, int learning_batch)
    {
        int counter = 0;
        for (int cornerY = 0; cornerY < originalImageDimension - 2; cornerY += 2)
        {
            for (int cornerX = 0; cornerX < originalImageDimension - 2; cornerX += 2)
            {
                feature2sumWeightError[cornerX, cornerY] = sumWeightError[learning_batch, counter];
                feature2sumWeightError[cornerX + 1, cornerY] = sumWeightError[learning_batch, counter];
                feature2sumWeightError[cornerX, cornerY + 1] = sumWeightError[learning_batch, counter];
                feature2sumWeightError[cornerX + 1, cornerY + 1] = sumWeightError[learning_batch, counter];
                counter++;
            }
        }
    }

    public float[] getPreparedImage(float[] originalImage)
    {
        float[,] reshapedImage = reshapeImage(originalImage, 28);

        float[,] featureLayer1 = GetFeatureLayer(reshapedImage, kernel1, 28);
        float[,] maxPooledImage1 = GetMaxPooledImage(featureLayer1, 24);
        maxPooledImage1cache = maxPooledImage1;

        float[,] featureLayer2 = GetFeatureLayer(maxPooledImage1, kernel2, 12);
        float[,] maxPooledImage2 = GetMaxPooledImage(featureLayer2, 8);

        return flattenImage(maxPooledImage2, 4);
    }

    public float[,] GetFeatureLayer(float[,] originalImage, float[,] kernel, int originalImageDimension)
    {
        //looptroguh images

        float[,] featureLayer = new float[originalImageDimension - 4, originalImageDimension - 4];

        for (int centerY = 2; centerY < originalImageDimension - 2; centerY++)
        {
            for (int centerX = 2; centerX < originalImageDimension - 2; centerX++)
            {
                float sum = 0;
                // looptrough kernel
                for (int y = -2; y < 2; y++)
                {
                    for (int x = -2; x < 2; x++)
                        sum += kernel[x + 2, y + 2] * originalImage[centerX + x, centerY + y];
                }

                featureLayer[centerX - 2, centerY - 2] = sum;
            }
        }

        return featureLayer;
    }

    public float[,] GetMaxPooledImage(float[,] originalImage, int originalImageDimension)
    {
        float[,] pooledImage = new float[originalImageDimension / 2, originalImageDimension / 2];
        for (int cornerY = 0; cornerY <= originalImageDimension - 2; cornerY += 2)
        {
            for (int cornerX = 0; cornerX <= originalImageDimension - 2; cornerX += 2)
            {
                float[] poolArray =
                {
                    originalImage[cornerX, cornerY],
                    originalImage[cornerX + 1, cornerY],
                    originalImage[cornerX, cornerY + 1],
                    originalImage[cornerX + 1, cornerY + 1]
                };

                pooledImage[cornerX / 2, cornerY / 2] = poolArray.Max();
            }
        }

        return pooledImage;
    }
}