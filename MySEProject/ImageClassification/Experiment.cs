using AConfig;
using NeoCortexApi.Entities;
using NeoCortexApi;
using NeoCortexApi.Network;
using NeoCortexApi.Utility;
using Daenet.ImageBinarizerLib;
using Daenet.ImageBinarizerLib.Entities;

namespace ConsoleApp
{
    /// <summary>
    /// Class Experiment contains methods for setting HTM Parameters
    /// and binarization of the images
    /// </summary>
    
    internal class Experiment
    {
        HtmConfig htmConfig;
        ArgsConfig expConfig;
                
        public Experiment(ArgsConfig config)
        {
            expConfig = config;
            htmConfig = config.htmConfig;
        }

        /// <summary>
        /// This method binarizes input images and works as an Encoder by
        /// encoding Input Images into arrays of 0 & 1.
        /// </summary>
        public void run()
        {
            int height = htmConfig.InputDimensions[0];
            int width = htmConfig.InputDimensions[1];

            // By default it only returns subdirectories one level deep. 
            var directories = Directory.GetDirectories(expConfig.inputFolder).ToList();

            // The key of the dictionary helps to keep track of which class the SDR belongs to
            (Dictionary<string, int[]> binaries, // List of Binarized images
                Dictionary<string, List<string>> inputsPath // Path of the list of images found in the given folder
            ) = imageBinarization(directories, width, height);

            
            // For training the Spatial Pooler with Black and White Images
            (Dictionary<string, int[]> sdrs, var cortexLayer) = SPTrain(htmConfig, binaries);
            
            // For training Spatial Pooler with colored images
            //(Dictionary<string, int[]> sdrs2, var cortexLayer2) = SPTrain(htmConfig, binaries, colorThreshold );

            HelpersTemp helperFunc = new HelpersTemp();

            Dictionary<string, double> listCorrelation = new();
            Dictionary<string, double> listInputCorrelation = new();
            foreach (KeyValuePair<string, List<string>> entry in inputsPath) // loop of the folder (classes) eg: Apple, banana, etc
            {
                var classLabel = entry.Key;
                var filePathList = entry.Value;
                var numberOfImages = filePathList.Count;

                for (int i = 0; i < numberOfImages; i++) // loop of the images inside the InputFolder
                {
                    if (!sdrs.TryGetValue(filePathList[i], out int[] sdr1)) continue;

                    foreach (KeyValuePair<string, List<string>> secondEntry in inputsPath)
                    { // loop of the folder (again)
                        var classLabel2 = secondEntry.Key;
                        var filePathList2 = secondEntry.Value;
                        var numberOfImages2 = filePathList2.Count;
                        for (int j = 0; j < numberOfImages2; j++) // loop of the images inside the InputFolder
                        {
                            if (!sdrs.TryGetValue(filePathList2[j], out int[] sdr2)) continue;
                            string fileNameofFirstImage = Path.GetFileNameWithoutExtension(filePathList[i]);
                            string fileNameOfSecondImage = Path.GetFileNameWithoutExtension(filePathList2[j]);
                            string temp = $"{classLabel + fileNameofFirstImage}__{classLabel2 + fileNameOfSecondImage}";

                            //For output similarity calculation
                            listCorrelation.Add(temp, (MathHelpers.CalcArraySimilarity(sdr1, sdr2)));
                            //For input similarity calculation (within same folder)
                            listInputCorrelation.Add(temp, Math.Round(MathHelpers.CalcArraySimilarity(binaries[filePathList[i]].IndexWhere((el) => el == 1), binaries[filePathList2[j]].IndexWhere((el) => el == 1)), 2));
                        }
                    }
                }
            }

            var classes = inputsPath.Keys.ToList();
            //helperFunc.printSimilarityMatrix(listCorrelation, "micro", classes);
            //helperFunc.printSimilarityMatrix(listCorrelation, "macro", classes);
            helperFunc.printSimilarityMatrix(listCorrelation, "both", classes); //Prints output similarity at console

            //To check the input similarity between two images of Label 'Nine'in InputFolder: 9_pic1.png and 9_pic3.png
            //Console.WriteLine("Input Similarity in Label Nine for images '9_pic1' and '9_pic3': " + (listInputCorrelation["Nine9_pic1__Nine9_pic3"])+" %");
            //To check the input similarity between two images of Label 'One' in InputFolder: 1_pic2.png and 1_pic3.png
            //Console.WriteLine("Input Similarity in Label One for images '1_pic1' and '1_pic3': " + (listInputCorrelation["One1_pic2__One1_pic3"])+ " %");
            //To check the input similarity between two images of Label 'Nine' and 'One' in InputFolder: 9_pic1.png and 1_pic1.png
            //Console.WriteLine("Input Similarity in Between Labels Nine and One for images '9_pic1' and '1_pic1': " + (listInputCorrelation["Nine9_pic1__One1_pic1"])+ " %");
            #region
            /// <summary>
            /// This helps to get the prediction of all images in the TestFolder in single execution of code
            /// making it efficient and save time.
            /// To enable the prediction code to run independently of the local machine, the method DirProject 
            /// gets the path of directory in which TestFolder is located.
            /// Images in TestFolder are iterated and paths of testing images in the TestFolder
            /// are fed to ReadImageData method for encoding.
            /// </summary>
            string MyDirPath = DirPath();  //returns path of directory in which TestFolder is located
            string TestFolderPath = MyDirPath + "\\TestFolder";  //generated full path of TestFolder

            //string array of paths of all testing images with .png file extention in TestFolder
            string[] imagePaths = Directory.GetFiles(@TestFolderPath, "*.png", SearchOption.AllDirectories); 
            var numberOfTestImages = imagePaths.Length;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"\nNumber of Images to be predicted in the TestFolder: {numberOfTestImages}");
            for (int i = 0; i < imagePaths.Length; i++) //loop of the images inside the TestFolder
            {
                string testingImageName = Path.GetFileNameWithoutExtension(imagePaths[i]);
                Console.BackgroundColor = ConsoleColor.Red;
                Console.ForegroundColor = ConsoleColor.Black;
                Console.WriteLine($"\n[{i+1}] Prediction Result for the Testing Image: {testingImageName}.png");
                Console.BackgroundColor = ConsoleColor.Black;
                //Console.ForegroundColor = ConsoleColor.Black;
                // Prediction Code
                // Testing image encoding,path of each iterated Testing Image is input to ReadImageData method
                int[] encodedInputImage = ReadImageData(imagePaths[i], width, height);
                var temp1 = cortexLayer.Compute(encodedInputImage, false);

                // This is a general way to get the SpatialPooler result from the layer.
                var activeColumns = cortexLayer.GetResult("sp") as int[];

                var sdrOfTestingImage = activeColumns.OrderBy(c => c).ToArray(); //SDR of presently iterated Testing Image

                string predictedLabel = PredictLabel(sdrOfTestingImage, sdrs);
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n============Input Image Prediction============");
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine($"\n>>Prediction status: {predictedLabel}"); //Displaying the prediction status obtained from Method "PredictLabel"
                Console.ForegroundColor = ConsoleColor.White;
            }
            /// <summary>
            /// Prediction Code done by Alam Sher Khan & Aiman Zehra
            /// The method PredictLabel compares the SDR of image to be predicted (Testing Image in TestFolder) with the SRDs 
            /// of the images used for learning and outputs the average, maximum and minimum similarity of Testing Image with 
            /// Images under each Learning Class (Label) and classify its Label with Maximum Similarity
            /// </summary>
            string PredictLabel(int[] sdrOfTestingImage, Dictionary<string, int[]> sdrs)
            {
                double similarityWithEachSDR = 0;
                double temp1 = 0;
                string label = "";
                foreach (KeyValuePair<string, List<string>> secondEntry in inputsPath)
                {
                    //List 'similarities' save the values of similarity of tested images with SDR of images in 
                    //the training labels and will later be used to calculate Max, Avg and Min similaries for the tested image
                    List<double> similarities = new List<double>();

                    // loop of each folder in the InputFolder
                    var classLabel2 = secondEntry.Key;
                    var filePathList2 = secondEntry.Value;
                    var numberOfImages2 = filePathList2.Count;
                    for (int j = 0; j < numberOfImages2; j++) // loop of each image in each category of Training Label
                    {
                        if (!sdrs.TryGetValue(filePathList2[j], out int[] sdr2)) continue;
                                                                       
                        //calculating the similarity between SDR of Testing Image with the SDR of the current iterated image (Training Dataset)
                        similarityWithEachSDR = Math.Round(MathHelpers.CalcArraySimilarity(sdrOfTestingImage, sdr2),2);
                        similarities.Add(similarityWithEachSDR); //saving similarity values in the list 'similarities'
                    }
                    //calculating the Maximum,Average and Minimum similarities of the Testing Image with Training Images in each Category (Label)
                    double maxSimilarity = Math.Round(similarities.Max(),2);
                    double avgSimilarity = Math.Round(similarities.Average(),2);
                    double minSimilarity = Math.Round(similarities.Min(),2);
                    
                    if (maxSimilarity > temp1)
                    {
                        temp1 = maxSimilarity;
                        label = $"{"The image is predicted as " + secondEntry.Key}";
                        //Similarity Threshold Value, it is selected on the basis of similarity matrix
                        //in the training phase based on HTM parameters given in htmconfig.json file
                        if (temp1<65.0) 
                        {
                            label = "The similarity between the tested image and training labels is lower than the required Similarity Threshold to qualify it for prediction, hence the tested image might not belong to the Training Dataset.";
                        }
                     }
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("\nSimilarity of Tested Image to Digit " + secondEntry.Key + ":");
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("> Maximum:" + maxSimilarity + " %, Average:" + avgSimilarity + " %, Minimum:" + minSimilarity + " %");
                   }
                //Display the highest similarity  of the Testing Image with the training category
                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.WriteLine("\n Highest Similarity is: " + temp1 + " % ");
                return label;
              }

        }
        //The method DirPath is used for returning the current path of Directory in which TestFolder
        //is placed
        public static string DirPath()
        {
            string DirPath = System.IO.Directory.GetCurrentDirectory();
            
            return DirPath;
        }
        #endregion
        private Tuple<Dictionary<string, int[]>, Dictionary<string, List<string>>> imageBinarization(List<string> directories, int width, int height)
        {
            Dictionary<string, List<string>> inputsPath = new Dictionary<string, List<string>>();
            Dictionary<string, int[]> binaries = new Dictionary<string, int[]>();

            foreach (var fullPath in directories)
            {
                string folderName = Path.GetFileName(fullPath);

                if (!inputsPath.ContainsKey(folderName))
                {
                    inputsPath[folderName] = new List<string>();
                }

                var filePathList = Directory.GetFiles(fullPath).Where(name => !name.EndsWith(".txt")).ToList();


                foreach (var filePath in filePathList)
                {

                    inputsPath[folderName].Add(filePath);

                    // Image binarization
                    int[] inputVector = ReadImageData(filePath, height, width);
                    // Convert inputVector to 2-D array
                    string[] savedVector = ConvertToString(inputVector, height, width);
                    // Write binarized data to a file
                    var baseDir = Path.GetDirectoryName(filePath);
                    var fileNameWithoutExt = Path.GetFileNameWithoutExtension(filePath);
                    var ext = "txt";

                    var fullFileName = $"{fileNameWithoutExt}.{ext}";
                    binaries.Add(filePath, inputVector);
                    System.IO.File.WriteAllLines(Path.Combine(baseDir, fullFileName), savedVector);
                }
            }
            return Tuple.Create(binaries, inputsPath);
        }

        private string[] ConvertToString(int[] inputVector, int height, int width)
        {
            string[] vs = new string[width];
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    vs[i] += inputVector[j * width + i].ToString() + ',';
                }
            }
            return vs;
        }

        /// <summary>
        /// Returns Binarized Image in integer array,color threshold not required for black and white images
        /// </summary>
        /// <param name="imagePath">Name of Image to be binarized</param>
        /// <param name="height">Height of Binarized Image</param>
        /// <param name="width">Width of Binarized Image</param>
        /// <returns></returns>

        public int[] ReadImageData(string imagePath, int height, int width)
        {
            var parameters = new BinarizerParams
            {
                InputImagePath = imagePath,
                ImageHeight = height,
                ImageWidth = width,
                //BlueThreshold = 200,
                //RedThreshold = 200,
                //GreenThreshold = 200
            };
            ImageBinarizer bizer = new ImageBinarizer(parameters);

            var doubleArray = bizer.GetArrayBinary();
            var hg = doubleArray.GetLength(1);
            var wd = doubleArray.GetLength(0);
            var intArray = new int[hg * wd];
            for (int j = 0; j < hg; j++)
            {
                for (int i = 0; i < wd; i++)
                {
                    intArray[j * wd + i] = (int)doubleArray[i, j, 0];
                }
            }
            return intArray;
        }
        /// <summary> Modified by Long Nguyen
        /// Pulling out SDRs after HPC fires a STABLE event when training the SP with list of patterns
        /// </summary>
        /// <param name="cfg"></param> Spatial Pooler configuration by HtmConfig style
        /// <param name="inputValues"></param> Binary input vector (pattern) list
        private static (Dictionary<string, int[]>, CortexLayer<object, object> cortexLayer) SPTrain(HtmConfig cfg, Dictionary<string, int[]> inputValues)
        {
            // Creates the htm memory.
            var mem = new Connections(cfg);
            bool isInStableState = false;
            // HPC extends the default Spatial Pooler algorithm.
            // The purpose of HPC is to set the SP in the new-born stage at the begining of the learning process.
            // In this stage the boosting is very active, but the SP behaves instable. After this stage is over
            // (defined by the second argument) the HPC is controlling the learning process of the SP.
            // Once the SDR generated for every input gets stable, the HPC will fire event that notifies your code
            // that SP is stable now.
            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, inputValues.Count * 40,
                (isStable, numPatterns, actColAvg, seenInputs) =>
                {
                    // Event should only be fired when entering the stable state.
                    // Ideal SP should never enter unstable state after stable state.
                    if (isStable == false)
                    {
                        Console.WriteLine($"INSTABLE STATE");
                        // This should usually not happen.
                        isInStableState = false;
                    }
                    else
                    {
                        Console.WriteLine($"STABLE STATE");
                        // Here you can perform any action if required.
                        isInStableState = true;
                    }
                });

            // It creates the instance of Spatial Pooler Multithreaded version.
            SpatialPooler sp = new SpatialPoolerMT(hpa);

            // Initializes the Spatial Pooler 
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            // mem.TraceProximalDendritePotential(true);

            // It creates the instance of the neo-cortex layer.
            // Algorithm will be performed inside of that layer.
            CortexLayer<object, object> cortexLayer = new CortexLayer<object, object>("L1");

            // Add encoder as the very first module. This model is connected to the sensory input cells
            // that receive the input. Encoder will receive the input and forward the encoded signal
            // to the next module.
            //cortexLayer.HtmModules.Add("encoder", encoder);

            // The next module in the layer is Spatial Pooler. This module will receive the output of the
            // encoder.
            cortexLayer.HtmModules.Add("sp", sp);

            // Learning process will take 1000 iterations (cycles)
            int maxSPLearningCycles = 1;
            // Save the result SDR into a list of array
            Dictionary<string, int[]> outputValues = new Dictionary<string, int[]>();

            for (int cycle = 0; cycle < maxSPLearningCycles; cycle++)
            {
                Console.WriteLine($"Cycle  ** {cycle} ** Stability: {isInStableState}");

                int iteration = 0;
                outputValues.Clear(); // Remove all elements in output SDR list

                // This trains the layer on input pattern.

                foreach (var input in inputValues)
                {
                    iteration++;

                    // Learn the input pattern.
                    // Output lyrOut is the output of the last module in the layer.

                    var lyrOut = cortexLayer.Compute(input.Value, true) as ComputeCycle;

                    // This is a general way to get the SpatialPooler result from the layer.
                    var activeColumns = cortexLayer.GetResult("sp") as int[];

                    var actCols = activeColumns.OrderBy(c => c).ToArray();

                    outputValues[input.Key] = actCols;
                }
                if (isInStableState)
                    break;
            }
            return (outputValues, cortexLayer);
        }
        
    }
}

