import torch
import torchvision
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA

def getDataLoaders(batch_size=64):
    # MNIST dataset download
    allTrainData = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    allTestData = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)

    # Filtering only numbers 0, 1, 2, 3 (MNIST-4)
    trainIndex = torch.where(allTrainData.targets < 4)[0]
    testIndex = torch.where(allTestData.targets < 4)[0]

    trainImages = allTrainData.data[trainIndex]
    trainLabels = allTrainData.targets[trainIndex]
    testImages = allTestData.data[testIndex]
    testLabels = allTestData.targets[testIndex]

    # Normalise images
    trainImagesNorm = trainImages.float() / 255.0
    testImagesNorm = testImages.float() / 255.0

    # Flatten images to list
    trainImagesFlat = torch.flatten(trainImagesNorm, start_dim=1)
    testImagesFlat = torch.flatten(testImagesNorm, start_dim=1)

    # PCA - take list of data and make vector of size 8,
    # which represents 8 most important pixels with maximum of differencies
    pca = PCA(n_components=8)
    trainPCA = pca.fit_transform(trainImagesFlat.numpy())
    testPCA = pca.transform(testImagesFlat.numpy())

    # Transfer to tensor
    trainTensor = torch.tensor(trainPCA, dtype=torch.float32)
    testTensor = torch.tensor(testPCA, dtype=torch.float32)

    # Joining labels with its new vectors
    trainDataset = TensorDataset(trainTensor, trainLabels)
    testDataset = TensorDataset(testTensor, testLabels)

    # Making DataLoader
    trainLoader = DataLoader(trainDataset, batch_size, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size, shuffle=False)

    return trainLoader, testLoader, pca

if __name__ == "__main__":
    trainLoader, testLoader, pca = getDataLoaders()
    trainBatch_x, trainBatch_y = next(iter(trainLoader))
    testBatch_x, testBatch_y = next(iter(testLoader))
    print(f"TrainLoader:")
    print(f"    {trainBatch_x.shape}")
    print(f"    {trainBatch_y.shape}")
    print(f"TestLoader:")
    print(f"    {testBatch_x.shape}")
    print(f"    {testBatch_y.shape}")