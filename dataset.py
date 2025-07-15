import ast
import time
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import polars as pl


def convert_csv_to_parquet(csv_path, parquet_path):
    df = pl.scan_csv(
        csv_path,
        dtypes={'path': pl.Utf8, 'label': pl.Utf8},
        null_values=None,
        low_memory=False
    ).collect(streaming=True)
    
    df.write_parquet(  # comment
            parquet_path,
            compression='zstd',
            compression_level=3,
            use_pyarrow=True,
            )


def read_path_labels(parquet_path):
    """
    Reads images paths  and their corresponding label or labels from the specified csv file.

    Args:
        csv_path (string): Root directory of csv file.

    Returns:
        tuple: A tuple containing:
            - paths (list): A numpy array of image paths, each representing an image path.
            - labels (list): A list containing list of labels for corresponded image.
    Note: 
        there must be no extra spaces between items in each line of csv file.
    """
    df = pl.read_parquet(
        parquet_path,
        use_pyarrow=True,
        parallel=True,  # parallel reading
        rechunk=False,  # جلوگیری از تخصیص مجدد حافظه
        row_index_name=None,  # ther is no index row
        low_memory=False  # use more memory
    )
    paths = df['path'].to_numpy()
    labels = [ast.literal_eval(label) for label in df['label'].to_numpy()]

    del df
    return paths, labels


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading images and labels from a directory structure.

    Attributes:
        paths (list): A list of image tensors loaded from the directory.
        labels (list): A list of labels corresponding to each image.
    """
    
    def __init__(self, csv_path: str, transform_: transforms.Compose):
        """
        Initializes the custom dataset by reading images paths and labels from the specified csv file.

        Args:
            csv_path (str): Root directory of csv file.
            transform_ (torchvision.transforms.transforms.Compose): The transform to be applied to the images.
        """

        super().__init__()
        self.paths, self.labels = read_path_labels(csv_path)
        self.transform = transform_
        

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of images (and labels) in the dataset.
        """

        return len(self.paths)
    
    def __getitem__(self, index):
        """
        Retrieves the image and label at the specified index.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'image': The image tensor at the specified index.
                - 'label': Labels (list) corresponding to the image.
        """
                
        try:
            with Image.open(self.paths[index]) as img:
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                                
                return {
                    'image': self.transform(img),
                    'label': self.labels[index]  # cab contain single or several labels
                }
        except Exception as e:
            print(f"error while opening image at index {index}, path: {self.paths[index]}")
            raise e 
            # return {
            #         'image': torch.zeros(3, 224, 224),
            #         'label': torch.zeros_like(self.labels_tensor[0])
            #     }


if __name__ == "__main__":
    t0 = time.time()
    
    convert_csv_to_parquet("test_data.csv", "test.parquet")
    
    t1 = time.time()
    
    print(f"converting csv to parquet took: {t1-t0:.3f} seconds")
    
    ds = CustomDataset("test.parquet", transforms.Compose([transforms.ToTensor()]))
    
    for i in range(ds.__len__()):
        ds[i]
    t2 = time.time()
    
    print(f"processing {ds.__len__()} images took: {t2-t1:.3f} seconds")
#test