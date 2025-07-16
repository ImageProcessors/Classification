import ast
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import polars as pl
# testing coment for git
def convert_csv_to_parquet(csv_path, parquet_path):
    """تبدیل بهینه CSV به Parquet با استفاده از Polars"""
    df = pl.read_csv(
        csv_path,
        schema_overrides={'path': pl.Utf8, 'label': pl.Utf8},
        null_values=None,
        low_memory=False
    )
    
    df.write_parquet(
        parquet_path,
        compression='zstd',
        compression_level=1,
        use_pyarrow=True,
        statistics=True
    )

def read_path_labels(parquet_path):
    """خواندن بهینه داده‌ها از Parquet با Polars"""
    df = pl.read_parquet(
        parquet_path,
        use_pyarrow=True,
        parallel=True
    )
    
    paths = df['path'].to_numpy()
    labels = [ast.literal_eval(label) for label in df['label'].to_numpy()]
    
    return paths, labels

class OptimizedDataset(Dataset):
    def __init__(self, parquet_path, transform=None):
        super().__init__()
        self.paths, self.labels = read_path_labels(parquet_path)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.contiguous())
        ])
        
        # تعیین اندازه تصویر پیش‌فرض
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            with Image.open(self.paths[index]) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # تبدیل لیبل به تنسور و اطمینان از یکسان بودن طول لیبل‌ها
                labels = np.array(self.labels[index], dtype=np.int64)
                return {
                    'image': self.transform(img),
                    'label': torch.tensor(labels, dtype=torch.long)
                }
        except Exception as e:
            print(f"Error loading {self.paths[index]}: {str(e)}")
            return {
                'image': torch.zeros(3, *self.dummy_size),
                'label': torch.zeros(1, dtype=torch.long)
            }

def optimized_collate(batch):
    """تابع جمع‌آوری بهینه برای دسته‌بندی"""
    images = torch.stack([item['image'] for item in batch])
    
    # پیدا کردن حداکثر طول لیبل در این بچ
    max_len = max(len(item['label']) for item in batch)
    
    # ایجاد تنسور لیبل‌ها با پدینگ
    labels = torch.full((len(batch), max_len), -1, dtype=torch.long)
    for i, item in enumerate(batch):
        labels[i, :len(item['label'])] = item['label']
    
    return {
        'image': images,
        'label': labels
    }

if __name__ == "__main__":
    # تست عملکرد
    csv_path = "D:\\labeling\\test\\fake_dataset_1m.csv"
    parquet_path = "D:\\labeling\\test\\optimized.parquet"
    
    try:
        # مرحله 1: تبدیل CSV به Parquet
        t0 = time.time()
        convert_csv_to_parquet(csv_path, parquet_path)
        t1 = time.time()
        print(f"CSV to Parquet conversion: {t1-t0:.2f} sec")
        
        # مرحله 2: ایجاد Dataset و DataLoader
        dataset = OptimizedDataset(
            parquet_path,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            num_workers=4,
            pin_memory=False,  # غیرفعال کردن pin_memory
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=optimized_collate
        )
        
        # مرحله 3: بنچمارک سرعت
        print("Starting data loading benchmark...")
        t_start = time.time()
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # انتقال دستی به GPU
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            total_samples += len(images)
            if batch_idx % 10 == 0:
                speed = total_samples / (time.time() - t_start)
                print(f"Processed {total_samples} samples | Speed: {speed:.1f} samples/sec")
        
        t_end = time.time()
        print(f"\nFinal Results:")
        print(f"Total samples: {total_samples}")
        print(f"Total time: {t_end-t_start:.2f} seconds")
        print(f"Average speed: {total_samples/(t_end-t_start):.1f} samples/sec")
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise