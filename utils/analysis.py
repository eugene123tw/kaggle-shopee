from .utils import read_csv


def visualise(data_root: str, csv_path: str):
    lines = read_csv(csv_path)


if __name__ == "__main__":
    visualise(
        data_root="/home/user/_DATASET/shopee-product-matching/train_images",
        csv_path="/home/user/_DATASET/shopee-product-matching/train.csv"
    )
