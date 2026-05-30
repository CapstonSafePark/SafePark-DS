import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = r"C:\Users\olcha\Safepark\SafePark-DS"

DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset_lane_crop")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
SAVE_PATH = os.path.join(WEIGHTS_DIR, "lane_crop_type_classifier.pth")

CLASS_NAMES = [
    "yellow_double",
    "yellow_single",
    "white_dotted",
    "white_solid",
    "none"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)


class LaneCropDataset(Dataset):
    def __init__(self, root_dir, class_names):
        self.root_dir = root_dir
        self.class_names = class_names
        self.samples = []

        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_dir):
                print("[경고] 폴더 없음:", class_dir)
                continue

            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(class_dir, filename), label_idx)
                    )

        print(root_dir, "이미지 개수:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        img = img.astype("float32") / 255.0
        img = np.transpose(img, (2, 0, 1))

        x = torch.from_numpy(img).float()
        y = torch.tensor(label, dtype=torch.long)

        return x, y


class LaneCropClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, epoch, num_epochs):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        _, preds = torch.max(outputs, 1)
        total_correct += (preds == y).sum().item()
        total_count += y.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / total_count, total_correct / total_count


def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    class_correct = [0 for _ in CLASS_NAMES]
    class_total = [0 for _ in CLASS_NAMES]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)

            _, preds = torch.max(outputs, 1)

            total_correct += (preds == y).sum().item()
            total_count += y.size(0)

            for label, pred in zip(y, preds):
                idx = label.item()
                class_total[idx] += 1
                if label.item() == pred.item():
                    class_correct[idx] += 1

    class_acc = {}

    for idx, class_name in enumerate(CLASS_NAMES):
        if class_total[idx] == 0:
            class_acc[class_name] = 0.0
        else:
            class_acc[class_name] = class_correct[idx] / class_total[idx]

    return total_loss / total_count, total_correct / total_count, class_acc


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    train_dir = os.path.join(DATASET_ROOT, "train")
    val_dir = os.path.join(DATASET_ROOT, "val")

    train_dataset = LaneCropDataset(train_dir, CLASS_NAMES)
    val_dataset = LaneCropDataset(val_dir, CLASS_NAMES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    model = LaneCropClassifier(num_classes=len(CLASS_NAMES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 20
    best_val_acc = 0.0

    print()
    print("차선 crop 분류기 학습 시작")
    print("train 개수:", len(train_dataset))
    print("val 개수:", len(val_dataset))
    print("저장 위치:", SAVE_PATH)
    print()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            num_epochs
        )

        val_loss, val_acc, class_acc = validate(
            model,
            val_loader,
            criterion
        )

        print()
        print(f"Epoch [{epoch}/{num_epochs}] 결과")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc : {train_acc:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val Acc   : {val_acc:.4f}")

        print("클래스별 정확도:")
        for class_name, acc in class_acc.items():
            print(f" - {class_name}: {acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "val_acc": best_val_acc
            }, SAVE_PATH)

            print("최고 성능 모델 저장 완료:", SAVE_PATH)

        print("-" * 60)

    print()
    print("학습 완료")
    print("최고 Val Acc:", best_val_acc)
    print("최종 저장 파일:", SAVE_PATH)


if __name__ == "__main__":
    main()